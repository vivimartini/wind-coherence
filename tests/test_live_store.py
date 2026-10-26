import pandas as pd

from gbwind.live import cycle_key, ingest_pending, publish_frame
from gbwind.object_store import LocalStore, S3Store


def test_publishing_the_same_cycle_twice_is_idempotent(tmp_path):
    store = LocalStore(tmp_path)
    frame = pd.DataFrame({"farm": ["a"], "member": [0], "mw": [1.5]})
    cycle = pd.Timestamp("2026-07-01T00")
    assert publish_frame(store, "aifs-ens", cycle, frame) == "written"
    assert publish_frame(store, "aifs-ens", cycle, frame) == "unchanged"
    files = list(tmp_path.rglob("*.parquet"))
    assert len(files) == 1
    assert not list(tmp_path.rglob("*.partial"))


def test_a_changed_cycle_replaces_the_object(tmp_path):
    store = LocalStore(tmp_path)
    cycle = pd.Timestamp("2026-07-01T00")
    publish_frame(store, "ifs-ens", cycle, pd.DataFrame({"mw": [1.0]}))
    assert publish_frame(store, "ifs-ens", cycle, pd.DataFrame({"mw": [2.0]})) == "written"
    assert len(list(tmp_path.rglob("*.parquet"))) == 1


def test_keys_cannot_escape_the_archive_root(tmp_path):
    store = LocalStore(tmp_path)
    try:
        store.put_bytes("../outside.txt", b"no")
    except ValueError:
        return
    raise AssertionError("escaped key was accepted")


def test_s3_put_is_idempotent_and_leaves_no_partial():
    class Memory:
        def __init__(self):
            self.objects = {}

        def head_object(self, Bucket, Key):
            if Key not in self.objects:
                raise KeyError(Key)
            return {"Metadata": self.objects[Key]["meta"]}

        def put_object(self, Bucket, Key, Body, Metadata):
            self.objects[Key] = {"body": Body, "meta": Metadata}

        def copy_object(self, Bucket, Key, CopySource, Metadata, MetadataDirective):
            src = self.objects[CopySource["Key"]]["body"]
            self.objects[Key] = {"body": src, "meta": Metadata}

        def delete_object(self, Bucket, Key):
            del self.objects[Key]

    client = Memory()
    store = S3Store("bucket", "prefix", client=client)
    assert store.put_bytes("archive/a.parquet", b"abc") == "written"
    assert store.put_bytes("archive/a.parquet", b"abc") == "unchanged"
    assert "prefix/archive/a.parquet" in client.objects
    assert not any(key.endswith(".partial") for key in client.objects)


def test_cycle_key_is_the_source_and_initialisation():
    assert cycle_key("aifs-ens", pd.Timestamp("2026-07-01T00")) == "archive/source=aifs-ens/init=2026-07-01T00.parquet"


def _frame(value=1.0):
    return pd.DataFrame({"farm": ["a", "b"], "member": [0, 0], "wind_u_100m": [value, value]})


def test_the_newest_source_cycle_is_ingested_even_when_the_clock_says_a_later_one(tmp_path):
    store = LocalStore(tmp_path)
    available = pd.date_range("2026-10-25T00", "2026-10-26T00", freq="6h")
    now = pd.Timestamp("2026-10-26T16:20", tz="UTC")
    events = ingest_pending(store, "aifs-ens", available, lambda cycle: _frame(), now)
    keys = [e["key"] for e in events]
    assert keys[-1] == cycle_key("aifs-ens", pd.Timestamp("2026-10-26T00"))
    assert all(e["status"] == "written" for e in events)
    assert not any("2026-10-26T06" in k for k in keys)


def test_a_second_run_with_nothing_new_is_a_normal_outcome(tmp_path):
    store = LocalStore(tmp_path)
    available = [pd.Timestamp("2026-10-26T00")]
    now = pd.Timestamp("2026-10-26T16:20")
    ingest_pending(store, "ifs-ens", available, lambda cycle: _frame(), now)
    again = ingest_pending(store, "ifs-ens", available, lambda cycle: _frame(), now)
    assert again == [{"event": "ingest", "source": "ifs-ens", "status": "no new cycle"}]
    assert len(list(store.root.rglob("*.parquet"))) == 1


def test_a_missed_run_is_caught_up_and_an_incomplete_cycle_is_left_for_later(tmp_path):
    store = LocalStore(tmp_path)
    available = [pd.Timestamp("2026-10-25T12"), pd.Timestamp("2026-10-25T18"), pd.Timestamp("2026-10-26T00")]
    now = pd.Timestamp("2026-10-26T10:00")

    def extract(cycle):
        return _frame(float("nan")) if cycle == pd.Timestamp("2026-10-26T00") else _frame()

    events = ingest_pending(store, "aifs-ens", available, extract, now)
    assert [e["status"] for e in events] == ["written", "written", "incomplete, retried next run"]
    later = ingest_pending(store, "aifs-ens", available, lambda cycle: _frame(), now)
    assert [e["key"] for e in later] == [cycle_key("aifs-ens", pd.Timestamp("2026-10-26T00"))]


def test_cycles_older_than_the_lookback_are_not_backfilled(tmp_path):
    store = LocalStore(tmp_path)
    available = [pd.Timestamp("2026-10-20T00"), pd.Timestamp("2026-10-26T00")]
    events = ingest_pending(store, "aifs-ens", available, lambda cycle: _frame(), pd.Timestamp("2026-10-26T12"))
    assert [e["key"] for e in events] == [cycle_key("aifs-ens", pd.Timestamp("2026-10-26T00"))]

