import pandas as pd

from gbwind.live import cycle_key, due_cycle, publish_frame
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


def test_due_cycle_is_the_latest_six_hourly_issue_after_the_lag():
    assert due_cycle(pd.Timestamp("2026-07-01T14:00")) == pd.Timestamp("2026-07-01T06:00")
    assert due_cycle(pd.Timestamp("2026-07-01T07:30")) == pd.Timestamp("2026-06-30T18:00")
    aware = pd.Timestamp("2026-07-01T14:00", tz="UTC")
    assert due_cycle(aware) == pd.Timestamp("2026-07-01T06:00")
