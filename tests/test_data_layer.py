import numpy as np
import pandas as pd

from gbwind.curtailment import (
    boa_min_level,
    build_farm_sp,
    centred_hourly,
    curtailed_units,
    negative_price_flags,
    pn_level,
)
from gbwind.elexon import sp_start_utc, tidy_b1610, tidy_levels
from gbwind.power import capacity_factor, hub_wind_speed


def test_sp_start_winter_and_summer():
    s = sp_start_utc(["2024-01-10", "2024-07-10"], [1, 1])
    assert s[0] == pd.Timestamp("2024-01-10 00:00")
    assert s[1] == pd.Timestamp("2024-07-09 23:00")


def test_sp_start_clock_change_days():
    # 31 March 2024 has 46 periods, 27 October 2024 has 50.
    spring = sp_start_utc(["2024-03-31"] * 46, range(1, 47))
    assert spring.iloc[-1] == pd.Timestamp("2024-03-31 22:30")
    autumn = sp_start_utc(["2024-10-27"] * 50, range(1, 51))
    assert autumn.iloc[0] == pd.Timestamp("2024-10-26 23:00")
    assert autumn.iloc[-1] == pd.Timestamp("2024-10-27 23:30")
    assert autumn.is_unique


def test_tidy_b1610_keeps_latest_run():
    rows = [
        {"bmUnit": "T_X-1", "settlementDate": "2024-01-10", "settlementPeriod": 1, "quantity": 10.0, "settlementRunType": "II"},
        {"bmUnit": "T_X-1", "settlementDate": "2024-01-10", "settlementPeriod": 1, "quantity": 12.0, "settlementRunType": "SF"},
    ]
    df = tidy_b1610(rows)
    assert len(df) == 1 and df["mwh"].iloc[0] == 12.0


def test_tidy_levels_parses_utc():
    rows = [{"bmUnit": "T_X-1", "timeFrom": "2024-07-10T01:00:00Z", "timeTo": "2024-07-10T01:30:00Z",
             "levelFrom": 5, "levelTo": 5, "acceptanceNumber": 1, "soFlag": False}]
    df = tidy_levels(rows, "BOALF")
    assert df["time_from"].iloc[0] == pd.Timestamp("2024-07-10 01:00")


def _boa(t0, t1, lf, lt, unit="U"):
    return pd.DataFrame({"bmunit": [unit], "time_from": [pd.Timestamp(t0)], "time_to": [pd.Timestamp(t1)],
                         "level_from": [lf], "level_to": [lt]})


def _pn(levels, unit="U", start="2024-01-01"):
    t = pd.date_range(start, periods=len(levels), freq="30min")
    return pd.DataFrame({"bmunit": unit, "time_from": t, "time_to": t + pd.Timedelta(minutes=30),
                         "level_from": levels, "level_to": levels})


def test_boa_min_level_marks_touched_periods_only():
    idx = pd.date_range("2024-01-01", periods=6, freq="30min")
    lvl = boa_min_level(_boa("2024-01-01 00:40", "2024-01-01 01:10", 50.0, 40.0), idx, ["U"])
    assert lvl["U"].notna().tolist() == [False, True, True, False, False, False]
    assert lvl["U"].iloc[1] == 40.0


def test_curtailment_threshold_is_fifteen_percent_of_notification():
    idx = pd.date_range("2024-01-01", periods=3, freq="30min")
    pn = pn_level(_pn([100.0, 100.0, 0.0]), idx, ["U"])
    for level, expected in ((90.0, False), (84.0, True)):
        boa = boa_min_level(_boa("2024-01-01 00:00", "2024-01-01 00:30", level, level), idx, ["U"])
        assert bool(curtailed_units(boa, pn)["U"].iloc[0]) is expected
    # No reduction is possible from a zero notification.
    boa = boa_min_level(_boa("2024-01-01 01:00", "2024-01-01 01:30", 0.0, 0.0), idx, ["U"])
    assert not curtailed_units(boa, pn)["U"].iloc[2]


def test_acceptance_without_notification_is_flagged():
    idx = pd.date_range("2024-01-01", periods=2, freq="30min")
    boa = boa_min_level(_boa("2024-01-01 00:00", "2024-01-01 00:30", 10.0, 10.0), idx, ["U"])
    pn = pn_level(pd.DataFrame(), idx, ["U"])
    assert curtailed_units(boa, pn)["U"].tolist() == [True, False]


def test_negative_price_needs_traded_volume():
    idx = pd.date_range("2024-01-01", periods=3, freq="30min")
    mid = pd.DataFrame({"sp_start": [idx[0], idx[0], idx[1], idx[2]],
                        "provider": ["APX", "N2EX", "APX", "APX"],
                        "price": [-5.0, 0.0, 30.0, -1.0], "volume": [100.0, 0.0, 50.0, 0.0]})
    assert negative_price_flags(mid, idx).tolist() == [True, False, False]


def _toy_b1610():
    sp = pd.date_range("2024-01-01", periods=8, freq="30min")
    rows = []
    for u, q in (("U1", 10.0), ("U2", 20.0), ("V1", 5.0)):
        for s in sp:
            rows.append({"bmunit": u, "sp_start": s, "mwh": q, "run": "SF"})
    df = pd.DataFrame(rows)
    return df[~((df["bmunit"] == "U2") & (df["sp_start"] == sp[3]))]


def test_farm_aggregation_metering_curtailment_and_price():
    b = _toy_b1610()
    boalf = _boa("2024-01-01 02:00", "2024-01-01 02:20", 2.0, 2.0, unit="V1")
    pn = _pn([10.0] * 8, unit="V1")
    idx = pd.date_range("2024-01-01", periods=8, freq="30min")
    mid = pd.DataFrame({"sp_start": [idx[6]], "provider": ["APX"], "price": [-3.0], "volume": [10.0]})
    t = build_farm_sp(b, boalf, {"farmA": ["U1", "U2"], "farmB": ["V1"]}, "2024-01-01", "2024-01-01 04:00",
                      pn=pn, mid=mid)
    a = t[t["farm"] == "farmA"].reset_index(drop=True)
    assert a.loc[0, "mw"] == 60.0
    assert not a.loc[3, "metered"] and np.isnan(a.loc[3, "mw"])
    bb = t[t["farm"] == "farmB"].reset_index(drop=True)
    assert bb["curtailed"].tolist() == [False] * 4 + [True] + [False] * 3
    assert bb["any_boa"].tolist() == bb["curtailed"].tolist()
    assert a["negative_price"].tolist() == [False] * 6 + [True, False]
    vals, adm = centred_hourly(t)
    assert not adm.loc["2024-01-01 03:00"].any()


def test_centred_hourly_requires_both_periods():
    b = _toy_b1610()
    t = build_farm_sp(b, pd.DataFrame(columns=["bmunit", "time_from", "time_to"]),
                      {"farmA": ["U1", "U2"]}, "2024-01-01", "2024-01-01 04:00")
    vals, adm = centred_hourly(t)
    assert list(vals.index) == list(pd.date_range("2024-01-01 01:00", periods=3, freq="h"))
    # 02:00 needs the periods starting 01:30 (missing U2) and 02:00.
    assert not adm.loc["2024-01-01 02:00", "farmA"]
    assert adm.loc["2024-01-01 01:00", "farmA"] and vals.loc["2024-01-01 01:00", "farmA"] == 60.0


def test_effective_capacity_ignores_flagged_periods_and_evaluation_window():
    from gbwind.portfolio import effective_capacity

    sp = pd.date_range("2026-01-01", "2026-06-01", freq="30min", inclusive="left")
    mw = np.linspace(0, 100, len(sp))
    df = pd.DataFrame({"farm": "f", "sp_start": sp, "mw": mw, "metered": True,
                       "curtailed": False, "negative_price": False})
    df.loc[df["sp_start"] >= "2026-05-12", "mw"] = 1000.0
    df.loc[df.index[:10], ["mw", "negative_price"]] = [5000.0, True]
    cap = effective_capacity(df, 0.995, "2026-05-12")["effective_capacity_mw"].iloc[0]
    pre = mw[(sp < pd.Timestamp("2026-05-12"))][10:]
    assert np.isclose(cap, np.quantile(pre, 0.995))


def test_power_curve_shape():
    cf = capacity_factor(np.array([0.0, 2.9, 3.0, 12.5, 20.0, 25.0]))
    assert cf.tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    mid = capacity_factor(np.linspace(3, 12.5, 50))
    assert np.all(np.diff(mid) > 0)


def test_hub_wind_extrapolation():
    ws = hub_wind_speed(10.0, 0.0, 10.0 / 10 ** (1 / 7), 0.0, 150.0)
    assert np.isclose(ws, 10.0 * 1.5 ** (1 / 7))
    assert np.isclose(hub_wind_speed(8.0, 0.0, 0.1, 0.0, 100.0), 8.0)
