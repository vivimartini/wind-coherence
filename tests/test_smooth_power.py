import numpy as np

from gbwind.power import capacity_factor, curve_wind_speed, farm_power_mw, smooth_capacity_factor


def test_smooth_curve_is_near_zero_in_calm_air_and_near_one_when_rated():
    assert smooth_capacity_factor(0.0) < 0.05
    assert smooth_capacity_factor(20.0) > 0.95


def test_smooth_curve_has_slope_where_the_exact_curve_is_flat():
    exact = capacity_factor(np.array([2.9, 3.1]))
    assert exact[0] == 0.0
    calm = smooth_capacity_factor(np.array([2.9, 3.1]))
    assert calm[1] > calm[0]


def test_smooth_curve_rises_through_the_cubic_region():
    mid = smooth_capacity_factor(np.array([7.0, 7.2]))
    assert mid[1] > mid[0]


def test_100m_wind_is_the_curve_input_even_when_the_hub_is_higher():
    assert np.isclose(curve_wind_speed(10.0, 0.0, 2.0, 0.0, 150.0), 10.0)


def test_missing_100m_wind_is_extrapolated_from_10m():
    ws = curve_wind_speed(np.nan, np.nan, 8.0, 0.0, 100.0)
    assert np.isclose(ws, 8.0 * 10 ** (1 / 7))


def test_a_farm_without_fitted_parameters_keeps_the_idealised_curve():
    fitted = farm_power_mw(20.0, 0.0, 5.0, 0.0, 80.0, 100.0, cut_in=3.0, rated=12.5, sharpness=6.0)
    plain = farm_power_mw(20.0, 0.0, 5.0, 0.0, 80.0, 100.0, cut_in=np.nan, rated=np.nan, sharpness=np.nan)
    assert fitted > 90
    assert np.isclose(plain, capacity_factor(20.0) * 100)
