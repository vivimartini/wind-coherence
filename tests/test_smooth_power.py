import numpy as np

from gbwind.power import capacity_factor, smooth_capacity_factor


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
