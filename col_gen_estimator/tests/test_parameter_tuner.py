import pytest
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal

from col_gen_estimator import RunningStat
from col_gen_estimator import Parameter


def test_running_stat():
    stat = RunningStat()
    assert_equal(stat.num_data_values(), 0)
    assert_equal(stat.mean(), 0.0)
    assert_equal(stat.variance(), 0.0)
    assert_equal(stat.standard_deviation(), 0.0)

    stat.push(5)
    assert_equal(stat.num_data_values(), 1)
    assert_equal(stat.mean(), 5.0)
    assert_equal(stat.variance(), 0.0)
    assert_equal(stat.standard_deviation(), 0.0)

    stat.push(10)
    assert_equal(stat.num_data_values(), 2)
    assert_almost_equal(stat.mean(), 7.5)
    assert_almost_equal(stat.variance(), 12.5)
    assert_almost_equal(stat.standard_deviation(), np.sqrt(12.5))


def test_parameter():
    parameter = Parameter(0.3, "Test", seed=42)

    parameter.add_value(1)
    parameter.add_value(2)
    parameter.set_switch_flag(1)

    assert_equal(parameter.get_best_value(), 1)
    assert_equal(parameter.get_best_value(), 2)

    parameter.adjust_score(10)
    assert_equal(parameter.get_best_value(), 1)

    parameter.adjust_score(15)
    assert_equal(parameter.get_best_value(), 2)


def test_parameter2():
    parameter = Parameter(0.3, "Test", seed=42)

    parameter.add_value(1)
    parameter.add_value(2)
    parameter.set_switch_flag(1)
    parameter.set_explore_count(10)

    for i in range(5):
        assert_equal(parameter.get_best_value(), 1)
        parameter.adjust_score(0.10)
        assert_equal(parameter.get_best_value(), 2)
        parameter.adjust_score(0.15)

    assert_equal(parameter.get_best_value(), 2)
