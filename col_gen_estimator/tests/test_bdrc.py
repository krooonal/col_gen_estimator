import pytest
import numpy as np

from numpy.testing import assert_array_equal
from numpy.testing import assert_equal

from col_gen_estimator import BDRMasterProblem


@pytest.fixture
def data():
    X = np.array([[1, 1, 1, 0],
                  [1, 1, 0, 1],
                  [0, 1, 1, 1],
                  [1, 0, 1, 1]], np.int8)
    y = np.array([1, 1, 0, 0], np.int8)
    return (X, y)


def test_mp_satisfies_clause():
    master_problem = BDRMasterProblem(
        C=10, p=1, optimization_problem_type='glop')
    assert_equal(master_problem.satisfies_clause(
        [0, 1, 1, 1, 0, 0, 0], [1, 2, 3]), 1)
    assert_equal(master_problem.satisfies_clause(
        [0, 1, 1, 1, 0, 0, 0], [2, 3]), 1)
    assert_equal(master_problem.satisfies_clause(
        [0, 1, 1, 1, 0, 0, 0], [0, 2, 3]), 0)
    assert_equal(master_problem.satisfies_clause(
        [0, 1, 1, 1, 0, 0, 0], [1, 2, 4]), 0)


def test_generate_lexicographic_clause():
    master_problem = BDRMasterProblem(
        C=10, p=1, optimization_problem_type='glop')
    assert_array_equal(master_problem.generate_lexicographic_clause(0), [])
    assert_array_equal(master_problem.generate_lexicographic_clause(5), [0, 2])
    assert_array_equal(master_problem.generate_lexicographic_clause(6), [1, 2])


def test_get_objective_coeff_mp(data):
    # Test how many zero examples satisfies the clause
    master_problem = BDRMasterProblem(
        C=10, p=1, optimization_problem_type='glop')
    master_problem.generate_mp(data[0], data[1])
    assert_equal(master_problem.get_objective_coeff_mp(clause=[0, 3]), 1)
    assert_equal(master_problem.get_objective_coeff_mp(clause=[2]), 2)
    assert_equal(master_problem.get_objective_coeff_mp(clause=[1, 2]), 1)
    assert_equal(master_problem.get_objective_coeff_mp(clause=[]), 2)
    assert_equal(master_problem.get_objective_coeff_mp(clause=[1, 3]), 1)
    assert_equal(master_problem.get_objective_coeff_mp(clause=[0, 1]), 0)


def test_get_clause_coeffs(data):
    # [positive examples] Coeff = 1 if example satisfies clause,
    # [complexity] Coeff = size of clause +1
    master_problem = BDRMasterProblem(
        C=10, p=1, optimization_problem_type='glop')
    assert_equal(master_problem.generated_, False)
    master_problem.generate_mp(data[0], data[1])
    assert_equal(master_problem.generated_, True)
    assert_array_equal(master_problem.get_clause_coeffs(
        clause=[0, 3]), [0, 1, 3])
    assert_array_equal(master_problem.get_clause_coeffs(clause=[2]), [1, 0, 2])
    assert_array_equal(master_problem.get_clause_coeffs(
        clause=[0, 1]), [1, 1, 3])


def test_duals(data):
    master_problem = BDRMasterProblem(
        C=10, p=1, optimization_problem_type='glop')
    master_problem.generate_mp(data[0], data[1])
    cc_dual, cs_duals = master_problem.solve_rmp()
    assert_equal(cc_dual, 0)
    assert_array_equal(cs_duals, [1, 1])


def test_add_column(data):
    master_problem = BDRMasterProblem(
        C=10, p=1, optimization_problem_type='glop')
    master_problem.generate_mp(data[0], data[1])
    clause = [0, 1]  # Clause that is only satisfied by positive examples.
    assert_equal(master_problem.add_column(clause), True)
    solved = master_problem.solve_ip()
    assert_equal(solved, True)
    assert_array_equal(master_problem.explanation, [[0, 1]])


def test_add_empty_column(data):
    master_problem = BDRMasterProblem(
        C=10, p=1, optimization_problem_type='glop')
    master_problem.generate_mp(data[0], data[1])
    clause = []
    assert_equal(master_problem.add_column(clause), False)


def test_regenerate_mp(data):
    # This test should cover the 'if' check at the start of 'generate_mp'
    # method.
    master_problem = BDRMasterProblem(
        C=10, p=1, optimization_problem_type='glop')
    assert_equal(master_problem.generated_, False)
    master_problem.generate_mp(data[0], data[1])
    assert_equal(master_problem.generated_, True)
    master_problem.generate_mp(data[0], data[1])
    assert_equal(master_problem.generated_, True)
