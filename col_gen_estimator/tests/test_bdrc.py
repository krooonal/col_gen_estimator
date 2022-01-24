import pytest
import numpy as np

from sklearn.datasets import load_iris
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose
from numpy.testing import assert_equal

from col_gen_estimator import BooleanDecisionRuleClassifier
from col_gen_estimator import BDRMasterProblem
from col_gen_estimator import BDRSubProblem



@pytest.fixture
def data():
    X = np.array(   [[1, 1, 1, 0], 
                    [1, 1, 0, 1],
                    [0, 1, 1, 1],
                    [1, 0, 1, 1]], np.int8)
    y = np.array([1, 1, 0, 0], np.int8)
    return (X,y)

def test_default_params():
    clf = BooleanDecisionRuleClassifier()
    assert clf.max_iterations == -1
    assert clf.C == 10
    assert clf.p == 1
    assert clf.rmp_solver_params == ""
    assert clf.master_ip_solver_params == ""
    assert clf.subproblem_params == ""
    assert clf.rmp_is_ip == True



def test_fit_predict(data):
    clf = BooleanDecisionRuleClassifier(max_iterations=3,C=10,p=1)
    clf.fit(data[0], data[1])
    assert hasattr(clf, 'is_fitted_')

    assert hasattr(clf, 'classes_')
    assert hasattr(clf, 'X_')
    assert hasattr(clf, 'y_')

    X = data[0]
    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)
    assert_array_equal(y_pred, data[1])

def test_mp_satisfies_clause():
    master_problem = BDRMasterProblem(C=10,p=1,optimization_problem_type='glop')
    assert_equal(master_problem.satisfies_clause([0,1,1,1,0,0,0], [1,2,3]), 1)
    assert_equal(master_problem.satisfies_clause([0,1,1,1,0,0,0], [2,3]), 1)
    assert_equal(master_problem.satisfies_clause([0,1,1,1,0,0,0], [0,2,3]), 0)
    assert_equal(master_problem.satisfies_clause([0,1,1,1,0,0,0], [1,2,4]), 0)

def test_generate_lexicographic_clause():
    master_problem = BDRMasterProblem(C=10,p=1,optimization_problem_type='glop')
    assert_array_equal(master_problem.generate_lexicographic_clause(0), [])
    assert_array_equal(master_problem.generate_lexicographic_clause(5), [0,2])
    assert_array_equal(master_problem.generate_lexicographic_clause(6), [1,2])

def test_get_objective_coeff_mp(data):
    # Test how many zero examples satisfies the clause
    master_problem = BDRMasterProblem(C=10,p=1,optimization_problem_type='glop')
    master_problem.generate_mp(data[0], data[1])
    assert_equal(master_problem.get_objective_coeff_mp( clause=[0,3]), 1)
    assert_equal(master_problem.get_objective_coeff_mp( clause=[2]), 2)
    assert_equal(master_problem.get_objective_coeff_mp( clause=[1,2]), 1)
    assert_equal(master_problem.get_objective_coeff_mp( clause=[]), 2)
    assert_equal(master_problem.get_objective_coeff_mp( clause=[1,3]), 1)
    assert_equal(master_problem.get_objective_coeff_mp( clause=[0,1]), 0)

def test_get_clause_coeffs(data):
    # [positive examples] Coeff = 1 if example satisfies clause, [complexity] Coeff = |clause| +1
    master_problem = BDRMasterProblem(C=10,p=1,optimization_problem_type='glop')
    master_problem.generate_mp(data[0], data[1])
    assert_array_equal(master_problem.get_clause_coeffs(clause=[0,3]), [0,1,3])
    assert_array_equal(master_problem.get_clause_coeffs(clause=[2]), [1,0,2])
    assert_array_equal(master_problem.get_clause_coeffs(clause=[0,1]), [1,1,3])

def test_duals(data):
    master_problem = BDRMasterProblem(C=10,p=1,optimization_problem_type='glop')
    master_problem.generate_mp(data[0], data[1])
    cc_dual, cs_duals = master_problem.solve_rmp()
    assert_equal(cc_dual, 0)
    assert_array_equal(cs_duals, [1, 1])

def test_add_column(data):
    master_problem = BDRMasterProblem(C=10,p=1,optimization_problem_type='glop')
    master_problem.generate_mp(data[0], data[1])
    clause = [0,1] # Clause that is only satisfied by positive examples.
    master_problem.add_column(clause)
    solved = master_problem.solve_ip()
    assert_equal(solved, True)
    assert_array_equal(master_problem.explanation, [[0,1]])

def test_sp_generate_column(data):
    subproblem = BDRSubProblem(D=10,optimization_problem_type='cbc')
    clauses = subproblem.generate_columns(data[0],data[1],dual_costs=(0,[1,1]))
    assert_array_equal(clauses, [[0,1]])

