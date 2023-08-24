import pytest

from sklearn.utils.estimator_checks import check_estimator

from col_gen_estimator import BooleanDecisionRuleClassifier
from col_gen_estimator import BooleanDecisionRuleClassifierWithHeuristic
from col_gen_estimator import DTreeClassifier


@pytest.mark.parametrize(
    "estimator",
    [BooleanDecisionRuleClassifier(),
     BooleanDecisionRuleClassifierWithHeuristic(),
     DTreeClassifier()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
