import pytest

from sklearn.utils.estimator_checks import check_estimator

from col_gen_estimator import TemplateEstimator
from col_gen_estimator import TemplateClassifier
from col_gen_estimator import TemplateTransformer


@pytest.mark.parametrize(
    "estimator",
    [TemplateEstimator(), TemplateTransformer(), TemplateClassifier()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
