"""
Binary classifier using boolean decision rule generation method.
TODO: Cite the paper and explain the algorithm.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from col_gen_classifier import ColGenClassifier, BaseMasterProblem, BaseSubProblem

class BDRMasterProblem(BaseMasterProblem):
    """ TODO
    """

    def __init__(self, C, P):
        """ TODO
        """
        self.C = C
        self.P = P
    
    def generate_mp(self,X,y,params):
        """ Generates the master problem model (RMP) and initializes the primal and dual solutions.
        """
        pass

    def add_column(self,column):
        """ Adds the given column to the master problem model.
        """
        pass

    def solve_rmp(self,solver_params):
        """ Solves the RMP with given solver params.
        """
        pass

    def solve_ip(self,solver_params):
        """ Solves the integer RMP with given solver params.
        """
        pass

class BDRSubProblem():
    """ Base class for subproblem. One needs to extend this for using with ColGenClassifier.
    """

    def generate_columns(self, X, y, dual_costs, params):
        """ Generates the new columns to be added to the RMP.
        """
        pass

class BooleanDecisionRuleClassifier(ColGenClassifier):
    """ Binary classifier using boolean decision rule generation method.

    Parameters
    ----------
    max_iterations : int, default='-1'
        Maximum column generation iterations. Negative values removes the iteration limit and the problem
        is solved till optimality.
    TODO: Documentation for C,P
    

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """
    def __init__(self, max_iterations='-1', C=10, P=10):
        super(BooleanDecisionRuleClassifier, self).__init__(max_iterations, 
            BDRMasterProblem(C, P), 
            BDRSubProblem(C))

    def predict(self, X):
        """ Predicts the class based on the solution of master problem. This method is abstract.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample.
        """
        pass

