"""
Base Column Genaration Classifier. One needs to extend this for implementing column generation based 
classifiers.
"""

from sklearn.base import BaseEstimator, ClassifierMixin

class BaseMasterProblem():
    """ Base class for master problem. One needs to extend this for using with ColGenClassifier.
    """

    def __init__(self):
        pass
    
    def generate_mp(self,X,y):
        """ Generates the master problem model (RMP) and initializes the primal and dual solutions.
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The input.
        y : ndarray, shape (n_samples,)
            The labels.
        """
        pass

    def add_column(self,column):
        """ Adds the given column to the master problem model.
        Parameters
        ----------
        column : object,
            The column to be added to the RMP.
        """
        pass

    def solve_rmp(self,solver_params=''):
        """ Solves the RMP with given solver params.
        Returns the dual costs.
        Parameters
        ----------
        solver_params : string, default='',
            The solver parameters for solving the RMP.
        """
        pass

    def solve_ip(self,solver_params=''):
        """ Solves the integer RMP with given solver params. 
        Returns false if the problem is not solved.
        Parameters
        ----------
        solver_params : string, default='',
            The solver parameters for solving the integer RMP.
        """
        pass

class BaseSubproblem():
    """ Base class for subproblem. One needs to extend this for using with ColGenClassifier.
    """

    def generate_columns(self, X, y, dual_costs, params):
        """ Generates the new columns to be added to the RMP.
        """
        pass

class ColGenClassifier(ClassifierMixin, BaseEstimator):
    """ Base Column Genaration Classifier. One needs to extend this for implementing column generation 
    based classifiers.

    Parameters
    ----------
    max_iterations : int, default=-1
        Maximum column generation iterations. Negative values removes the iteration limit and the problem
        is solved till optimality.
    master_problem : Instance of BaseMasterProblem, default = BaseSubproblem()
    subproblem : Instance of BaseSubproblem, default = BaseSubproblem()
    rmp_is_ip : boolean, default = False
        True if the master problem has integer variables.
    rmp_solver_params: string, default = "",
        Solver parameters for solving restricted master problem (rmp).
    master_ip_solver_params: string, default = "",
        Solver parameters for solving the integer master problem.
    subproblem_params: string, default = "",
        Parameters for solving the subproblem.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """
    def __init__(self, max_iterations=-1, 
            master_problem = BaseMasterProblem(), 
            subproblem = BaseSubproblem(),
            rmp_is_ip = False,
            rmp_solver_params = "", 
            master_ip_solver_params = "",
            subproblem_params = ""):
        self.max_iterations = max_iterations
        self.master_problem = master_problem
        self.subproblem = subproblem
        self.rmp_is_ip = rmp_is_ip
        self.rmp_solver_params = rmp_solver_params
        self.master_ip_solver_params = master_ip_solver_params
        self.subproblem_params = subproblem_params
        
    def fit(self, X, y):
        """Runs the column generation loop.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        return self

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
