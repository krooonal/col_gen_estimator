"""
Base Column Genaration Classifier. One needs to extend this for implementing
column generation based classifiers.
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelEncoder


class BaseMasterProblem():
    """ Base class for master problem. One needs to extend this for using with
    ColGenClassifier.
    """

    def __init__(self):
        pass

    def generate_mp(self, X, y):
        """ Generates the master problem model (RMP) and initializes the primal
        and dual solutions.
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The input.
        y : ndarray, shape (n_samples,)
            The labels.
        """
        pass

    def add_column(self, column):
        """ Adds the given column to the master problem model.
        Parameters
        ----------
        column : object,
            The column to be added to the RMP.
        """
        pass

    def solve_rmp(self, solver_params=''):
        """ Solves the RMP with given solver params.
        Returns the dual costs.
        Parameters
        ----------
        solver_params : string, default='',
            The solver parameters for solving the RMP.
        """
        pass

    def solve_ip(self, solver_params=''):
        """ Solves the integer RMP with given solver params.
        Returns false if the problem is not solved.
        Parameters
        ----------
        solver_params : string, default='',
            The solver parameters for solving the integer RMP.
        """
        pass


class BaseSubproblem():
    """ Base class for subproblem. One needs to extend this for using with
    ColGenClassifier.
    """

    def generate_columns(self, X, y, dual_costs, params):
        """ Generates the new columns to be added to the RMP.
        """
        pass


class ColGenClassifier(ClassifierMixin, BaseEstimator):
    """ Base Column Genaration Classifier. One needs to extend this for
    implementing column generation based classifiers.

    Parameters
    ----------
    max_iterations : int, default=-1
        Maximum column generation iterations. Negative values removes the
        iteration limit and the problem is solved till optimality.
    master_problem : Instance of BaseMasterProblem, default = BaseSubproblem()
    subproblem : Instance of BaseSubproblem, default = BaseSubproblem()
    rmp_is_ip : boolean, default = False
        True if the master problem has integer variables.
    rmp_solver_params: string, default = "",
        Solver parameters for solving restricted master problem (rmp).
    master_ip_solver_params: string, default = "",
        Solver parameters for solving the integer master problem.
    subproblem_params: list of string, default = [""],
        Parameters for solving the subproblem.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    performed_iter_ : int
        Total number of iterations performed in column generation loop.
    mp_optimal: bool
        Set to true if none of the subproblems could generate any column in an
        iteration.
    num_col_added_sp_: list of ints (size = number of subproblems)
        Count of number of columns added to the master problem by each
        subproblem.
    """

    def __init__(self, max_iterations=-1,
                 master_problem=BaseMasterProblem(),
                 subproblems=[BaseSubproblem()],
                 rmp_is_ip=False,
                 rmp_solver_params="",
                 master_ip_solver_params="",
                 subproblem_params=[""]):
        self.max_iterations = max_iterations
        self.master_problem = master_problem
        self.subproblems = subproblems
        self.rmp_is_ip = rmp_is_ip
        self.rmp_solver_params = rmp_solver_params
        self.master_ip_solver_params = master_ip_solver_params
        self.subproblem_params = subproblem_params
        assert len(subproblems) == len(subproblem_params)

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
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        # Convert the y into integer class lables.
        self.label_encoder_ = LabelEncoder()
        self.processed_y_ = self.label_encoder_.fit_transform(y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        if len(self.classes_) <= 1:
            raise ValueError(
                "Classifier can't train when only one class is present.")

        # Initiate the master and subproblems
        self.master_problem.generate_mp(X, self.processed_y_)

        self.performed_iter_ = 0
        self.mp_optimal_ = False
        self.num_col_added_sp_ = [0]*len(self.subproblems)

        for iter in range(self.max_iterations):
            self.performed_iter_ += 1
            dual_costs = self.master_problem.solve_rmp(self.rmp_solver_params)

            rmp_updated = False
            sp_ind = 0
            for sp_ind in range(len(self.subproblems)):
                generated_columns = self.subproblems[sp_ind].generate_columns(
                    X, self.processed_y_, dual_costs,
                    self.subproblem_params[sp_ind])

                rmp_updated = False
                for column in generated_columns:
                    col_added = self.master_problem.add_column(column)
                    self.num_col_added_sp_[sp_ind] += 1 if col_added else 0
                    rmp_updated = rmp_updated or col_added
                if rmp_updated:
                    break

            if not rmp_updated:
                print("RMP not updated. exiting the loop.")
                self.mp_optimal_ = True
                break

        if self.rmp_is_ip:
            solved = self.master_problem.solve_ip(self.master_ip_solver_params)
            assert solved, "RMP integer program couldn't be solved."

        self.is_fitted_ = True
        # Return the classifier
        return self

    def predict(self, X):
        """ Predicts the class based on the solution of master problem. This
        method is abstract.

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
