"""
Base Column Genaration Classifier. One needs to extend this for implementing
column generation based classifiers.
"""

from time import time

from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelEncoder

from ._parameter_tuner import Parameter


class BaseMasterProblem():
    """ Base class for master problem. One needs to extend this for using with
    ColGenClassifier.
    Parameters
    ----------
    solver_str : string, default='glop'
        Describes the solver used for solving the masterproblem.

    Attributes
    ----------
    solver_ : MPSolver from OR-Tools,
        The solver used for solving the masterproblem.
    """

    def __init__(self, solver_str='glop'):
        self.solver_ = pywraplp.Solver.CreateSolver(solver_str)
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

    def rmp_objective_improved(self):
        """ (Optional) Returns True if objective value of rmp is improved. 
        This is used for computing the number of improving iterations.
        """
        return False


class BaseSubproblem():
    """ Base class for subproblem. One needs to extend this for using with
    ColGenClassifier.
    Parameters
    ----------
    solver_str : string, default='cbc'
        Describes the solver used for solving the subproblem. This is only used
        when MPSolver interface is used for solving subproblem.

    Attributes
    ----------
    solver_ : MPSolver from OR-Tools,
        The solver used for solving the subproblem.
    cp_solver_ : CpSolver from OR-Tools,
        Sometimes we solve subproblems using OR-Tools CP-SAT solver instead
        of MPSolver.
    tunable_params_ : dict (string->Parameter)
        Dictionary of all tunable solver parameters.
    solve_times_ : list(float)
        Wall times for solving instances. Only filled when CP-SAT solver is
        used for solving subproblems.
    gaps_ : list(float)
        Relative gaps after solving instances. Only filled when CP-SAT solver
        is used for solving subproblems.
    """

    def __init__(self, solver_str='cbc') -> None:
        self.cp_solver_ = cp_model.CpSolver()
        self.solver_ = pywraplp.Solver.CreateSolver(solver_str)
        self.tunable_params_ = BaseSubproblem.get_tunable_params()
        self.solve_times_ = []
        self.gaps_ = []

    @staticmethod
    def get_tunable_params():
        """ Returns the list of all parameters we tune in the cp_solver_"""
        params = {}
        # interleave_search = Parameter(0.3, 'interleave_search', 15189)
        # interleave_search.add_value(False)
        # interleave_search.add_value(True)
        # interleave_search.set_switch_flag(1)
        # params['interleave_search'] = interleave_search

        presolve_iterations = Parameter(0.3, 'max_presolve_iterations', 1564)
        presolve_iterations.add_value(3)
        presolve_iterations.add_value(4)
        presolve_iterations.add_value(2)
        presolve_iterations.set_explore_count(15)
        params['max_presolve_iterations'] = presolve_iterations

        probing_level = Parameter(0.3, 'cp_model_probing_level', 987234)
        probing_level.add_value(2)
        probing_level.add_value(1)
        probing_level.set_switch_flag(1 << 1)
        params['cp_model_probing_level'] = probing_level

        lns_workers = Parameter(0.3, 'min_num_lns_workers', 25646)
        lns_workers.add_value(2)
        lns_workers.add_value(0)
        lns_workers.set_switch_flag(1 << 2)
        params['min_num_lns_workers'] = lns_workers
        return params

    def generate_columns(self, X, y, dual_costs, params):
        """ Generates the new columns to be added to the RMP.
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The input.
        y : ndarray, shape (n_samples,)
            The labels.
        dual_costs:
            The dual costs and other information needed to update the
            subproblem.
        params:
            The parameters for the subproblem. (TODO: Remove this.)
        """
        pass

    def update_subproblem(self, X, y, dual_costs):
        """ (Optional) Updates the subproblem model. This is useful when we add
        more constraints to the master problem. The subproblem needs to be
        updated to handle new dual cost information.
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The input.
        y : ndarray, shape (n_samples,)
            The labels.
        dual_costs:
            The dual costs and other information needed to update the
            subproblem.
        """
        return

    def solve_model(self, model, time_limit=30, callback=None):
        """ Solves the given model. This method reuses the information
        generated from previous solves of the subproblem to tune the solver
        parameters. As of now it only tunes parameters of the CP solver.
        Parameters
        ----------
        model: cp_model.CpModel
            The model to solve.
        time_limit: int
            Time limit to solve the model in seconds.
        callback: cp_model.CpSolverSolutionCallback
            Optional callback to capture intermediate solutions.
        """
        if self.cp_solver_ == None:
            return

        self.cp_solver_.parameters.num_search_workers = 7
        self.cp_solver_.parameters.max_time_in_seconds = time_limit

        # self.cp_solver_.parameters.interleave_search = self.tunable_params_[
        #     'interleave_search'].get_best_value()
        self.cp_solver_.parameters.min_num_lns_workers = self.tunable_params_[
            'min_num_lns_workers'].get_best_value()
        self.cp_solver_.parameters.cp_model_probing_level = self.tunable_params_[
            'cp_model_probing_level'].get_best_value()
        self.cp_solver_.parameters.max_presolve_iterations = self.tunable_params_[
            'max_presolve_iterations'].get_best_value()

        # for key in self.tunable_params_:
        #     self.tunable_params_[key].print_stats()

        status = self.cp_solver_.Solve(model, callback)

        time_to_solve = self.cp_solver_.WallTime()
        time_score = time_to_solve / time_limit
        gap_score = 1
        if status == cp_model.OPTIMAL:
            gap_score = 0
        elif status == cp_model.FEASIBLE:
            objective = self.cp_solver_.ObjectiveValue()
            bound = self.cp_solver_.BestObjectiveBound()
            gap = abs(objective - bound)
            time_score = 1
            if objective * bound >= 0:
                gap_score = gap / max(abs(objective), abs(bound))
        else:
            time_score = 1
            gap_score = 1

        total_score = time_score + gap_score
        # print("Time to solve: ", time_to_solve)
        # print("Time score: ", time_score)
        # print("Gap score: ", gap_score)
        # print("Total score: ", total_score)
        self.solve_times_.append(time_to_solve)
        self.gaps_.append(gap_score)

        # self.tunable_params_['interleave_search'].adjust_score(-total_score)
        # self.tunable_params_['min_num_lns_workers'].adjust_score(-total_score)
        # self.tunable_params_[
        #     'cp_model_probing_level'].adjust_score(-total_score)
        # self.tunable_params_[
        #     'max_presolve_iterations'].adjust_score(-total_score)

        for key in self.tunable_params_:
            self.tunable_params_[key].adjust_score(-total_score)
            # self.tunable_params_[key].print_stats()

        return


class ColGenClassifier(ClassifierMixin, BaseEstimator):
    """ Base Column Genaration Classifier. One needs to extend this for
    implementing column generation based classifiers.

    Parameters
    ----------
    max_iterations : int, default=-1
        Maximum column generation iterations. Negative values removes the
        iteration limit.
    time_limit : int, default=-1
        Time limit in seconds. Negative values removes the time limit.
    master_problem : Instance of BaseMasterProblem, default = BaseSubproblem()
    subproblems : List of list of instances of BaseSubproblem according to
        their levels. The subproblems are called in increasing values of their
        levels. The list of subproblems at level i+1 is given by
        subproblems[i+1]. The subproblems[i+1] are called only if the
        subproblems[i] fail to generate any columns. Typically, the subproblem
        heuristics are passed at lower levels and the exact methods are passed
        at the higher levels. Default = [[BaseSubproblem()]]
    rmp_is_ip : boolean, default = False
        True if the master problem has integer variables.
    rmp_solver_params: string, default = "",
        Solver parameters for solving restricted master problem (rmp).
    master_ip_solver_params: string, default = "",
        Solver parameters for solving the integer master problem.
    subproblem_params: list of list of string, default = [""],
        Parameters for solving the subproblems (as per their levels).

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
                 time_limit=-1,
                 master_problem=BaseMasterProblem(),
                 subproblems=[[BaseSubproblem()]],
                 rmp_is_ip=False,
                 rmp_solver_params="",
                 master_ip_solver_params="",
                 subproblem_params=[[""]]):
        self.max_iterations = max_iterations
        self.time_limit = time_limit
        self.master_problem = master_problem
        self.subproblems = subproblems
        self.rmp_is_ip = rmp_is_ip
        self.rmp_solver_params = rmp_solver_params
        self.master_ip_solver_params = master_ip_solver_params
        self.subproblem_params = subproblem_params
        assert len(subproblems) == len(subproblem_params)
        for level in range(len(subproblems)):
            assert len(subproblems[level]) == len(subproblem_params[level])

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
        t_start = time()
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        has_time_limit = self.time_limit > 0
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
        self.num_improving_iter_ = 0
        self.mp_optimal_ = False
        self.time_limit_reached_ = False
        self.num_col_added_sp_ = []
        self.time_spent_sp_ = []
        self.time_add_col_ = 0.0
        self.time_spent_master_ = 0.0
        for level in range(len(self.subproblems)):
            self.num_col_added_sp_.append([0]*len(self.subproblems[level]))
            self.time_spent_sp_.append([0.0]*len(self.subproblems[level]))

        self.iter_ = 0
        while True:
            if self.max_iterations > 0 and self.iter_ >= self.max_iterations:
                break
            self.iter_ += 1
            print("Iteration number: ", self.iter_)
            print("Time elapsed: ", time() - t_start)
            master_time_start = time()
            dual_costs = self.master_problem.solve_rmp(self.rmp_solver_params)
            self.time_spent_master_ += time() - master_time_start
            if self.master_problem.rmp_objective_improved():
                self.num_improving_iter_ += 1

            # TODO:Remove this.
            if hasattr(self.master_problem, 'reset_timer_'):
                if self.master_problem.reset_timer_:
                    t_start = time()
                    self.master_problem.reset_timer_ = False
                    print("Reset timer.")

            rmp_updated = False
            # Update the subproblems
            for sp_level in range(len(self.subproblems)):
                for sp_ind in range(len(self.subproblems[sp_level])):
                    sp_time_start = time()
                    self.subproblems[sp_level][sp_ind] \
                        .update_subproblem(
                            X, self.processed_y_, dual_costs)
                    sp_time_end = time()
                    self.time_spent_sp_[
                        sp_level][sp_ind] += sp_time_end - sp_time_start

            for sp_level in range(len(self.subproblems)):
                # TODO: do this in parallel.
                rmp_updated = False
                for sp_ind in range(len(self.subproblems[sp_level])):
                    self.time_elapsed_ = time() - t_start
                    if has_time_limit and self.time_elapsed_ > self.time_limit:
                        self.time_limit_reached_ = True
                        break
                    sp_time_start = time()
                    generated_columns = self.subproblems[sp_level][sp_ind] \
                        .generate_columns(
                        X, self.processed_y_, dual_costs,
                        self.subproblem_params[sp_level][sp_ind])
                    sp_time_end = time()
                    self.time_spent_sp_[
                        sp_level][sp_ind] += sp_time_end - sp_time_start

                    col_add_start = time()
                    for column in generated_columns:
                        col_added = self.master_problem.add_column(column)
                        self.num_col_added_sp_[
                            sp_level][sp_ind] += 1 if col_added else 0
                        rmp_updated = rmp_updated or col_added
                    self.time_add_col_ += time() - col_add_start
                    if rmp_updated:
                        break
                self.time_elapsed_ = time() - t_start
                if self.time_limit_reached_:
                    print("Time limit reached!")
                    break
                if rmp_updated:
                    break

            if not rmp_updated:
                print("RMP not updated. exiting the loop.")
                if not self.time_limit_reached_:
                    self.mp_optimal_ = True
                break

        self.time_spent_master_ip_ = 0.0
        if self.rmp_is_ip:
            master_ip_start_time = time()
            solved = self.master_problem.solve_ip(self.master_ip_solver_params)
            self.time_spent_master_ip_ += time() - master_ip_start_time
            assert solved, "RMP integer program couldn't be solved."

        self.is_fitted_ = True
        self.time_elapsed_ = time() - t_start
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
