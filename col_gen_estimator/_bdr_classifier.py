"""
Binary classifier using boolean decision rule generation method.

Reference: Sanjeeb Dash, Oktay Gunluk, and Dennis Wei. Boolean decision rules
via column generation. In Advances in Neural Information Processing Systems,
volume 31. Curran Associates, Inc., 2018.

The model is designed for binary classification. The model generates a boolean
decision rule in DNF (Disjunctive Normal Form, OR-of-ANDs). An example of a DNF
rule with two clauses is "(x1 AND x2) OR (x3 AND x5 AND x7)". These rules
return True when the example belongs to the class and False otherwise. These
binary rules also serve as explanation for the output class.

Sets
P set of all positive examples in the dataset.
Z set of all negative examples in the dataset.
K set of all possible clauses.
K_i set of all possible clauses satisfied by input i.

Parameters
c_k complexity of the clause k. Typically set to 1 + length of the clause.
C complexity of the explanation.
p misclassification cost weight for false negatives.

Decision Variables
w_k binary variable indicating if the clause k is selected.
xi_i binary variable indicating if the input i does not satisfy any selected
     clauses.

z_{MIP} = min   sum_{i in P} p*xi_i + sum_{i in Z} sum_{k in K_i} w_k   (obj)
          s.t.  xi_i + sum_{k in K_i} w_k >= 1, i in P                  (1a)
                sum_{k in K} c_k*w_k >= C                               (1b)
                xi_i >= 0, i in P                                       (1c)
                w_k in {0,1}, k in K                                    (1d)

The first sum in the objective is the penalty for the misclassified positive
examples, and the second part is the penalty for the misclassified negative
examples. Constraints (1a) ensure that each positive example either satisfies
at least one of the selected clause or the associated penalty variable is set
to 1. Constraints (1b) control the overall complexity of the generated DNF
formula. Clearly, the number of clauses is exponential, so the columns for w_k
are generated through a subproblem.

The reduced cost for a column $w_k$ is given by
sum_{i in mathcal{Z}} delta_i - sum_{i in mathcal{P}} mu_i delta_i + lambda
c_k where mu_i (>= 0) are the dual costs for constraints (1a) and lambda (>= 0)
is the dual cost for constraint (1c). The binary variable delta_i is set to 1
if the ith example satisfies the clause w_k.

Sets
P  set of all positive examples in the dataset.
Z  set of all negative examples in the dataset.
J  set of all the features.
S_i  the zero-valued features in sample i, S_i = {j in J: x_{ij} = 0}.

Parameters
D maximum complexity of the generated clause.

Decision Variables
z_j binary variable indicating if the feature j is present in the selected
    clause.
delta_i binary variable indicating if the input i is satisfied by the generated
        clause.

z_{CG} = min    lambda*(1 + sum_{j in J} z_j) -
                    - sum_{i in P} mu_i*delta_i + sum_{i in Z} delta_i  (obj)
         s.t.   delta_i + z_j <= 1, j in S_i, i in P                    (2a)
                delta_i >= 1 - sum_{j in S_i}z_j, i in Z                (2b)
                sum_{j in J} z_j <= D                                   (2c)
                z_j in {0,1}, j in J                                    (2d)
                delta_i >= 0, i in Z                                    (2e)

In the objective, the term 1 + sum_{j in J} z_j is equal to the generated
clause complexity, i.e., to 1 plus the clause size. This objective function
corresponds to the reduced cost of the clause generated. Negative objective
values identify clauses with a negative reduced cost, and the clause can be
added to the restricted master problem (RMP) in (1). The clause is constructed
by selecting the words for which the corresponding z variables are set to 1.
The constraints (2a) - (2b) relate the selected features with the delta
variables. Constraint (2c) controls the complexity of the clause being
generated.  Note that here the variables delta_i are binary but need not be
encoded as binary, as it is implied.
"""
import numpy as np
from bitarray.util import int2ba
from ortools.linear_solver import pywraplp
from ortools.linear_solver import linear_solver_pb2
from ._col_gen_classifier import BaseMasterProblem
from ._col_gen_classifier import BaseSubproblem
from ._col_gen_classifier import ColGenClassifier


def get_params_from_string(params):
    """ Given the params in string form, returns the MPSolverParameters.
    Parameters
    ----------
    params : string,
        The solver parameters in string format.
    """
    # TODO: Implement this method.
    solver_params = pywraplp.MPSolverParameters()
    print(params)
    return solver_params


class BDRMasterProblem(BaseMasterProblem):
    """The master problem for boolean decision rule generation described in
    'Boolean decision rules via column generation' by Sanjeeb Dash et. al.
    2018. This extends the BaseMasterProblem for column generation classifier.

    Parameters
    ----------
    C : int, default=10,
        A parameter used for controlling the overall complexity of decision
        rule.
    p : float, default=1,
        A parameter used for balancing the penalty between false negatives and
        false positives.
        Higher value of p would result in more penalty for the false negatives.
    optimization_problem_type : string, default='glop',
        Describes the solver used for solving the master problem.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`generate_mp`. The inputs should only
        contain values in {0,1}.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`generate_mp`. The labels should only
        contain values in {0,1}.
    solver_ : MPSolver from OR-Tools,
        The solver used for solving the master problem.
    clause_vars_ : list(int),
        Stores the indices of clause variables w_k generated so far.
    xi_vars_ : list(int),
        Stores the indices of positive penalty variables xi.
    clause_satisfaction_ : list(int)
        Stores the indices of constraints (1a).
    clause_complexity_: int
        Index of constraint (1b).
    clause_dict_ : dictionary, (int->list)
        Dictionary of clauses generated so far. The keys are the indices of the
        corresponding variables and the values are the lists containing the
        indices of features that make the clause.
    generated_ : boolean,
        True when the master problem model has been generated. False otherwise.
    """

    def __init__(self, C=10, p=1, optimization_problem_type='glop'):
        self.generated_ = False
        self.C_ = C
        self.p_ = p
        self.solver_ = pywraplp.Solver.CreateSolver(optimization_problem_type)

        # Vars
        self.clause_vars_ = None
        self.xi_vars_ = None

        # Constraints
        self.clause_satisfaction_ = None
        self.clause_complexity_ = None

        self.clause_dict_ = {}

    def generate_mp(self, X, y):
        """Generates the master problem model (RMP) and initializes the primal
        and dual solutions.
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The input. The inputs should only contain values in {0,1}.
        y : ndarray, shape (n_samples,)
            The labels. The labels should only contain values in {0,1}.
        """
        if self.generated_:
            return

        self.X_ = X
        self.y__ = y
        infinity = self.solver_.infinity()
        # If not specified, Gurobi uses all the threads available. For solving
        # the LP, we only need one thread.
        self.solver_.SetNumThreads(1)

        n_positive_examples = len(list(filter(lambda x: (x > 0), y)))

        # Initial number of clasuses to start with.
        n_clauses = 1

        objective = self.solver_.Objective()
        self.xi_vars_ = [None]*n_positive_examples
        for i in range(n_positive_examples):
            # This need not be a boolean variable.
            # The upper bound is not needed.
            xi_var = self.solver_.IntVar(0.0, infinity, 'p_'+str(i))
            self.xi_vars_[i] = xi_var.index()
            objective.SetCoefficient(xi_var, self.p_)

        self.clause_vars_ = [None]*n_clauses
        for i in range(n_clauses):
            # This need not be a boolean variable.
            # The upper bound is not needed.
            clause_var = self.solver_.IntVar(0.0, infinity, 'c_'+str(i))
            self.clause_vars_[i] = clause_var.index()
            clause = self.generate_lexicographic_clause(i)
            self.clause_dict_[self.clause_vars_[i]] = clause
            obj_coeff = self.get_objective_coeff_mp(clause)
            objective.SetCoefficient(clause_var, obj_coeff)
        objective.SetMinimization()

        # Add constraints
        self.clause_satisfaction_ = [None]*n_positive_examples
        clause_satisfaction_cons = [None]*n_positive_examples
        clause_complexity_cons = self.solver_.Constraint(
            -infinity, self.C_, "clause_complexity")
        self.clause_complexity_ = clause_complexity_cons.index()
        for i in range(n_positive_examples):
            clause_satisfaction_cons[i] = self.solver_.Constraint(
                1, infinity, "clause_satisfaction_" + str(i))
            self.clause_satisfaction_[i] = clause_satisfaction_cons[i].index()
            xi_var = self.solver_.variable(self.xi_vars_[i])
            clause_satisfaction_cons[i].SetCoefficient(xi_var, 1)
        for i in range(n_clauses):
            clause_var = self.solver_.variable(self.clause_vars_[i])
            coeffs = self.get_clause_coeffs(
                self.clause_dict_[self.clause_vars_[i]])
            for j in range(n_positive_examples):
                clause_satisfaction_cons[j].SetCoefficient(
                    clause_var, coeffs[j])
            clause_complexity_cons.SetCoefficient(clause_var, coeffs[-1])

        self.generated_ = True

    def add_column(self, clause):
        """Adds the given column to the master problem model. Retuns True if
        the column is added to the master problem.

        Boolean decision rule generation is a special case where we take the
        clause as input instead of the column coefficients.

        Parameters
        ----------
        clause : list(int),
            The clause for which the column is to be generated.
        """
        assert self.generated_

        if len(clause) == 0:
            return False

        # Avoid adding duplicates.
        assert clause not in self.clause_dict_.values()

        n_clauses = len(self.clause_vars_)
        n_positive_examples = self.solver_.NumConstraints() - 1
        clause_var = self.solver_.IntVar(
            0.0, self.solver_.infinity(), 'c_'+str(n_clauses))
        self.clause_vars_.append(clause_var.index())
        self.clause_dict_[clause_var.index()] = clause
        obj_coeff = self.get_objective_coeff_mp(clause)
        self.solver_.Objective().SetCoefficient(clause_var, obj_coeff)

        # Constraint coeffs
        coeffs = self.get_clause_coeffs(clause)
        assert n_positive_examples == len(self.clause_satisfaction_)
        for j in range(n_positive_examples):
            clause_satisfaction_cons = self.solver_.constraint(
                self.clause_satisfaction_[j])
            clause_satisfaction_cons.SetCoefficient(clause_var, coeffs[j])
        clause_comp_constraint = self.solver_.constraint(
            self.clause_complexity_)
        clause_comp_constraint.SetCoefficient(clause_var, coeffs[-1])
        return True

    def solve_rmp(self, solver_params=''):
        """Solves the RMP with given solver params.
        Returns the dual costs of constraints (1b) and (1a) in a tuple.
        Parameters
        ----------
        solver_params : string, default='',
            The solver parameters for solving the RMP.
        """
        assert self.generated_

        # TODO(krunalp): Warm start LP
        # Solve lp
        result_status = self.solver_.Solve(
            get_params_from_string(solver_params))
        n_positive_examples = self.solver_.NumConstraints() - 1

        # TODO: Hide this under optional logging flag.
        print('Number of variables RMIP = %d' % self.solver_.NumVariables())
        print('Number of constraints RMIP = %d' %
              self.solver_.NumConstraints())
        if result_status == pywraplp.Solver.OPTIMAL:
            print('RMP Optimal objective value = %f' %
                  self.solver_.Objective().Value())

        # Dual costs
        clause_comp_cons = self.solver_.constraint(self.clause_complexity_)
        cc_dual = abs(clause_comp_cons.dual_value())
        cs_duals = []
        for i in range(n_positive_examples):
            clause_satisfaction_cons = self.solver_.constraint(
                self.clause_satisfaction_[i])
            cs_duals.append(clause_satisfaction_cons.dual_value())
        return (cc_dual, cs_duals)

    def solve_ip(self, solver_params=''):
        """Solves the integer RMP with given solver params.
        Returns True if the explanation is generated.
        Parameters
        ----------
        solver_params : string, default='',
            The solver parameters for solving the integer RMP.
        """
        assert self.generated_

        # We can use sat here since all the coefficients and variables are
        # integer. We can also use gurobi. But sat is 2-3 times faster.
        # TODO: Solver type should be a parameter.
        solver = pywraplp.Solver.CreateSolver("sat")

        # We have to load the model from LP solver.
        # This copy is not avoidable with OR-Tools since we are now switching
        # form solving LP to solving IP.
        model_proto = linear_solver_pb2.MPModelProto()
        self.solver_.ExportModelToProto(model_proto)
        solver.LoadModelFromProto(model_proto)

        num_xi_vars = len(self.xi_vars_)

        result_status = solver.Solve(get_params_from_string(solver_params))
        print('Problem solved in %f milliseconds' % solver.wall_time())
        self.explanation = []

        has_solution = (
            result_status == pywraplp.Solver.OPTIMAL or
            result_status == pywraplp.Solver.FEASIBLE)
        assert has_solution

        # TODO: Store solution instead of printing it.
        print("Integer RMP Objective = ", solver.Objective().Value())
        all_vars = solver.variables()
        assert len(all_vars) == num_xi_vars + len(self.clause_vars_)
        for i in range(num_xi_vars, len(all_vars)):
            if all_vars[i].solution_value() > 0:
                orig_var = self.solver_.variable(
                    self.clause_vars_[i-num_xi_vars])
                self.explanation.append(self.clause_dict_[orig_var.index()])
        return True

    @staticmethod
    def satisfies_clause(entry, clause):
        """ Given the entry and clause, returns 1 if the entry satisfies the
        clause and 0 oterwise.
        Parameters
        ----------
        entry : ndarray, shape(n_features)
            The input example. The array should have values in {0,1}.
        clause : list(int),
            The list containing the indices of features present in the clause.
        """
        for index in clause:
            if entry[index] == 0:
                return 0
        return 1

    def get_objective_coeff_mp(self, clause):
        """ Returns the number of zero examples satisfying the clause.
        Parameters
        ----------
        clause : list(int),
            The list containing the indices of features present in the clause.
        """
        num_entries_satisfying_clause = 0
        for i in range(len(self.X_)):
            entry = self.X_[i]
            target = self.y__[i]
            # Only consider zero examples.
            if target == 1:
                continue
            num_entries_satisfying_clause += self.satisfies_clause(
                entry, clause)
        return num_entries_satisfying_clause

    def get_clause_coeffs(self, clause):
        """ Given the clause, returns the coefficients for corresponding
        variable in RMP.
        Parameters
        ----------
        clause : list(int),
            The list containing the indices of features present in the clause.
        """
        # no of rows = No of positive examples + 1.
        coeffs = []
        for i in range(len(self.X_)):
            entry = self.X_[i]
            target = self.y__[i]
            # Only consider positive examples.
            if target == 0:
                continue
            coeffs.append(self.satisfies_clause(entry, clause))
        # last constraint
        coeffs.append(len(clause) + 1)
        return coeffs

    @staticmethod
    def generate_lexicographic_clause(index):
        """ Generates the 'index'th clause as per lexicogrphical index.
        Parameters
        ----------
        index : int.
        """
        ind = int2ba(index)
        ind = ind[::-1]
        clause = [i for i in range(len(ind)) if ind[i]]
        return clause


class BDRSubProblem(BaseSubproblem):
    """The  subproblem for boolean decision rule generation described in
    'Boolean decision rules via column generation' by Sanjeeb Dash et. al.
    2018. This extends the BaseSubproblem for column generation classifier.

    Parameters
    ----------
    D : int,
        A parameter used for controlling the complexity of the clause being
        generated.
    optimization_problem_type : string, default='cbc'
        Describes the solver used for solving the subproblem (2).

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`generate_columns`. The inputs should
        only contain values in {0,1}.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`generate_columns`. The labels should
        only contain values in {0,1}.
    solver_ : MPSolver from OR-Tools,
        The solver used for solving the subproblem.
    delta_vars_ : list(int),
        Stores the indices of delta variables in (2).
    z_vars_ : list(int),
        Stores the indices of z variables in (2).
    generated_ : boolean,
        True when the master problem model has been generated. False otherwise.
    """

    def __init__(self, D, optimization_problem_type='cbc'):
        self.D_ = D
        self.solver_ = pywraplp.Solver.CreateSolver(optimization_problem_type)
        self.generated_ = False

        # Vars
        self.delta_vars_ = None
        self.z_vars_ = None

    def create_submip(self, cc_dual, cs_duals):
        """Creates the model for the subproblem. This should be called only
        once for a given problem.
        Parameters
        ----------
        cc_dual : float,
            dual cost of clause complexity constraint (1b).
        cs_duals : list(float),
            dual costs of clause satisfaction constraints (1a).
        """
        # TODO: Implement this method.
        return

    def update_objective(self, cc_dual, cs_duals):
        """Updates the objective of the generated subproblem. This can be
        called only after the create_submip method has been called.
        Parameters
        ----------
        cc_dual : float,
            dual cost of clause complexity constraint (1b).
        cs_duals : list(float),
            dual costs of clause satisfaction constraints (1a).
        """
        # TODO: Implement this method.
        return

    def generate_columns(self, X, y, dual_costs, params=""):
        """Generates the new columns to be added to the RMP.
        In this case instead of directly generating the coefficients, this
        method returns the list of generated clauses. The Master problem can
        find the coefficients from it.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The input. The inputs should only contain values in {0,1}.
        y : ndarray, shape (n_samples,)
            The labels. The labels should only contain values in {0,1}.
        dual_costs : tuple(float), size=2
            Dual costs of constraints (1b) and (1a) in a tuple.
        params : string, default=""
            Solver parameters.
        """
        # TODO: Implement this method.
        return []


class BooleanDecisionRuleClassifier(ColGenClassifier):
    """Binary classifier using boolean decision rule generation method.

    Parameters
    ----------
    max_iterations : int, default=-1
        Maximum column generation iterations. Negative values removes the
        iteration limit and the problem is solved till optimality.
    C : int,default=10,
        A parameter used for controlling the overall complexity of decision
        rule.
    p : float,default=1
        A parameter used for balancing the penalty between false negatives and
        false positives. Higher value of p would result in more penalty for the
        false negatives.
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
                 C=10,
                 p=1,
                 rmp_solver_params="",
                 master_ip_solver_params="",
                 subproblem_params=""):
        super(BooleanDecisionRuleClassifier, self).__init__(
            max_iterations,
            master_problem=BDRMasterProblem(
                C, p, 'glop'),
            subproblem=BDRSubProblem(
                C, 'cbc'),
            rmp_is_ip=True,
            rmp_solver_params=rmp_solver_params,
            master_ip_solver_params=master_ip_solver_params,
            subproblem_params=subproblem_params)
        # Not used after this but we have to store them for sklearn
        # compatibility.
        self.C = C
        self.p = p

    def _more_tags(self):
        return {'X_types': ['categorical'],
                'non_deterministic': True,
                'binary_only': True}

    def predict(self, X):
        """Predicts the class based on the solution of master problem. This
        method is abstract.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples. The inputs should only contain values in {0,1}.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample. The labels only contain values in {0,1}.
        """
        # TODO: Implement this method.
        return np.zeros(X.shape[0], dtype=int)
