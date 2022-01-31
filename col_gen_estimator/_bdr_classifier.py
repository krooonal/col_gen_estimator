"""
Binary classifier using boolean decision rule generation method.

Reference: Sanjeeb Dash, Oktay Gunluk, and Dennis Wei. Boolean decision rules via column generation. 
In Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018.

The model is designed for binary classification. The model generates a boolean decision rule in DNF 
(Disjunctive Normal Form, OR-of-ANDs). An example of a DNF rule with two clauses is "(x1 AND x2) OR (x3 AND
x5 AND x7)". These rules return True when the example belongs to the class and False otherwise. These 
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
xi_i binary variable indicating if the input i does not satisfy any selected clauses.

z_{MIP} = min   sum_{i in P} p*xi_i + sum_{i in Z} sum_{k in K_i} w_k   (obj)
          s.t.  xi_i + sum_{k in K_i} w_k >= 1, i in P                  (1a)
                sum_{k in K} c_k*w_k >= C                               (1b)
                xi_i >= 0, i in P                                       (1c)
                w_k in {0,1}, k in K                                    (1d) 

The first sum in the objective is the penalty for the misclassified positive examples, and the second 
part is the penalty for the misclassified negative examples. Constraints (1a) ensure that each positive 
example either satisfies at least one of the selected clause or the associated penalty variable is set 
to 1. Constraints (1b) control the overall complexity of the generated DNF formula. Clearly, the number 
of clauses is exponential, so the columns for w_k are generated through a subproblem. 

The reduced cost for a column $w_k$ is given by 
\sum_{i in mathcal{Z}} delta_i - sum_{i in mathcal{P}} mu_i delta_i + lambda c_k
where mu_i (>= 0) are the dual costs for constraints (1a) and lambda (>= 0) is the dual cost for 
constraint (1c). The binary variable delta_i is set to 1 if the ith example satisfies the clause w_k.

Sets 
P  set of all positive examples in the dataset. 
Z  set of all negative examples in the dataset. 
J  set of all the features.
S_i  the zero-valued features in sample i, S_i = {j in J: x_{ij} = 0}.

Parameters
D maximum complexity of the generated clause.

Decision Variables
z_j binary variable indicating if the feature j is present in the selected clause.
delta_i binary variable indicating if the input i is satisfied by the generated clause.

z_{CG} = min    lambda*(1 + sum_{j in J} z_j) - 
                    - sum_{i in P} mu_i*delta_i + sum_{i in Z} delta_i  (obj)
         s.t.   delta_i + z_j <= 1, j in S_i, i in P                    (2a)
                delta_i >= 1 - sum_{j in S_i}z_j, i in Z                (2b)
                sum_{j in J} z_j <= D                                   (2c)
                z_j in {0,1}, j in J                                    (2d) 
                delta_i >= 0, i in Z                                    (2e)

In the objective, the term 1 + sum_{j in J} z_j is equal to the generated clause complexity, 
i.e., to 1 plus the clause size. This objective function corresponds to the reduced cost of the 
clause generated. Negative objective values identify clauses with a negative reduced cost, and 
the clause can be added to the restricted master problem (RMP) in (1). The clause is constructed 
by selecting the words for which the corresponding z variables are set to 1. The constraints (2a) - (2b) 
relate the selected features with the delta variables. Constraint (2c) controls the complexity of 
the clause being generated.  Note that here the variables delta_i are binary but need not be encoded 
as binary, as it is implied. 
"""
import numpy as np
from ortools.linear_solver import pywraplp
from ._col_gen_classifier import BaseMasterProblem
from ._col_gen_classifier import BaseSubproblem
from ._col_gen_classifier import ColGenClassifier

class BDRMasterProblem(BaseMasterProblem):
    """The master problem for boolean decision rule generation described in 
    'Boolean decision rules via column generation' by Sanjeeb Dash et. al. 2018.
    This extends the BaseMasterProblem for column generation classifier.

    Parameters
    ----------
    C : int, default=10,
        A parameter used for controlling the overall complexity of decision rule.
    p : float, default=1,
        A parameter used for balancing the penalty between false negatives and false positives. 
        Higher value of p would result in more penalty for the false negatives.
    optimization_problem_type : string, default='glop',
        Describes the solver used for solving the master problem.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`generate_mp`. The inputs should only contain values in {0,1}.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`generate_mp`. The labels should only contain values in {0,1}.
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
        Dictionary of clauses generated so far. The keys are the indices of the corresponding variables
        and the values are the lists containing the indices of features that make the clause.
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
    
    def generate_mp(self,X,y):
        """Generates the master problem model (RMP) and initializes the primal and dual solutions.
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The input. The inputs should only contain values in {0,1}.
        y : ndarray, shape (n_samples,)
            The labels. The labels should only contain values in {0,1}.
        """
        #TODO: Implement this method.
        return

    def add_column(self,clause):
        """Adds the given column to the master problem model. 
        This is a special case where we take the clause as input instead of column.
        Parameters
        ----------
        clause : list(int),
            The clause for which the column is to be generated.
        """
        #TODO: Implement this method.
        return True
        
    def solve_rmp(self,solver_params=''):
        """Solves the RMP with given solver params. 
        Returns the dual costs of constraints (1b) and (1a) in a tuple.
        Parameters
        ----------
        solver_params : string, default='',
            The solver parameters for solving the RMP.
        """
        #TODO: Implement this method.
        # Dual costs
        cc_dual = 0
        cs_duals = []
        return (cc_dual,cs_duals)

    def solve_ip(self,solver_params=''):
        """Solves the integer RMP with given solver params.
        Returns True if the explanation is generated.
        Parameters
        ----------
        solver_params : string, default='',
            The solver parameters for solving the integer RMP.
        """
        #TODO: Implement this method.
        return True

class BDRSubProblem(BaseSubproblem):
    """The  subproblem for boolean decision rule generation described in 
    'Boolean decision rules via column generation' by Sanjeeb Dash et. al. 2018.
    This extends the BaseSubproblem for column generation classifier.

    Parameters
    ----------
    D : int,
        A parameter used for controlling the complexity of the clause being generated.
    optimization_problem_type : string, default='cbc'
        Describes the solver used for solving the subproblem (2).

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`generate_columns`. The inputs should only contain values in {0,1}.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`generate_columns`. The labels should only contain values in {0,1}.
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
        """Creates the model for the subproblem. This should be called only once for a given problem.
        Parameters
        ----------
        cc_dual : float,
            dual cost of clause complexity constraint (1b).
        cs_duals : list(float),
            dual costs of clause satisfaction constraints (1a).
        """
        #TODO: Implement this method.
        return
    
    def update_objective(self, cc_dual, cs_duals):
        """Updates the objective of the generated subproblem. This can be called only after the 
        Parameters
        ----------
        cc_dual : float,
            dual cost of clause complexity constraint (1b).
        cs_duals : list(float),
            dual costs of clause satisfaction constraints (1a).
        """
        #TODO: Implement this method.
        return
        
    def generate_columns(self, X, y, dual_costs, params=""):
        """Generates the new columns to be added to the RMP.
        In this case instead of directly generating the coefficients, this method 
        returns the list of generated clauses. The Master problem can find the coefficients from it.

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
        #TODO: Implement this method.
        return []

class BooleanDecisionRuleClassifier(ColGenClassifier):
    """Binary classifier using boolean decision rule generation method.

    Parameters
    ----------
    max_iterations : int, default=-1
        Maximum column generation iterations. Negative values removes the iteration limit and the problem
        is solved till optimality.
    C : int,default=10,
        A parameter used for controlling the overall complexity of decision rule.
    p : float,default=1
        A parameter used for balancing the penalty between false negatives and false positives. 
        Higher value of p would result in more penalty for the false negatives.
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
            rmp_solver_params = "", 
            master_ip_solver_params = "",
            subproblem_params = ""):
        super(BooleanDecisionRuleClassifier, self).__init__(max_iterations, 
            master_problem = BDRMasterProblem(C, p, 'glop'), 
            subproblem = BDRSubProblem(C, 'cbc'),
            rmp_is_ip = True,
            rmp_solver_params = rmp_solver_params, 
            master_ip_solver_params = master_ip_solver_params,
            subproblem_params = subproblem_params)
        # Not used after this but we have to store them for sklearn compatibility.
        self.C = C
        self.p = p
    
    def _more_tags(self):
        return {'X_types': ['categorical'],
                'non_deterministic': True,
                'binary_only': True}

    def predict(self, X):
        """Predicts the class based on the solution of master problem. This method is abstract.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples. The inputs should only contain values in {0,1}.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample. The labels only contain values in {0,1}.
        """
        #TODO: Implement this method.
        return np.zeros(X.shape[0], dtype=int)
