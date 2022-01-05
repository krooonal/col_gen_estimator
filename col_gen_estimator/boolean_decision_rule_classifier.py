"""
Binary classifier using boolean decision rule generation method.
TODO: Cite the paper and explain the algorithm.
"""
import numpy as np
from bitarray.util import int2ba
from ortools.linear_solver import pywraplp
from ortools.linear_solver import linear_solver_pb2
from col_gen_classifier import ColGenClassifier, BaseMasterProblem, BaseSubProblem

class BDRMasterProblem(BaseMasterProblem):
    """ TODO
    """

    def __init__(self, C, P, optimization_problem_type):
        """ TODO
        """
        self.generated = False
        self.C = C
        self.P = P
        self.solver = pywraplp.Solver.CreateSolver(optimization_problem_type)
    
        # Vars
        self.clause_vars = None
        self.xi_vars = None

        # Constraints
        self.clause_satisfaction = None
        self.clause_complexity = None
        
        self.clause_dict = {}
    
    def generate_mp(self,X,y,params):
        """ Generates the master problem model (RMP) and initializes the primal and dual solutions.
        """
        self.X = X
        self.y = y
        infinity = self.solver.infinity()
        self.solver.SetNumThreads(1)
        
        n_positive_examples = int(sum(y))
        
        n_clauses = 1 # Initial number of clasuses to start with.
        
        objective = self.solver.Objective()
        # Add variables
        self.xi_vars = [None]*n_positive_examples
        # Add objective coeff
        for i in range(n_positive_examples):
            # This need not be a boolean variable. It doesn't it need the upper bound.
            self.xi_vars[i] = self.solver.IntVar(0.0, infinity, 'p_'+str(i))
            # Regular objective coefficient here is 1. But we can try different weights.
            objective.SetCoefficient(self.xi_vars[i], self.P)
        
        self.clause_vars = [None]*n_clauses
        for i in range(n_clauses):
            self.clause_vars[i] = self.solver.IntVar(0.0,infinity,'c_'+str(i))
            clause_var = self.clause_vars[i]
            clause = self.GetClauseFromVarInd(i)
            self.clause_dict[clause_var] = clause
            obj_coeff = self.GetObjectiveCoeffMP(clause)
            objective.SetCoefficient(clause_var, obj_coeff)
        objective.SetMinimization()

        # Add constraints
        self.clause_satisfaction = [None]*n_positive_examples
        self.clause_complexity = self.solver.Constraint(-infinity, self.C, "clause_complexity")
        for i in range(n_positive_examples):
            self.clause_satisfaction[i] = self.solver.Constraint(1, infinity, "clause_satisfaction_" + str(i))
            self.clause_satisfaction[i].SetCoefficient(self.xi_vars[i] , 1)
        for i in range(n_clauses):
            clause_var = self.clause_vars[i]
            coeffs = self.GetCoefficientsForClause(self.clause_dict[clause_var])
            for j in range(n_positive_examples):
                self.clause_satisfaction[j].SetCoefficient(clause_var, coeffs[j])
            self.clause_complexity.SetCoefficient(clause_var, coeffs[-1])
        
        self.generated = True

    def add_column(self,clause):
        """ Adds the given column to the master problem model. 
        This is a special case where we take the clause as input instead of column.
        """
        if not self.generated:
            return
        if len(clause) == 0:
            return False
    
        if clause in self.clause_dict.values():
            print("Regenerated the clause ", clause)
            return False
        
        n_clauses = len(self.clause_vars)
        n_positive_examples = self.solver.NumConstraints() - 1
        clause_var = self.solver.IntVar(0.0,self.solver.infinity(),'c_'+str(n_clauses))
        self.clause_vars.append(clause_var)
        self.clause_dict[clause_var] = clause
        obj_coeff = self.GetObjectiveCoeffMP(clause)
        self.solver.Objective().SetCoefficient(clause_var, obj_coeff)

        # Constraint coeffs
        coeffs = self.GetCoefficientsForClause(clause)
        assert(n_positive_examples == len(self.clause_satisfaction))
        for j in range(n_positive_examples):
            self.clause_satisfaction[j].SetCoefficient(clause_var, coeffs[j])
        self.clause_complexity.SetCoefficient(clause_var, coeffs[-1])

        print("Added a column: ", clause_var.name(), " ", clause)
        return True
        

    def solve_rmp(self,solver_params):
        """ Solves the RMP with given solver params.
        """
        if not self.generated:
            return
        # TODO(krunalp): Warm start LP
        # Solve lp
        result_status = self.solver.Solve(solver_params)
        n_positive_examples = self.solver.NumConstraints() - 1

        # TODO: Hide this under optional logging flag.
        print('LP Problem solved in %f milliseconds' % self.solver.wall_time())
        print('Number of variables RMIP = %d' % self.solver.NumVariables())
        print('Number of constraints RMIP = %d' % self.solver.NumConstraints())
        if result_status == pywraplp.Solver.OPTIMAL:
            print('Optimal objective value = %f' % self.solver.Objective().Value())
            num_unsatisfied_cons = 0
            for xi in self.xi_vars:
                if xi.solution_value() > 0:
                    num_unsatisfied_cons += 1
            print("Number of unsatisfied positives: ", num_unsatisfied_cons)
            for cvar in self.clause_vars:
                if cvar.solution_value() > 0:
                    print(cvar.name(), " ", self.clause_dict[cvar],  " " , cvar.solution_value())
                if cvar.ReducedCost() > 0:
                    print(cvar.name(), " ", self.clause_dict[cvar], " rc " , cvar.ReducedCost())

        # Dual costs
        cc_dual = abs(self.clause_complexity.dual_value())
        cs_duals = []
        for i in range(n_positive_examples):
            cs_duals.append(self.clause_satisfaction[i].dual_value())
        return (cc_dual,cs_duals)

    def solve_ip(self,solver_params):
        """ Solves the integer RMP with given solver params.
        Returns True if the explanation is generated.
        """
        if not self.generated:
            return
        # We can use sat here since all the coefficients and variables are integer. 
        # We can also use gurobi. But sat is 2-3 times faster.
        solver = pywraplp.Solver.CreateSolver("sat") 
        
        # We have to load the model from LP solver. This copy is not avoidable with OR-Tools.
        model_proto = linear_solver_pb2.MPModelProto()
        self.solver.ExportModelToProto(model_proto)
        solver.LoadModelFromProto(model_proto)
        
        num_xi_vars = len(self.xi_vars)
        solver.SetTimeLimit(30000)
        result_status = solver.Solve(solver_params)
        print('Problem solved in %f milliseconds' % solver.wall_time())
        self.explanation = []
        
        n_positive_examples = int(sum(self.y))
        n_samples = self.X.shape[0]
        n_zero_examples = n_samples - n_positive_examples
        if result_status != pywraplp.Solver.OPTIMAL and result_status != pywraplp.Solver.FEASIBLE:
            return False
        print("Objective = ", solver.Objective().Value())
        all_vars = solver.variables()
        num_unsatisfied_positives = 0
        for i in range(len(all_vars)):
            if i < num_xi_vars:
                if all_vars[i].solution_value() > 0:
                    num_unsatisfied_positives += 1
            elif all_vars[i].solution_value() > 0:
                orig_var = self.clause_vars[i-num_xi_vars]
                print(orig_var.name(),"=", all_vars[i].solution_value(), 
                    " ", self.clause_dict[orig_var])
                self.explanation.append(self.clause_dict[orig_var])
        return True

    @staticmethod
    def SatisfiesDNF(self, entry, clause):
        """ Given the entry and clause, check if the entry satisfies the clause
        """
        for index in clause:
            if entry[index] == 0:
                return 0
        return 1

    def GetObjectiveCoeffMP(self, clause):
        """ Returns the number of zero examples satisfying the clause
        """
        num_entries_satisfying_clause = 0
        for i in range(len(self.X)):
            entry = self.X[i]
            target = self.y[i]
            # Only consider zero examples.
            if target == 1:
                continue
            num_entries_satisfying_clause += self.SatisfiesDNF(entry, clause)
        return num_entries_satisfying_clause

    # Function: 
    def GetCoefficientsForClause(self, clause):
        """ Given the clause, returns its coefficients
        """
        # no of rows = No of positive examples + 1.
        coeffs = []
        for i in range(len(self.X)):
            entry = self.X[i]
            target = self.y[i]
            # Only consider positive examples.
            if target == 0:
                continue
            coeffs.append(self.SatisfiesDNF(entry, clause))
        # last constraint
        coeffs.append(len(clause) + 1)
        return coeffs
    
    @staticmethod
    def GetClauseFromVarInd(var_ind):
        """ Generates the 'var_ind'th clause as per lexicogrphical index. 
        """
        ind = int2ba(var_ind)
        ind = ind[::-1]
        clause = [i for i in range(len(ind)) if ind[i]]
        return clause

class BDRSubProblem(BaseSubProblem):
    """ Base class for subproblem. One needs to extend this for using with ColGenClassifier.
    """

    def __init__(self, D):
        self.D = D
        self.solver = None
        self.generated = False
    
        # Vars
        self.delta_vars = None
        self.z_vars = None
    
    def create_submip(self, cc_dual, cs_duals):
        self.solver = pywraplp.Solver.CreateSolver('gurobi')
        assert(self.solver.SetSolverSpecificParametersAsString("Threads = 2"))
        infinity = self.solver.infinity()

        n_words = self.X.shape[1]
        n_samples = self.X.shape[0]

        # We assume the positive dual cost in this model.
        assert(cc_dual >= 0, "Negative clause complexity dual")

        self.z_vars = [None]*n_words
        objective = self.solver.Objective()
        for i in range(n_words):
            self.z_vars[i] = self.solver.BoolVar('z_'+str(i))
            objective.SetCoefficient(self.z_vars[i], cc_dual)
        
        self.delta_vars = [None]*n_samples
        # Add objective coeff
        n_positive_examples = 0
        n_zero_examples = 0
        for i in range(n_samples):
            # This need not be a boolean variable.
            self.delta_vars[i] = self.solver.BoolVar('d_'+str(i))
            obj_coeff = 1
            target = self.y[i]
            if target == 1:
                obj_coeff *= -cs_duals[n_positive_examples]
                n_positive_examples += 1
            else:
                n_zero_examples += 1
            objective.SetCoefficient(self.delta_vars[i], obj_coeff)

        objective.SetOffset(cc_dual)
        objective.SetMinimization()
        
        # Constraints
        zero_count = 0
        for i in range(n_samples):
            if self.y[i] == 1:
                for j in range(n_words):
                    if self.X[i,j] == 0:
                        cons = self.solver.Constraint(0, 1, "pos_" + str(i))
                        cons.SetCoefficient(self.delta_vars[i] , 1)
                        cons.SetCoefficient(self.z_vars[j] , 1)
            else:
                cons = self.solver.Constraint(1, infinity, "zero_" + str(i))
                cons.SetCoefficient(self.delta_vars[i] , 1)
                for j in range(n_words):
                    if self.X[i,j] == 0:
                        cons.SetCoefficient(self.z_vars[j] , 1)
                zero_count += 1

        cc_const = self.solver.Constraint(0, self.D, "clause_complexity")
        for j in range(n_words):
            cc_const.SetCoefficient(self.z_vars[j],1)
        self.generated = True
    
    def update_objective(self, cc_dual, cs_duals):
        """ Updates the objective of the generated subproblem.
        """
        assert(self.generated, "Update objective called before generating the subproblem.")
        n_words = self.X.shape[1]
        n_samples = self.X.shape[0]
        objective = self.solver.Objective()
        for i in range(n_words):
            objective.SetCoefficient(self.z_vars[i], cc_dual)
        
        n_positive_examples = 0
        n_zero_examples = 0
        for i in range(n_samples):
            obj_coeff = 1
            target = self.y[i]
            if target == 1:
                obj_coeff *= -cs_duals[n_positive_examples]
                n_positive_examples += 1
            else:
                n_zero_examples += 1
            objective.SetCoefficient(self.delta_vars[i], obj_coeff)

        objective.SetOffset(cc_dual)
        objective.SetMinimization()
        
    def generate_columns(self, X, y, dual_costs, params):
        """ Generates the new columns to be added to the RMP.
        In this case instead of directly generating the coefficients, this method 
        returns the generated clause. The Master problem can find the coefficients from it.
        """
        cc_dual = dual_costs[0]
        cs_duals = dual_costs[1]

        if self.generated:
            self.update_objective(cc_dual, cs_duals)
        else:
            self.create_submip(cc_dual,cs_duals)

        # Solve sub problem
        result_status = self.solver.Solve(params.solver_params)
        
        # Try to generate the clause if negative objective
        if result_status != pywraplp.Solver.OPTIMAL and result_status != pywraplp.Solver.FEASIBLE:
            print("No solution in sp found.")
            return []

        print('Optimal objective value = %f' % self.solver.Objective().Value())
        clause = []
        # TODO: This threshold should be a parameter.
        if self.solver.Objective().Value() < -1e-6:
            for i in range(len(self.z_vars)):
                if self.z_vars[i].solution_value() > 0:
                    clause.append(i)

        return [clause]

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
    def __init__(self, max_iterations='-1', 
            C=10, 
            P=10,
            rmp_solver_params = "", 
            master_ip_solver_params = "",
            subproblem_params = ""):
        super(BooleanDecisionRuleClassifier, self).__init__(max_iterations, 
            master_problem = BDRMasterProblem(C, P), 
            subproblem = BDRSubProblem(C),
            rmp_is_ip = True,
            rmp_solver_params = rmp_solver_params, 
            master_ip_solver_params = master_ip_solver_params,
            subproblem_params = subproblem_params)

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
        explanation = self.master_problem.explanation
        # Check if the input satisfies any of the clause in explanation.
        y_predict = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            for clause in explanation:
                if self.master_problem.SatisfiesDNF(X[i], clause):
                    y_predict[i] = 1
                    break
        return y_predict

