"""
Binary classifier using boolean decision rule generation method.
TODO: Cite the paper and explain the algorithm.
"""
import numpy as np
from bitarray.util import int2ba
from ortools.linear_solver import pywraplp
from ortools.linear_solver import linear_solver_pb2
from sklearn.utils.validation import check_array, check_is_fitted
from ._col_gen_classifier import BaseMasterProblem
from ._col_gen_classifier import BaseSubproblem
from ._col_gen_classifier import ColGenClassifier

class SerializableMPSolver:
    """ Wrapper on MPSolver that is serializable.
    """
    def __init__(self, optimization_problem_type):
        self.optimization_problem_type = optimization_problem_type
        self.solver = pywraplp.Solver.CreateSolver(optimization_problem_type)
    
    def infinity(self):
        return self.solver.infinity()
    def SetNumThreads(self, num):
        return self.solver.SetNumThreads(num)
    def Objective(self):
        return self.solver.Objective()
    def IntVar(self, lb, ub, name):
        return self.solver.IntVar(lb,ub,name)
    def BoolVar(self, name):
        return self.solver.BoolVar(name)
    def Constraint(self, lb, ub, name):
        return self.solver.Constraint(lb,ub,name)
    def NumConstraints(self):
        return self.solver.NumConstraints()
    def NumVariables(self):
        return self.solver.NumVariables()
    def Solve(self, params):
        return self.solver.Solve(params)
    def ExportModelToProto(self, model_proto):
        return self.solver.ExportModelToProto(model_proto)
    def LoadModelFromProto(self, model_proto):
        return self.solver.LoadModelFromProto(model_proto)
    def variable(self, index):
        return self.solver.variable(index)
    def constraint(self, index):
        return self.solver.constraint(index)
    def variables(self):
        return self.solver.variables()
    def constraints(self):
        return self.solver.constraints()
    
    def __getstate__(self):
        model_proto = linear_solver_pb2.MPModelProto()
        self.solver.ExportModelToProto(model_proto)
        return [model_proto, self.optimization_problem_type]

    def __setstate__(self,state):
        print("Called __setstate__")
        model_proto = state[0]
        self.optimization_problem_type = state[1]
        self.solver = pywraplp.Solver.CreateSolver(self.optimization_problem_type)
        self.solver.LoadModelFromProto(model_proto)

class BDRMasterProblem(BaseMasterProblem):
    """ TODO
    """

    def __init__(self, C, P, optimization_problem_type):
        """ TODO
        """
        self.generated = False
        self.C = C
        self.P = P
        self.solver = SerializableMPSolver(optimization_problem_type)
    
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
        if self.generated:
            return
        self.X = X
        self.y = y
        infinity = self.solver.infinity()
        self.solver.SetNumThreads(1)
        
        n_positive_examples = len(list(filter(lambda x: (isinstance(x, int) and x > 0), y)))
        
        n_clauses = 1 # Initial number of clasuses to start with.
        
        objective = self.solver.Objective()
        # Add variables
        self.xi_vars = [None]*n_positive_examples
        # Add objective coeff
        for i in range(n_positive_examples):
            # This need not be a boolean variable. It doesn't it need the upper bound.
            xi_var = self.solver.IntVar(0.0, infinity, 'p_'+str(i))
            self.xi_vars[i] = xi_var.index()
            # Regular objective coefficient here is 1. But we can try different weights.
            objective.SetCoefficient(xi_var, self.P)
        
        self.clause_vars = [None]*n_clauses
        for i in range(n_clauses):
            clause_var = self.solver.IntVar(0.0,infinity,'c_'+str(i))
            self.clause_vars[i] = clause_var.index()
            clause = self.GetClauseFromVarInd(i)
            self.clause_dict[self.clause_vars[i]] = clause
            obj_coeff = self.GetObjectiveCoeffMP(clause)
            objective.SetCoefficient(clause_var, obj_coeff)
        objective.SetMinimization()

        # Add constraints
        self.clause_satisfaction = [None]*n_positive_examples
        clause_satisfaction_cons = [None]*n_positive_examples
        clause_complexity_cons = self.solver.Constraint(-infinity, self.C, "clause_complexity")
        self.clause_complexity = clause_complexity_cons.index()
        for i in range(n_positive_examples):
            clause_satisfaction_cons[i] = self.solver.Constraint(1, infinity, "clause_satisfaction_" + str(i))
            self.clause_satisfaction[i] = clause_satisfaction_cons[i].index()
            xi_var = self.solver.variable(self.xi_vars[i])
            clause_satisfaction_cons[i].SetCoefficient(xi_var , 1)
        for i in range(n_clauses):
            clause_var = self.solver.variable(self.clause_vars[i])
            coeffs = self.GetCoefficientsForClause(self.clause_dict[self.clause_vars[i]])
            for j in range(n_positive_examples):
                clause_satisfaction_cons[j].SetCoefficient(clause_var, coeffs[j])
            clause_complexity_cons.SetCoefficient(clause_var, coeffs[-1])
        
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
        self.clause_vars.append(clause_var.index())
        self.clause_dict[clause_var.index()] = clause
        obj_coeff = self.GetObjectiveCoeffMP(clause)
        self.solver.Objective().SetCoefficient(clause_var, obj_coeff)

        # Constraint coeffs
        coeffs = self.GetCoefficientsForClause(clause)
        assert n_positive_examples == len(self.clause_satisfaction)
        for j in range(n_positive_examples):
            clause_satisfaction_cons = self.solver.constraint(self.clause_satisfaction[j])
            clause_satisfaction_cons.SetCoefficient(clause_var, coeffs[j])
        clause_comp_constraint = self.solver.constraint(self.clause_complexity)
        clause_comp_constraint.SetCoefficient(clause_var, coeffs[-1])

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
        # print('LP Problem solved in %f milliseconds' % self.solver.wall_time())
        print('Number of variables RMIP = %d' % self.solver.NumVariables())
        print('Number of constraints RMIP = %d' % self.solver.NumConstraints())
        if result_status == pywraplp.Solver.OPTIMAL:
            # TODO: Record solution in class for proper pickle effect.
            print('Optimal objective value = %f' % self.solver.Objective().Value())
            num_unsatisfied_cons = 0
            for xi_index in self.xi_vars:
                xi = self.solver.variable(xi_index)
                if xi.solution_value() > 0:
                    num_unsatisfied_cons += 1
            print("Number of unsatisfied positives: ", num_unsatisfied_cons)
            for cvar_index in self.clause_vars:
                cvar = self.solver.variable(cvar_index)
                if cvar.solution_value() > 0:
                    print(cvar.name(), " ", self.clause_dict[cvar],  " " , cvar.solution_value())
                if cvar.ReducedCost() > 0:
                    print(cvar.name(), " ", self.clause_dict[cvar], " rc " , cvar.ReducedCost())

        # Dual costs
        clause_comp_cons = self.solver.constraint(self.clause_complexity)
        cc_dual = abs(clause_comp_cons.dual_value())
        cs_duals = []
        for i in range(n_positive_examples):
            clause_satisfaction_cons = self.solver.constraint(self.clause_satisfaction[i])
            cs_duals.append(clause_satisfaction_cons.dual_value())
        return (cc_dual,cs_duals)

    def solve_ip(self,solver_params):
        """ Solves the integer RMP with given solver params.
        Returns True if the explanation is generated.
        """
        if not self.generated:
            return
        # We can use sat here since all the coefficients and variables are integer. 
        # We can also use gurobi. But sat is 2-3 times faster.
        # TODO: Solver type should be a parameter.
        solver = pywraplp.Solver.CreateSolver("sat") 
        
        # We have to load the model from LP solver. This copy is not avoidable with OR-Tools.
        model_proto = linear_solver_pb2.MPModelProto()
        self.solver.ExportModelToProto(model_proto)
        solver.LoadModelFromProto(model_proto)
        
        num_xi_vars = len(self.xi_vars)
        solver.SetTimeLimit(30000)

        result_status = solver.Solve(self.get_params(solver_params))
        print('Problem solved in %f milliseconds' % solver.wall_time())
        self.explanation = []
        
        if result_status != pywraplp.Solver.OPTIMAL and result_status != pywraplp.Solver.FEASIBLE:
            return False
        # TODO: Store solution instead of printing it.
        print("Objective = ", solver.Objective().Value())
        all_vars = solver.variables()
        assert len(all_vars) == num_xi_vars + len(self.clause_vars) 
        num_unsatisfied_positives = 0
        for i in range(len(all_vars)):
            if i < num_xi_vars:
                if all_vars[i].solution_value() > 0:
                    num_unsatisfied_positives += 1
            elif all_vars[i].solution_value() > 0:
                orig_var = self.solver.variable(self.clause_vars[i-num_xi_vars])
                print(orig_var.name(),"=", all_vars[i].solution_value(), 
                    " ", self.clause_dict[orig_var.index()])
                self.explanation.append(self.clause_dict[orig_var.index()])
        return True
    
    @staticmethod
    def get_params(params):
        """ Given the params in string form, returns the MPSolverParameters."""
        # TODO: Implement this method.
        solver_params = pywraplp.MPSolverParameters()
        print(params)
        return solver_params

    @staticmethod
    def SatisfiesDNF(entry, clause):
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

class BDRSubProblem(BaseSubproblem):
    """ Base class for subproblem. One needs to extend this for using with ColGenClassifier.
    """

    def __init__(self, D):
        self.D = D
        self.solver = SerializableMPSolver('cbc')
        self.generated = False
    
        # Vars
        self.delta_vars = None
        self.z_vars = None
    
    def create_submip(self, cc_dual, cs_duals):
        infinity = self.solver.infinity()

        n_words = self.X.shape[1]
        n_samples = self.X.shape[0]

        # We assume the positive dual cost in this model.
        assert cc_dual >= 0, "Negative clause complexity dual" 

        self.z_vars = [None]*n_words
        objective = self.solver.Objective()
        for i in range(n_words):
            z_var = self.solver.BoolVar('z_'+str(i))
            self.z_vars[i] = z_var.index()
            objective.SetCoefficient(z_var, cc_dual)
        
        self.delta_vars = [None]*n_samples
        # Add objective coeff
        n_positive_examples = 0
        n_zero_examples = 0
        for i in range(n_samples):
            # This need not be a boolean variable.
            delta_var = self.solver.BoolVar('d_'+str(i))
            self.delta_vars[i] = delta_var.index()
            obj_coeff = 1
            target = self.y[i]
            if target == 1:
                obj_coeff *= -cs_duals[n_positive_examples]
                n_positive_examples += 1
            else:
                n_zero_examples += 1
            objective.SetCoefficient(delta_var, obj_coeff)

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
        assert self.generated, "Update objective called before generating the subproblem." 
        n_words = self.X.shape[1]
        n_samples = self.X.shape[0]
        objective = self.solver.Objective()
        for i in range(n_words):
            z_var = self.solver.variable(self.z_vars[i])
            objective.SetCoefficient(z_var, cc_dual)
        
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
            delta_var = self.solver.variable(self.delta_vars[i])
            objective.SetCoefficient(delta_var, obj_coeff)

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
                z_var = self.solver.variable(self.z_vars[i])
                if z_var.solution_value() > 0:
                    clause.append(i)

        return [clause]

class BooleanDecisionRuleClassifier(ColGenClassifier):
    """ Binary classifier using boolean decision rule generation method.

    Parameters
    ----------
    max_iterations : int, default=-1
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
    def __init__(self, max_iterations=-1, 
            C=10, 
            P=10,
            rmp_solver_params = "", 
            master_ip_solver_params = "",
            subproblem_params = ""):
        super(BooleanDecisionRuleClassifier, self).__init__(max_iterations, 
            master_problem = BDRMasterProblem(C, P, 'cbc'), 
            subproblem = BDRSubProblem(C),
            rmp_is_ip = True,
            rmp_solver_params = rmp_solver_params, 
            master_ip_solver_params = master_ip_solver_params,
            subproblem_params = subproblem_params)
        self.C = C
        self.P = P
    
    def _more_tags(self):
        return {'X_types': ['categorical'],
                'non_deterministic': True,
                'binary_only': True}

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
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        explanation = self.master_problem.explanation
        # Check if the input satisfies any of the clause in explanation.
        y_predict = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            for clause in explanation:
                if self.master_problem.SatisfiesDNF(X[i], clause):
                    y_predict[i] = 1
                    break
        return self.label_encoder.inverse_transform(y_predict)

