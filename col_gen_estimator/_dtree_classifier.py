"""
TODO: Documentation of the file.
"""
import numpy as np
import math
import itertools
from bitarray.util import int2ba
from ortools.linear_solver import pywraplp
from ortools.linear_solver import linear_solver_pb2
from sklearn.utils.validation import check_array, check_is_fitted
from ._col_gen_classifier import BaseMasterProblem
from ._col_gen_classifier import BaseSubproblem
from ._col_gen_classifier import ColGenClassifier


class Path:
    """TODO: Documentation.
    """

    def __init__(self) -> None:
        self.leaf_id = -1
        self.node_ids = []
        self.splits = []
        self.cost = 0
        self.id = -1
        self.target = -1

    def set_leaf(self, leaf_id, depth):
        """TODO: Documentation.
        For a full binary tree, adds the nodes corresponding to the given
        leaf_id."""
        self.leaf_id = leaf_id
        self.node_ids = []
        count = 2**(depth) - 1 + leaf_id
        while(count > 0):
            father = math.floor((count-1) / 2)
            self.node_ids.append(father)
            count = father


class Node:
    """TODO: Documentation.
    """

    def __init__(self) -> None:
        self.id = -1
        self.candidate_splits = []


class Leaf:
    """TODO: Documentation.
    """

    def __init__(self) -> None:
        self.id = id
        self.left_nodes = []
        self.right_nodes = []

    def create_leaf(self, id, depth) -> None:
        """TODO: Documentation.
        """
        self.id = id
        self.left_nodes = []
        self.right_nodes = []
        count = 2**(depth) - 1 + id
        while(count > 0):
            father = math.floor((count-1) / 2)
            if count % 2 == 0:
                self.right_nodes.append(father)
            else:
                self.left_nodes.append(father)
            count = father


class Split:
    """TODO: Documentation.
    """

    def __init__(self) -> None:
        self.id = -1
        self.feature = -1
        self.threshold = -1


def row_satisfies_path(X, leaf, splits, row, path):
    """TODO: Documentation.
    """
    split_ids = path.splits
    node_ids = path.node_ids
    for i in range(len(node_ids)):
        split = splits[split_ids[i]]
        feature = split.feature
        threshold = split.threshold
        if node_ids[i] in leaf.left_nodes:
            if X[row, feature] > threshold:
                return False
        else:  # Right node in the path
            if X[row, feature] <= threshold:
                return False
    return True


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


class DTreeMasterProblem(BaseMasterProblem):
    """TODO: Documentation.
    """

    def __init__(self, initial_paths, leaves, nodes, splits):
        super().__init__()
        self.solver_ = pywraplp.Solver.CreateSolver('glop')
        self.generated_ = False
        self.paths_ = initial_paths
        self.leaves_ = leaves
        self.nodes_ = nodes
        self.splits_ = splits

    def generate_mp(self, X, y):
        """TODO: Documentation.
        """
        if self.generated_:
            return

        self.X_ = X
        self.y__ = y
        infinity = self.solver_.infinity()
        self.solver_.SetNumThreads(1)

        objective = self.solver_.Objective()
        self.path_vars_ = [None]*len(self.paths_)
        for i in range(len(self.paths_)):
            xp_var = self.solver_.IntVar(0.0, infinity, 'p_'+str(i))
            self.path_vars_[i] = xp_var.index()
            self.paths_[i].id = xp_var.index()
            objective.SetCoefficient(xp_var, self.paths_[i].cost)
        objective.SetMaximization()

        # leaf constraints
        self.leaf_cons_ = [None]*len(self.leaves_)
        for leaf in self.leaves_:
            self.leaf_cons_[leaf.id] = self.solver_.Constraint(
                1, 1, "leaf_"+str(leaf.id))
            for path in self.paths_:
                if path.leaf_id == leaf.id:
                    xp_var = self.solver_.variable(path.id)
                    self.leaf_cons_[leaf.id].SetCoefficient(xp_var, 1)

        # row constraints
        n_rows = self.X_.shape[0]
        self.row_cons_ = [None]*n_rows
        for r in range(n_rows):
            self.row_cons_[r] = self.solver_.Constraint(1, 1, "row_"+str(r))
            for path in self.paths_:
                if row_satisfies_path(X, self.leaves_[path.leaf_id],
                                      self.splits_, r, path):
                    xp_var = self.solver_.variable(path.id)
                    self.row_cons_[r].SetCoefficient(xp_var, 1)

        # consistency constraints
        self.ns_constraints_ = {}
        for leaf in self.leaves_:
            self.ns_constraints_[leaf.id] = {}
            nodes = leaf.left_nodes + leaf.right_nodes
            for node_id in nodes:
                node = self.nodes_[node_id]
                self.ns_constraints_[leaf.id][node.id] = {}
                for split in node.candidate_splits:

                    ns_constraint = self.solver_.Constraint(
                        0, 0, "ns_"+str(node.id)+"_"+str(split)+"_"
                        + str(leaf.id))

                    dummy_var = self.solver_.BoolVar("r_ns_" +
                                                     str(node.id) +
                                                     "_"+str(split) +
                                                     "_"+str(leaf.id))
                    ns_constraint.SetCoefficient(dummy_var, -1)

                    for path in self.paths_:
                        for i in range(len(path.node_ids)):
                            if path.leaf_id == leaf.id\
                                and path.node_ids[i] == node.id \
                                    and path.splits[i] == split:
                                xp_var = self.solver_.variable(path.id)
                                ns_constraint.SetCoefficient(xp_var, 1)

                    self.ns_constraints_[
                        leaf.id][node.id][split] = ns_constraint
        # print(self.solver_.ExportModelAsLpFormat(False))
        self.generated_ = True

    def add_column(self, path):
        """TODO: Documentation.
        """
        objective = self.solver_.Objective()
        infinity = self.solver_.infinity()
        xp_var = self.solver_.IntVar(
            0.0, infinity, 'p_'+str(len(self.path_vars_)))
        self.path_vars_.append(xp_var.index())
        path.id = xp_var.index()
        objective.SetCoefficient(xp_var, path.cost)
        self.paths_.append(path)

        # leaf constraints
        for leaf in self.leaves_:
            if path.leaf_id == leaf.id:
                self.leaf_cons_[leaf.id].SetCoefficient(xp_var, 1)

        # row constraints
        n_rows = self.X_.shape[0]
        for r in range(n_rows):
            if row_satisfies_path(self.X_, self.leaves_[path.leaf_id],
                                  self.splits_, r, path):
                self.row_cons_[r].SetCoefficient(xp_var, 1)

        # consistency constraints
        for leaf in self.leaves_:
            nodes = leaf.left_nodes + leaf.right_nodes
            for node_id in nodes:
                node = self.nodes_[node_id]
                for split in node.candidate_splits:

                    ns_constraint = self.ns_constraints_[
                        leaf.id][node.id][split]

                    for i in range(len(path.node_ids)):
                        if path.leaf_id == leaf.id\
                            and path.node_ids[i] == node.id \
                                and path.splits[i] == split:
                            ns_constraint.SetCoefficient(xp_var, 1)
        return True

    def solve_rmp(self, solver_params=''):
        """TODO: Documentation.
        """
        assert self.generated_

        result_status = self.solver_.Solve(
            get_params_from_string(solver_params))
        n_paths = len(self.paths_)

        # TODO: Hide this under optional logging flag.
        print('Number of variables RMIP = %d' % self.solver_.NumVariables())
        print('Number of constraints RMIP = %d' %
              self.solver_.NumConstraints())
        if result_status == pywraplp.Solver.OPTIMAL:
            print('RMP Optimal objective value = %f' %
                  self.solver_.Objective().Value())

            for var in self.solver_.variables():
                print(var.name(), var.solution_value())

            leaf_duals = []
            for lid in range(len(self.leaves_)):
                leaf_duals.append(self.leaf_cons_[lid].dual_value())

            row_duals = []
            n_rows = self.X_.shape[0]
            for r in range(n_rows):
                row_duals.append(self.row_cons_[r].dual_value())

            ns_duals = {}
            for leaf in self.leaves_:
                ns_duals[leaf.id] = {}
                nodes = leaf.left_nodes + leaf.right_nodes
                for node_id in nodes:
                    node = self.nodes_[node_id]
                    ns_duals[leaf.id][node.id] = {}
                    for split in node.candidate_splits:
                        ns_duals[leaf.id][node.id][split] = \
                            self.ns_constraints_[
                            leaf.id][node.id][split].dual_value()
        else:
            print(result_status)
            print(self.solver_.ExportModelAsLpFormat(False))

        return (leaf_duals, row_duals, ns_duals)

    def solve_ip(self, solver_params=''):
        """Solves the integer RMP with given solver params.
        Returns True if the tree is generated.
        Parameters
        ----------
        solver_params : string, default='',
            The solver parameters for solving the integer RMP.
        """
        assert self.generated_

        # We can use sat here since all the coefficients and variables are
        # integer.
        # TODO: Solver type should be a parameter.
        solver = pywraplp.Solver.CreateSolver("sat")

        # We have to load the model from LP solver.
        # This copy is not avoidable with OR-Tools since we are now switching
        # form solving LP to solving IP.
        model_proto = linear_solver_pb2.MPModelProto()
        self.solver_.ExportModelToProto(model_proto)
        solver.LoadModelFromProto(model_proto)

        num_path_vars = len(self.path_vars_)

        result_status = solver.Solve(get_params_from_string(solver_params))
        print('Problem solved in %f milliseconds' % solver.wall_time())
        self.selected_paths = []

        has_solution = (
            result_status == pywraplp.Solver.OPTIMAL or
            result_status == pywraplp.Solver.FEASIBLE)
        assert has_solution

        # TODO: Store solution instead of printing it.
        print("Integer RMP Objective = ", solver.Objective().Value())
        all_vars = solver.variables()

        for path in self.paths_:
            path_var = solver.variable(path.id)
            if path_var.solution_value() > 0:
                self.selected_paths.append(path)
        return True


class DTreeSubProblem(BaseSubproblem):
    """TODO: Documentation."""

    def __init__(self, leaf, nodes, splits, targets,
                 depth, optimization_problem_type='cbc') -> None:
        super().__init__()
        self.leaf_id_ = leaf.id
        self.leaf_ = leaf
        self.nodes_ = nodes
        self.splits_ = splits
        self.targets_ = targets
        self.tree_depth_ = depth
        self.solver_ = pywraplp.Solver.CreateSolver(optimization_problem_type)
        self.generated_ = False
        self.z_vars_ = None

    def create_submip(self, leaf_dual, row_duals, ns_duals):
        """TODO: Documentation.
        """
        assert not self.generated_, "SP is already created."
        infinity = self.solver_.infinity()

        n_rows = self.X_.shape[0]
        self.z_vars_ = [None]*n_rows
        self.y_vars_ = [None]*n_rows
        objective = self.solver_.Objective()
        for i in range(n_rows):
            z_var = self.solver_.BoolVar('z_'+str(i))
            objective.SetCoefficient(z_var, 1)
            self.z_vars_[i] = z_var.index()

            y_var = self.solver_.BoolVar('y_'+str(i))
            row_dual = row_duals[i]
            objective.SetCoefficient(y_var, -row_dual)
            self.y_vars_[i] = y_var.index()

        self.u_vars_ = {}
        self.leaf_
        nodes = self.leaf_.left_nodes + self.leaf_.right_nodes
        all_splits = []
        for node_id in nodes:
            node = self.nodes_[node_id]
            self.u_vars_[node.id] = {}
            for split in node.candidate_splits:
                all_splits.append(split)
                u_var = self.solver_.BoolVar(
                    'u_'+str(node.id)+'_'+str(split))
                ns_dual = ns_duals[self.leaf_id_][node.id][split]
                objective.SetCoefficient(u_var, -ns_dual)
                self.u_vars_[node.id][split] = u_var.index()

        objective.SetOffset(-leaf_dual)
        objective.SetMaximization()

        # Constraints
        # For each node, exactly one feature is selected
        for node_id in nodes:
            node = self.nodes_[node_id]
            one_feature = self.solver_.Constraint(
                1, 1, "one_feature_" + str(node.id))
            for split in node.candidate_splits:
                u_var = self.solver_.variable(self.u_vars_[node.id][split])
                one_feature.SetCoefficient(u_var, 1)

        # Each feature is selected at max once in a path
        all_splits = list(dict.fromkeys(all_splits))
        for split in all_splits:
            unique_split = self.solver_.Constraint(
                0, 1, "unique_split_" + str(split))
            for node in self.nodes_:
                if node.id not in nodes:
                    continue
                if split not in node.candidate_splits:
                    continue
                u_var = self.solver_.variable(self.u_vars_[node.id][split])
                unique_split.SetCoefficient(u_var, 1)

        # Row follows correct path (upper and lower bound on y var)
        for i in range(n_rows):
            y_lb_cons = self.solver_.Constraint(
                -infinity, self.tree_depth_-1, "row_" + str(i))
            y_var = self.solver_.variable(self.y_vars_[i])
            y_lb_cons.SetCoefficient(y_var, -1)
            for node in self.nodes_:
                if node.id not in nodes:
                    continue
                rn_cons = self.solver_.Constraint(
                    -infinity, 0, "rn_" + str(i) + '_' + str(node.id))

                rn_cons.SetCoefficient(y_var, 1)
                if node.id in self.leaf_.left_nodes:
                    for split in node.candidate_splits:
                        if self.row_satisfies_split(i, self.splits_[split]):
                            u_var = self.solver_.variable(
                                self.u_vars_[node.id][split])
                            rn_cons.SetCoefficient(u_var, -1)
                            y_lb_cons.SetCoefficient(u_var, 1)
                elif node.id in self.leaf_.right_nodes:
                    for split in node.candidate_splits:
                        if not self.row_satisfies_split(i,
                                                        self.splits_[split]):
                            u_var = self.solver_.variable(
                                self.u_vars_[node.id][split])
                            rn_cons.SetCoefficient(u_var, -1)
                            y_lb_cons.SetCoefficient(u_var, 1)

        # only one class selected
        self.w_vars_ = [None]*len(self.targets_)
        single_target_cons = self.solver_.Constraint(0, 1, 'single_target')
        for i in range(len(self.targets_)):
            target = self.targets_[i]
            w_var = self.solver_.BoolVar('w_' + str(target))
            self.w_vars_[i] = w_var.index()
            single_target_cons.SetCoefficient(w_var, 1)
        # relation between w,y,z vars
        for i in range(n_rows):
            z_var = self.solver_.variable(self.z_vars_[i])
            y_var = self.solver_.variable(self.y_vars_[i])
            yz_cons = self.solver_.Constraint(-infinity, 0, "yz_" + str(i))
            yz_cons.SetCoefficient(z_var, 1)
            yz_cons.SetCoefficient(y_var, -1)
            correct_class = self.y_[i]
            w_var = self.solver_.variable(self.w_vars_[correct_class])
            wz_cons = self.solver_.Constraint(-infinity, 0, "wz_" + str(i))
            wz_cons.SetCoefficient(z_var, 1)
            wz_cons.SetCoefficient(w_var, -1)

        self.generated_ = True

    def update_objective(self, leaf_dual, row_duals, ns_duals):
        """TODO: Documentation.
        """
        objective = self.solver_.Objective()
        n_rows = self.X_.shape[0]
        for i in range(n_rows):
            z_var = self.solver_.variable(self.z_vars_[i])
            objective.SetCoefficient(z_var, 1)

            y_var = self.solver_.variable(self.y_vars_[i])
            row_dual = row_duals[i]
            objective.SetCoefficient(y_var, -row_dual)

        nodes = self.leaf_.left_nodes + self.leaf_.right_nodes
        for node_id in nodes:
            node = self.nodes_[node_id]
            for split in node.candidate_splits:
                u_var = self.solver_.variable(self.u_vars_[node.id][split])
                ns_dual = ns_duals[self.leaf_id_][node.id][split]
                objective.SetCoefficient(u_var, -ns_dual)

        objective.SetOffset(-leaf_dual)
        objective.SetMaximization()
        return

    def generate_columns(self, X, y, dual_costs, params=""):
        """TODO: Documentation.
        """
        self.X_ = X
        self.y_ = y
        leaf_dual = dual_costs[0][self.leaf_id_]
        row_duals = dual_costs[1]
        ns_duals = dual_costs[2]

        self.X_ = X
        self.y__ = y
        self.solver_.SetNumThreads(1)

        if self.generated_:
            self.update_objective(leaf_dual, row_duals, ns_duals)
        else:
            self.create_submip(leaf_dual, row_duals, ns_duals)

        # Solve sub problem
        result_status = self.solver_.Solve(get_params_from_string(params))

        # Empty column is always feasible.
        has_solution = (result_status == pywraplp.Solver.OPTIMAL or
                        result_status == pywraplp.Solver.FEASIBLE)
        assert has_solution

        # print(self.solver_.ExportModelAsLpFormat(False))

        # TODO: This threshold should be a parameter.
        print("Subproblem objective = ", self.solver_.Objective().Value())
        if self.solver_.Objective().Value() <= 1e-6:
            return []

        nodes = self.leaf_.left_nodes + self.leaf_.right_nodes
        path = Path()
        path.leaf_id = self.leaf_id_
        path.node_ids = []
        path.splits = []
        path.cost = 0
        path.id = -1
        for node in self.nodes_:
            if node.id not in nodes:
                continue
            path.node_ids.append(node.id)
            for split in node.candidate_splits:
                u_var = self.solver_.variable(self.u_vars_[node.id][split])
                if u_var.solution_value() > 0.5:
                    path.splits.append(split)
                    break
        for target in self.targets_:
            w_var = self.solver_.variable(self.w_vars_[target])
            if w_var.solution_value() > 0.5:
                path.target = target
                break
        n_rows = self.X_.shape[0]
        for i in range(n_rows):
            z_var = self.solver_.variable(self.z_vars_[i])
            path.cost += z_var.solution_value()
            if z_var.solution_value() > 0.5:
                print(self.z_vars_[i], z_var)
        return [path]

    def row_satisfies_split(self, row, split):
        """TODO: Documentation.
        """
        feature = split.feature
        threshold = split.threshold
        if self.X_[row, feature] <= threshold:
            return True
        return False


class DTreeClassifier(ColGenClassifier):
    """TODO: Documentation.
    """

    def __init__(self, initial_paths=[], leaves=[], nodes=[], splits=[],
                 tree_depth=1,
                 targets=[], max_iterations=-1, rmp_is_ip=True,
                 rmp_solver_params="",
                 master_ip_solver_params="", subproblem_params=""):

        self.initial_paths = initial_paths
        self.leaves = leaves
        self.nodes = nodes
        self.splits = splits
        self.tree_depth = tree_depth
        self.targets = targets
        self.subproblem_params = subproblem_params
        split_ids = []
        for split in splits:
            split_ids.append(split.id)
        # Each node must be in sequence.
        node_ids = []
        node_id = 0
        for node in nodes:
            assert node.id == node_id
            node_ids.append(node.id)
            node_id += 1

        # Each leaf must be in sequence.
        leaf_ids = []
        leaf_id = 0
        for leaf in leaves:
            assert leaf.id == leaf_id
            leaf_ids.append(leaf.id)
            leaf_id += 1
        # Each path can only contain nodes and leaves provided.
        for path in self.initial_paths:
            assert path.leaf_id in leaf_ids
            for node_id in path.node_ids:
                assert node_id in node_ids
            for split_id in path.splits:
                assert split_id in split_ids
            assert path.target in targets
            assert len(path.node_ids) == self.tree_depth
            assert len(path.splits) == self.tree_depth

        self.master_problem = DTreeMasterProblem(
            self.initial_paths, leaves, nodes, splits)
        self.subproblems = []
        all_subproblem_params = []
        for leaf in leaves:
            subproblem = DTreeSubProblem(
                leaf, nodes, splits, targets, self.tree_depth, 'cbc')
            self.subproblems.append(subproblem)
            all_subproblem_params.append(subproblem_params)
        super().__init__(max_iterations, self.master_problem, self.subproblems,
                         rmp_is_ip, rmp_solver_params, master_ip_solver_params,
                         all_subproblem_params)

    def _more_tags(self):
        """TODO: Documentation.
        """
        return {'X_types': ['categorical'],
                'non_deterministic': True,
                'binary_only': True}

    def predict(self, X):
        """TODO: Documentation.
        """
        selected_paths = self.master_problem.selected_paths
        # Check for each row, which path it satisfies. There should be exactly
        # one.
        num_rows = X.shape[0]
        y_predict = np.zeros(X.shape[0], dtype=int)
        for row in range(num_rows):
            for path in selected_paths:
                if row_satisfies_path(X, self.leaves[path.leaf_id],
                                      self.splits, row, path):
                    y_predict[row] = path.target
                    break
        return self.label_encoder_.inverse_transform(y_predict)
