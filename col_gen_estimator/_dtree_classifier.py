"""
Decision tree classifier using column generation.

Reference: Murat Firat, Guillaume Crognier, Adriana F. Gabor, C.A.J. Hurkens,
and Yingqian Zhang.
Column generation based heuristic for learning classification trees.
Computers and Operations Research, 116, apr 2020.

The algorithm takes candidate splits for each node of the decision tree as
inputs. A path is a set of nodes from root node to a leaf with a fixed split
for each node and a assigned target. The master problem uses a binary variable
for each possible path. The goal is to select a set of path that are consistent
among themselves and result in the best accuracy.

Since there are exponentially many paths, the path variables are generated
using column generation by a subproblem.

We only work with complete trees. That is all the paths in the tree have the
same depth d. Each internal node has two children.

We assume a fixed ordering of the nodes and leaves in the tree. The root node
is numbered 0 (ID of the node). The left child of the node (i) is (2i + 1) and
the right child of the node (i) is (2i + 2). Leaves are numberd from (0) to
(2^d) from left to right.

Master problem:
Sets
R set of rows in data file.
F set of features in data file.
N_{lf} , N_{int} leaf and internal (non-leaf) nodes in the decision tree.
p_{BT}(l) path to leaf l in binary tree.
DP_l$ set of decision paths ending in leaf l.
DP_l(j) subset of paths in DP_l, such that j in p_{BT}(l).
R^{l}(p) subset of rows directed to leaf l through path p.
S_j set of decision splits for node j.

Parameters
k depth of the decision tree.
CP(p) number of correct predictions/true positives for path p.

Decision Variables
x_p binary variable indicating that path p is assigned to leaf l.
rho_{j,a} binary variable indicating that split a has been assigned to node j.


Max     sum_{l in N_{lf}} sum_{p in DP_{l}} CP(p)x_p
s.t.    sum_{p in DP_{l}} x_p = 1,  forall l in  N_{lf}             (alpha)
		sum_{lin N_{lf}} sum_{p in DP_{l}:r in R^l(p)} x_p = 1
                                        forall r in R               (beta)
		sum_{p in DP_{l}:s(j) = a} x_p = rho_{j,a},
                                        forall l in N_{lf},
                                            j in p_{BT}(l) and N_{int},
                                            a in S_j                (gamma)
		x_p in {0,1},  forall p in DP_{l}, l in N_{lf}
		rho_{j,a} in {0,1} forall j in N_{int}, a in S_j

The objective maximizes the accuracy. The (alpha) constraints ensure that
exactly one path is selected for each leaf. The (beta) constraints ensure
that exactly one path is selected for each row in the dataset. They are implied
by the (alpha) and (gamma) constraints, but they are needed for stronger LP
relaxation. The (gamma) constraints ensure that the selected paths are
consistent (have the same split on common nodes).

Subproblem: (Modified from the original paper)
Sets
R & set of all rows in data file.
F set of features in data file.
Y set of all class labels.
S_j set of decision splits for node j.
LC(l) set of internal nodes with left child in p_{BT}(l).
RC(l) set of internal nodes with right child in p_{BT}(l).
T(r) set of splits for which row r returns a TRUE (feature <= threshold)
F(r) set of splits for which row r returns a FALSE: (feature > threshold)

Parameters
alpha_l dual cost of (alpha) constraint in the master problem for leaf l.
beta_r dual cost of (beta) constraint in the master problem for row r.
gamma_{l,j,a} dual cost of (gamma) constraint in the master problem for
        leaf l, node j, split a.

Decision Variables
y_r binary variable indicating that row r reaches leaf l.
z_r binary variable indicating that row r reaches leaf l and has the same
    target as the path being generated.
u_{j,a} binary variable indicating that split a is assigned to node j.

DPP(l):
Max sum_{r in R} z_r - alpha_l
    - sum_{j in p_{BT}(l)} sum_{a in S_j} gamma_{l,j,a} u_{j,a}
    - sum_{r in R} beta_r y_r
s.t. sum_{a in S_j} u_{j,a} = 1,  forall j in  RC(l) or LC(l)               (1)
	 y_r <= sum_{a in S_j and T(r)} u_{j,a}, forall j in LC(l), r in R      (2)
	 y_r <= sum_{a in S_j and F(r)} u_{j,a} , forall j in RC(l), r in R     (3)
	 sum_{j in LC(l)} sum_{a in S_j and T(r)} u_{j,a}
        + sum_{j in RC(l)} sum_{a in S_j and F(r)} u_{j,a} - (k-1) <= y_r,
                                            forall r in R                   (4)
	 sum_{j in p_{BT}(l)} u_{j,a} <= 1, forall a in (Union of all splits)   (5)
     z_r <= y_r, forall r in R                                              (6)
	 z_r <= w_t, forall r in R , t = correct class for row r                (7)
	 sum_{t \in Y} w_t = 1                                                  (8)
	 z_r in {0,1},  forall r in R                                           (9)
     y_r in {0,1},  forall r in R                                          (10)
	 u_{j,a} in {0,1} forall j in RC(l) or LC(l), a in S_j                 (11)

We have one subproblem for each leaf. This is different from the original paper
where there is one subproblem for each leaf and target combination.

The objective maximizes the reduced cost. The constraints (1) ensure that
exactly one split is selected for each node. The constraints (2)(3)(4) ensure
that the row follows the path when the corresponding splits are selected. The
constraints (5) ensure that a split is only selected once in a path. The
constraints (6)(7) ensure that the z variable is set only if the row follows
the path (y variable is set) and the right target is selected (w variable is
set). The constraint (8) ensures that exactly one target is selected for the
path being generated.
"""
import numpy as np
import math
import random
import multiprocessing as mp
from ortools.linear_solver import pywraplp
from ortools.linear_solver import linear_solver_pb2
from ortools.sat.python import cp_model
from math import floor, ceil
from sklearn.utils.validation import check_array, check_is_fitted
from ._col_gen_classifier import BaseMasterProblem
from ._col_gen_classifier import BaseSubproblem
from ._col_gen_classifier import ColGenClassifier

from time import time


class Row:
    """To make the processing faster, we store some row related information in
    this class.
    Attributes
    ----------
    id: (int) ID of the row.
    reachable_nodes: (set(int)) IDs of the nodes the row can reach.
    reachable_paths = (set(int)) IDs of the paths the row can follow.
    reachable_leaves = (set(int)) IDs of the leaves the row can reach.
    target: (int) Target of the row.
    left_splits: (set(int)) IDs of the splits where the feature value of the
        row is smaller or equal to the threshold.
    right_splits: (set(int)) IDs of the splits where the feature value of
        the row is greater than the threshold.
    removed_from_master: (bool) True if the row is removed from the master.
        This can happen if the row can only reach at max two leaves or the row
        is similar to some other row in the dataset with respect to the splits
        being considered.
    removed_from_sp: (bool) True if the row is removed from the master and the
        subproblem. This can happen if the row is similar to some other row in
        the dataset with respect to the splits being considered.
    weight: (int) Typically = 1. Some rows can have higher weights if we remove
        the other rows that are similar to this row from the dataset.
    """

    def __init__(self) -> None:
        self.id = -1
        self.reachable_nodes = set()
        self.reachable_paths = set()
        self.reachable_leaves = set()
        self.target = -1
        # List of satisfied splits.
        self.left_splits = set()
        # List of unsatisfied splits.
        self.right_splits = set()
        self.removed_from_master = False
        self.removed_from_sp = False
        self.weight = 1


class Path:
    """All path related information.
    Attributes
    ----------
    leaf_id: (int) ID of the associated leaf.
    node_ids: (list(int)) IDs of the nodes in the path.
    splits: (list(int)) IDs of the selected splits for nodes in the path.
        The order of splits must match the order of nodes.
    cost: (int) Number of rows following the path with correct target.
    id: (int) ID of the path.
    target: (int) Associated target of the path.
    """

    def __init__(self) -> None:
        self.leaf_id = -1
        self.node_ids = []
        self.splits = []
        self.cost = 0
        self.id = -1
        self.target = -1
        self.initial = False

    def is_same_as(self, path):
        """ Returns true if the current path is same as the path in the
        argument.
        Parameters
        ----------
        path: (Path) The other path being compared to.
        """
        if self.leaf_id != path.leaf_id:
            return False

        if self.target != path.target:
            return False

        if self.cost != path.cost:
            return False

        if len(self.node_ids) != len(path.node_ids):
            return False

        for i in range(len(self.node_ids)):
            node_id = self.node_ids[i]
            split_id = self.splits[i]
            found = False
            for j in range(len(path.node_ids)):
                if path.node_ids[j] == node_id and path.splits[j] == split_id:
                    found = True
                    break
            if not found:
                return False

        return True

    def print_path(self):
        """Prints the attributes of the path."""
        print("-------------------")
        print("ID: ", self.id)
        print("Nodes: ", self.node_ids)
        print("Splits: ", self.splits)
        print("Leaf: ", self.leaf_id)
        print("Target: ", self.target)
        print("Cost: ", self.cost)
        print("-------------------")

    def set_leaf(self, leaf_id, depth):
        """
        For a full binary tree, populates the nodes corresponding to the given
        leaf_id.

        Parameters
        ----------
        leaf_id: (int) The ID of the leaf.
        depth: (int) Depth of the path.
        """
        self.leaf_id = leaf_id
        self.node_ids = []
        count = 2**(depth) - 1 + leaf_id
        while(count > 0):
            father = math.floor((count-1) / 2)
            self.node_ids.append(father)
            count = father


class Node:
    """All node related information.

    Attributes
    ----------
    id: (int) ID of the node.
    candidate_splits: (list(int)) IDs of the candidate splits that can be used
        at this node.
    candidate_splits_count: (list(int)) Count of how many times the
        corresponding split was seen in initialization. This is useful to trim
        the candidate_splits list. Used only at the initialization and not
        during training.
    last_split: (int) ID of the last split added to the candidate_splits. Used
        only at the initialization and not during training.
    parent: (int) ID of the parent node. Parent of the root node is -1.
    # child is -1 for the leaves.
    left_child: (int) ID of the left child node. -1 if the child is a leaf.
    right_child: (int) ID of the right child node. -1 if the child is a leaf.
    children_are_leaves: (bool) True if the children of this node are leaves.
    """

    def __init__(self) -> None:
        self.id = -1
        self.candidate_splits = []
        self.candidate_splits_count = []
        self.last_split = -1
        # parent is -1 for the root node.
        self.parent = -1
        # child is -1 for the leaves.
        self.left_child = -1
        self.right_child = -1
        self.children_are_leaves = False


class Leaf:
    """All leaf related information.
    Attributes
    ----------
    id: (int) ID of the leaf.
    left_nodes: (list(int)) Set of nodes that take left branch to reach this
        leaf.
    right_nodes: (list(int)) Set of nodes that take right branch to reach this
        leaf.
    """

    def __init__(self) -> None:
        self.id = id
        self.left_nodes = []
        self.right_nodes = []

    def create_leaf(self, id, depth) -> None:
        """Given the id and depth of the leaf, populates the left_nodes and
        right_nodes attributes.
        Parameters
        ----------
        id: (int) ID of the leaf.
        depth: (int) Depth of the tree.
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
    """All split related information.
    Attributes
    ----------
    id: (int) ID of the split.
    feature: (int) Index of the feature used for the split.
    threshold: (float) Threshold value for the split. The row takes left branch
        if the feature <= threshold.
    left_rows: (set(int)) IDs of the rows that take left branch on this split.
        This is used for faster computing of the rows that follow a particular
        path.
    right_rows: (set(int)) IDs of the rows that take right branch on this
        split.
    left_cuts: (set(int)) Indices of the cuts added as (beta) constraints that
        correspopnd to a row that take left branch on this split. This is used
        for faster computing of the cuts that follow a particular path.
    right_cuts: (set(int)) Indices of the cuts added as (beta) constraints that
        correspopnd to a row that take left branch on this split.
    """

    def __init__(self) -> None:
        self.id = -1
        self.feature = -1
        self.threshold = -1
        # In reduced split case, some splits are removed.
        self.removed = False
        self.left_rows = set()
        self.right_rows = set()
        self.left_cuts = set()
        self.right_cuts = set()


def get_satisfied_rows(path, leaf, splits):
    """Returns the set of satisfied rows for the given path.
    We use set intersection to compute this faster. This requires that we store
    the set of rows that take a specific branch on each split.
    Parameters
    ----------
    path: (Path) the path.
    leaf: (Leaf) the leaf of the path. We pass this because the leaf contains
        the information about which nodes take left branch for this path.
    splits: (list(Split)) splits to get the rows that take the correct branch.
    """

    satisfied_rows = set()
    split_ids = path.splits
    node_ids = path.node_ids
    split_0 = splits[split_ids[0]]
    if node_ids[0] in leaf.left_nodes:
        satisfied_rows = split_0.left_rows
    else:
        satisfied_rows = split_0.right_rows
    for i in range(1, len(node_ids)):
        split = splits[split_ids[i]]
        if node_ids[i] in leaf.left_nodes:
            satisfied_rows = satisfied_rows.intersection(split.left_rows)
        else:  # Right node in the path
            satisfied_rows = satisfied_rows.intersection(split.right_rows)

    return satisfied_rows


def get_satisfied_cuts(path, leaf, splits):
    """Returns the set of satisfied cut (beta) rows for the given path.
    We use set intersection to compute this faster. This requires that we store
    the set of cuts that take a specific branch on each split.
    Parameters
    ----------
    path: (Path) the path.
    leaf: (Leaf) the leaf of the path. We pass this because the leaf contains
        the information about which nodes take left branch for this path.
    splits: (list(Split)) splits to get the cuts that take the correct branch.
    """

    satisfied_cuts = set()
    split_ids = path.splits
    node_ids = path.node_ids
    split_0 = splits[split_ids[0]]
    if node_ids[0] in leaf.left_nodes:
        satisfied_cuts = split_0.left_cuts
    else:
        satisfied_cuts = split_0.right_cuts
    for i in range(1, len(node_ids)):
        split = splits[split_ids[i]]
        if node_ids[i] in leaf.left_nodes:
            satisfied_cuts = satisfied_cuts.intersection(split.left_cuts)
        else:  # Right node in the path
            satisfied_cuts = satisfied_cuts.intersection(split.right_cuts)

    return satisfied_cuts


def row_satisfies_path(X_row, leaf, splits, path):
    """Returns true if the passed row follows the path. The row may have
    different target than the path.
    Parameters
    ----------
    X_row:  ndarray, shape (1, n_features)
        The datapoint representing the row.
    leaf: (Leaf) The leaf of the path.
    splits: (list(Split)) splits to get the rows that take the correct branch.
    path: (Path) The path.
    """
    # path.print_path()
    split_ids = path.splits
    node_ids = path.node_ids
    for i in range(len(node_ids)):
        split = splits[split_ids[i]]
        feature = split.feature
        threshold = split.threshold
        if node_ids[i] in leaf.left_nodes:
            if X_row[feature] > threshold:
                return False
        else:  # Right node in the path
            if X_row[feature] <= threshold:
                return False
    return True


def get_params_from_string(params):
    """ Given the params in string form, returns the MPSolverParameters.
    This method is not yet implemented completely.
    By default we keep the incrementality on and use the primal simplex as the
    lp algorithm.
    Parameters
    ----------
    params : string,
        The solver parameters in string format.
    """
    # TODO: Implement this method.
    solver_params = pywraplp.MPSolverParameters()
    solver_params.SetIntegerParam(
        pywraplp.MPSolverParameters.INCREMENTALITY,
        pywraplp.MPSolverParameters.INCREMENTALITY_ON)
    solver_params.SetIntegerParam(
        pywraplp.MPSolverParameters.LP_ALGORITHM,
        pywraplp.MPSolverParameters.PRIMAL)
    print(params)
    return solver_params


class CutGenerator(cp_model.CpSolverSolutionCallback):
    """Track intermediate solutions.
    Attributes
    ----------
    path_vars: (list(int)) Indices of path vars in the solver.
    split_vars: (list(int)) Indices of split vars in the solver.
    solution_count: (int) Count of all solutions seen so far. They may not have
        the desired objective value.
    cuts: list(list(bool)), shape (num cuts, num paths)
        List of all generated cuts. Each cut is a list of boolean representing
        whether or not the cut follows the path.
    cut_rows: list(list(bool)), shape (num cuts, num splits)
        List of all generated cuts. Each cut is a list of boolean representing
        whether or not the cut takes the left branch on the splits.
    orig_path_coeffs: (list(float)), shape (num paths)
        List of original path coefficients in the objective. We compare the
        solutions using this coefficients to get the original objective value
        without the scaling. The solution represents a valid cut if the
        original objective value is greater than 1.
    """

    def __init__(self, path_vars, split_vars, orig_path_coeffs):
        cp_model.CpSolverSolutionCallback.__init__(self)
        # path_vars are updated after each col gen iter.
        self.path_vars = path_vars
        self.split_vars = split_vars
        self.solution_count = 0
        self.cuts_ = []
        self.cut_rows = []
        self.orig_path_coeffs = orig_path_coeffs

    def on_solution_callback(self):
        self.solution_count += 1
        orig_obj = 0.0
        cut = [False] * len(self.path_vars)
        cut_row = [False] * len(self.split_vars)
        for i in range(len(self.path_vars)):
            v = self.path_vars[i]
            if self.Value(v) > 0:
                orig_obj += self.orig_path_coeffs[i]
                cut[i] = True
        for i in range(len(self.split_vars)):
            v = self.split_vars[i]
            if self.Value(v) > 0:
                cut_row[i] = True
        if orig_obj > 1 + 1e-6:
            self.cuts_.append(cut)
            self.cut_rows.append(cut_row)

    def solution_count(self):
        return self.solution_count

    def cuts(self):
        return (self.cuts_, self.cut_rows)


class DTreeMasterProblem(BaseMasterProblem):
    """TODO: Documentation.
    """

    def __init__(self, initial_paths, leaves, nodes, splits,
                 beta_constraints_as_cuts=False,
                 generate_cuts=False,
                 num_cuts_round=0,
                 solver_type='glop',
                 data_rows=None):
        super().__init__()
        # TODO: Switch to glop for testing.
        self.solver_ = pywraplp.Solver.CreateSolver(solver_type)
        self.cut_gen_model_ = cp_model.CpModel()
        self.cut_obj_scale = 1000000
        self.generated_ = False
        self.paths_ = initial_paths
        self.generated_paths_ = set()
        self.leaves_ = leaves
        self.nodes_ = nodes
        self.splits_ = splits
        self.data_rows = data_rows
        self.num_cuts_round = num_cuts_round
        self.generate_cuts = generate_cuts
        self.beta_constraints_as_cuts = beta_constraints_as_cuts
        self.total_cuts_added = 0
        self.cut_gen_time = 0.0
        self.rmp_objective_ = 0.0
        self.iter_ = 0
        # Experimental
        self.last_reset_iter_ = 0
        self.reset_timer_ = False

    def create_cut_gen_model(self):
        """Creates the model for cut generation using a CP-SAT solver."""
        # Takes value 1 if row follows left branch on split (satisfies split)
        self.cut_split_vars_ = [None]*len(self.splits_)
        for i in range(len(self.splits_)):
            self.cut_split_vars_[
                i] = self.cut_gen_model_.NewBoolVar('z_'+str(i))

        # Takes value 1 if row satisfies the path.
        self.cut_path_vars_ = [None]*len(self.paths_)
        for i, path in enumerate(self.paths_):
            self.cut_path_vars_[
                i] = self.cut_gen_model_.NewBoolVar('p_'+str(i))
            literals = []
            split_ids = path.splits
            node_ids = path.node_ids
            leaf = self.leaves_[path.leaf_id]
            for j in range(len(node_ids)):
                split_id = split_ids[j]
                if node_ids[j] in leaf.left_nodes:
                    literals.append(self.cut_split_vars_[split_id])
                else:  # Right node in the path
                    literals.append(self.cut_split_vars_[split_id].Not())
            negated_literals = [x.Not() for x in literals]

            # p => AND(splits)
            self.cut_gen_model_.AddBoolAnd(
                literals).OnlyEnforceIf(self.cut_path_vars_[i])
            # ~p => NOT(AND(splits)) = OR(NOT(splits))
            self.cut_gen_model_.AddBoolOr(
                negated_literals).OnlyEnforceIf(self.cut_path_vars_[i].Not())

        # Split dependency:
        # Split_i: f <= t1,
        # Split_j: f <= t2, t1 < t2,
        # Split_i => Split_j
        for i in range(len(self.splits_)):
            split_i = self.splits_[i]
            threshold_i = split_i.threshold
            split_i_var = self.cut_split_vars_[i]
            for j in range(i+1, len(self.splits_)):
                split_j = self.splits_[j]
                if split_i.feature != split_j.feature:
                    continue
                threshold_j = split_j.threshold
                split_j_var = self.cut_split_vars_[j]
                assert threshold_i != threshold_j
                if threshold_i < threshold_j:
                    self.cut_gen_model_.AddImplication(
                        split_i_var, split_j_var)
                else:
                    self.cut_gen_model_.AddImplication(
                        split_j_var, split_i_var)

    def generate_mp(self, X, y):
        """TODO: Documentation.
        """
        if self.generated_:
            return

        self.X_ = X
        self.y_ = y
        self.create_cut_gen_model()
        infinity = self.solver_.infinity()
        self.solver_.SetNumThreads(1)

        objective = self.solver_.Objective()
        self.path_vars_ = [None]*len(self.paths_)
        for i in range(len(self.paths_)):
            xp_var = self.solver_.IntVar(0.0, infinity, 'p_'+str(i))
            self.path_vars_[i] = xp_var.index()
            self.paths_[i].id = xp_var.index()
            # TODO: Compute cost of the path here?
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

        if self.data_rows == None:
            self.data_rows = self.preprocess_rows()
        else:
            self.compute_satisfied_paths()
        # row constraints
        n_rows = self.X_.shape[0]
        self.row_cons_ = [None]*n_rows
        self.added_row = [False]*n_rows
        for r in range(n_rows):
            if self.beta_constraints_as_cuts:
                break
            # continue
            if self.data_rows[r].removed_from_master:
                continue
            self.added_row[r] = True
            self.row_cons_[
                r] = self.solver_.Constraint(-infinity, 1, "row_"+str(r))
            for path in self.paths_:
                if row_satisfies_path(X[r], self.leaves_[path.leaf_id],
                                      self.splits_, path):
                    xp_var = self.solver_.variable(path.id)
                    self.row_cons_[r].SetCoefficient(xp_var, 1)

        # Cut constraints to be populated later.
        self.cut_cons_ = []

        # consistency constraints
        self.ns_vars = {}
        self.ns_constraints_ = {}
        for leaf in self.leaves_:
            self.ns_constraints_[leaf.id] = {}
            nodes = leaf.left_nodes + leaf.right_nodes
            for node_id in nodes:
                node = self.nodes_[node_id]
                self.ns_constraints_[leaf.id][node.id] = {}
                for split in node.candidate_splits:

                    ns_var_ind = self.ns_vars[(node.id, split)] \
                        if (node.id, split) in self.ns_vars \
                        else self.solver_.IntVar(0.0, infinity, "r_ns_" +
                                                 str(node.id) +
                                                 "_"+str(split)).index()
                    # else self.solver_.BoolVar("r_ns_" + str(node.id) +
                    #                           "_"+str(split)).index()
                    self.ns_vars[(node.id, split)] = ns_var_ind

                    ns_var = self.solver_.variable(ns_var_ind)

                    ns_constraint = self.solver_.Constraint(
                        0, 0, "ns_"+str(node.id)+"_"+str(split)+"_"
                        + str(leaf.id))

                    ns_constraint.SetCoefficient(ns_var, -1)

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

    def store_lp_solution(self):
        """Store the solution values in the class."""
        self.rmp_objective_ = self.solver_.Objective().Value()
        self.path_solution_values = {}
        for xp_var_id in self.path_vars_:
            xp_var = self.solver_.variable(xp_var_id)
            self.path_solution_values[xp_var_id] = xp_var.solution_value()

    def compute_satisfied_paths(self):
        for r, data_row in enumerate(self.data_rows):
            for path in self.paths_:
                if row_satisfies_path(self.X_[r], self.leaves_[path.leaf_id],
                                      self.splits_, path):
                    data_row.reachable_paths.add(path.id)

    def preprocess_rows(self, aggressive=False):
        """TODO: Documentation."""
        # preprocess nodes first
        n_leaves = len(self.leaves_)
        depth = int(math.log2(n_leaves))
        for node in self.nodes_:
            if node.id == 0:
                node.parent = -1
            else:
                node.parent = math.floor((node.id-1) / 2)

            node.left_child = 2 * node.id + 1
            node.right_child = 2 * node.id + 2
            if node.left_child >= 2**depth - 1:
                node.left_child -= (2**(depth) - 1)
                node.right_child -= (2**(depth) - 1)
                node.children_are_leaves = True

            # print(node.id, node.parent, node.left_child,
            #       node.right_child, node.children_are_leaves)

        # print("Splits:")
        # for split in self.splits_:
        #     feature = split.feature
        #     threshold = split.threshold
        #     print(split.id, feature, threshold)
        data_rows = []
        n_rows = self.X_.shape[0]
        removed_count = 0
        average_leaf_reach = 0.0
        for r in range(n_rows):
            data_row = Row()
            for split in self.splits_:
                feature = split.feature
                threshold = split.threshold
                if self.X_[r, feature] <= threshold:
                    data_row.left_splits.add(split.id)
                else:
                    data_row.right_splits.add(split.id)
            # Record which nodes it can reach.
            for node in self.nodes_:
                if node.parent == -1 or node.id in data_row.reachable_nodes:
                    data_row.reachable_nodes.add(node.id)
                    for split_id in node.candidate_splits:
                        if split_id in data_row.left_splits:
                            if node.children_are_leaves:
                                data_row.reachable_leaves.add(
                                    node.left_child)
                            else:
                                data_row.reachable_nodes.add(
                                    node.left_child)
                        else:
                            if node.children_are_leaves:
                                data_row.reachable_leaves.add(
                                    node.right_child)
                            else:
                                data_row.reachable_nodes.add(
                                    node.right_child)

            average_leaf_reach += len(data_row.reachable_leaves)
            # print(r, data_row.reachable_leaves)
            if len(data_row.reachable_leaves) <= 2:
                # print("Row reaches fewer leaves: ",
                #       r, data_row.reachable_leaves)
                # print(data_row.left_splits)
                # print(data_row.reachable_nodes)
                # print(self.X_[r])
                data_row.removed_from_master = True
                removed_count += 1

            # Record which paths it can follow.
            for path in self.paths_:
                if row_satisfies_path(self.X_[r], self.leaves_[path.leaf_id],
                                      self.splits_, path):
                    data_row.reachable_paths.add(path.id)

            data_rows.append(data_row)

        print("Fewer leaves removed rows: ", removed_count)
        average_leaf_reach /= n_rows
        print("Average leaf reach: ", average_leaf_reach)

        # Quadratic loop
        if not aggressive:
            return data_rows
        for r1 in range(n_rows):
            if data_rows[r1].removed_from_master:
                continue
            for r2 in range(r1+1, n_rows):
                if data_rows[r2].removed_from_master:
                    continue

                if data_rows[r1].reachable_nodes != \
                        data_rows[r2].reachable_nodes:
                    continue

                found_mismatch_split = False
                valid_splits = set()
                for node_id in data_rows[r1].reachable_nodes:
                    node = self.nodes_[node_id]
                    for split_id in node.candidate_splits:
                        valid_splits.add(split_id)

                for split_id in valid_splits:
                    left_membership_r1 = split_id in data_rows[r1].left_splits
                    left_membership_r2 = split_id in data_rows[r2].left_splits
                    if left_membership_r1 != left_membership_r2:
                        found_mismatch_split = True
                        break

                if not found_mismatch_split:
                    data_rows[r2].removed_from_master = True
                    removed_count += 1
                    # print("Rows are similar: ", r1, r2)

                # if data_rows[r1].left_splits == data_rows[r2].left_splits:
                #     data_rows[r2].removed_from_master = True
                #     removed_count += 1
                #     print("Rows are similar: ", r1, r2)

        print("Total removed rows: ", removed_count)
        return data_rows

    def add_column(self, path):
        """TODO: Documentation.
        """
        # print("Adding path")
        # path.print_path()
        objective = self.solver_.Objective()
        infinity = self.solver_.infinity()
        xp_var = self.solver_.IntVar(
            0.0, infinity, 'p_'+str(len(self.path_vars_)))
        self.path_vars_.append(xp_var.index())
        path.id = xp_var.index()
        objective.SetCoefficient(xp_var, path.cost)
        self.paths_.append(path)
        self.generated_paths_.add(path.id)

        # leaf constraints
        for leaf in self.leaves_:
            if path.leaf_id == leaf.id:
                self.leaf_cons_[leaf.id].SetCoefficient(xp_var, 1)

        # row constraints
        satisfied_rows = get_satisfied_rows(
            path, self.leaves_[path.leaf_id], self.splits_)
        for r in satisfied_rows:
            self.data_rows[r].reachable_paths.add(path.id)
            if not self.added_row[r]:
                continue
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

        # Cut constraints
        satisfied_cuts = get_satisfied_cuts(
            path, self.leaves_[path.leaf_id], self.splits_)
        for c in satisfied_cuts:
            self.cut_cons_[c].SetCoefficient(xp_var, 1)

        # Cut generation model.
        new_cut_path_var = self.cut_gen_model_.NewBoolVar(
            'p_'+str(len(self.cut_path_vars_)))
        self.cut_path_vars_.append(new_cut_path_var)
        literals = []
        split_ids = path.splits
        node_ids = path.node_ids
        leaf = self.leaves_[path.leaf_id]
        for j in range(len(node_ids)):
            split_id = split_ids[j]
            if node_ids[j] in leaf.left_nodes:
                literals.append(self.cut_split_vars_[split_id])
            else:  # Right node in the path
                literals.append(self.cut_split_vars_[split_id].Not())
        negated_literals = [x.Not() for x in literals]

        # p => AND(splits)
        self.cut_gen_model_.AddBoolAnd(
            literals).OnlyEnforceIf(new_cut_path_var)
        # ~p => NOT(AND(splits)) = OR(NOT(splits))
        self.cut_gen_model_.AddBoolOr(
            negated_literals).OnlyEnforceIf(new_cut_path_var.Not())
        return True

    def add_cuts(self):
        # Create the objective for the cut generation model
        path_coeffs = []
        orig_path_coeffs = []
        for path in self.paths_:
            id = path.id
            orig_path_coeff = self.path_solution_values[id]
            path_coeff = int(
                floor(orig_path_coeff * self.cut_obj_scale))
            path_coeffs.append(path_coeff)
            orig_path_coeffs.append(orig_path_coeff)
        obj = cp_model.LinearExpr.WeightedSum(self.cut_path_vars_, path_coeffs)
        self.cut_gen_model_.Maximize(obj)

        sp_solver = cp_model.CpSolver()
        sp_solver.parameters.num_search_workers = 4

        # Solve
        cut_generator = CutGenerator(
            self.cut_path_vars_, self.cut_split_vars_, orig_path_coeffs)
        status = sp_solver.Solve(self.cut_gen_model_, cut_generator)
        print("Cut generation status: ", sp_solver.StatusName(status))
        print("Cut generation objective: ", sp_solver.ObjectiveValue())
        cuts, cut_rows = cut_generator.cuts()
        if len(cuts) == 0:
            return []

        # Update splits to store cut information.
        num_existing_cuts = len(self.cut_cons_)
        for i, cut_row in enumerate(cut_rows):
            cut_index = num_existing_cuts + i
            for split in self.splits_:
                if cut_row[split.id]:
                    split.left_cuts.add(cut_index)
                else:
                    split.right_cuts.add(cut_index)

        for cut in cuts:
            r = len(self.cut_cons_)
            self.cut_cons_.append(self.solver_.Constraint(
                1, 1, "cut_"+str(r)))
            self.added_row.append(True)

            for i, path in enumerate(self.paths_):
                if cut[i]:
                    xp_var = self.solver_.variable(path.id)
                    self.cut_cons_[r].SetCoefficient(xp_var, 1)

        return cut_rows

    def get_satisfied_path_ids(self, row):
        """TODO: Documentation."""
        return list(self.data_rows[row].reachable_paths)

    def violated_row_constraint(self, satisfied_path_ids):
        """TODO: Documentation."""
        path_sum = 0.0
        for path_id in satisfied_path_ids:
            path_sum += self.path_solution_values[path_id]
        if abs(path_sum - 1) > 1e-6:
            return True
        return False

    def add_beta_cuts(self):
        assert self.generated_
        n_rows = self.X_.shape[0]
        # added_cuts = False
        infinity = self.solver_.infinity()
        num_cuts = 0
        for r in range(n_rows):
            if self.data_rows[r].removed_from_master:
                continue
            if num_cuts >= 10:
                break
            if self.added_row[r]:
                continue
            satisfied_path_ids = self.get_satisfied_path_ids(r)
            if (self.violated_row_constraint(satisfied_path_ids)):
                # added_cuts = True
                num_cuts += 1
                self.added_row[r] = True
                self.row_cons_[r] = self.solver_.Constraint(
                    -infinity, 1, "row_"+str(r))
                for path_id in satisfied_path_ids:
                    xp_var = self.solver_.variable(path_id)
                    self.row_cons_[r].SetCoefficient(xp_var, 1)
        return num_cuts

    def remove_gen_vars(self):
        print("Removing the generated paths. Setting UB to 0.")
        self.last_reset_iter_ = self.iter_
        self.reset_timer_ = True
        for path in self.paths_:
            if not path.initial:
                xp_var = self.solver_.variable(path.id)
                xp_var.SetUb(0)

    def solve_rmp(self, solver_params=''):
        """TODO: Documentation.
        """
        assert self.generated_
        self.prev_rmp_objective_ = self.rmp_objective_
        self.iter_ += 1

        self.rmp_result_status = self.solver_.Solve(
            get_params_from_string(solver_params))

        # TODO: Hide this under optional logging flag.
        print('Number of variables RMIP = %d' % self.solver_.NumVariables())
        print('Number of constraints RMIP = %d' %
              self.solver_.NumConstraints())
        newly_added_cuts = []
        if self.iter_ % 10 == 0:
            for i in range(self.num_cuts_round):
                if self.rmp_result_status == pywraplp.Solver.OPTIMAL:
                    # print("Before cut iter ", i)
                    # print(self.solver_.ExportModelAsLpFormat(False))
                    self.store_lp_solution()
                    t_start = time()
                    cut_rows = []
                    num_beta_cuts = 0
                    if self.beta_constraints_as_cuts:
                        num_beta_cuts = self.add_beta_cuts()
                    if self.generate_cuts and num_beta_cuts == 0:
                        cut_rows = self.add_cuts()
                    t_end = time()
                    self.cut_gen_time += t_end - t_start
                    self.total_cuts_added += len(cut_rows) + num_beta_cuts
                    print("Beta cuts added: ", num_beta_cuts)
                    print("Total cuts added: ", self.total_cuts_added)
                    print("Time spent in cut gen: ", self.cut_gen_time)
                    if (len(cut_rows) > 0) or num_beta_cuts > 0:
                        # Experiment: Cuts added. Remove the generated vars.
                        # self.remove_gen_vars()
                        cuts_params = get_params_from_string(solver_params)
                        cuts_params.SetIntegerParam(
                            pywraplp.MPSolverParameters.LP_ALGORITHM,
                            pywraplp.MPSolverParameters.DUAL)
                        self.rmp_result_status = self.solver_.Solve(
                            cuts_params)
                        if len(cut_rows) > 0:
                            newly_added_cuts.extend(cut_rows)
                    else:
                        break
                else:
                    break
        leaf_duals = []
        row_duals = []
        cut_duals = []
        ns_duals = {}
        if self.rmp_result_status == pywraplp.Solver.OPTIMAL:
            self.store_lp_solution()
            print('RMP Optimal objective value = %f' %
                  self.solver_.Objective().Value())

            # for var in self.solver_.variables():
            #     print(var.name(), var.solution_value())

            for lid in range(len(self.leaves_)):
                leaf_duals.append(self.leaf_cons_[lid].dual_value())

            n_rows = self.X_.shape[0]
            for r in range(n_rows):
                if self.added_row[r]:
                    row_duals.append(self.row_cons_[r].dual_value())
                else:
                    row_duals.append(0)

            for cut_cons in self.cut_cons_:
                cut_duals.append(cut_cons.dual_value())

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
            print("Master problem not solved correctly:",
                  self.rmp_result_status)
            # print(self.solver_.ExportModelAsLpFormat(False))

        return (leaf_duals, row_duals, ns_duals, cut_duals, newly_added_cuts)

    def rmp_objective_improved(self):
        if self.rmp_result_status == pywraplp.Solver.OPTIMAL:
            return self.prev_rmp_objective_ < self.rmp_objective_
        else:
            return False

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
                 depth, data_rows=None,
                 optimization_problem_type='sat') -> None:
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
        self.yc_vars_ = None
        self.data_rows = data_rows
        self.cut_rows_starting_index = 0
        self.iter_ = 0

    def create_submip(self, leaf_dual, row_duals, ns_duals, cut_duals):
        """TODO: Documentation.
        """
        assert not self.generated_, "SP is already created."
        infinity = self.solver_.infinity()
        self.solver_.SetNumThreads(7)

        n_rows = self.X_.shape[0]
        # Binary variables indicating that row reaches the leaf and has the
        # correct target.
        self.z_vars_ = [None]*n_rows
        # Binary variables indicating that row reaches the leaf.
        self.y_vars_ = [None]*n_rows
        objective = self.solver_.Objective()
        for i in range(n_rows):
            obj_coeff = 1
            if self.data_rows != None:
                if self.data_rows[i].removed_from_sp:
                    continue
                obj_coeff = self.data_rows[i].weight
            z_var = self.solver_.BoolVar('z_'+str(i))
            objective.SetCoefficient(z_var, obj_coeff)
            self.z_vars_[i] = z_var.index()

            y_var = self.solver_.BoolVar('y_'+str(i))
            row_dual = row_duals[i]
            objective.SetCoefficient(y_var, -row_dual)
            self.y_vars_[i] = y_var.index()

        num_cuts = len(cut_duals)
        self.yc_vars_ = [None]*num_cuts
        if num_cuts > 0:
            for i in range(num_cuts):
                yc_var = self.solver_.BoolVar('yc_'+str(i))
                cut_dual = cut_duals[i]
                objective.SetCoefficient(yc_var, -cut_dual)
                self.yc_vars_[i] = yc_var.index()

        # Binary variables u_j_a indicating that split s is assigned to the
        # node j.
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
        # all_splits = list(dict.fromkeys(all_splits))
        # for split in all_splits:
        #     unique_split = self.solver_.Constraint(
        #         0, 1, "unique_split_" + str(split))
        #     for node_id in nodes:
        #         node = self.nodes_[node_id]
        #         if split not in node.candidate_splits:
        #             continue
        #         u_var = self.solver_.variable(self.u_vars_[node.id][split])
        #         unique_split.SetCoefficient(u_var, 1)

        # Row follows correct path (upper and lower bound on y var)
        for i in range(n_rows):
            if self.data_rows != None:
                if self.data_rows[i].removed_from_sp:
                    continue
            y_lb_cons = self.solver_.Constraint(
                -infinity, self.tree_depth_-1, "row_" + str(i))
            y_var = self.solver_.variable(self.y_vars_[i])
            y_lb_cons.SetCoefficient(y_var, -1)
            for node_id in nodes:
                node = self.nodes_[node_id]
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
        # Binary variables indicating that the class is selected.
        self.w_vars_ = [None]*len(self.targets_)
        single_target_cons = self.solver_.Constraint(0, 1, 'single_target')
        for i in range(len(self.targets_)):
            target = self.targets_[i]
            w_var = self.solver_.BoolVar('w_' + str(target))
            self.w_vars_[i] = w_var.index()
            single_target_cons.SetCoefficient(w_var, 1)
        # relation between w,y,z vars
        for i in range(n_rows):
            if self.data_rows != None:
                if self.data_rows[i].removed_from_sp:
                    continue
            z_var = self.solver_.variable(self.z_vars_[i])
            y_var = self.solver_.variable(self.y_vars_[i])
            # z <= y (row reaches leaf)
            yz_cons = self.solver_.Constraint(-infinity, 0, "yz_" + str(i))
            yz_cons.SetCoefficient(z_var, 1)
            yz_cons.SetCoefficient(y_var, -1)

            # z <= w (row has correct target)
            correct_class = self.y_[i]
            w_var = self.solver_.variable(self.w_vars_[correct_class])
            wz_cons = self.solver_.Constraint(-infinity, 0, "wz_" + str(i))
            wz_cons.SetCoefficient(z_var, 1)
            wz_cons.SetCoefficient(w_var, -1)

        self.generated_ = True

    def update_objective(self, leaf_dual, row_duals, ns_duals, cut_duals):
        """TODO: Documentation.
        """
        objective = self.solver_.Objective()
        n_rows = self.X_.shape[0]
        for i in range(n_rows):
            obj_coeff = 1
            if self.data_rows != None:
                if self.data_rows[i].removed_from_sp:
                    continue
                obj_coeff = self.data_rows[i].weight
            z_var = self.solver_.variable(self.z_vars_[i])
            objective.SetCoefficient(z_var, obj_coeff)

            y_var = self.solver_.variable(self.y_vars_[i])
            row_dual = row_duals[i]
            objective.SetCoefficient(y_var, -row_dual)

        num_cuts = len(cut_duals)
        existing_cuts = len(self.yc_vars_)
        if num_cuts > 0:
            for i in range(num_cuts):
                yc_var = None
                if i >= existing_cuts:
                    yc_var = self.solver_.BoolVar('yc_'+str(i))
                    self.yc_vars_.append(yc_var.index())
                else:
                    yc_var = self.solver_.variable(self.yc_vars_[i])
                cut_dual = cut_duals[i]
                objective.SetCoefficient(yc_var, -cut_dual)

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

    def add_cut_rows(self, cut_rows):
        assert self.generated_
        infinity = self.solver_.infinity()
        nodes = self.leaf_.left_nodes + self.leaf_.right_nodes
        n_cut_rows = len(cut_rows)
        # Row follows correct path (upper and lower bound on y var)
        for i in range(n_cut_rows):
            cut_index = i + self.cut_rows_starting_index
            yc_lb_cons = self.solver_.Constraint(
                -infinity, self.tree_depth_-1, "cut_" + str(cut_index))
            # yc_var is created in update_objective method.
            yc_var = self.solver_.variable(self.yc_vars_[cut_index])
            yc_lb_cons.SetCoefficient(yc_var, -1)
            for node_id in nodes:
                node = self.nodes_[node_id]
                cn_cons = self.solver_.Constraint(
                    -infinity, 0, "cn_" + str(cut_index) + '_' + str(node.id))

                cn_cons.SetCoefficient(yc_var, 1)
                if node.id in self.leaf_.left_nodes:
                    for split in node.candidate_splits:
                        if cut_rows[i][split]:
                            u_var = self.solver_.variable(
                                self.u_vars_[node.id][split])
                            cn_cons.SetCoefficient(u_var, -1)
                            yc_lb_cons.SetCoefficient(u_var, 1)
                elif node.id in self.leaf_.right_nodes:
                    for split in node.candidate_splits:
                        if not cut_rows[i][split]:
                            u_var = self.solver_.variable(
                                self.u_vars_[node.id][split])
                            cn_cons.SetCoefficient(u_var, -1)
                            yc_lb_cons.SetCoefficient(u_var, 1)
        self.cut_rows_starting_index += n_cut_rows

    def generate_columns(self, X, y, dual_costs, params=""):
        """TODO: Documentation.
        """

        # Solve sub problem
        result_status = self.solver_.Solve(get_params_from_string(params))

        has_solution = (result_status == pywraplp.Solver.OPTIMAL or
                        result_status == pywraplp.Solver.FEASIBLE)

        # The current path is always feasible.
        # # TODO: Fix this. Sat solver returns infeasible sometimes
        # (even without cuts).
        if not has_solution:
            return []
            # print(self.solver_.ExportModelAsLpFormat(False))
        assert has_solution, "Result status: " + str(result_status)

        # if self.leaf_id_ == 4:
        #     sp_mps = self.solver_.ExportModelAsMpsFormat(fixed_format=False,
        #                                                  obfuscated=False)
        #     name = "sp_" + str(self.tree_depth_) + "_" + \
        #         str(self.iter_) + ".mps"
        #     self.iter_ += 1
        #     f = open(name, "w")
        #     f.write(sp_mps)
        #     f.close()

        # TODO: This threshold should be a parameter.
        print("Subproblem ", self.leaf_id_, " objective = ",
              self.solver_.Objective().Value())
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
            obj_coeff = 1
            if self.data_rows != None:
                if self.data_rows[i].removed_from_sp:
                    continue
                obj_coeff = self.data_rows[i].weight
            z_var = self.solver_.variable(self.z_vars_[i])

            path.cost += obj_coeff * z_var.solution_value()
            # if z_var.solution_value() > 0.5:
            #     print(self.z_vars_[i], z_var)
        return [path]

    def update_subproblem(self, X, y, dual_costs):
        self.X_ = X
        self.y_ = y
        leaf_dual = dual_costs[0][self.leaf_id_]
        row_duals = dual_costs[1]
        ns_duals = dual_costs[2]
        cut_duals = dual_costs[3]
        cut_rows = dual_costs[4]

        if self.generated_:
            self.update_objective(leaf_dual, row_duals, ns_duals, cut_duals)
        else:
            self.create_submip(leaf_dual, row_duals, ns_duals, cut_duals)

        if len(cut_rows) > 0:
            self.add_cut_rows(cut_rows)

    def row_satisfies_split(self, row, split):
        """TODO: Documentation.
        """
        feature = split.feature
        threshold = split.threshold
        if self.X_[row, feature] <= threshold:
            return True
        return False


class PathGenerator(cp_model.CpSolverSolutionCallback):
    """Track intermediate solutions of sp.
    Attributes
    ----------
    TODO
    """

    def __init__(self, u_vars, w_vars, z_vars, y_vars,
                 yc_vars,
                 leaf_dual,
                 leaf, targets,
                 nodes,
                 orig_obj_coeffs):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.u_vars = u_vars
        self.w_vars = w_vars
        self.z_vars = z_vars
        self.y_vars = y_vars
        self.yc_vars = yc_vars
        self.leaf = leaf
        self.leaf_dual = leaf_dual
        self.targets = targets
        self.nodes = nodes
        self.orig_obj_coeffs = orig_obj_coeffs
        self.solution_count = 0
        self.paths = []
        self.original_objective = 0.0

    def on_solution_callback(self):
        self.solution_count += 1
        original_objective = 0.0
        path_cost = 0.0
        for z_var in self.z_vars:
            if z_var == None:
                continue
            path_cost += self.orig_obj_coeffs[z_var] * \
                self.Value(z_var)

        original_objective += path_cost

        for y_var in self.y_vars:
            if y_var == None:
                continue
            original_objective += self.orig_obj_coeffs[y_var] * \
                self.Value(y_var)

        for yc_var in self.yc_vars:
            original_objective += self.orig_obj_coeffs[yc_var] * \
                self.Value(yc_var)

        nodes = self.leaf.left_nodes + self.leaf.right_nodes
        for node_id in nodes:
            node = self.nodes[node_id]
            for split in node.candidate_splits:
                u_var = self.u_vars[node.id][split]
                original_objective += self.orig_obj_coeffs[u_var] * \
                    self.Value(u_var)

        original_objective -= self.leaf_dual

        self.original_objective = original_objective
        if original_objective <= 1e-6:
            return

        nodes = self.leaf.left_nodes + self.leaf.right_nodes
        path = Path()
        path.leaf_id = self.leaf.id
        path.node_ids = []
        path.splits = []
        path.cost = path_cost
        path.id = -1
        for node_id in nodes:
            node = self.nodes[node_id]
            path.node_ids.append(node.id)
            for split in node.candidate_splits:
                u_var = self.u_vars[node.id][split]
                if self.Value(u_var) > 0.5:
                    path.splits.append(split)
                    break
        for target in self.targets:
            w_var = self.w_vars[target]
            if self.Value(w_var) > 0.5:
                path.target = target
                break
        self.paths.append(path)
        # print("Path cost", path_cost)


class DTreeSubProblemSat(BaseSubproblem):
    """TODO: Documentation."""

    def __init__(self, leaf, nodes, splits, targets,
                 depth, data_rows=None) -> None:
        super().__init__()
        self.leaf_id_ = leaf.id
        self.leaf_ = leaf
        self.nodes_ = nodes
        self.splits_ = splits
        self.targets_ = targets
        self.tree_depth_ = depth
        self.model_ = cp_model.CpModel()
        self.objective_scale = 100000
        self.orig_obj_coeffs = {}
        self.generated_ = False
        self.z_vars_ = None
        self.yc_vars_ = None
        self.data_rows = data_rows
        self.cut_rows_starting_index = 0

    def scale_obj_coeff(self, value):
        return int(round(value * self.objective_scale))

    def create_submip(self, leaf_dual, row_duals, ns_duals, cut_duals):
        """TODO: Documentation.
        """
        assert not self.generated_, "SP is already created."

        # obj = cp_model.LinearExpr()

        n_rows = self.X_.shape[0]
        # Binary variables indicating that row reaches the leaf and has the
        # correct target.
        self.z_vars_ = [None]*n_rows
        # Binary variables indicating that row reaches the leaf.
        self.y_vars_ = [None]*n_rows
        obj_vars = []
        obj_coeffs = []

        for i in range(n_rows):
            z_coeff = 1
            if self.data_rows != None:
                if self.data_rows[i].removed_from_sp:
                    continue
                z_coeff = self.data_rows[i].weight
            z_var = self.model_.NewBoolVar('z_'+str(i))
            self.orig_obj_coeffs[z_var] = z_coeff
            z_coeff = self.scale_obj_coeff(z_coeff)
            self.z_vars_[i] = (z_var)
            # obj += cp_model.LinearExpr.Term(z_var, z_coeff)
            obj_vars.append(z_var)
            obj_coeffs.append(z_coeff)

            y_var = self.model_.NewBoolVar('y_'+str(i))
            self.orig_obj_coeffs[y_var] = -row_duals[i]
            y_coeff = self.scale_obj_coeff(-row_duals[i])
            self.y_vars_[i] = (y_var)
            # obj += cp_model.LinearExpr.Term(y_var, y_coeff)
            obj_vars.append(y_var)
            obj_coeffs.append(y_coeff)

        num_cuts = len(cut_duals)
        self.yc_vars_ = [None]*num_cuts
        if num_cuts > 0:
            for i in range(num_cuts):
                yc_var = self.model_.NewBoolVar('yc_'+str(i))
                self.orig_obj_coeffs[yc_var] = -cut_duals[i]
                yc_coeff = self.scale_obj_coeff(-cut_duals[i])
                self.yc_vars_[i] = yc_var
                # obj += cp_model.LinearExpr.Term(yc_var, yc_coeff)
                obj_vars.append(yc_var)
                obj_coeffs.append(yc_coeff)

        # Binary variables u_j_a indicating that split s is assigned to the
        # node j.
        self.u_vars_ = {}
        nodes = self.leaf_.left_nodes + self.leaf_.right_nodes
        all_splits = []
        for node_id in nodes:
            node = self.nodes_[node_id]
            self.u_vars_[node.id] = {}
            for split in node.candidate_splits:
                all_splits.append(split)
                u_var = self.model_.NewBoolVar(
                    'u_'+str(node.id)+'_'+str(split))
                self.orig_obj_coeffs[u_var] = \
                    -ns_duals[self.leaf_id_][node.id][split]
                u_coeff = self.scale_obj_coeff(
                    -ns_duals[self.leaf_id_][node.id][split])
                self.u_vars_[node.id][split] = u_var
                # obj += cp_model.LinearExpr.Term(u_var, u_coeff)
                obj_vars.append(u_var)
                obj_coeffs.append(u_coeff)

        # obj += -self.scale_obj_coeff(leaf_dual)
        obj = cp_model.LinearExpr.WeightedSum(obj_vars, obj_coeffs) - leaf_dual
        self.model_.Maximize(obj)

        # Constraints
        # For each node, exactly one feature is selected
        for node_id in nodes:
            node = self.nodes_[node_id]
            cons_vars = []
            for split in node.candidate_splits:
                u_var = self.u_vars_[node.id][split]
                cons_vars.append(u_var)
            self.model_.Add(cp_model.LinearExpr.Sum(cons_vars) == 1)

        # Each feature is selected at max once in a path
        # all_splits = list(dict.fromkeys(all_splits))
        # for split in all_splits:
        #     cons_vars = []
        #     for node_id in nodes:
        #         node = self.nodes_[node_id]
        #         if split not in node.candidate_splits:
        #             continue
        #         u_var = self.u_vars_[node.id][split]
        #         cons_vars.append(u_var)
        #     self.model_.Add(cp_model.LinearExpr.Sum(cons_vars) <= 1)

        # Row follows correct path (upper and lower bound on y var)
        for i in range(n_rows):
            if self.data_rows != None:
                if self.data_rows[i].removed_from_sp:
                    continue
            y_lb_expr = cp_model.LinearExpr.Term(self.y_vars_[i], -1)
            for node_id in nodes:
                node = self.nodes_[node_id]
                rn_expr = cp_model.LinearExpr.Term(self.y_vars_[i], 1)

                if node.id in self.leaf_.left_nodes:
                    for split in node.candidate_splits:
                        if self.row_satisfies_split(i, self.splits_[split]):
                            u_var = self.u_vars_[node.id][split]
                            rn_expr += cp_model.LinearExpr.Term(u_var, -1)
                            y_lb_expr += cp_model.LinearExpr.Term(u_var, 1)
                elif node.id in self.leaf_.right_nodes:
                    for split in node.candidate_splits:
                        if not self.row_satisfies_split(i,
                                                        self.splits_[split]):
                            u_var = self.u_vars_[node.id][split]
                            rn_expr += cp_model.LinearExpr.Term(u_var, -1)
                            y_lb_expr += cp_model.LinearExpr.Term(u_var, 1)
                self.model_.Add(rn_expr <= 0)
            self.model_.Add(y_lb_expr <= self.tree_depth_-1)

        # only one class selected
        # Binary variables indicating that the class is selected.
        self.w_vars_ = [None]*len(self.targets_)
        for i in range(len(self.targets_)):
            target = self.targets_[i]
            w_var = self.model_.NewBoolVar('w_'+str(target))
            self.w_vars_[i] = w_var
        # Note: This can be added as <= 1 inequality because of the objective.
        self.model_.Add(cp_model.LinearExpr.Sum(self.w_vars_) == 1)

        # relation between w,y,z vars
        for i in range(n_rows):
            if self.data_rows != None:
                if self.data_rows[i].removed_from_sp:
                    continue
            # z <= y (row reaches leaf)
            self.model_.Add(self.z_vars_[i] <= self.y_vars_[i])

            # z <= w (row has correct target)
            correct_class = self.y_[i]
            self.model_.Add(self.z_vars_[i] <= self.w_vars_[correct_class])

        self.generated_ = True

    def update_objective(self, leaf_dual, row_duals, ns_duals, cut_duals):
        """TODO: Documentation.
        """
        assert self.generated_, "Called update_objective before generating SP"
        # obj = cp_model.LinearExpr()
        # TODO: This method is slow. Make it faster.
        obj_vars = []
        obj_coeffs = []

        n_rows = self.X_.shape[0]

        for i in range(n_rows):
            z_coeff = 1
            if self.data_rows != None:
                if self.data_rows[i].removed_from_sp:
                    continue
                z_coeff = self.data_rows[i].weight
            z_var = self.z_vars_[i]
            self.orig_obj_coeffs[z_var] = z_coeff
            z_coeff = self.scale_obj_coeff(z_coeff)

            # obj += cp_model.LinearExpr.Term(z_var, z_coeff)
            obj_vars.append(z_var)
            obj_coeffs.append(z_coeff)

            y_coeff = self.scale_obj_coeff(-row_duals[i])
            y_var = self.y_vars_[i]
            self.orig_obj_coeffs[y_var] = -row_duals[i]
            # obj += cp_model.LinearExpr.Term(y_var, y_coeff)
            obj_vars.append(y_var)
            obj_coeffs.append(y_coeff)

        num_cuts = len(cut_duals)
        existing_cuts = len(self.yc_vars_)
        if num_cuts > 0:
            for i in range(num_cuts):
                yc_var = None
                if i >= existing_cuts:
                    yc_var = self.model_.NewBoolVar('yc_'+str(i))
                    self.yc_vars_.append(yc_var)
                else:
                    yc_var = self.yc_vars_[i]
                yc_coeff = self.scale_obj_coeff(-cut_duals[i])
                self.orig_obj_coeffs[yc_var] = -cut_duals[i]
                # obj += cp_model.LinearExpr.Term(yc_var, yc_coeff)
                obj_vars.append(yc_var)
                obj_coeffs.append(yc_coeff)

        # Binary variables u_j_a indicating that split s is assigned to the
        # node j.
        nodes = self.leaf_.left_nodes + self.leaf_.right_nodes
        for node_id in nodes:
            node = self.nodes_[node_id]
            for split in node.candidate_splits:
                u_coeff = self.scale_obj_coeff(
                    -ns_duals[self.leaf_id_][node.id][split])
                u_var = self.u_vars_[node.id][split]
                self.orig_obj_coeffs[u_var] = \
                    -ns_duals[self.leaf_id_][node.id][split]
                # obj += cp_model.LinearExpr.Term(u_var, u_coeff)
                obj_vars.append(u_var)
                obj_coeffs.append(u_coeff)

        # obj -= self.scale_obj_coeff(leaf_dual)
        obj = cp_model.LinearExpr.WeightedSum(obj_vars, obj_coeffs) - leaf_dual
        self.model_.Maximize(obj)
        return

    def add_cut_rows(self, cut_rows):
        assert self.generated_
        nodes = self.leaf_.left_nodes + self.leaf_.right_nodes
        n_cut_rows = len(cut_rows)
        # Row follows correct path (upper and lower bound on y var)
        for i in range(n_cut_rows):
            cut_index = i + self.cut_rows_starting_index
            yc_lb_expr = cp_model.LinearExpr.Term(self.yc_vars_[cut_index], -1)
            for node_id in nodes:
                node = self.nodes_[node_id]
                cn_expr = cp_model.LinearExpr.Term(self.yc_vars_[cut_index], 1)

                if node.id in self.leaf_.left_nodes:
                    for split in node.candidate_splits:
                        if cut_rows[i][split]:
                            u_var = self.u_vars_[node.id][split]
                            cn_expr += cp_model.LinearExpr.Term(u_var, -1)
                            yc_lb_expr += cp_model.LinearExpr.Term(u_var, 1)
                elif node.id in self.leaf_.right_nodes:
                    for split in node.candidate_splits:
                        if not cut_rows[i][split]:
                            u_var = self.u_vars_[node.id][split]
                            cn_expr += cp_model.LinearExpr.Term(u_var, -1)
                            yc_lb_expr += cp_model.LinearExpr.Term(u_var, 1)
                self.model_.Add(cn_expr <= 0)
            self.model_.Add(yc_lb_expr <= self.tree_depth_-1)
        self.cut_rows_starting_index += n_cut_rows

    def update_subproblem(self, X, y, dual_costs):
        cut_rows = dual_costs[4]
        if len(cut_rows) == 0:
            return
        self.X_ = X
        self.y_ = y
        leaf_dual = dual_costs[0][self.leaf_id_]
        row_duals = dual_costs[1]
        ns_duals = dual_costs[2]
        cut_duals = dual_costs[3]

        if self.generated_:
            self.update_objective(leaf_dual, row_duals, ns_duals, cut_duals)
        else:
            self.create_submip(leaf_dual, row_duals, ns_duals, cut_duals)

        if len(cut_rows) > 0:
            self.add_cut_rows(cut_rows)

    def row_satisfies_split(self, row, split):
        """TODO: Documentation.
        """
        feature = split.feature
        threshold = split.threshold
        if self.X_[row, feature] <= threshold:
            return True
        return False

    def generate_columns(self, X, y, dual_costs, params=""):
        """TODO: Documentation.
        """
        self.X_ = X
        self.y_ = y
        leaf_dual = dual_costs[0][self.leaf_id_]
        row_duals = dual_costs[1]
        ns_duals = dual_costs[2]
        cut_duals = dual_costs[3]
        if self.generated_:
            self.update_objective(leaf_dual, row_duals, ns_duals, cut_duals)
        else:
            self.create_submip(leaf_dual, row_duals, ns_duals, cut_duals)

        # Solve sub problem
        sp_solver = cp_model.CpSolver()
        sp_solver.parameters.num_search_workers = 7

        # Solve
        leaf_dual = dual_costs[0][self.leaf_id_]
        path_generator = PathGenerator(self.u_vars_, self.w_vars_, self.z_vars_,
                                       self.y_vars_,
                                       self.yc_vars_, leaf_dual, self.leaf_,
                                       self.targets_,
                                       self.nodes_,
                                       self.orig_obj_coeffs)
        status = sp_solver.Solve(self.model_, path_generator)
        # print(self.leaf_id_, " Path generation status: ",
        #       sp_solver.StatusName(status))
        print(self.leaf_id_, " Path generation objective: ",
              path_generator.original_objective)
        paths = path_generator.paths
        return paths


class DTreeSubProblemHeuristic(BaseSubproblem):
    """TODO: Documentation."""

    def __init__(self, leaves, nodes, splits, targets,
                 depth, data_rows=None) -> None:
        super().__init__()
        self.leaves_ = leaves
        self.nodes_ = nodes
        self.splits_ = splits
        self.targets_ = targets
        self.tree_depth_ = depth
        self.data_rows = data_rows
        self.num_cuts = 0
        self.failed_rounds = 0
        self.success_rounds = 0

    def generate_columns(self, X, y, dual_costs, params=""):
        """ TODO: Documentation."""
        # Generate random columns and return the ones with postive reduced
        # cost.
        generated_paths = []
        # return generated_paths
        for iter in range(100):
            path = Path()
            # Pick a leaf
            leaf = random.choice(self.leaves_)
            path.leaf_id = leaf.id

            # Pick splits for each node.
            node_ids = leaf.left_nodes + leaf.right_nodes
            path.node_ids = node_ids
            path.splits = []
            success = True
            for node_id in node_ids:
                node = self.nodes_[node_id]
                candidate_splits = node.candidate_splits.copy()
                for used_split in path.splits:
                    if used_split in candidate_splits:
                        candidate_splits.remove(used_split)
                if not candidate_splits:
                    success = False
                    break
                chosen_split = random.choice(candidate_splits)
                path.splits.append(chosen_split)

            if not success:
                self.failed_rounds += 1
                continue

            # Compute the best target.
            n_rows = X.shape[0]

            possible_targets = {}
            best_target = -1
            best_target_count = 0
            row_satisfies_path_array = [False]*n_rows

            satisfied_rows = get_satisfied_rows(path, leaf, self.splits_)
            # print(satisfied_rows)
            for row in satisfied_rows:
                row_satisfies_path_array[row] = True
                row_weight = 1
                if self.data_rows != None:
                    row_weight = self.data_rows[row].weight
                    if self.data_rows[row].removed_from_sp:
                        continue

                target = y[row]
                if target in possible_targets.keys():
                    possible_targets[target] += row_weight
                else:
                    possible_targets[target] = row_weight

                if possible_targets[target] > best_target_count:
                    best_target_count = possible_targets[target]
                    best_target = target
            path.target = best_target
            path.cost = best_target_count

            already_generated = False
            for old_path in generated_paths:
                if path.is_same_as(old_path):
                    already_generated = True
                    break

            if already_generated:
                self.failed_rounds += 1
                continue

            # Evaluate the reduced cost.
            reduced_cost = self.get_reduced_cost(
                X, y, dual_costs, path, row_satisfies_path_array)
            if reduced_cost > 1e-6:
                # print("Generated new path: ", len(generated_paths))
                # path.print_path()
                generated_paths.append(path)
            else:
                self.failed_rounds += 1

        self.success_rounds += len(generated_paths)

        return generated_paths

    def update_subproblem(self, X, y, dual_costs):
        cut_rows = dual_costs[4]

        for i, cut_row in enumerate(cut_rows):
            cut_index = self.num_cuts + i
            for split in self.splits_:
                if cut_row[split.id]:
                    split.left_cuts.add(cut_index)
                else:
                    split.right_cuts.add(cut_index)
        self.num_cuts += len(cut_rows)

    def get_reduced_cost(self, X, y, dual_costs, path,
                         row_satisfies_path_array):
        """TODO: Documentation."""

        n_rows = X.shape[0]
        leaves_dual = dual_costs[0]
        row_duals = dual_costs[1]
        ns_duals = dual_costs[2]
        cut_duals = dual_costs[3]

        reduced_cost = 0
        try:
            reduced_cost -= leaves_dual[path.leaf_id]
        except IndexError:
            print("List index out of error: ", path.leaf_id, len(leaves_dual))

        reduced_cost += path.cost

        for row in range(n_rows):
            if row_satisfies_path_array[row]:
                reduced_cost -= row_duals[row]

        satisfied_cuts = get_satisfied_cuts(
            path, self.leaves_[path.leaf_id], self.splits_)
        for c in satisfied_cuts:
            reduced_cost -= cut_duals[c]

        for i in range(len(path.node_ids)):
            node_id = path.node_ids[i]
            split_id = path.splits[i]
            reduced_cost -= ns_duals[path.leaf_id][node_id][split_id]

        return reduced_cost


class DTreeSubProblemOld(BaseSubproblem):
    """TODO: Documentation."""

    def __init__(self, leaf, nodes, splits, target,
                 depth, data_rows=None,
                 optimization_problem_type='sat') -> None:
        super().__init__()
        self.leaf_id_ = leaf.id
        self.leaf_ = leaf
        self.nodes_ = nodes
        self.splits_ = splits
        self.target_ = target
        self.tree_depth_ = depth
        self.solver_ = pywraplp.Solver.CreateSolver(optimization_problem_type)
        self.generated_ = False
        self.yc_vars_ = None
        self.data_rows = data_rows
        self.cut_rows_starting_index = 0
        self.iter_ = 0

    def create_submip(self, leaf_dual, row_duals, ns_duals, cut_duals):
        """TODO: Documentation.
        """
        assert not self.generated_, "SP is already created."
        infinity = self.solver_.infinity()
        self.solver_.SetNumThreads(7)

        n_rows = self.X_.shape[0]
        # Binary variables indicating that row reaches the leaf.
        self.y_vars_ = [None]*n_rows
        objective = self.solver_.Objective()
        for i in range(n_rows):
            obj_coeff = 1
            if self.data_rows != None:
                if self.data_rows[i].removed_from_sp:
                    continue
                obj_coeff = self.data_rows[i].weight

            y_var = self.solver_.BoolVar('y_'+str(i))
            row_dual = row_duals[i]
            if self.y_[i] == self.target_:
                objective.SetCoefficient(y_var, 1-row_dual)
            else:
                objective.SetCoefficient(y_var, -row_dual)
            self.y_vars_[i] = y_var.index()

        num_cuts = len(cut_duals)
        self.yc_vars_ = [None]*num_cuts
        if num_cuts > 0:
            for i in range(num_cuts):
                yc_var = self.solver_.BoolVar('yc_'+str(i))
                cut_dual = cut_duals[i]
                objective.SetCoefficient(yc_var, -cut_dual)
                self.yc_vars_[i] = yc_var.index()

        # Binary variables u_j_a indicating that split s is assigned to the
        # node j.
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
        # all_splits = list(dict.fromkeys(all_splits))
        # for split in all_splits:
        #     unique_split = self.solver_.Constraint(
        #         0, 1, "unique_split_" + str(split))
        #     for node_id in nodes:
        #         node = self.nodes_[node_id]
        #         if split not in node.candidate_splits:
        #             continue
        #         u_var = self.solver_.variable(self.u_vars_[node.id][split])
        #         unique_split.SetCoefficient(u_var, 1)

        # Row follows correct path (upper and lower bound on y var)
        for i in range(n_rows):
            if self.data_rows != None:
                if self.data_rows[i].removed_from_sp:
                    continue
            y_lb_cons = self.solver_.Constraint(
                -infinity, self.tree_depth_-1, "row_" + str(i))
            y_var = self.solver_.variable(self.y_vars_[i])
            y_lb_cons.SetCoefficient(y_var, -1)
            for node_id in nodes:
                node = self.nodes_[node_id]
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

        self.generated_ = True

    def update_objective(self, leaf_dual, row_duals, ns_duals, cut_duals):
        """TODO: Documentation.
        """
        objective = self.solver_.Objective()
        n_rows = self.X_.shape[0]
        for i in range(n_rows):
            obj_coeff = 1
            if self.data_rows != None:
                if self.data_rows[i].removed_from_sp:
                    continue
                # obj_coeff = self.data_rows[i].weight

            y_var = self.solver_.variable(self.y_vars_[i])
            row_dual = row_duals[i]
            if self.y_[i] == self.target_:
                objective.SetCoefficient(y_var, 1-row_dual)
            else:
                objective.SetCoefficient(y_var, -row_dual)

        num_cuts = len(cut_duals)
        existing_cuts = len(self.yc_vars_)
        if num_cuts > 0:
            for i in range(num_cuts):
                yc_var = None
                if i >= existing_cuts:
                    yc_var = self.solver_.BoolVar('yc_'+str(i))
                    self.yc_vars_.append(yc_var.index())
                else:
                    yc_var = self.solver_.variable(self.yc_vars_[i])
                cut_dual = cut_duals[i]
                objective.SetCoefficient(yc_var, -cut_dual)

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

    def add_cut_rows(self, cut_rows):
        assert self.generated_
        infinity = self.solver_.infinity()
        nodes = self.leaf_.left_nodes + self.leaf_.right_nodes
        n_cut_rows = len(cut_rows)
        # Row follows correct path (upper and lower bound on y var)
        for i in range(n_cut_rows):
            cut_index = i + self.cut_rows_starting_index
            yc_lb_cons = self.solver_.Constraint(
                -infinity, self.tree_depth_-1, "cut_" + str(cut_index))
            # yc_var is created in update_objective method.
            yc_var = self.solver_.variable(self.yc_vars_[cut_index])
            yc_lb_cons.SetCoefficient(yc_var, -1)
            for node_id in nodes:
                node = self.nodes_[node_id]
                cn_cons = self.solver_.Constraint(
                    -infinity, 0, "cn_" + str(cut_index) + '_' + str(node.id))

                cn_cons.SetCoefficient(yc_var, 1)
                if node.id in self.leaf_.left_nodes:
                    for split in node.candidate_splits:
                        if cut_rows[i][split]:
                            u_var = self.solver_.variable(
                                self.u_vars_[node.id][split])
                            cn_cons.SetCoefficient(u_var, -1)
                            yc_lb_cons.SetCoefficient(u_var, 1)
                elif node.id in self.leaf_.right_nodes:
                    for split in node.candidate_splits:
                        if not cut_rows[i][split]:
                            u_var = self.solver_.variable(
                                self.u_vars_[node.id][split])
                            cn_cons.SetCoefficient(u_var, -1)
                            yc_lb_cons.SetCoefficient(u_var, 1)
        self.cut_rows_starting_index += n_cut_rows

    def generate_columns(self, X, y, dual_costs, params=""):
        """TODO: Documentation.
        """

        # Solve sub problem
        result_status = self.solver_.Solve(get_params_from_string(params))

        has_solution = (result_status == pywraplp.Solver.OPTIMAL or
                        result_status == pywraplp.Solver.FEASIBLE)

        # The current path is always feasible.
        # # TODO: Fix this. Sat solver returns infeasible sometimes
        # (even without cuts).
        if not has_solution:
            return []
            # print(self.solver_.ExportModelAsLpFormat(False))
        assert has_solution, "Result status: " + str(result_status)

        # if self.leaf_id_ == 4:
        #     sp_mps = self.solver_.ExportModelAsMpsFormat(fixed_format=False,
        #                                                  obfuscated=False)
        #     name = "sp_" + str(self.tree_depth_) + "_" + \
        #         str(self.iter_) + ".mps"
        #     self.iter_ += 1
        #     f = open(name, "w")
        #     f.write(sp_mps)
        #     f.close()

        # TODO: This threshold should be a parameter.
        print("Subproblem ", self.leaf_id_, " objective = ",
              self.solver_.Objective().Value())
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
        path.target = self.target_
        n_rows = self.X_.shape[0]
        for i in range(n_rows):
            obj_coeff = 1
            if self.data_rows != None:
                if self.data_rows[i].removed_from_sp:
                    continue
                # obj_coeff = self.data_rows[i].weight
            y_var = self.solver_.variable(self.y_vars_[i])
            if self.y_[i] == self.target_:
                path.cost += 1 * y_var.solution_value()
        return [path]

    def update_subproblem(self, X, y, dual_costs):
        self.X_ = X
        self.y_ = y
        leaf_dual = dual_costs[0][self.leaf_id_]
        row_duals = dual_costs[1]
        ns_duals = dual_costs[2]
        cut_duals = dual_costs[3]
        cut_rows = dual_costs[4]

        if self.generated_:
            self.update_objective(leaf_dual, row_duals, ns_duals, cut_duals)
        else:
            self.create_submip(leaf_dual, row_duals, ns_duals, cut_duals)

        if len(cut_rows) > 0:
            self.add_cut_rows(cut_rows)

    def row_satisfies_split(self, row, split):
        """TODO: Documentation.
        """
        feature = split.feature
        threshold = split.threshold
        if self.X_[row, feature] <= threshold:
            return True
        return False


class DTreeClassifier(ColGenClassifier):
    """Decision Tree classifier using column generation.

    Parameters
    ----------
    initial_paths: list(Path), default=[],
        List of paths used to initialize the master problem. The user must
        ensure that a valid tree can be formed using the initial paths.
    leaves: list(Leaf), default=[],
        List of leaves in the tree.
    nodes: list(Node), default=[],
        List of nodes in the tree.
    splits: list(Split), default=[],
        List of split checks used in the nodes.
    tree_depth: int, default=1,
        Depth of the tree.
    targets: list(int), default=[],
        List of target ids. They must start from 0.
    max_iterations: int, default=-1
        Maximum column generation iterations. Negative values removes the
        iteration limit and the problem is solved till optimality.
    time_limit: int, default=-1,
        Time limit in seconds for training. Negative values removes the
        time limit and the problem is solved till optimality.
    num_master_cuts_round: int, default=3,
        Number of times the master problem adds cuts in an iteration.
    master_beta_constraints_as_cuts: bool, default=False,
        If True, adds existing beta constraints (constraints for data rows) as
        cutting planes in the master problem.
    master_generate_cuts: bool, default=False,
        If True, master problem generates new beta cuts using SAT solver.
    data_rows: list(Row), default=None,
        Preprocessed data rows. The preprocessed rows help with faster running
        times.
    use_old_sp: bool, default=False,
        If True, uses the old subproblem model published in Firat et. al. 2020.
    master_solver_type: str, default='glop',
        Solver for RMP from OR-Tools. Use 'glop' for tests. See OR-Tools
        documentation for other possible values.
    rmp_solver_params: string, default = "",
        Solver parameters for solving restricted master problem (rmp).
    master_ip_solver_params: string, default = "",
        Solver parameters for solving the integer master problem.
    subproblem_params: list of strings, default = [""],
        Parameters for solving the subproblem.
    """

    def __init__(self, initial_paths=[], leaves=[], nodes=[], splits=[],
                 tree_depth=1,
                 targets=[], max_iterations=-1,
                 time_limit=-1,
                 num_master_cuts_round=3,
                 master_beta_constraints_as_cuts=False,
                 master_generate_cuts=False,
                 data_rows=None,
                 use_old_sp=False,
                 master_solver_type='glop',
                 rmp_solver_params="",
                 master_ip_solver_params="", subproblem_params=""):

        self.initial_paths = initial_paths
        self.leaves = leaves
        self.nodes = nodes
        self.splits = splits
        self.tree_depth = tree_depth
        self.targets = targets
        self.subproblem_params = subproblem_params
        self.num_master_cuts_round = num_master_cuts_round
        self.master_beta_constraints_as_cuts = master_beta_constraints_as_cuts
        self.master_generate_cuts = master_generate_cuts
        self.data_rows = data_rows
        split_ids = []
        for split in splits:
            split_ids.append(split.id)
        # Each node must be in sequence.
        node_ids = []
        node_id = 0
        for node in self.nodes:
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
            path.initial = True
            assert path.leaf_id in leaf_ids
            for node_id in path.node_ids:
                assert node_id in node_ids
            for split_id in path.splits:
                assert split_id in split_ids, "split id " + str(split_id)
            assert path.target in targets
            assert len(path.node_ids) == self.tree_depth
            assert len(path.splits) == self.tree_depth

        self.master_problem = DTreeMasterProblem(
            self.initial_paths, self.leaves, self.nodes, self.splits,
            num_cuts_round=num_master_cuts_round,
            beta_constraints_as_cuts=master_beta_constraints_as_cuts,
            generate_cuts=master_generate_cuts,
            solver_type=master_solver_type,
            data_rows=data_rows)
        self.subproblems = []
        all_subproblem_params = []
        heuristic = DTreeSubProblemHeuristic(
            self.leaves, self.nodes, self.splits, self.targets,
            self.tree_depth, data_rows=data_rows)
        self.subproblems.append([heuristic])
        all_subproblem_params.append([""])
        self.subproblems.append([])
        all_subproblem_params.append([])
        for leaf in self.leaves:
            if use_old_sp:
                for target in self.targets:
                    subproblem = DTreeSubProblemOld(
                        leaf, self.nodes, self.splits, target,
                        self.tree_depth, optimization_problem_type='sat',
                        data_rows=data_rows)
                    self.subproblems[1].append(subproblem)
                    all_subproblem_params[1].append(subproblem_params)
            else:
                # subproblem = DTreeSubProblem(
                #     leaf, self.nodes, self.splits, self.targets,
                #     self.tree_depth, optimization_problem_type='sat',
                #     data_rows=data_rows)
                subproblem = DTreeSubProblemSat(
                    leaf, self.nodes, self.splits, self.targets,
                    self.tree_depth,
                    data_rows=data_rows)
                self.subproblems[1].append(subproblem)
                all_subproblem_params[1].append(subproblem_params)

        rmp_is_ip = True
        super().__init__(max_iterations, time_limit,
                         self.master_problem, self.subproblems,
                         rmp_is_ip, rmp_solver_params, master_ip_solver_params,
                         all_subproblem_params)

    def _more_tags(self):
        return {'X_types': ['categorical'],
                'non_deterministic': True,
                'binary_only': True}

    def predict(self, X):
        """Predicts the class based on the solution of master problem. 

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples. The inputs should only contain numeric values.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        selected_paths = self.master_problem.selected_paths
        # Check for each row, which path it satisfies. There should be exactly
        # one.
        num_rows = X.shape[0]
        y_predict = np.zeros(X.shape[0], dtype=int)
        for row in range(num_rows):
            for path in selected_paths:
                if row_satisfies_path(X[row], self.leaves[path.leaf_id],
                                      self.splits, path):
                    y_predict[row] = path.target
                    break
        return self.label_encoder_.inverse_transform(y_predict)
