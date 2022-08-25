import pytest
import numpy as np

from numpy.testing import assert_array_equal
from numpy.testing import assert_equal
from numpy.testing import assert_raises

from col_gen_estimator import DTreeMasterProblem
from col_gen_estimator import DTreeSubProblem
from col_gen_estimator import DTreeClassifier
from col_gen_estimator import Split
from col_gen_estimator import Node
from col_gen_estimator import Leaf
from col_gen_estimator import Path


def test_leaf_creation():
    leaf = Leaf()
    leaf.create_leaf(0, 2)
    assert_array_equal(leaf.left_nodes, [1, 0])
    assert_array_equal(leaf.right_nodes, [])

    leaf.create_leaf(1, 2)
    assert_array_equal(leaf.left_nodes, [0])
    assert_array_equal(leaf.right_nodes, [1])

    leaf.create_leaf(2, 2)
    assert_array_equal(leaf.left_nodes, [2])
    assert_array_equal(leaf.right_nodes, [0])

    leaf.create_leaf(3, 2)
    assert_array_equal(leaf.left_nodes, [])
    assert_array_equal(leaf.right_nodes, [2, 0])


def test_path_set_leaf():
    path = Path()
    path.set_leaf(0, 2)
    assert_array_equal(path.node_ids, [1, 0])
    assert(path.leaf_id, 0)

    path.set_leaf(1, 2)
    assert_array_equal(path.node_ids, [1, 0])
    assert(path.leaf_id, 1)

    path.set_leaf(2, 2)
    assert_array_equal(path.node_ids, [2, 0])
    assert(path.leaf_id, 2)

    path.set_leaf(3, 2)
    assert_array_equal(path.node_ids, [2, 0])
    assert(path.leaf_id, 3)


@pytest.fixture
def data():
    X = np.array([[1, 1, 1],
                  [1, 1, 0],
                  [1, 0, 1],
                  [1, 0, 0],
                  [0, 1, 1],
                  [0, 1, 0],
                  [0, 0, 1],
                  [0, 0, 0], ], np.int8)
    y = np.array([1, 1, 0, 0, 0, 1, 0, 1], np.int8)

    splitA = Split()
    splitA.id = 0
    splitA.feature = 0
    splitA.threshold = 0.5

    splitB = Split()
    splitB.id = 1
    splitB.feature = 1
    splitB.threshold = 0.5

    splitC = Split()
    splitC.id = 2
    splitC.feature = 2
    splitC.threshold = 0.5

    root_node = Node()
    root_node.id = 0
    root_node.candidate_splits = [0]

    left_node = Node()
    left_node.id = 1
    left_node.candidate_splits = [1, 2]

    right_node = Node()
    right_node.id = 2
    right_node.candidate_splits = [1, 2]

    leaf_0 = Leaf()
    leaf_0.create_leaf(0, 2)

    leaf_1 = Leaf()
    leaf_1.create_leaf(1, 2)

    leaf_2 = Leaf()
    leaf_2.create_leaf(2, 2)

    leaf_3 = Leaf()
    leaf_3.create_leaf(3, 2)

    path_0 = Path()
    path_0.set_leaf(0, 2)
    path_0.splits = [1, 0]
    path_0.cost = 1
    path_0.target = 1

    path_1 = Path()
    path_1.set_leaf(1, 2)
    path_1.splits = [1, 0]
    path_1.cost = 1
    path_1.target = 0

    path_2 = Path()
    path_2.set_leaf(2, 2)
    path_2.splits = [2, 0]
    path_2.cost = 1
    path_2.target = 0

    path_3 = Path()
    path_3.set_leaf(3, 2)
    path_3.splits = [2, 0]
    path_3.cost = 1
    path_3.target = 1

    path_4 = Path()
    path_4.set_leaf(0, 2)
    path_4.splits = [2, 0]
    path_4.cost = 2
    path_4.target = 1

    path_5 = Path()
    path_5.set_leaf(1, 2)
    path_5.splits = [2, 0]
    path_5.cost = 2
    path_5.target = 0

    path_6 = Path()
    path_6.set_leaf(2, 2)
    path_6.splits = [1, 0]
    path_6.cost = 2
    path_6.target = 0

    path_7 = Path()
    path_7.set_leaf(3, 2)
    path_7.splits = [1, 0]
    path_7.cost = 2
    path_7.target = 1

    paths = [path_0, path_1, path_2, path_3, path_4, path_5, path_6, path_7]
    nodes = [root_node, left_node, right_node]
    leaves = [leaf_0, leaf_1, leaf_2, leaf_3]
    splits = [splitA, splitB, splitC]
    targets = [0, 1]

    return (X, y, paths, nodes, leaves, splits, targets)


def test_master_prob(data):
    X = data[0]
    y = data[1]
    paths = data[2]
    nodes = data[3]
    leaves = data[4]
    splits = data[5]
    master_problem = DTreeMasterProblem(paths, leaves, nodes, splits)
    assert_equal(master_problem.generated_, False)
    master_problem.generate_mp(X, y)
    assert_equal(master_problem.generated_, True)
    duals = master_problem.solve_rmp()
    assert(len(duals), 3)
    leaf_duals = duals[0]
    row_duals = duals[1]
    ns_duals = duals[2]
    assert_array_equal(leaf_duals, [1, 0, 2, 1])
    assert_array_equal(row_duals, [0, -1, 0, 0, 0, 1, 0, 0])
    assert_equal(ns_duals[1][1][2], 2)
    assert_equal(ns_duals[3][2][1], 2)


def test_master_prob_add_col(data):
    X = data[0]
    y = data[1]
    paths = data[2][:7]
    nodes = data[3]
    leaves = data[4]
    splits = data[5]
    master_problem = DTreeMasterProblem(paths, leaves, nodes, splits)
    assert_equal(master_problem.generated_, False)
    master_problem.generate_mp(X, y)
    assert_equal(master_problem.generated_, True)
    duals = master_problem.solve_rmp()
    assert(len(duals), 3)
    leaf_duals = duals[0]
    row_duals = duals[1]
    ns_duals = duals[2]
    assert_array_equal(leaf_duals, [1, 0, 0, 0])
    assert_array_equal(row_duals, [0, 0, 1, 0, 0, 1, 0, 0])
    assert_equal(ns_duals[1][1][2], 2)

    for leaf in leaves:
        for node in nodes:
            if node.id not in leaf.left_nodes + leaf.right_nodes:
                continue
            for split_id in node.candidate_splits:
                print("leaf={leaf}, node={node}, split={split},\
                      dual={dual}".format(
                    leaf=leaf.id, node=node.id, split=split_id,
                    dual=ns_duals[leaf.id][node.id][split_id]))
    path_7 = data[2][-1]
    master_problem.add_column(path_7)
    duals = master_problem.solve_rmp()
    assert(len(duals), 3)
    leaf_duals = duals[0]
    row_duals = duals[1]
    ns_duals = duals[2]
    assert_array_equal(leaf_duals, [1, 0, 2, 1])
    assert_array_equal(row_duals, [0, -1, 0, 0, 0, 1, 0, 0])
    assert_equal(ns_duals[1][1][2], 2)
    assert_equal(ns_duals[3][2][1], 2)


def test_subproblem(data):
    X = data[0]
    y = data[1]
    paths = data[2]
    nodes = data[3]
    leaves = data[4]
    splits = data[5]
    targets = data[6]
    subproblem = DTreeSubProblem(leaves[0], nodes, splits, targets, depth=2)

    leaf_duals = [2, 3, 1, 2]
    row_duals = [0, 1, 0, 0, 0, -1, 0, 0]
    ns_duals = {}

    ns_duals[0] = {}
    zero_nodes = leaves[0].left_nodes + leaves[0].right_nodes
    for node in nodes:
        if node.id not in zero_nodes:
            continue
        ns_duals[0][node.id] = {}
        for split_id in node.candidate_splits:
            ns_duals[0][node.id][split_id] = 0.0
    duals = (leaf_duals, row_duals, ns_duals)
    new_paths = subproblem.generate_columns(X, y, duals)
    assert_equal(len(new_paths), 1)
    path = new_paths[0]
    assert_equal(path.leaf_id, 0)
    assert_array_equal(path.node_ids, [0, 1])
    assert_array_equal(path.splits, [0, 2])
    assert_equal(path.cost, 2)
    assert_equal(path.target, 1)


def test_subproblem2(data):
    X = data[0]
    y = data[1]
    paths = data[2]
    nodes = data[3]
    leaves = data[4]
    splits = data[5]
    targets = data[6]
    subproblem = DTreeSubProblem(leaves[0], nodes, splits, targets, depth=2)

    leaf_duals = [1, 0, 2, 1]
    row_duals = [0, -1, 0, 0, 0, 1, 0, 0]
    ns_duals = {}

    ns_duals[0] = {}
    zero_nodes = leaves[0].left_nodes + leaves[0].right_nodes
    for node in nodes:
        if node.id not in zero_nodes:
            continue
        ns_duals[0][node.id] = {}
        for split_id in node.candidate_splits:
            ns_duals[0][node.id][split_id] = 0.0
    duals = (leaf_duals, row_duals, ns_duals)
    new_paths = subproblem.generate_columns(X, y, duals)
    assert_array_equal(new_paths, [])


def test_subproblem3(data):
    X = data[0]
    y = data[1]
    paths = data[2]
    nodes = data[3]
    leaves = data[4]
    splits = data[5]
    targets = data[6]
    subproblem = DTreeSubProblem(leaves[2], nodes, splits, targets, depth=2)

    leaf_duals = [1, 0, 0, 0]
    row_duals = [0, 0, 1, 0, 0, 1, 0, 0]
    ns_duals = {}
    for node in nodes:
        print(node.candidate_splits)

    ns_duals[2] = {}
    two_nodes = leaves[2].left_nodes + leaves[2].right_nodes
    for node in nodes:
        if node.id not in two_nodes:
            continue
        ns_duals[2][node.id] = {}
        for split_id in node.candidate_splits:
            ns_duals[2][node.id][split_id] = 0.0
    ns_duals[2][0][0] = 1.0
    duals = (leaf_duals, row_duals, ns_duals)
    new_paths = subproblem.generate_columns(X, y, duals)
    assert_array_equal(new_paths, [])


def test_subproblem4(data):
    X = data[0]
    y = data[1]
    paths = data[2]
    nodes = data[3]
    leaves = data[4]
    splits = data[5]
    targets = data[6]
    subproblem = DTreeSubProblem(leaves[3], nodes, splits, targets, depth=2)

    leaf_duals = [1, 0, 0, 0]
    row_duals = [0, 0, 1, 0, 0, 1, 0, 0]
    ns_duals = {}
    for node in nodes:
        print(node.candidate_splits)

    ns_duals[3] = {}
    two_nodes = leaves[3].left_nodes + leaves[3].right_nodes
    for node in nodes:
        if node.id not in two_nodes:
            continue
        ns_duals[3][node.id] = {}
        for split_id in node.candidate_splits:
            ns_duals[3][node.id][split_id] = 0.0
    duals = (leaf_duals, row_duals, ns_duals)
    new_paths = subproblem.generate_columns(X, y, duals)
    assert_equal(len(new_paths), 1)
    path = new_paths[0]
    assert_equal(path.leaf_id, 3)
    assert_array_equal(path.node_ids, [0, 2])
    assert_array_equal(path.splits, [0, 1])
    assert_equal(path.cost, 2)
    assert_equal(path.target, 1)


def test_fit_predict(data):
    X = data[0]
    y = data[1]
    paths = data[2][:7]
    nodes = data[3]
    leaves = data[4]
    splits = data[5]
    clf = DTreeClassifier(paths, leaves, nodes, splits,
                          tree_depth=2, targets=[0, 1], max_iterations=3)
    clf.fit(X, y)
    assert hasattr(clf, 'is_fitted_')
    assert hasattr(clf, 'classes_')
    assert hasattr(clf, 'X_')
    assert hasattr(clf, 'y_')
    assert hasattr(clf, 'mp_optimal_')
    assert hasattr(clf, 'performed_iter_')
    assert hasattr(clf, 'num_col_added_sp_')

    assert_equal(clf.mp_optimal_, True)
    assert_equal(clf.performed_iter_, 2)
    assert_equal(clf.num_col_added_sp_[0], 0)
    assert_equal(clf.num_col_added_sp_[1], 0)
    assert_equal(clf.num_col_added_sp_[2], 0)
    assert_equal(clf.num_col_added_sp_[3], 1)

    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)
    assert_array_equal(y_pred, y)
