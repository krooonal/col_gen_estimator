"""TODO: Documentation."""

import random
from random import choice
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn import tree

from sklearn import preprocessing

from time import time

from col_gen_estimator import DTreeClassifier
from col_gen_estimator import Split
from col_gen_estimator import Node
from col_gen_estimator import Leaf
from col_gen_estimator import Path


def main():
    random.seed(10)
    # Load data from URL using pandas read_csv method
    link = 'http://archive.ics.uci.edu/ml/machine-learning-databases/' + \
        'tic-tac-toe/tic-tac-toe.data'
    df = pd.read_csv(link,
                     header=None,
                     names=["TopLeft", "TopMiddle", "TopRight",
                            "MiddleLeft", "MiddleMiddle", "MiddleRight",
                            "BottomLeft", "BottomMiddle", "BottomRight",
                            "Class"])

    print(df.head())

    # Convert labels to numbers
    le = preprocessing.LabelEncoder()
    for col in df.columns:
        df[col] = le.fit_transform(df[col])

    ohe = preprocessing.OneHotEncoder(handle_unknown='ignore')
    # print(df.head())

    features = (list(df.columns[:-1]))
    X = ohe.fit_transform(df[features])
    print(X.shape)

    # X = df[features]
    y = df['Class']

    experiment_max_depth = 4

    t0 = time()
    print("DecisionTree")

    combined_splits = {}
    combined_nodes = {}
    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=i)
        dt = tree.DecisionTreeClassifier(
            max_depth=experiment_max_depth, random_state=99)
        clf_dt = dt.fit(X_train, y_train)

        n_nodes = clf_dt.tree_.node_count
        children_left = clf_dt.tree_.children_left
        children_right = clf_dt.tree_.children_right
        feature = clf_dt.tree_.feature
        threshold = clf_dt.tree_.threshold
        targets = clf_dt.tree_.value
        is_leaves = get_is_leaves(n_nodes, children_left, children_right)
        all_splits = get_all_splits(
            n_nodes, feature, threshold, is_leaves)
        combined_splits = merge_all_splits(combined_splits, all_splits)
        all_nodes = get_all_nodes(
            n_nodes, feature, threshold, is_leaves, combined_splits)
        combined_nodes = merge_all_nodes(combined_nodes, all_nodes)

    # split dataset to 60% training and 40% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0)
    print(X_train.shape, y_train.shape)
    print(X.shape, y.shape)

    dt = tree.DecisionTreeClassifier(
        max_depth=experiment_max_depth, random_state=99)

    # clf_dt = dt.fit(X_train, y_train)
    clf_dt = dt.fit(X, y)

    print("Train Acurracy: ", clf_dt.score(X_train, y_train))
    print("Test Acurracy: ", clf_dt.score(X_test, y_test))
    print("Full Acurracy: ", clf_dt.score(X, y))
    t1 = time()
    print("time elapsed: ", t1-t0)

    n_nodes = clf_dt.tree_.node_count
    children_left = clf_dt.tree_.children_left
    children_right = clf_dt.tree_.children_right
    feature = clf_dt.tree_.feature
    threshold = clf_dt.tree_.threshold
    targets = clf_dt.tree_.value
    n_clases = clf_dt.tree_.n_classes

    is_leaves = get_is_leaves(n_nodes, children_left, children_right)

    # Create all used splits
    all_splits = get_all_splits(n_nodes, feature, threshold, is_leaves)
    combined_splits = merge_all_splits(combined_splits, all_splits)
    # Create node and add correspondning split to candidate splits
    all_nodes = get_all_nodes(
        n_nodes, feature, threshold, is_leaves, combined_splits)
    combined_nodes = merge_all_nodes(combined_nodes, all_nodes)

    all_leaves, all_paths = get_all_leaves_paths(
        children_left, children_right, feature, threshold,
        targets, is_leaves, combined_splits, combined_nodes)

    paths = get_paths_list(all_paths)

    leaves = get_leaves_list(all_leaves)

    nodes = get_nodes_list(combined_nodes)

    splits = get_splits_list(combined_splits)

    clf = DTreeClassifier(paths, leaves, nodes, splits,
                          tree_depth=experiment_max_depth,
                          targets=[0, 1], max_iterations=70)
    clf.fit(X.toarray(), y)

    print(clf.mp_optimal_, clf.performed_iter_)
    for i in range(len(all_leaves)):
        print(i, clf.num_col_added_sp_[i])
    print("Train Acurracy: ", clf_dt.score(X_train, y_train))
    print("Test Acurracy: ", clf_dt.score(X_test, y_test))
    print("Full Acurracy: ", clf_dt.score(X, y))


def get_splits_list(all_splits):
    splits = [None]*len(all_splits)
    print("splits")
    for key, split in all_splits.items():
        print(split.id, split.feature, split.threshold)
        splits[split.id] = split
    return splits


def get_nodes_list(all_nodes):
    nodes = [None]*len(all_nodes)
    print("Nodes")
    for key, node in all_nodes.items():
        print(node.id, node.candidate_splits)
        nodes[node.id] = node
    return nodes


def get_leaves_list(all_leaves):
    leaves = [None]*len(all_leaves)
    print("Leaves")
    for key, leaf in all_leaves.items():
        print(leaf.id, leaf.left_nodes, leaf.right_nodes)
        leaves[leaf.id] = leaf
    return leaves


def get_paths_list(all_paths):
    paths = []
    for key, path in all_paths.items():
        #print(path.node_ids, path.leaf_id, path.target)
        paths.append(path)
    return paths


def get_all_leaves_paths(children_left, children_right, feature, threshold,
                         targets, is_leaves, all_splits, all_nodes):
    all_leaves = {}
    all_paths = {}

    stack = [(0, [], [])]  # start with the root node id (0) and empty path
    while len(stack) > 0:
        # print(stack)
        # `pop` ensures each node is only visited once
        node_id, nodes, splits = stack.pop()

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            # tree_node_id = all_nodes[node_id].id
            new_nodes = nodes.copy()
            new_nodes.append(node_id)
            f = feature[node_id]
            t = threshold[node_id]
            corresponding_split = all_splits[(f, t)]
            new_splits = splits.copy()
            new_splits.append(corresponding_split.id)

            stack.append((children_left[node_id], new_nodes, new_splits))
            stack.append((children_right[node_id], new_nodes, new_splits))
        else:
            is_leaves[node_id] = True
            # Create leaves
            leaf = Leaf()
            leaf.id = len(all_leaves)
            all_leaves[node_id] = leaf
            tree_node_ids = []
            for n in nodes:
                tree_node_id = all_nodes[n].id
                tree_node_ids.append(tree_node_id)
                if children_left[n] in nodes or children_left[n] == node_id:
                    leaf.left_nodes.append(tree_node_id)
                else:
                    leaf.right_nodes.append(tree_node_id)

            # Create paths
            path = Path()
            path.node_ids = tree_node_ids
            path.splits = splits
            path.leaf_id = leaf.id
            path.target = np.argmax(targets[node_id], axis=1)[0]
            path.cost = targets[node_id][0][path.target]
            print(path.target, targets[node_id])
            all_paths[node_id] = path
            # print("Created path ")
            # path.print_path()
    return all_leaves, all_paths


def get_is_leaves(n_nodes, children_left, children_right):
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    for n_id in range(n_nodes):
        if children_left[n_id] == children_right[n_id]:
            is_leaves[n_id] = True
    return is_leaves


def get_all_splits(n_nodes, feature, threshold, is_leaves):
    all_splits = {}
    split_ids = []
    for n_id in range(n_nodes):
        if is_leaves[n_id]:
            continue
        f = feature[n_id]
        t = threshold[n_id]
        if (f, t) not in all_splits.keys():
            split = Split()
            split.feature = f
            split.threshold = t
            split.id = len(all_splits)
            all_splits[(f, t)] = split
            split_ids.append(split.id)
    return all_splits


def merge_all_splits(splits_dict1, splits_dict2):
    all_splits = splits_dict1
    for key, value in splits_dict2.items():
        if key in all_splits:
            continue
        f = key[0]
        t = key[1]
        split = Split()
        split.feature = f
        split.threshold = t
        split.id = len(all_splits)
        all_splits[(f, t)] = split
    return all_splits


def get_all_nodes(n_nodes, feature, threshold, is_leaves, all_splits):
    all_nodes = {}
    for n_id in range(n_nodes):
        if is_leaves[n_id]:
            continue
        node = Node()
        node.id = len(all_nodes)
        f = feature[n_id]
        t = threshold[n_id]
        corresponding_split = all_splits[(f, t)]
        node.candidate_splits.append(corresponding_split.id)
        all_nodes[n_id] = node
    return all_nodes


def merge_all_nodes(nodes_dict1, nodes_dict2):
    all_nodes = nodes_dict1
    for key, node2 in nodes_dict2.items():
        if key in all_nodes:
            node = all_nodes[key]
            for split_id in node2.candidate_splits:
                if split_id not in node.candidate_splits:
                    node.candidate_splits.append(split_id)
            all_nodes[key] = node
        else:
            node = Node()
            node.id = len(all_nodes)
            node.candidate_splits = node2.candidate_splits
            all_nodes[key] = node
    return all_nodes


if __name__ == "__main__":
    main()
