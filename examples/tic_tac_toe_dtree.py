"""TODO: Documentation."""

import math
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
import col_gen_estimator


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

    # split dataset to 60% training and 40% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0)
    print(X_train.shape, y_train.shape)
    print(X.shape, y.shape)

    combined_splits = {}
    combined_nodes = {}
    max_num_nodes = 2**experiment_max_depth - 1
    for node_id in range(max_num_nodes):
        node = Node()
        node.id = node_id
        node.candidate_splits = []
        combined_nodes[node_id] = node
    for i in range(100):
        X_train_train, _, y_train_train, _ = train_test_split(
            X_train, y_train, test_size=0.3, random_state=i)
        dt = tree.DecisionTreeClassifier(
            max_depth=experiment_max_depth, random_state=99)
        clf_dt = dt.fit(X_train_train, y_train_train)

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
            children_left, children_right, n_nodes,
            feature, threshold, is_leaves, combined_splits)
        combined_nodes = merge_all_nodes(combined_nodes, all_nodes)

    dt = tree.DecisionTreeClassifier(
        max_depth=experiment_max_depth, random_state=99)

    # X_train = X
    # y_train = y
    clf_dt = dt.fit(X_train, y_train)
    # clf_dt = dt.fit(X, y)

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

    is_leaves = get_is_leaves(n_nodes, children_left, children_right)

    # Create all used splits
    all_splits = get_all_splits(n_nodes, feature, threshold, is_leaves)
    splits = get_splits_list(combined_splits)
    combined_splits = merge_all_splits(combined_splits, all_splits)
    # Create node and add correspondning split to candidate splits
    all_nodes = get_all_nodes(
        children_left, children_right, n_nodes, feature, threshold, is_leaves,
        combined_splits)
    combined_nodes = merge_all_nodes(combined_nodes, all_nodes)

    # If some node never appeared, populate its candidate splits here.
    for node_id in range(max_num_nodes):
        if not combined_nodes[node_id].candidate_splits:
            combined_nodes[node_id].candidate_splits.append(0)
    nodes = get_nodes_list(combined_nodes)

    all_leaves, all_paths = get_all_leaves_paths(
        combined_nodes, experiment_max_depth, splits, X_train.toarray(),
        y_train.to_numpy())

    paths = get_paths_list(all_paths)

    leaves = get_leaves_list(all_leaves)

    clf = DTreeClassifier(paths, leaves, nodes, splits,
                          tree_depth=experiment_max_depth,
                          targets=[0, 1], max_iterations=50)
    clf.fit(X_train.toarray(), y_train.to_numpy())

    print(clf.mp_optimal_, clf.performed_iter_)
    for i in range(len(all_leaves)):
        print(i, clf.num_col_added_sp_[i])
    print("Train Acurracy: ", clf.score(X_train, y_train))
    print("Test Acurracy: ", clf.score(X_test, y_test))
    print("Full Acurracy: ", clf.score(X, y))


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
        # print(path.node_ids, path.leaf_id, path.target)
        paths.append(path)
    return paths


def get_all_leaves_paths(all_nodes, depth, splits, X, y):
    all_leaves = {}
    all_paths = {}

    num_leaves = 2**depth

    for leaf_id in range(num_leaves):
        path = Path()
        path.id = len(all_paths)
        leaf = Leaf()
        leaf.id = leaf_id
        path.leaf_id = leaf_id
        path.node_ids = []
        count = 2**(depth) - 1 + leaf_id
        while(count > 0):
            father = math.floor((count-1) / 2)
            path.node_ids.append(father)
            selected_split = all_nodes[father].candidate_splits[-1]
            path.splits.append(selected_split)
            if count == 2*father + 1:
                leaf.left_nodes.append(father)
            else:
                leaf.right_nodes.append(father)
            count = father
        # Compute target and cost of the path.
        best_target = 0
        best_cost = 0
        target_counts = {}
        for row in range(X.shape[0]):
            if row_satisfies_path(splits, X, path, leaf, row):
                row_target = y[row]
                if row_target in target_counts:
                    target_counts[row_target] += 1
                else:
                    target_counts[row_target] = 1
                if target_counts[row_target] > best_cost:
                    best_cost = target_counts[row_target]
                    best_target = row_target
        path.cost = best_cost
        path.target = best_target

        all_leaves[leaf_id] = leaf
        all_paths[path.id] = path
    return all_leaves, all_paths


def row_satisfies_path(splits, X, path, leaf, row):
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


def get_all_nodes(children_left, children_right,
                  n_nodes, feature, threshold, is_leaves, all_splits):
    all_nodes = {}
    node_id_mapping = {}
    node_id_mapping[0] = 0
    for n_id in range(n_nodes):
        if is_leaves[n_id]:
            continue
        node = Node()
        node.id = node_id_mapping[n_id]
        f = feature[n_id]
        t = threshold[n_id]
        corresponding_split = all_splits[(f, t)]
        node.candidate_splits.append(corresponding_split.id)
        all_nodes[node.id] = node
        # Update the node_id_mapping for children
        node_id_mapping[children_left[n_id]] = 2*node_id_mapping[n_id] + 1
        node_id_mapping[children_right[n_id]] = 2*node_id_mapping[n_id] + 2
    return all_nodes


def merge_all_nodes(nodes_dict1, nodes_dict2):
    # Combined nodes should have keys of all possible nodes.
    combined_nodes = nodes_dict1
    for key, node2 in nodes_dict2.items():
        if key in combined_nodes:
            node = combined_nodes[key]
            for split_id in node2.candidate_splits:
                if split_id not in node.candidate_splits:
                    node.candidate_splits.append(split_id)
            combined_nodes[key] = node
        else:
            print("Unexpected node. ", key)
    return combined_nodes


if __name__ == "__main__":
    main()
