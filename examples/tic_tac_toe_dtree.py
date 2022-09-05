"""TODO: Documentation."""

from random import random
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
    # for col in df.columns:
    #     df[col] = ohe.fit_transform(df[col])

    # print(df.head())

    features = (list(df.columns[:-1]))
    X = ohe.fit_transform(df[features])
    print(X.shape)

    # X = df[features]
    y = df['Class']

    # split dataset to 60% training and 40% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0)
    print(X_train.shape, y_train.shape)
    print(X.shape, y.shape)
    print(X)

    experiment_max_depth = 2

    t0 = time()
    print("DecisionTree")

    dt = tree.DecisionTreeClassifier(
        max_depth=experiment_max_depth, random_state=99)

    # clf_dt = dt.fit(X_train, y_train)
    clf_dt = dt.fit(X, y)

    print("Train Acurracy: ", clf_dt.score(X_train, y_train))
    print("Test Acurracy: ", clf_dt.score(X_test, y_test))
    print("Full Acurracy: ", clf_dt.score(X, y))
    t1 = time()
    print("time elapsed: ", t1-t0)

    tt0 = time()
    print("cross result========")
    scores = cross_val_score(dt, X, y, cv=3)
    print(scores)
    print(scores.mean())
    tt1 = time()
    print("time elapsed: ", tt1-tt0)

    n_nodes = clf_dt.tree_.node_count
    children_left = clf_dt.tree_.children_left
    children_right = clf_dt.tree_.children_right
    feature = clf_dt.tree_.feature
    threshold = clf_dt.tree_.threshold
    targets = clf_dt.tree_.value
    n_clases = clf_dt.tree_.n_classes

    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    for n_id in range(n_nodes):
        if children_left[n_id] == children_right[n_id]:
            is_leaves[n_id] = True

    paths_to_nodes = [None]*n_nodes
    # Create all used splits
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

    # Create node and add correspondning split to candidate splits
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
        # Randomly choose two more splits
        for iter in range(2):
            chosen = choice(split_ids)
            while(chosen in node.candidate_splits):
                chosen = choice(split_ids)
            node.candidate_splits.append(chosen)

        # node.candidate_splits = split_ids

        all_nodes[n_id] = node

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
            print("Created path ", path.node_ids, node_id, leaf.id, path.cost)

    paths = []
    leaves = [None]*len(all_leaves)
    nodes = [None]*len(all_nodes)
    splits = [None]*len(all_splits)
    targets = []

    for key, path in all_paths.items():
        print(path.node_ids, path.leaf_id, path.target)
        paths.append(path)

    print("Leaves")
    for key, leaf in all_leaves.items():
        print(leaf.id, leaf.left_nodes, leaf.right_nodes)
        leaves[leaf.id] = leaf

    print("Nodes")
    for key, node in all_nodes.items():
        print(node.id, node.candidate_splits)
        nodes[node.id] = node

    print("splits")
    for key, split in all_splits.items():
        print(split.id, split.feature, split.threshold)
        splits[split.id] = split

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


if __name__ == "__main__":
    main()
