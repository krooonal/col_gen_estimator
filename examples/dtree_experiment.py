"""Code to launch decision tree experiments.
Usage:
python dtree_experiment.py \
-T <Training data file> \
-t <Testing data file> \
-S <CSV separator> \
-d <Depth of decision tree> \
-s <Time limit in seconds> \
-i <Column Generation iteration limit> \
-r <Results directory>
"""

import math
import random
import pandas as pd
import numpy as np
import getopt
import os
import sys
import csv

from sklearn.model_selection import train_test_split

from sklearn import tree

from sklearn.preprocessing import LabelEncoder

from time import time

from col_gen_estimator import DTreeClassifier
from col_gen_estimator import Split
from col_gen_estimator import Node
from col_gen_estimator import Leaf
from col_gen_estimator import Path
from col_gen_estimator import Row


class Results:
    def __init__(self) -> None:
        self.name = ""
        self.train_accuracy = 0.0
        self.test_accuracy = 0.0
        self.preprocessing_time = 0.0
        self.training_time = 0.0
        self.master_time = 0.0
        self.mster_cuts_time = 0.0
        self.master_num_cuts = 0
        self.sp_time_heuristic = 0.0
        self.sp_time_ip = 0.0
        self.sp_cols_added_heuristic = 0
        self.sp_cols_added_ip = 0
        self.col_add_time = 0.0
        self.master_ip_time = 0.0
        self.total_iterations = 0
        self.num_improving_iter = 0
        self.heuristic_hit_rate = 0.0


def main(argv):

    random.seed(10)
    train_file = ""
    test_file = ""
    experiment_max_depth = 2
    max_iterations = -1
    time_limit = 600
    preprocessing_time_limit = 200
    sep = ','
    header = 0
    results_dir = "results/"
    REDUCED_SPLITS = True

    try:
        opts, args = getopt.getopt(argv, "hT:t:S:H:d:i:s:r:",
                                   ["train_file=",
                                    "test_file=",
                                    "sep=",
                                    "header=",
                                    "depth=",
                                    "iterations=",
                                    "time_limit=",
                                    "results_dir="])
    except getopt.GetoptError:
        print_options()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print_options()
            sys.exit()
        elif opt in ("-T", "--train_file"):
            train_file = arg
        elif opt in ("-t", "--test_file"):
            test_file = arg
        elif opt in ("-S", "--sep"):
            sep = arg
        elif opt in ("-H", "--header"):
            header = int(arg)
            if header < 0:
                header = None
        elif opt in ("-d", "--depth"):
            experiment_max_depth = int(arg)
        elif opt in ("-i", "--iterations"):
            max_iterations = int(arg)
        elif opt in ("-s", "--time_limit"):
            time_limit = int(arg)
            preprocessing_time_limit = time_limit / 3
        elif opt in ("-r", "--results_dir"):
            results_dir = "results/" + arg + "/"

    train_df = pd.read_csv(train_file, sep=sep, header=header)
    test_df = pd.read_csv(test_file, sep=sep, header=header)

    print(train_df.head())

    features = (list(train_df.columns[:-1]))

    X_train = train_df[features]
    y_train = train_df.iloc[:, -1]
    X_test = test_df[features]
    y_test = test_df.iloc[:, -1]

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)
    print("Y after transform: ", y_train[:5])

    use_old_sp = False

    is_large_dataset = False
    if X_train.shape[0] >= 5000:
        is_large_dataset = True

    t0 = time()

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    combined_splits = {}
    combined_nodes = {}
    max_num_nodes = 2**experiment_max_depth - 1
    for node_id in range(max_num_nodes):
        node = Node()
        node.id = node_id
        node.candidate_splits = []
        combined_nodes[node_id] = node
    q_root = int(150 / max_num_nodes)
    q_node = int(100 / max_num_nodes)

    for i in range(300):
        if preprocessing_time_limit > 0 and time() - t0 > preprocessing_time_limit:
            break
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

    print("DecisionTree")
    d_tree_start_time = time()
    dt = tree.DecisionTreeClassifier(
        max_depth=experiment_max_depth, random_state=99)

    clf_dt = dt.fit(X_train, y_train)
    train_accuracy = clf_dt.score(X_train, y_train)
    test_accuracy = clf_dt.score(X_test, y_test)

    print("Train Acurracy: ", train_accuracy)
    print("Test Acurracy: ", test_accuracy)
    t1 = time()
    print("time elapsed: ", t1-d_tree_start_time)

    cart_results = Results()
    cart_results.name = "CART"
    cart_results.train_accuracy = train_accuracy
    cart_results.test_accuracy = test_accuracy
    cart_results.training_time = t1-d_tree_start_time

    n_nodes = clf_dt.tree_.node_count
    children_left = clf_dt.tree_.children_left
    children_right = clf_dt.tree_.children_right
    feature = clf_dt.tree_.feature
    threshold = clf_dt.tree_.threshold

    is_leaves = get_is_leaves(n_nodes, children_left, children_right)

    # Create all used splits
    all_splits = get_all_splits(n_nodes, feature, threshold, is_leaves)

    combined_splits = merge_all_splits(combined_splits, all_splits)
    splits = get_splits_list(combined_splits)
    # Create node and add correspondning split to candidate splits
    all_nodes = get_all_nodes(
        children_left, children_right, n_nodes, feature, threshold, is_leaves,
        combined_splits)
    combined_nodes = merge_all_nodes(combined_nodes, all_nodes, 1000)

    # If some node never appeared, populate its candidate splits here.
    for node_id in range(max_num_nodes):
        if not combined_nodes[node_id].candidate_splits:
            combined_nodes[node_id].candidate_splits.append(0)
            combined_nodes[node_id].candidate_splits_count.append(0)
            combined_nodes[node_id].last_split = 0

    all_leaves, all_paths = get_all_leaves_paths(
        combined_nodes, experiment_max_depth, splits, X_train.to_numpy(),
        y_train)

    paths = get_paths_list(all_paths)
    leaves = get_leaves_list(all_leaves)
    splits = get_splits_list(combined_splits)
    nodes = get_nodes_list(combined_nodes)

    if REDUCED_SPLITS:
        nodes = reduce_splits(nodes, splits, q_root, q_node)

    # for split in splits:
    #     print("Split removed ", split.id, split.removed)

    # Add more paths for initialization.
    for i in range(100):
        # EXP: Enable initialization.
        # break
        if preprocessing_time_limit > 0 and time() - t0 > preprocessing_time_limit:
            break
        X_train_train, _, y_train_train, _ = train_test_split(
            X_train, y_train, test_size=0.2, random_state=i)
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
        # all_splits = get_all_splits(
        #     n_nodes, feature, threshold, is_leaves)
        # combined_splits = merge_all_splits(combined_splits, all_splits)
        # splits = get_splits_list(combined_splits)
        tree_nodes = get_all_nodes(
            children_left, children_right, n_nodes,
            feature, threshold, is_leaves, combined_splits)
        # combined_nodes = merge_all_nodes(combined_nodes, all_nodes)
        # combined_nodes = merge_all_nodes_last_split(combined_nodes, all_nodes)
        for leaf in leaves:
            path = get_path_for_leaf(
                leaf, tree_nodes, experiment_max_depth, splits,
                X_train.to_numpy(),
                y_train)
            if path == None:
                continue
            found = False
            for p in paths:
                if path.is_same_as(p):
                    found = True
                    break
            if not found:
                path.id = len(paths)
                paths.append(path)

    # splits = get_splits_list(combined_splits)
    # nodes = get_nodes_list(combined_nodes)
    targets = np.unique(y_train)  # y_train.unique()
    splits = add_rows_to_splits(splits, X_train.to_numpy())
    # EXP: Set default aggressive mode to True/False.
    use_aggressive_mode = is_large_dataset
    if preprocessing_time_limit > 0 and (time() - t0) > preprocessing_time_limit:
        use_aggressive_mode = False
    data_rows = preprocess(nodes, splits, X_train.to_numpy(),
                           y_train, experiment_max_depth,
                           aggressive=use_aggressive_mode,
                           time_limit=preprocessing_time_limit - (time() - t0))
    t2 = time()
    preprocessing_time = t2-t0
    print("Total preprocessing time: ", t2-t0)
    time_limit -= preprocessing_time

    t0 = time()
    clf = DTreeClassifier(paths.copy(), leaves.copy(), nodes.copy(),
                          splits.copy(),
                          tree_depth=experiment_max_depth,
                          targets=targets.copy(),
                          data_rows=data_rows.copy(),
                          max_iterations=max_iterations,
                          time_limit=time_limit,
                          num_master_cuts_round=10,
                          master_beta_constraints_as_cuts=True,
                          master_generate_cuts=False,
                          use_old_sp=use_old_sp)
    clf.fit(X_train.to_numpy(), y_train)

    print(clf.mp_optimal_, clf.iter_, clf.time_elapsed_)
    train_accuracy = clf.score(X_train, y_train)
    print("Train Accuracy: ", train_accuracy)
    test_accuracy = clf.score(X_test, y_test)
    print("Test Accuracy: ", test_accuracy)
    t1 = time()
    print("time elapsed: ", t1-t0)

    cg1_results = Results()
    cg1_results.name = "CG1"
    cg1_results.train_accuracy = train_accuracy
    cg1_results.test_accuracy = test_accuracy
    cg1_results.preprocessing_time = preprocessing_time
    cg1_results.training_time = clf.time_elapsed_
    cg1_results.master_time = clf.time_spent_master_
    cg1_results.mster_cuts_time = clf.master_problem.cut_gen_time
    cg1_results.master_num_cuts = clf.master_problem.total_cuts_added
    cg1_results.sp_time_heuristic = clf.time_spent_sp_[0][0]
    cg1_results.sp_time_ip = sum(clf.time_spent_sp_[1])
    cg1_results.sp_cols_added_heuristic = clf.num_col_added_sp_[0][0]
    cg1_results.sp_cols_added_ip = sum(clf.num_col_added_sp_[1])
    cg1_results.master_ip_time = clf.time_spent_master_ip_
    cg1_results.total_iterations = clf.iter_
    cg1_results.num_improving_iter = clf.num_improving_iter_
    cg1_results.col_add_time = clf.time_add_col_
    failed_heuristic_rounds = clf.subproblems[0][0].failed_rounds
    successful_heuristic_rounds = clf.subproblems[0][0].success_rounds
    cg1_results.heuristic_hit_rate = 0.0
    if (successful_heuristic_rounds + failed_heuristic_rounds) > 0:
        cg1_results.heuristic_hit_rate = float(successful_heuristic_rounds) / \
            (successful_heuristic_rounds + failed_heuristic_rounds)

    added_rows = []
    for r in range(len(data_rows)):
        if clf.master_problem.added_row[r]:
            added_rows.append(r)

    print("Total added rows: ", len(added_rows))
    print("Last reset iter = ", clf.master_problem.last_reset_iter_)

    attrs = vars(cg1_results)
    print('\n'.join("%s: %s" % item for item in attrs.items()))

    train_file_name = os.path.basename(train_file)

    prefix = results_dir + 'T_' + train_file_name + \
        '_d_' + str(experiment_max_depth)
    results_filename = prefix + '_results.csv'
    with open(results_filename, 'w', newline='') as f:
        # fieldnames lists the headers for the csv.
        w = csv.DictWriter(f, fieldnames=sorted(vars(cart_results)))
        w.writeheader()
        w.writerow({k: getattr(cart_results, k)
                    for k in vars(cart_results)})
        w.writerow({k: getattr(cg1_results, k)
                    for k in vars(cg1_results)})


def print_options():
    print('experiment.py')
    print(' -T <train_file>')
    print(' -t <test_file>')
    print(' -S <sep>')
    print(' -H <header>')
    print(' -d <depth>')
    print(' -i <iterations>')
    print(' -s <time_limit>')
    print(' -r <results_dir>')


def add_rows_to_splits(splits, X):
    n_rows = X.shape[0]
    for r in range(n_rows):
        for split in splits:
            feature = split.feature
            threshold = split.threshold
            if X[r, feature] <= threshold:
                split.left_rows.add(r)
            else:
                split.right_rows.add(r)

    return splits


def preprocess(nodes, splits, X, y, depth, aggressive=False, time_limit=150):
    """TODO: Documentation."""
    t_start = time()
    # preprocess nodes first. Assign the parent and children.
    nodes = preprocess_nodes(nodes, depth)

    for node in nodes:
        print(node.id, node.parent, node.left_child,
              node.right_child, node.children_are_leaves)

    data_rows = []
    n_rows = X.shape[0]
    removed_count_master = 0
    removed_count_sp = 0
    average_leaf_reach = 0.0
    for r in range(n_rows):
        data_row = Row()
        data_row.target = y[r]
        # TODO: Double computation.
        for split in splits:
            if r in split.left_rows:
                data_row.left_splits.add(split.id)
            else:
                data_row.right_splits.add(split.id)
        # Record which nodes it can reach.
        for node in nodes:
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
        # EXP: Enable/Disable preprocessing.
        if len(data_row.reachable_leaves) <= 2:
            data_row.removed_from_master = True
            removed_count_master += 1

        data_rows.append(data_row)

    print("Fewer leaves removed rows: ", removed_count_master)
    average_leaf_reach /= n_rows
    print("Average leaf reach: ", average_leaf_reach)

    # Quadratic loop
    if not aggressive:
        return data_rows
    for r1 in range(n_rows):
        if time_limit > 0 and time()-t_start > time_limit:
            break
        if data_rows[r1].removed_from_sp:
            continue
        for r2 in range(r1+1, n_rows):
            if time_limit > 0 and time()-t_start > time_limit:
                break
            if data_rows[r2].removed_from_sp:
                continue

            if data_rows[r1].reachable_nodes != \
                    data_rows[r2].reachable_nodes:
                continue

            found_mismatch_split = False
            valid_splits = set()
            for node_id in data_rows[r1].reachable_nodes:
                node = nodes[node_id]
                for split_id in node.candidate_splits:
                    valid_splits.add(split_id)

            for split_id in valid_splits:
                left_membership_r1 = split_id in data_rows[r1].left_splits
                left_membership_r2 = split_id in data_rows[r2].left_splits
                if left_membership_r1 != left_membership_r2:
                    found_mismatch_split = True
                    break

            if not found_mismatch_split:
                if not data_rows[r1].removed_from_master:
                    data_rows[r2].removed_from_master = True
                    removed_count_master += 1
                if y[r1] == y[r2]:
                    data_rows[r2].removed_from_sp = True
                    removed_count_sp += 1
                    data_rows[r1].weight += 1

                # print("Rows are similar: ", r1, r2)

    print("Total removed rows master: ", removed_count_master)
    print("Total removed rows sp: ", removed_count_sp)
    return data_rows


def preprocess_nodes(nodes, depth):
    for node in nodes:
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

    return nodes


def get_splits_list(all_splits, print_splits=False):
    splits = [None]*len(all_splits)
    if print_splits:
        print("splits")
    for key, split in all_splits.items():
        if print_splits:
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
            selected_split = all_nodes[father].last_split
            path.splits.append(selected_split)
            if count == 2*father + 1:
                leaf.left_nodes.append(father)
            else:
                leaf.right_nodes.append(father)
            count = father
        # Compute target and cost of the path.
        best_target = y[0]
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
        # path.print_path()
        all_leaves[leaf_id] = leaf
        all_paths[path.id] = path
    return all_leaves, all_paths


def get_path_for_leaf(leaf, tree_nodes, depth, splits, X, y):
    path = Path()
    path.id = -1
    path.leaf_id = leaf.id
    path.node_ids = []
    count = 2**(depth) - 1 + leaf.id
    while(count > 0):
        father = math.floor((count-1) / 2)
        path.node_ids.append(father)
        if father not in tree_nodes:
            return None
        selected_split = tree_nodes[father].last_split
        if selected_split == -1:
            return None
        if splits[selected_split].removed:
            return None
        path.splits.append(selected_split)
        count = father
    # Compute target and cost of the path.
    best_target = y[0]
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
    return path


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
        if (f, t) in all_splits:
            corresponding_split = all_splits[(f, t)]
            node.candidate_splits.append(corresponding_split.id)
            node.candidate_splits_count.append(1)
            node.last_split = corresponding_split.id
        else:
            node.last_split = -1
        all_nodes[node.id] = node
        # Update the node_id_mapping for children
        node_id_mapping[children_left[n_id]] = 2*node_id_mapping[n_id] + 1
        node_id_mapping[children_right[n_id]] = 2*node_id_mapping[n_id] + 2
    return all_nodes


def merge_all_nodes(nodes_dict1, nodes_dict2, count=1):
    # Combined nodes should have keys of all possible nodes.
    # Assumes that all nodes in nodes_dict2 have only one split.
    combined_nodes = nodes_dict1
    for key, node2 in nodes_dict2.items():
        if key in combined_nodes:
            node = combined_nodes[key]
            node.last_split = node2.last_split
            for split_id in node2.candidate_splits:
                if split_id not in node.candidate_splits:
                    node.candidate_splits.append(split_id)
                    node.candidate_splits_count.append(count)
                else:
                    split_ind = node.candidate_splits.index(split_id)
                    node.candidate_splits_count[split_ind] += count
            combined_nodes[key] = node
        else:
            print("Unexpected node. ", key)
    return combined_nodes


def merge_all_nodes_last_split(nodes_dict1, nodes_dict2):
    # Combined nodes should have keys of all possible nodes.
    # Assumes that all nodes in nodes_dict2 have only one split.
    combined_nodes = nodes_dict1
    for key, node2 in nodes_dict2.items():
        if key in combined_nodes:
            node = combined_nodes[key]
            node.last_split = node2.last_split
            combined_nodes[key] = node
        else:
            print("Unexpected node. ", key)
    return combined_nodes


def reduce_splits(nodes, splits, q_root, q_node):
    print(q_root, q_node)
    used_split_ids = set()
    final_nodes = []
    for node in nodes:
        q_value = q_node
        if node.id == 0:
            q_value = q_root

        list1 = node.candidate_splits_count
        list2 = node.candidate_splits
        print(list1, list2)
        list1, list2 = (list(t) for t in zip(
            *sorted(zip(list1, list2), reverse=True)))
        print(list1[:q_value], list2[:q_value])
        node.candidate_splits_count = list1[:q_value]
        node.candidate_splits = list2[:q_value]
        final_nodes.append(node)
        for split_id in node.candidate_splits:
            used_split_ids.add(split_id)

    for split in splits:
        # Modified in place.
        split.removed = True
        if split.id in used_split_ids:
            split.removed = False

    return final_nodes  # , final_splits


if __name__ == "__main__":
    main(sys.argv[1:])
