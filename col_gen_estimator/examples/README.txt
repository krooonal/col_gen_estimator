.. _general_examples:

General examples
================

Example command to launch decision tree experiment from the examples directory:

python dtree_experiment.py \
-T datasets/small_datasets/tic_tac_toe/given_splits/tic-tac-toe.csv.train0.csv \
-t datasets/small_datasets/tic_tac_toe/given_splits/tic-tac-toe.csv.test0.csv \
-S ';' \
-d 4 \
-s 10

The experiment statistics can be found in the 'results' directory.
