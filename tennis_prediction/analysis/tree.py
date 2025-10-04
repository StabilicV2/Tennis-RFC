import os
import joblib
import numpy as np
from sklearn.tree import export_graphviz
import pydot

#! THIS SCRIPT WILL NOT WORK

# --- CONFIG ---
MODEL_PATH = "../results/small_tennis_model.pkl"
TARGET_DEPTH = 4
MAX_DEPTH = 4
OUTPUT_FILE = "tree_with_leaf.svg"

feature_names = [
    "Player Aces", "Player Double Faults", "Player Break Points Saved",
    "Player Break Points Faced", "Opponent Aces", "Opponent Double Faults",
    "Opponent Break Points Saved", "Opponent Break Points Faced",
    "Player Rank", "Player Rank Points", "Opponent Rank", "Opponent Rank Points",
    "Carpet Surface", "Clay Surface", "Grass Surface", "Hard Surface",
    "Tournament Level A", "Tournament Level D", "Tournament Level F",
    "Tournament Level G", "Tournament Level M", "Best of 3", "Best of 5"
]

class_names = ["Win", "Loss"]

# --- Load model ---
model = joblib.load(MODEL_PATH)

def has_leaf_at_or_below_depth(tree, target_depth):
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    n_nodes = tree.tree_.node_count

    depths = np.zeros(n_nodes, dtype=int)
    for i in range(1, n_nodes):
        parent = np.where((children_left == i) | (children_right == i))[0][0]
        depths[i] = depths[parent] + 1

    is_leaf = children_left == children_right
    return np.any((depths <= target_depth) & is_leaf)

# --- Search for tree with a shallow leaf ---
matching_tree_index = None
for idx, tree in enumerate(model.estimators_):
    if has_leaf_at_or_below_depth(tree, TARGET_DEPTH):
        matching_tree_index = idx
        break

if matching_tree_index is not None:
    tree = model.estimators_[matching_tree_index]
    dot_data = export_graphviz(
        tree,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        max_depth=MAX_DEPTH,
        filled=True,
        rounded=True
    )
    (graph,) = pydot.graph_from_dot_data(dot_data)
    graph.write_svg(OUTPUT_FILE)
    print(f"Exported tree #{matching_tree_index} with a leaf at or below depth {TARGET_DEPTH} to '{OUTPUT_FILE}'")
else:
    print(f"No tree with a leaf at or below depth {TARGET_DEPTH} was found.")