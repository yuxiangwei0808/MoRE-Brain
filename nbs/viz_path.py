import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.gridspec import GridSpec

def create_complete_binary_tree(num_leaves):
    """Creates a NetworkX DiGraph representing a complete binary tree."""
    if num_leaves <= 0 or (num_leaves & (num_leaves - 1) != 0):
        raise ValueError("Number of leaves must be a positive power of 2.")

    height = int(math.log2(num_leaves))
    total_nodes = 2**(height + 1) - 1

    G = nx.DiGraph()
    G.add_nodes_from(range(total_nodes))

    for i in range(total_nodes):
        left_child = 2 * i + 1
        right_child = 2 * i + 2
        if left_child < total_nodes:
            G.add_edge(i, left_child)
        if right_child < total_nodes:
            G.add_edge(i, right_child)

    # Identify leaf nodes (nodes in the last level)
    first_leaf_node_index = 2**height - 1
    leaf_nodes = list(range(first_leaf_node_index, total_nodes))

    # Map leaf index (0 to N-1) to actual node ID
    leaf_index_to_node_id = {i: node_id for i, node_id in enumerate(leaf_nodes)}

    return G, leaf_index_to_node_id

def get_path_to_leaf_by_index(G, leaf_index, leaf_index_to_node_id):
    """Finds the path from the root (node 0) to the leaf specified by its index."""
    if leaf_index not in leaf_index_to_node_id:
        raise ValueError(f"Invalid leaf index: {leaf_index}")

    target_node = leaf_index_to_node_id[leaf_index]
    path_nodes = [target_node]
    current_node = target_node

    while current_node != 0:
        # Find the parent (works for the 0, 1, 2... indexing)
        parent = math.floor((current_node - 1) / 2)
        path_nodes.append(parent)
        current_node = parent

    return path_nodes[::-1] 

def visualize_path(paths, path_color='red', base_color='gray', node_color='skyblue'):
    num_leaves = 16

    G, leaf_index_to_node_id = create_complete_binary_tree(num_leaves)
    all_path_edges = []
    for k, i in paths.items():
        path_nodes = get_path_to_leaf_by_index(G, i, leaf_index_to_node_id)
        # Convert node sequence to edges (u, v)
        path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))
        all_path_edges.extend(path_edges)

    plt.figure(figsize=(12, 8))
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')

    # 1. Draw all nodes
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color=node_color, alpha=0.9)

    # 2. Draw all edges of the tree faintly
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color=base_color, width=1.0)

    # 3. Draw the edges belonging to any root-to-leaf path more prominently
    nx.draw_networkx_edges(G, pos, edgelist=all_path_edges,
                           width=1.5, alpha=0.8, edge_color=path_color)

    # 4. Draw labels (optional, can be cluttered)
    # nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title(f"Complete Binary Tree with {num_leaves} Leaves - All Root-to-Leaf Paths Highlighted")
    plt.axis('off') # Turn off the axis
    plt.show()


def analyze_path_distribution(path_sums):
    """
    Analyze and visualize the distribution of highest paths across all trees.
    
    Args:
        path_sums: Array of path sums for all trees, shape (n_trees, n_paths)
    """
    if path_sums.ndim == 1:
        highest_path_indices = path_sums
    else:
        highest_path_indices = np.argmax(path_sums, axis=1)
    path_frequencies = np.bincount(highest_path_indices, minlength=16) / len(highest_path_indices)
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(16), path_frequencies, color='skyblue')
    plt.title('Distribution of Highest Path Indices Across All Trees')
    plt.xlabel('Path Index')
    plt.ylabel('Frequency')
    plt.xticks(range(16))
    plt.grid(axis='y', alpha=0.75)
    plt.show()

def visualize_category_distribution(top1_label):
    """
    Visualize the distribution of top 1 paths across different categories.
    
    Args:
        top1_label: Dictionary mapping category names to their respective path indices.
    """
    plt.figure(figsize=(12, 6))
    for label, paths in top1_label.items():
        plt.hist(paths, bins=16, alpha=0.5, label=label)
    
    plt.title('Distribution of Top 1 Paths Across Categories')
    plt.xlabel('Path Index')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.show()


if __name__ == '__main__':
    np.random.seed(42)  # For reproducible results
    top1_arr = np.load('Viz/subj01/path_top1/shap_attrs_path_top1.npy')
    top1_arr = top1_arr.flatten()

    label_matrix = np.load('data/NSD/coco_labels_val.npy')[:1000]
    label_matrix = np.concatenate([label_matrix, label_matrix, label_matrix], axis=0)
    labels = ['accessory', 'animal', 'appliance', 'electronic', 'food', 'furniture', 'indoor', 'kitchen', 'outdoor', 'person', 'sports', 'vehicle']
    pt = [3, 14, 3, 13, 12, 10, 8, 3, 6, 2, 0, 3, 3]

    top1_label = {}
    for i in range(len(labels)):
        idx = np.where(label_matrix[:, i] == 1)[0]
        top1_label[labels[i]] = top1_arr[idx]
    visualize_category_distribution(top1_label)

    # get the most frequent path for each label
    most_frequent_paths = {label: np.bincount(paths).argmax() for label, paths in top1_label.items()}
    most_frequent_paths = {labels[i]: pt[i] for i in range(len(labels))}

    visualize_path(most_frequent_paths)

    # Analyze the distribution of highest paths
    analyze_path_distribution(top1_arr)
