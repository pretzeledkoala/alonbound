#!/usr/bin/env python3
"""
High-Speed Search for P₃-Intersecting Family Counterexample
===========================================================

This script analyzes a specific graph, chosen by the user from a library,
to find the largest P₃-intersecting family of its subgraphs and determines
if the density |F|/2^e(G) exceeds the current best of 17/128.

A P₃-intersecting family is a collection of graphs where every pair of graphs
in the family has an intersection that contains a path of length 3 (a simple
path with 3 edges and 4 vertices).

HFT Optimization Note: The core logic has been refactored to use lightweight,
hashable frozensets of edge tuples instead of heavy networkx.Graph objects
in all performance-critical loops. This minimizes object creation overhead
and leverages highly optimized native set operations, drastically improving
speed while ensuring correctness.

Author: Optimized implementation for counterexample search
"""

import networkx as nx
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict
import random
import time
from networkx.algorithms import bipartite

# Import the graph definitions from the separate library file
try:
    from graphs_library import define_graphs_to_test
except ImportError:
    print("Error: Could not find 'graphs_library.py'. Please ensure it is in the same directory.")
    exit()

# ============================================================================
# Part 1: Core Graph Utilities (Optimized for Speed)
# ============================================================================

def find_p3_from_edgeset(edgeset, memo):
    """
    Efficiently check if a given graph, represented by an edge set, contains a P₃.
    This is the most performance-critical function.

    Args:
        edgeset: A frozenset of edge tuples representing the graph.
        memo: Dictionary for memoization.
        
    Returns:
        bool: True if a path of length 3 exists, False otherwise.
    """
    # The edgeset is already a canonical, hashable key.
    if edgeset in memo:
        return memo[edgeset]

    # A path of length 3 requires at least 3 edges.
    if len(edgeset) < 3:
        memo[edgeset] = False
        return False

    # Optimization: Build a lightweight adjacency list on-the-fly.
    # This is much faster than creating a full networkx object.
    adj = defaultdict(set)
    nodes = set()
    for u, v in edgeset:
        adj[u].add(v)
        adj[v].add(u)
        nodes.add(u)
        nodes.add(v)

    # A path of length 3 also requires at least 4 vertices.
    if len(nodes) < 4:
        memo[edgeset] = False
        return False
    
    # Check all possible P₃ paths: iterate through edges to find x-u-v-y pattern.
    for u, v in edgeset:
        # Using the local adjacency list is much faster than graph.neighbors().
        u_neighbors = adj[u] - {v}
        v_neighbors = adj[v] - {u}
        
        if not u_neighbors or not v_neighbors:
            continue

        for x in u_neighbors:
            for y in v_neighbors:
                if x != y:
                    memo[edgeset] = True
                    return True
    
    memo[edgeset] = False
    return False


def check_new_edgeset_against_family(new_edgeset, family_edgesets, memo):
    """
    Efficiently verify if a new graph (as an edgeset) can be added to an
    existing P₃-intersecting family.

    Args:
        new_edgeset: The frozenset of edges for the candidate graph.
        family_edgesets: A list of frozensets for the graphs in the family.
        memo: Memoization cache for find_p3.
        
    Returns:
        bool: True if new_edgeset can be added, False otherwise.
    """
    for existing_edgeset in family_edgesets:
        # Intersection is a highly optimized native set operation.
        intersection_edgeset = new_edgeset.intersection(existing_edgeset)
        
        if not find_p3_from_edgeset(intersection_edgeset, memo):
            return False
    
    return True

# ============================================================================
# Part 2: Search Strategy and Main Loop
# ============================================================================

def find_all_p3_edgesets(base_graph):
    """
    Finds all unique path of length 3 subgraphs within the base graph.

    Returns:
        list: A list of frozensets of edges, each representing a unique P₃.
    """
    p3_edge_sets = set()
    # Pre-calculating the adjacency list is faster than repeated calls to neighbors().
    adj = {n: set(base_graph.neighbors(n)) for n in base_graph.nodes()}

    for u, v in base_graph.edges():
        u_neighbors = adj[u] - {v}
        v_neighbors = adj[v] - {u}
        for x in u_neighbors:
            for y in v_neighbors:
                if x != y:
                    # Create a canonical representation of the P₃'s edges.
                    edge1 = tuple(sorted((x, u)))
                    edge2 = tuple(sorted((u, v)))
                    edge3 = tuple(sorted((v, y)))
                    p3_edges = frozenset(sorted([edge1, edge2, edge3]))
                    p3_edge_sets.add(p3_edges)
    return list(p3_edge_sets)


def prune_family_edgesets(family_edgesets, memo):
    """
    Takes a family of edgesets and removes members until it is P₃-intersecting.
    """
    working_family = family_edgesets.copy()
    
    while True:
        conflicts = []
        for es1, es2 in itertools.combinations(working_family, 2):
            intersection = es1.intersection(es2)
            if not find_p3_from_edgeset(intersection, memo):
                conflicts.append((es1, es2))

        if not conflicts:
            break

        conflict_counts = defaultdict(int)
        for es1, es2 in conflicts:
            conflict_counts[es1] += 1
            conflict_counts[es2] += 1
        
        key_to_remove = max(conflict_counts, key=conflict_counts.get)
        working_family.remove(key_to_remove)

    return working_family


def search_for_counterexample(base_graph, name):
    """
    Main search algorithm using a hybrid "seed-and-augment" and "prune-and-augment"
    strategy to ensure both baseline and complex families are found.
    """
    print(f"\n=== Analyzing {name} ===")
    start_time = time.time()
    
    eG = base_graph.number_of_edges()
    memo = {}
    
    print(f"Base graph: {base_graph.number_of_nodes()} vertices, {eG} edges")
    
    # Generate all subgraphs as edgesets once for maximum efficiency.
    print("Generating all subgraph edgesets...")
    all_subgraph_edgesets = []
    base_edges = tuple(tuple(sorted(e)) for e in base_graph.edges())
    for i in range(1 << eG):
        edgeset = frozenset(base_edges[j] for j in range(eG) if (i >> j) & 1)
        all_subgraph_edgesets.append(edgeset)
    print(f"Generated {len(all_subgraph_edgesets)} subgraph edgesets.")

    F_best = []
    best_density = 0.0

    # --- STRATEGY 1: Seed and Augment (Guarantees baseline) ---
    print("\n--- Strategy 1: Seed and Augment ---")
    all_p3_seeds = find_all_p3_edgesets(base_graph)
    print(f"Found {len(all_p3_seeds)} unique P₃ paths to test as seeds.")

    for i, p3_seed_edges in enumerate(all_p3_seeds):
        # Constructing the trivial family is much faster with set operations.
        F_trivial = [es for es in all_subgraph_edgesets if p3_seed_edges.issubset(es)]
        
        candidates = [es for es in all_subgraph_edgesets if not p3_seed_edges.issubset(es)]
        candidates.sort(key=len, reverse=True)
        
        for new_es in candidates:
            if check_new_edgeset_against_family(new_es, F_trivial, memo):
                F_trivial.append(new_es)
        
        if len(F_trivial) > len(F_best):
            F_best = F_trivial
            best_density = len(F_best) / (2**eG)
            print(f"Seed {i+1} gave new best family! Size: {len(F_best)}, Density: {best_density:.8f}")

    # --- STRATEGY 2: Prune and Augment (Finds complex families) ---
    print("\n--- Strategy 2: Prune and Augment ---")
    all_subgraphs_by_size = sorted(all_subgraph_edgesets, key=len, reverse=True)
    
    for k in range(eG, 2, -1):
        print(f"\nTesting Kernel: All subgraphs with >= {k} edges")
        
        F_kernel = [es for es in all_subgraphs_by_size if len(es) >= k]
        if not F_kernel: continue
        
        F_valid_kernel = prune_family_edgesets(F_kernel, memo)
        
        F_augmented = F_valid_kernel.copy()
        # Create a set of keys for fast lookups.
        augmented_keys = set(F_augmented)
        candidates = [es for es in all_subgraphs_by_size if es not in augmented_keys]
        
        for new_es in candidates:
            if check_new_edgeset_against_family(new_es, F_augmented, memo):
                F_augmented.append(new_es)
        
        if len(F_augmented) > len(F_best):
            F_best = F_augmented
            best_density = len(F_best) / (2**eG)
            print(f"Kernel k={k} gave new best family! Size: {len(F_best)}, Density: {best_density:.8f}")

    elapsed = time.time() - start_time
    print(f"\nSearch completed for {name} in {elapsed:.2f} seconds")
    
    if F_best:
        return F_best, best_density
    else:
        print("No valid P₃-intersecting family found")
        return None, 0


# ============================================================================
# Part 3: Visualization
# ============================================================================

def visualize_best_result(base_graph, name, family_edgesets, density):
    """
    Create a plot visualizing the best result found.
    Converts lightweight edgesets back to networkx graphs only for plotting.
    """
    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Best Result from {name}: Density = {density:.6f}', fontsize=16)
    
    # Pre-calculate layout for consistent node positions
    pos = None
    try:
        if bipartite.is_bipartite(base_graph):
            top_nodes = {n for n, d in base_graph.nodes(data=True) if d.get("bipartite") == 0}
            pos = nx.bipartite_layout(base_graph, top_nodes)
        else:
            pos = nx.spring_layout(base_graph, seed=42)
    except:
        pos = nx.spring_layout(base_graph, seed=42)

    # First subplot: Complete base graph
    ax = axes[0, 0]
    nx.draw(base_graph, pos, ax=ax, with_labels=True, node_color='skyblue', 
            node_size=700, font_size=10, font_weight='bold', width=2)
    ax.set_title(f'Base Graph: {name}\n(V={base_graph.number_of_nodes()}, E={base_graph.number_of_edges()})')
    
    # Select up to 3 sample graphs from family for visualization
    family_edgesets.sort(key=len)
    sample_edgesets = family_edgesets[:min(3, len(family_edgesets))]

    # Plot sample graphs
    for i, edgeset in enumerate(sample_edgesets):
        row = (i + 1) // 2
        col = (i + 1) % 2
        ax = axes[row, col]
        
        # Draw base graph in light grey as background
        nx.draw_networkx_nodes(base_graph, pos, ax=ax, node_color='#cccccc', node_size=700)
        nx.draw_networkx_edges(base_graph, pos, ax=ax, edge_color='#e0e0e0', width=1.5)
        nx.draw_networkx_labels(base_graph, pos, ax=ax, font_size=10)
        
        # Draw sample graph edges in bright color
        nx.draw_networkx_edges(base_graph, pos, ax=ax, edgelist=list(edgeset), edge_color='royalblue', width=2.5)
        
        ax.set_title(f'Sample Graph {i+1} from Family (E={len(edgeset)})')
    
    # Hide unused subplot if we have fewer than 3 samples
    if len(sample_edgesets) < 3:
        axes[1, 1].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# ============================================================================
# Part 4: Main Execution Block
# ============================================================================

def main():
    """Main execution function."""
    # Define target density
    TARGET_DENSITY = 17 / 128
    
    print("=" * 60)
    print("P₃-INTERSECTING FAMILY COUNTEREXAMPLE SEARCH")
    print("=" * 60)
    print(f"Target density to beat: {TARGET_DENSITY:.8f} ({17}/{128})")
    
    # Get the dictionary of graphs to test
    graphs_to_test = define_graphs_to_test()
    graph_list = list(graphs_to_test.items())

    # Prompt user to select a graph
    print("\nPlease select a graph to analyze:")
    for i, (name, g) in enumerate(graph_list):
        print(f"  {i+1}: {name} (V={g.number_of_nodes()}, E={g.number_of_edges()})")
    
    choice = -1
    while choice < 1 or choice > len(graph_list):
        try:
            raw_choice = input(f"Enter number (1-{len(graph_list)}): ")
            choice = int(raw_choice)
        except (ValueError, IndexError):
            print("Invalid input. Please enter a number from the list.")

    # Get the chosen graph
    name, graph = graph_list[choice - 1]

    # Search for counterexample
    family_edgesets, density = search_for_counterexample(graph, name)
    
    if family_edgesets is not None:
        print(f"\n" + "-" * 60)
        print(f"FINAL RESULTS FOR: {name}")
        print("-" * 60)
        print(f"Best family size found: {len(family_edgesets)}")
        print(f"Base graph edges: {graph.number_of_edges()}")
        print(f"Density: {len(family_edgesets)}/2^{graph.number_of_edges()} = {density:.8f}")
        print(f"Target density: {TARGET_DENSITY:.8f}")
        
        if density > TARGET_DENSITY:
            print("\n🎉 NEW RECORD! This density exceeds the target! 🎉")
        else:
            print("\nThis density does not exceed the target.")
        
        # Visualize the result
        visualize_best_result(graph, name, family_edgesets, density)
        
    else:
        print(f"\nNo valid P₃-intersecting family could be constructed for {name}")


if __name__ == "__main__":
    main()
