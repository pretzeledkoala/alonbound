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

def find_p3(graph, memo):
    """
    Efficiently check if a given graph contains a path of length 3 (P₄).
    
    A path of length 3 consists of 4 vertices and 3 edges (e.g., a-b-c-d).
    This is the most performance-critical function. Uses memoization to cache
    results based on the graph's edge set.
    
    Args:
        graph: A networkx.Graph object
        memo: Dictionary for memoization to cache results
        
    Returns:
        bool: True if a path of length 3 exists, False otherwise
    """
    # Create a unique, hashable key based on the graph's edge set.
    # Sorting edges makes the key canonical.
    edges_key = frozenset(tuple(sorted(e)) for e in graph.edges())
    
    # Check cache first
    if edges_key in memo:
        return memo[edges_key]
    
    # A path of length 3 requires at least 3 edges and 4 vertices.
    if graph.number_of_edges() < 3 or graph.number_of_nodes() < 4:
        memo[edges_key] = False
        return False
    
    # Check all possible P₃ paths: iterate through edges to find x-u-v-y pattern
    for u, v in graph.edges():
        # Get neighbors of u (excluding v) and neighbors of v (excluding u)
        u_neighbors = set(graph.neighbors(u)) - {v}
        v_neighbors = set(graph.neighbors(v)) - {u}
        
        # If either has no other neighbors, this edge cannot be the middle of a P₃
        if not u_neighbors or not v_neighbors:
            continue

        # Check if there exist x ∈ N(u) and y ∈ N(v) such that x, u, v, y are distinct.
        # This forms the path x-u-v-y.
        for x in u_neighbors:
            # We need to ensure y is not x. The sets u_neighbors and v_neighbors
            # might overlap.
            for y in v_neighbors:
                if x != y:
                    # Found a path of length 3: x-u-v-y
                    memo[edges_key] = True
                    return True
    
    # No path of length 3 found
    memo[edges_key] = False
    return False


def get_graph_intersection(graph1, graph2):
    """
    Compute the intersection of two graphs.
    
    Args:
        graph1, graph2: networkx.Graph objects
        
    Returns:
        networkx.Graph: New graph containing only edges present in both inputs
    """
    # Get edge sets
    e1 = set(frozenset(e) for e in graph1.edges())
    e2 = set(frozenset(e) for e in graph2.edges())
    
    # Compute intersection
    common_edges = e1.intersection(e2)
    
    # Create intersection graph
    G_int = nx.Graph()
    
    # Add all nodes from the base graph to ensure consistency
    G_int.add_nodes_from(graph1.nodes())
    
    # Add common edges
    G_int.add_edges_from(common_edges)
    
    return G_int


def check_new_graph_against_family(new_graph, family, memo):
    """
    Efficiently verify if a new graph can be added to an existing P₃-intersecting family.
    
    For a graph to be added, its intersection with every graph already in the 
    family must contain a path of length 3.
    
    Args:
        new_graph: The candidate graph to add
        family: List of graphs in the existing valid family
        memo: Memoization cache for find_p3
        
    Returns:
        bool: True if new_graph can be added, False otherwise
    """
    for graph_in_family in family:
        # Compute intersection
        G_int = get_graph_intersection(new_graph, graph_in_family)
        
        # Check if intersection contains a path of length 3
        if not find_p3(G_int, memo):
            return False
    
    return True

# ============================================================================
# Part 2: Search Strategy and Main Loop
# ============================================================================

def prune_family(family, memo):
    """
    Takes a family of graphs and removes members until it is P₃-intersecting.
    Uses a greedy heuristic: repeatedly remove the graph involved in the most
    conflicts.

    Args:
        family: A list of networkx.Graph objects.
        memo: The memoization cache for find_p3.

    Returns:
        A valid P₃-intersecting family (a subset of the input family).
    """
    
    working_family = family.copy()
    
    while True:
        conflicts = []
        # Find all conflicting pairs
        for g1, g2 in itertools.combinations(working_family, 2):
            intersection = get_graph_intersection(g1, g2)
            if not find_p3(intersection, memo):
                conflicts.append((g1, g2))

        if not conflicts:
            # No conflicts found, the family is valid
            break

        # Count which graphs are involved in the most conflicts
        conflict_counts = defaultdict(int)
        for g1, g2 in conflicts:
            # Use a canonical key for the graphs (sorted frozenset of edges)
            key1 = frozenset(tuple(sorted(e)) for e in g1.edges())
            key2 = frozenset(tuple(sorted(e)) for e in g2.edges())
            conflict_counts[key1] += 1
            conflict_counts[key2] += 1
        
        # Find the graph to remove (the one with the highest conflict count)
        max_conflicts = -1
        key_to_remove = None
        for key, count in conflict_counts.items():
            if count > max_conflicts:
                max_conflicts = count
                key_to_remove = key
        
        # Remove the most problematic graph from our working family
        working_family = [g for g in working_family if frozenset(tuple(sorted(e)) for e in g.edges()) != key_to_remove]

    return working_family


def search_for_counterexample(base_graph, name):
    """
    Main search algorithm using a prune-then-augment strategy.
    
    Args:
        base_graph: The base networkx.Graph to analyze
        name: Name of the graph for reporting
        
    Returns:
        tuple: (best_family, density) or (None, 0) if no valid family found
    """
    print(f"\n=== Analyzing {name} ===")
    start_time = time.time()
    
    eG = base_graph.number_of_edges()
    memo = {}
    
    print(f"Base graph: {base_graph.number_of_nodes()} vertices, {eG} edges")
    
    # Generate all subgraphs once
    print("Generating all subgraphs...")
    all_subgraphs = []
    base_edges = list(base_graph.edges())
    nodes = list(base_graph.nodes())
    for i in range(1 << eG):
        subgraph = nx.Graph()
        subgraph.add_nodes_from(nodes)
        edges_to_add = [base_edges[j] for j in range(eG) if (i >> j) & 1]
        subgraph.add_edges_from(edges_to_add)
        all_subgraphs.append(subgraph)
    print(f"Generated {len(all_subgraphs)} subgraphs.")

    F_best = []
    best_density = 0.0

    # Test different high-density kernels
    for k in range(eG, 2, -1):
        print(f"\n--- Testing Kernel: All subgraphs with >= {k} edges ---")
        
        # 1. Create the kernel
        F_kernel = [g for g in all_subgraphs if g.number_of_edges() >= k]
        if not F_kernel:
            continue
        print(f"Initial kernel size: {len(F_kernel)}")

        # 2. Prune the kernel to make it a valid P₃-intersecting family
        F_valid_kernel = prune_family(F_kernel, memo)
        print(f"Size after pruning: {len(F_valid_kernel)}")

        # 3. Greedily augment the valid kernel
        F_augmented = F_valid_kernel.copy()
        candidates = [g for g in all_subgraphs if g not in F_augmented]
        # Sort candidates to test larger graphs first
        candidates.sort(key=lambda g: g.number_of_edges(), reverse=True)
        
        added_count = 0
        for g_new in candidates:
            if check_new_graph_against_family(g_new, F_augmented, memo):
                F_augmented.append(g_new)
                added_count += 1
        
        if added_count > 0:
            print(f"Greedy augmentation added {added_count} graphs.")
        
        # 4. Update best result found so far
        if len(F_augmented) > len(F_best):
            F_best = F_augmented
            best_density = len(F_best) / (2**eG)
            print(f"Found new best family! Size: {len(F_best)}, Density: {best_density:.8f}")

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

def visualize_best_result(base_graph, name, family, density):
    """
    Create a plot visualizing the best result found.
    
    Args:
        base_graph: The base networkx.Graph
        name: Name of the graph
        family: The best family found
        density: Calculated density
    """
    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Best Result from {name}: Density = {density:.6f}', fontsize=16)
    
    # Pre-calculate layout for consistent node positions
    pos = None
    try:
        if bipartite.is_bipartite(base_graph):
            # Get the partitions for bipartite layout
            top_nodes = {n for n, d in base_graph.nodes(data=True) if d.get("bipartite") == 0}
            pos = nx.bipartite_layout(base_graph, top_nodes)
        else:
            pos = nx.spring_layout(base_graph, seed=42)
    except:
        # Fallback for any other errors
        pos = nx.spring_layout(base_graph, seed=42)

    
    # First subplot: Complete base graph
    ax = axes[0, 0]
    nx.draw(base_graph, pos, ax=ax, with_labels=True, node_color='skyblue', 
            node_size=700, font_size=10, font_weight='bold', width=2)
    ax.set_title(f'Base Graph: {name}\n(V={base_graph.number_of_nodes()}, E={base_graph.number_of_edges()})')
    
    # Select up to 3 sample graphs from family for visualization
    # Prioritize showing graphs with fewer edges if possible
    family.sort(key=lambda g: g.number_of_edges())
    sample_graphs = family[:min(3, len(family))]

    # Plot sample graphs
    for i, sample in enumerate(sample_graphs):
        row = (i + 1) // 2
        col = (i + 1) % 2
        ax = axes[row, col]
        
        # Draw base graph in light grey as background
        nx.draw_networkx_nodes(base_graph, pos, ax=ax, node_color='#cccccc', node_size=700)
        nx.draw_networkx_edges(base_graph, pos, ax=ax, edge_color='#e0e0e0', width=1.5)
        nx.draw_networkx_labels(base_graph, pos, ax=ax, font_size=10)
        
        # Draw sample graph edges in bright color
        nx.draw_networkx_edges(sample, pos, ax=ax, edge_color='royalblue', width=2.5)
        
        ax.set_title(f'Sample Graph {i+1} from Family (E={sample.number_of_edges()})')
    
    # Hide unused subplot if we have fewer than 3 samples
    if len(sample_graphs) < 3:
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
    family, density = search_for_counterexample(graph, name)
    
    if family is not None:
        print(f"\n" + "-" * 60)
        print(f"FINAL RESULTS FOR: {name}")
        print("-" * 60)
        print(f"Best family size found: {len(family)}")
        print(f"Base graph edges: {graph.number_of_edges()}")
        print(f"Density: {len(family)}/2^{graph.number_of_edges()} = {density:.8f}")
        print(f"Target density: {TARGET_DENSITY:.8f}")
        
        if density > TARGET_DENSITY:
            print("\n🎉 NEW RECORD! This density exceeds the target! 🎉")
        else:
            print("\nThis density does not exceed the target.")
        
        # Visualize the result
        visualize_best_result(graph, name, family, density)
        
    else:
        print(f"\nNo valid P₃-intersecting family could be constructed for {name}")


if __name__ == "__main__":
    main()
