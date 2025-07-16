#!/usr/bin/env python3
"""
Graph Library for P₃-Intersecting Family Search
================================================

This file contains the definitions for various graphs to be tested.
The `define_graphs_to_test` function returns a dictionary of named
networkx.Graph objects.
"""

import networkx as nx

def define_graphs_to_test():
    """
    Defines a collection of graphs to be tested.
    
    Returns:
        dict: A dictionary where keys are graph names and values are
              networkx.Graph objects.
    """
    graphs = {}

    # K_3,3
    G_k33 = nx.complete_bipartite_graph(3, 3)
    graphs['K_3,3'] = G_k33

    # Christofides Graph (Corrected)
    # This is a K_{2,3} graph with a pendant vertex attached to one of the
    # vertices in the partition of size 2.
    G_chris = nx.Graph()
    # Define partitions for K_2,3
    U = ['u1', 'u2']
    V = ['v1', 'v2', 'v3']
    # Add pendant vertex
    W = ['w']
    
    # Add nodes with the 'bipartite' attribute set for proper visualization
    G_chris.add_nodes_from(U, bipartite=0)
    G_chris.add_nodes_from(V, bipartite=1)
    G_chris.add_node(W[0], bipartite=1) # The pendant vertex joins the larger partition
    
    # Add edges for K_2,3
    for u_node in U:
        for v_node in V:
            G_chris.add_edge(u_node, v_node)
    
    # Add the pendant edge (connecting a node from partition 1 to partition 0)
    G_chris.add_edge(W[0], U[0])
    
    graphs['Christofides Graph (K_2,3 + pendant)'] = G_chris
    
    # K_2,4
    G_k24 = nx.complete_bipartite_graph(2, 4)
    graphs['K_2,4'] = G_k24
    
    # K_2,2,2 (Octahedron)
    G_octa = nx.complete_multipartite_graph(2, 2, 2)
    graphs['K_2,2,2 (Octahedron)'] = G_octa

    # Add other custom graphs here
    # Example:
    # G_custom = nx.Graph()
    # G_custom.add_edges_from([(0,1), (1,2), (2,3)])
    # graphs['My Custom Graph'] = G_custom

    return graphs
