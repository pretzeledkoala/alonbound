import networkx as nx
import itertools
import matplotlib.pyplot as plt

# Step 1: Create a function that checks if two graphs share a P3

def is_edge(i, j, input) -> bool:
    G1 = input[i]
    G2 = input[j]
    R = nx.intersection(G1, G2)
    my_nodes = R.nodes()
    perms = itertools.permutations(my_nodes, 4)
    for perm in perms:
        if R.has_edge(perm[0], perm[1]):
            if R.has_edge(perm[1], perm[2]):
                if R.has_edge(perm[2], perm[3]):
                    return True
    return False

# main function: finds the biggest collection of subgraphs in which any two share a P3
def compute(K, M) -> int:
    edges = list(K.edges())
    n = len(edges)
    # Enumerate all 2^n subgraphs
    subgraphs = []
    for bits in range(2**n):
        bits_str = format(bits, f'0{n}b')
        G = nx.Graph()
        G.add_nodes_from(K)
        G.add_edges_from([edges[i] for i in range(n) if bits_str[i] == '1'])
        if (len(list(G.edges())) >= M):
            subgraphs.append(G)

    # Make a graph connecting subgraphs if their intersection includes a copy of P3
    Supergraph = nx.Graph()
    size = len(subgraphs)
    Supergraph.add_nodes_from(range(size))

    for i in range(size):
        j = i + 1
        while (j < size):
            if is_edge(i, j, subgraphs):
                Supergraph.add_edge(i, j)
            j = j + 1
    # Return the size of the biggest clique
    return max(len(c) for c in nx.find_cliques(Supergraph))

# Start with a complete graph on 6 vertices
J = nx.complete_graph(6)
e = 7 # number of edges to remove
edges = list(J.edges())
edge_lists = itertools.combinations(edges, e)
i = 0
# Iterate over graphs with the right number of edges
for edge_list in edge_lists:
    i = i + 1
    if (i % 100 == 0): # Making sure that the code is working
        print(i)
    K = nx.complete_graph(6)
    K.remove_edges_from(edge_list)
    big = compute(K, 5)
    if (big > 34):
        nx.draw(K)
        plt.show()
