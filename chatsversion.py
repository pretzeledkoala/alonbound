import networkx as nx
import itertools
import matplotlib.pyplot as plt

# Step 1: Generate our graph with k vertices and n edges
K = nx.complete_bipartite_graph(3, 3) # change input graph as needed
edges = list(K.edges())
n = len(edges)
M = 6

# Step 2: Enumerate all 2^n subgraphs
subgraphs = []
for bits in range(2**n):
    bits_str = format(bits, f'0{n}b')
    G = nx.Graph()
    G.add_nodes_from(K)
    G.add_edges_from([edges[i] for i in range(n) if bits_str[i] == '1'])
    if (len(list(G.edges())) >= M):
        subgraphs.append(G)

Supergraph = nx.Graph()
size = len(subgraphs)
Supergraph.add_nodes_from(range(size))

def is_edge(i, j) -> bool:
    G1 = subgraphs[i]
    G2 = subgraphs[j]
    R = nx.intersection(G1, G2)
    my_nodes = R.nodes()
    perms = itertools.permutations(my_nodes, 4)
    for perm in perms:
        if R.has_edge(perm[0], perm[1]):
            if R.has_edge(perm[1], perm[2]):
                if R.has_edge(perm[2], perm[3]):
                    return True
    return False
for i in range(size):
    j = i + 1
    while (j < size):
        if is_edge(i, j):
            Supergraph.add_edge(i, j)
        j = j + 1
nx.draw(Supergraph)
plt.show()
print(max(len(c) for c in nx.find_cliques(Supergraph)))
