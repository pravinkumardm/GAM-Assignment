import networkx as nx
import numpy as np
import os
from pathlib import Path

def cal_mod_gain(adjmat, resolution=1):
    # Create a zeros matrix of the same shape as the adjacency_matrix
    out_mat = np.zeros(adjmat.shape)
    # Defining the variables for the formula that do not change for the entire iterations
    m = adjmat.shape[0]

    for i in range(adjmat.shape[0]):
        for j in range(adjmat.shape[0]):
            # Defining variables that change for every value:
            # k_in is the variable defining the number of edges(sum of weights between the node k and Community C)
            k_in = adjmat[i][j]
            p1 = k_in / (2*m)

            # k_in defines sum of all the weights of the edges for k
            k_i = sum(adjmat[i])
            # sigma_tot defines the sum of all weights for the Community C
            sigma_tot = sum(adjmat.T[j])
            p2 = resolution * ((sigma_tot * k_i)/(2*m**2))
            # The modularity gain is updated for each i,j
            out_mat[i][j] = p1-p2

    # Returning the modularity gain matrix that is of the same shape as adjacency matrix
    return out_mat

def louvain_communities(adj, threshold=0.0000001):
    # Defining the loop variable
    count = 0

    # Creating the initial list of communities
    communities = [{i} for i in range(adj.shape[0])]
    print(f"Number of Communities at start : {len(communities)}")

    while count<20: #Setting an upper limit as generally due to O(nlogn) the larger graphs also converge within 10 iterations
        # only for first iteration
        if count == 0:
            new_adj = adj.copy()
        
        # For every iteration
        # compute the gains using the modularity gains calculation function
        gains = cal_mod_gain(new_adj)
        # print("Gains computed")

        # If none of the gains are greater than the threshold, then stopping the iterations
        if not np.any(gains > threshold):
            print(f"Exiting the loop as gains are lower than threshold\nAfter Iteration {count}: {len(communities)}")
            break
        
        # Phase 1 of the louvain algorithm, getting the maximum gain for each row and 
        # creating the sets of communities and appending the main communities list
        for k, value in enumerate(gains):
            filt = (value == max(value)) & (value > 0)
            out = set(sorted([k] + np.where(filt)[0].tolist()))
            if out not in communities:
                communities.append(out)

        # The communities list may have multiple duplicates, removing all the duplicates so that only the final consolidated communities remain.
        communities = sorted(communities, key=len, reverse=True)
        for i, fset in enumerate(communities[:-1]):
            for j, sset in enumerate(communities[i+1:]):
                if len(sset & fset) > 0:
                    communities[i] = fset | sset
                    communities.remove(sset)

        # Phase 2 - Creating a new adjacency matrix, that uses the consolidated communities
        new_adj = np.zeros((len(communities), len(communities)))

        # Updating the weights of the consolidated communities from the main adjacency matrix
        for i, fset in enumerate(communities[:-1]):
            for j, sset in enumerate(communities[i+1:], start=i+1):
                f = list(fset)
                s = list(sset)
                new_adj[i][j] = adj[f][:,s].sum()
        # The above code gives an upper triangular matrix only with all diagonal elements and lower triangle set to zero. 
        # Creating a symmetric adjacency matrix
        new_adj = new_adj + new_adj.T

        # Updating the count of the iterations
        count += 1
        print(f'After Iteration {count}: {len(communities)}')
    return communities


if __name__ == "__main__":
    folder = Path(__file__).parent
    network_data = folder / r"email-Eu-core.txt"
    graph = nx.read_edgelist(network_data, nodetype=int)
    adjacency_matrix = nx.to_numpy_array(graph)
    communities = louvain_communities(adjacency_matrix)
    print(f"Final number of Communities: {len(communities)}")
