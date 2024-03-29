import random
import numpy as np
import math
import networkx as nx

class ERModel:
    def __init__(self, n, p) -> None:
        self.n = n
        self.p = p
        self.generate_graph()
        self.generate_edge_list()
        self.generate_adj_list()
        self.g = nx.Graph(self.adj_mat)

    def generate_graph(self):
        self.adj_mat = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i+1,self.n):
                self.adj_mat[i][j] = random.choices([0,1], [1-self.p, self.p])[0]
        self.adj_mat = self.adj_mat + self.adj_mat.T
    
    def generate_edge_list(self):
        ones = np.where(self.adj_mat == 1)
        self.edge_list = [x for x in zip(ones[0], ones[1])]
    
    def generate_adj_list(self):
        self.adj_list = {}
        for i in range(self.n):
            val = np.where(self.adj_mat[i]==1)[0].tolist()
            self.adj_list[i] = val
    
    def get_largest_component(self):
        components = nx.connected_components(self.g)
        subgraph = self.g.subgraph(max(components, key=len))
        return subgraph.nodes, subgraph.edges
    
    def get_triangles(self):
        return sum(nx.triangles(self.g).values()) / 3

class PageRank:
    def __init__(self, adj_mat, iterations=10000, n_surfers=100) -> None:
        self.adj_mat = adj_mat
        self.iterations = iterations
        self.n_surfers = n_surfers
        self.len_of_nodes = len(adj_mat)
        self.page_rank = np.ones(self.len_of_nodes) / self.len_of_nodes
        self.generate_adj_list()
        self.generate_edge_list()
        self.generate_parent_list()

    def generate_edge_list(self):
        ones = np.where(self.adj_mat == 1)
        self.edge_list = [x for x in zip(ones[0], ones[1])]

    def generate_adj_list(self):
        self.adj_list = {}
        for i in range(self.len_of_nodes):
            val = np.where(self.adj_mat[i]==1)[0].tolist()
            self.adj_list[i] = val

    def generate_parent_list(self):
        self.parent_list = {}
        for i in range(self.len_of_nodes):
            val = np.where(self.adj_mat.T[i]==1)[0].tolist()
            self.parent_list[i] = val
    
    def iterate_pagerank(self):
        for i in range(self.iterations):
            print(f"Iteration: {i}", end=" | ")
            if i == 0:
                surf = np.ones(self.len_of_nodes) * self.n_surfers
            new_rank = np.zeros(self.len_of_nodes)
            for j in range(self.len_of_nodes):
                children = self.adj_list[j]
                for each in children:
                    new_rank[each] += math.ceil(surf[j] / len(children))
            print(f'Updated Rank: {new_rank} | Old Rank: {surf} Matching updated and old: {all(new_rank == surf)}')
            
            if all(surf == new_rank):
                total = sum(surf)
                page_rank = [round((x*100)/total, 2) for x in surf]
                print(f'Final Page Ranks : {page_rank}')
                break
            surf = new_rank

def iterate_ERgraphs(n, p, iterations):
    print(f'========================= n:{n}, p:{p}, iterations: {iterations}==========================')
    model_lc = []
    for _ in range(iterations):
        ergraph = ERModel(n,p)
        lcc, _ = ergraph.get_largest_component()
        model_lc.append(len(lcc))
    if n*p < 1:
        print(f'\tTrue size of LCC (Mean for {iterations} iterations): {sum(model_lc) / len(model_lc)} : From Formula(O(log(n))): {math.log(n)} Truesize > O(log(n)) : {sum(model_lc) / len(model_lc) > math.log(n)}')
    elif n*p == 1:
        print(f'\tTrue size of LCC (Mean for {iterations} iterations): {sum(model_lc) / len(model_lc)} : From Formula(n^(2/3)): {n**(2/3)} Truesize == n^(2/3) : {sum(model_lc) / len(model_lc) == int(n**(2/3))}')
    else:
        print(f'\tTrue size of LCC (Mean for {iterations} iterations): {sum(model_lc) / len(model_lc)} : From Formula(n): {n} Truesize == n : {sum(model_lc) / len(model_lc) == int(n)}')
        
if __name__ == "__main__":
    iterate_ERgraphs(100, 0.04, 100)
    iterate_ERgraphs(100, 0.5, 100)

    print(f'===================================================')
    adjmat = np.array([
            [0,1,1,1,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0,1,0],
            [0,1,0,0,0,1,0,0,0,0],
            [1,0,0,0,0,0,0,0,0,1],
            [0,0,0,1,0,0,0,0,0,0],
            [0,0,0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,1,1,0],
            [0,0,0,0,0,1,0,0,0,0],
            [0,0,0,0,0,0,0,1,0,1],
            [0,0,0,0,1,0,0,0,1,0],
        ])
    pg = PageRank(adj_mat=adjmat)
    pg.iterate_pagerank()
