import networkx as nx
from sklearn.metrics.cluster import fowlkes_mallows_score,adjusted_mutual_info_score
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# get code_location and traverse to datsets
# Loading the files for EU-Core Dataset
folder = Path(__file__).parent
network_data = folder / r"email-Eu-core.txt"
data_labels = folder / r"email-Eu-core-department-labels.txt"

# Creating a graph from the net_data, as a directed graph
graph = nx.read_edgelist(network_data, nodetype=int, create_using=nx.DiGraph())


# Question 1
# Identify the largest weakly connected component and on the largest connected component and compute communities according to any community detection approach you select. You may choose an approach which produces disjoint communities or overlapping communities. Compare the set of communities with the set of ground truth communities using a suitable method. Clearly explain how you compared the two sets of communities, this may either be a method already proposed in the literature or your own (provide relevant citation clearly). Comment on how similar or different, the two sets of communities are, based on your score and comparison method. 

#Answer 
# Steps to complete question 1
# 1. Identify the largest weakly connected component 
# 2. Compute communities in the largest weakly connected component using any community detection approach (Using the Louvain Community Detection Algorithm)
# 3. Get Ground Truth Communities
# 4. Compare communities with Ground Truth Communities

# Step 1-1: Identify the largest weakly connected component and create a graph using the largest connected component
largest_connected_component = max(nx.weakly_connected_components(graph), key=len)
lcc_graph = graph.subgraph(largest_connected_component)

# Step 1-2: Compute communities in the largest weakly connected component using any community detection approach
computed_communities = sorted(nx.algorithms.community.louvain_communities(lcc_graph))

# Step 1-3: Get Ground Truth Communities
# One way for the ground truth communities is to use the department labels to get the ground truth
with open(data_labels) as fi:
    labeldata = fi.readlines()

gt_labels = {}
for i in labeldata:
    person, dept = [int(x) for x in i.strip().split()]
    if dept not in gt_labels:
        gt_labels[dept] = []
    if person in largest_connected_component:
        gt_labels[dept].append(person)

gt_communities = list(gt_labels.values())

# Step 1-4: Compare communities with Ground Truth Communities
# converting each list of sets to list of sorted lists and then comparing the computed_communities
sorted_computed_communities = sorted([sorted(list(x)) for x in computed_communities])
sorted_ground_truth_communities = sorted([sorted(x) for x in gt_communities])

computed_labels = []
gt_labels = []

for node in list(largest_connected_component):
    for idx, community in enumerate(sorted_computed_communities):
        if node in community:
            computed_labels.append(idx)
    for idx, community in enumerate(sorted_ground_truth_communities):
        if node in community:
            gt_labels.append(idx)

# ars = round(adjusted_rand_score(computed_labels, gt_labels), 5)
# The adjusted rand score is : {ars}
# The normalized mutual info score is : {nmis}
# nmis = round(normalized_mutual_info_score(computed_labels, gt_labels), 5)




fms = round(fowlkes_mallows_score(computed_labels, gt_labels), 5)
amis = round(adjusted_mutual_info_score(computed_labels, gt_labels), 5)
print(f'''
The adjusted fowlkes mallows score is : {fms} 
The adjusted mutual information score is : {amis}''')