# Recovering libraries

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Basic Libraries
import numpy as np
import pandas as pd

# Libraries for Node2Vec
import networkx as nx
from node2vec import Node2Vec

# Libraries for the Random Walk
import random


# Function for getting sequences using random walk
# input: a graph, a node, a size of the sequence
# output: a list of sequences

def convert(set):
    return sorted(set)

def random_walk(g, node, walk_lenght):
    rwalk = [node]
    for i in range(walk_lenght-1):
        temp = g.neighbors(node, mode="out")
        temp = convert(set(temp) - set(rwalk))
        if temp == 0:
            break
        new_node = random.choice(temp)
        rwalk.append(new_node)
    return rwalk

def rw_n_times(g, node, walk_lenght, n_times):
    list_rw = []
    for i in range(0, n_times):
        rw = random_walk(g, node, walk_lenght)
        list_rw.append(rw)
    return list_rw

# Function for building paths with the algorithm Node2Vec
# input: a graph, dimensions: Embedding dimensions, walk_length: Number of nodes in each walk, num_walks: Number of walks per node
#        p: Return hyper parameter, q: Inout parameter, workers: Number of workers for parallel execution 
# output: a plot

def nodeTOvec(g, n_dim, w_len, n_walk, pi, qi):
    aux = []
    node2vec = Node2Vec(g, dimensions=n_dim, walk_length=w_len, num_walks=n_walk, p = pi, q = qi)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    for i in range(0,len(g.degree())):
        aux.append(model.wv.most_similar(i))
    paths = []
    for i in range(0,len(aux)):
        a = []
        for j in range(0,len(aux[i])):
            a.append(int(aux[i][j][0]))
        paths.append(a)
    return paths

