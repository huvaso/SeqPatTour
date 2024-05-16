#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Script for extracting traversal sequences using iGraph, NetworkX and Networkit

# Libraries for buiding graphs (igraph is quickly than networkx)
import igraph as ig
from igraph import *

# Libraries for buiding graphs (networkit is quickly than igraph)
import networkit as nk

# Libraries for buiding graphs
import networkx as nx

# Libraries for getting the elapsed time
from timeit import default_timer as timer

# Function for finding all paths from a source to a target. If target is EMPTY, all targets are searched
# input: graph, start and end vertices, options (in, out, all)
# output: all paths
# Note: very greedy

def find_all_paths2(G, start, end, vn = []):
    vn = vn if type(vn) is list else [vn]
    #vn = list(set(vn)-set([start,end]))
    path  = []
    paths = []
    queue = [(start, end, path)]
    while queue:
        start, end, path = queue.pop()
        path = path + [start]

        if start not in vn:
            for node in set(G.neighbors(start,mode='OUT')).difference(path):
                queue.append((node, end, path))

            if start == end and len(path) > 0:              
                paths.append(path)
            else:
                pass
        else:
            pass
    return paths

# Function for finding all paths from a source to a target
# input: graph, start V1 and end V2 vertices
# output: all paths starting at V1 and ending at V2

def get_paths_in_out(g, v1, v2):
    paths = g.get_all_simple_paths(v1,v2)
    return paths


# Function for finding all paths from a source to a target using iGraph
# input: graph, start V1 and end V2 vertices
# output: all paths starting at V1 and ending at V2
# Note: This method is not adecuate for our purposes (all paths)

def traversal_ig_simple(g, start, end): # 
    tseq = []
    for i in range(0, end):
        path = g.get_all_simple_paths(start, i)
        if len(path) > 0:
            tseq.append(path)
    return tseq

# Function for finding all shortest paths from a source to a target using iGraph - Dijkstra algo
# input: graph, start V1 and end V2 vertices
# output: all paths starting at V1 and ending at V2

def traversals_ig_dij(g, start, end):
    tseq = []
    for i in range(0, end):
        if i != start:
            path = g.get_shortest_paths(start, i, mode='out', output='vpath') #It was fixed in OUT
            #print(path[0])
            if len(path[0]) > 0:
                tseq.append(path[0])
    return tseq

# Function for finding all paths from a source to a target using Neworkit and A* algorithm
# input: graph, start V1 and end V2 vertices
# output: all paths starting at V1 and ending at V2

def traversals_nk_A(g, start, end):
    heuristic = [0 for _ in range(g.upperNodeIdBound())]
    tseq = []
    for i in range(0, end):
        astar = nk.distance.AStar(g, heuristic, start, i)
        astar.run()
        if len(astar.getPath()) > 0:
            p = []
            p.append(start)
            r = astar.getPath()
            r.append(i)
            a = p + r
            tseq.append(a)
    return tseq

# Function for finding all paths from a source to a target using Neworkit and BFS algorithm
# input: graph, start V1 and end V2 vertices
# output: all paths starting at V1 and ending at V2

def traversals_nk_bfs(g, start, end):
    tseq = []
    for i in range(0, end):
        bfs = nk.distance.BFS(g, start, True, False, i)
        bfs.run()
        if len(bfs.getPath(i)) > 0:
            r = bfs.getPath(i)
            tseq.append(r)
    return tseq

# Function for finding all paths from a source to a target using Neworkit and Dijkstra algorithm
# input: graph, start V1 and end V2 vertices
# output: all paths starting at V1 and ending at V2

def traversals_nk_dij(g, start, end):
    tseq = []
    dij = nk.distance.Dijkstra(g, start, True, False, end)
    dij.run()
    for i in range(0, end):
        if len(dij.getPath(i)) > 0:
            r = dij.getPath(i)
            tseq.append(r)
    return tseq

# Function for finding all paths from a source to each node in NetworkX and shorted_path algorithm
# input: graph, and the start vertex 
# output: all paths starting at V1 and ending at each node

def traversals_nx_shortest(gx, start):
    tseq = []
    path_single = nx.single_source_shortest_path(gx, start)
    for i in path_single:
        #if len(path_single[i]) > 0:
            #print(path_single[i])
        tseq.append(path_single[i])
    return tseq

# Function for finding all paths from a source to each node in NetworkX and Dijkstra algorithm
# input: graph, and the start vertex 
# output: all paths starting at V1 and ending at each node

def traversals_nx_dij(gx, start):
    tseq = []
    path_single = nx.single_source_dijkstra_path(gx, start)
    for i in path_single:
        #if len(path_single[i]) > 0:
            #print(path_single[i])
        tseq.append(path_single[i])
    return tseq

# Function for finding all paths from a source to each node using different libraries and methods
# input: graphs, and the start and end vertices 
# output: the name, the time and pahts for each method and libraries

def extraction_paths_from_graph(gi, gk, gx, start, end):
    lst_paths = []
    ### Paths extraction using Networkit A
    aux =[]
    aux.append('nk_a')
    start_t = timer()
    tseq_a_nk = traversals_nk_A(gk, start, end)
    end_t = timer()
    time_a_nk_paths = (end_t - start_t)
    aux.append(time_a_nk_paths)
    aux.append(len(tseq_a_nk))
    aux.append(tseq_a_nk)
    lst_paths.append(aux)
    ## Paths extraction using Networkit BFS
    aux =[]
    aux.append('nk_bfs')
    start_t = timer()
    tseq_bfs_nk = traversals_nk_bfs(gk, start, end)
    end_t = timer()
    time_bfs_nk_paths = (end_t - start_t)
    aux.append(time_bfs_nk_paths)
    aux.append(len(tseq_bfs_nk))
    aux.append(tseq_bfs_nk)
    lst_paths.append(aux)
    ## Paths extraction using Networkit Dijkstra
    aux =[]
    aux.append('nk_dij')
    start_t = timer()
    tseq_dij_nk = traversals_nk_dij(gk, start, end)
    end_t = timer()
    time_dij_nk_paths = (end_t - start_t)
    aux.append(time_dij_nk_paths)
    aux.append(len(tseq_dij_nk))
    aux.append(tseq_dij_nk)
    lst_paths.append(aux)

    ### Paths extraction using NetworkX Dijkstra
    aux =[]
    aux.append('nx_dij')
    start_t = timer()
    tseq_dij_nx = traversals_nx_dij(gx, start)
    end_t = timer()
    time_dij_nx_paths = (end_t - start_t)
    aux.append(time_dij_nx_paths)
    aux.append(len(tseq_dij_nx))
    aux.append(tseq_dij_nx)
    lst_paths.append(aux)
    ### Paths extraction using NetworkX Dijkstra
    aux =[]
    aux.append('nx_shor')
    start_t = timer()
    start_t = timer()
    tseq_short_nx = traversals_nx_shortest(gx, start)
    end_t = timer()
    time_short_nx_paths = (end_t - start_t)
    aux.append(time_short_nx_paths)
    aux.append(len(tseq_short_nx))
    aux.append(tseq_short_nx)
    lst_paths.append(aux)

    ### Paths extraction using iGraph Dijkstra
    aux =[]
    aux.append('ni_dij')
    start_t = timer()
    tseq_dij_ig = traversals_ig_dij(gi, start, end)
    end_t = timer()
    time_dij_ni_paths = (end_t - start_t)
    aux.append(time_dij_ni_paths)
    aux.append(len(tseq_dij_ig))
    aux.append(tseq_dij_ig)
    lst_paths.append(aux)

    return lst_paths