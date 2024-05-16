#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Script for extracting frequent sequences under constraints

# Basic Libraries
import numpy as np
import pandas as pd

# Libraries for extracting sequential patterns using Seq2Pat
from sequential.seq2pat import Seq2Pat, Attribute

# Libraries for getting the elapsed time
from timeit import default_timer as timer

# Librarires for building collections
from collections import Counter

# Libraries for extracting sequential patterns using Prefixspan
from prefixspan import PrefixSpan

# Libraries for extracting sequential patterns using Spam
from sequence_mining.spam import SpamAlgo


# Function for building a sequences from paths
# input: sequence of locations 
# output: a sequence of items (following the classical approach of sequnetial pattern mining - Han et al.)
# Library: https://github.com/fidelity/seq2pat
# Stpeps:
# 1. To build sequences from paths
# 2. To create a sequences of timestamps (consecutive times, i.e., 1, 2, 3, ...)
# 3. Extracting sequential patterns with a minimal support

def paths2seq(paths):
    # Creating sequences from paths
    seq2pat = Seq2Pat(paths)
    # Creating a sequences of timestamps (consecutive numbers from 1 to n)
    # We can change these consecutive numbers for values representig the hours/days/weeks passed between each location
    # It will be used for extracting patterns with consecutive timestamps
    num = []
    for i, sublist in enumerate(paths):
        num.append([])
        count = 1
        for item in sublist:
            num[i].append(count)
            count +=1
    timestamp = Attribute(num)
    # Adding a restriction for extract consecutive items (places)
    span_constraint = seq2pat.add_constraint(1 <= timestamp.span() <= 2)
    return seq2pat

# Function for extracting frequent sequences using Seq2Pat
# input: sequnces of locations, a minimal support (minsup)
# output: patterns appearing at least minsup times

def spm_old(p2s, minsup):
    return p2s.get_patterns(min_frequency=minsup)

def join_columns(row):
    return row['patterns'] + [row['supp']]

def spm(p2s, minsup, big_graph):
    patt = p2s.get_patterns(min_frequency=minsup)
    verf_patt = verify_patt_graph(list2dataframe(patt), big_graph)
    verf_patt['joined_list'] = verf_patt.apply(join_columns, axis=1)
    return verf_patt['joined_list'].to_list()

# Function for recovering the vertex ID instead the vertex index
# input: pattern like [a, b, 234], a graph
# output: patterns with the ID like [id_a, id_b, 234]
# Note: The last value of the pattern is the absolute support (the times pattern appears in the dataset)

def recovering_id(patterns, g):
    patt_id = []
    for i, sublist in enumerate(patterns):
        aux = []
        for j in range(0,(len(sublist))):
            #print(patterns[i][j])
            aux.append(g.vs[patterns[i][j]]['id'])
        patt_id.append(aux)
    return patt_id

# Function for recovering the vertex ID instead the vertex index
# input: pattern like [a, b, 234], a graph
# output: patterns with the ID like [id_a, id_b, 234]
# Note: The last value of the pattern is the absolute support (the times pattern appears in the dataset)

def recovering_id_with_supp(patterns, g):
    patt_id = []
    for i, sublist in enumerate(patterns):
        patt_id.append([])
        for j in range(0,(len(sublist)-1)):
            patt_id[i].append(g.vs[patterns[i][j]]['id'])
        patt_id[i].append(sublist[(len(sublist)-1)])
    return patt_id

# Function for building a dataframe for each pattern (as an example)
# input: a pattern index (or position)
# output: a dataframe containg the locations following in the pattern
# Note: it is needed to reseat the index in the dataframe

def df2fig(vertices, id_pattern, patt_id):
    data_to_viz = vertices.loc[vertices['id'].isin(patt_id[id_pattern])]
    data_to_viz.reset_index(inplace=True)
    return data_to_viz

# Function for counting items in a sequence
# input: a list of sequences
# output: a dictionary index(item) : value (frequence)

def count_items_seq(list_seq):
    freq = {}
    for i in list_seq:
        for j in i: 
            if j in freq:
                freq[j] += 1
            else:
                freq[j] = 1
    return freq

# Function for building a vector of indices and a vector of frequency from a doctionary
# input: a dictionary
# output: two vectors 

def dic2lists(dic):
    keys = []
    values = []
    for key in dic.keys():
        keys.append(str(key))
    for value in dic.values():
        values.append(value)
    return keys, values

# Function for adding a label to each sequential pattern (for a Patricia trie buidling purposes)
# input: two set of patterns to be compared
# output: a list of patterns with labels 

def del_consecutive_duplicates(cadena):
    return ''.join(char for i, char in enumerate(cadena) if i == 0 or char != cadena[i - 1])

def merge_lists_with_suffix(gr_1, gr_2):
    result = []
    suffix_1 = 1
    suffix_2 = 2

    def add_suffix(row, suffix):
        return [f"{value}.{suffix}" for value in row]

    # Agregar sufijo .1 a las listas en gr_1
    for row in gr_1:
        result.append(add_suffix(row, suffix_1))

    # Agregar sufijo .2 a las listas en gr_2
    for row in gr_2:
        result.append(add_suffix(row, suffix_2))

    # Reemplazar sufijos si hay duplicados entre gr_1 y gr_2
    occurrences = {}
    for row in result:
        for value in row:
            base_value = int(value.split('.')[0])
            if base_value in occurrences:
                occurrences[base_value].append(value.split('.')[1])
            else:
                occurrences[base_value] = [value.split('.')[1]]
                

    for key, values in occurrences.items():
        if len(values) > 1:
            for i, row in enumerate(result):
                for j, value in enumerate(row):
                    if int(value.split('.')[0]) == key:
                        final_suffix = "".join(values)
                        final_suffix = del_consecutive_duplicates(final_suffix)
                        result[i][j] = f"{key}." + final_suffix

    ## convertir a float
    final = []
    for mlist in result:
        final.append([float(valor) for valor in mlist])

    return final

# Function for building a dataframe from list of patterns and their supports
# input: a list of patterns
# output: a dataframe patterns - support

def list2dataframe(list_patt):
    patterns = []
    supp = []
    for i in list_patt:
        patterns.append([str(x) for x in i[0:len(i)-1]])
        supp.append(i[len(i)-1])
    df = pd.DataFrame({'patterns': patterns,'supp': supp})
    return df

# Function for recovering the vertex ID instead the vertex index
# input: pattern like [a, b, 234], a graph
# output: patterns with the ID like [id_a, id_b, 234]
# Note: The last value of the pattern is the absolute support (the times pattern appears in the dataset)

def recovering_id_nk(patterns, g):
    patt_id = []
    for i, sublist in enumerate(patterns):
        aux = []
        for j in range(0,(len(sublist))):
            #print(patterns[i][j])
            aux.append(g.vs[patterns[i][j]]['id'])
        patt_id.append(aux)
    return patt_id

# Function for extracting data for patterns mining analysis
# input: a dataframe of sequences from diferent traversal technique, the min, max supp and the step
# output: a dataframe with the number of pattenrs and the execution time for each traversal technique

def stat_supp_patt_by_traversal_old(p2s_a, p2s_bfs, p2s_dij, supp_min_max, supp_min_min, step):
    lst_supp_patt = []
    for i in range(supp_min_min, supp_min_max, step):
        lst_aux = []
        if i >= supp_min_max:
            break
        start = timer()
        patterns_a = spm(p2s_a, i)
        end = timer()
        time_a_patt = (end - start)
        start = timer()
        patterns_bfs = spm(p2s_bfs, i)
        end = timer()
        time_bfs_patt = (end - start)
        start = timer()
        patterns_dij = spm(p2s_dij, i)
        end = timer()
        time_dij_patt = (end - start)
        lst_aux.append(i)
        lst_aux.append(len(patterns_a))
        lst_aux.append(len(patterns_bfs))
        lst_aux.append(len(patterns_dij))
        lst_aux.append(time_a_patt)
        lst_aux.append(time_bfs_patt)
        lst_aux.append(time_dij_patt)
        lst_supp_patt.append(lst_aux)
    return lst_supp_patt

def stat_supp_patt_by_traversal(lst_p2s, supp_min_max, supp_min_min, step, big_graph):
    lst_supp_patt = []
    for i in range(supp_min_min, supp_min_max, step):
        for m in range(len(lst_p2s)):
            lst_aux = []
            if i >= supp_min_max:
                break
            start = timer()
            patterns = spm(lst_p2s[m][1], i, big_graph)
            end = timer()
            time_patt = (end - start)
            lst_aux.append(lst_p2s[m][0])
            lst_aux.append(i)
            lst_aux.append(len(patterns))
            lst_aux.append(time_patt)
            lst_supp_patt.append(lst_aux)
    return lst_supp_patt

# Function for calculating metrics concernant the traversal and pattern mining extraction
# input: list of patterns, the name of the TSE, the execution time (Patt, TSE) 
# output: all paths starting at V1 and ending at V2

def freq_patt_by_traversal(lst_patterns, name):
    sizes = [len(sublist) for sublist in lst_patterns['patterns']]
    freq = Counter(sizes)
    lst_res_patt = []
    aux = []
    aux.append(name)
    aux.append(len(lst_patterns))
    #dic_freq = {}
    #dic_freq['name'] = 'bfs'
    #for key, value in freq.items():
    #    dic_freq[key] = value
    aux.append(freq)
    lst_res_patt.append(aux)
    return lst_res_patt


# Function for building sequences (for seq2pat) from each path list
# input: list of paths list and a graph
# output: a list of list of sequences, one for each traversal method

def recovering_id_set_paths(lst_paths_by_lib, big_graph):
    lst_res = []
    for i in range(len(lst_paths_by_lib)):
        aux = []
        p2s = paths2seq(recovering_id(lst_paths_by_lib[i][3],big_graph))
        aux.append(lst_paths_by_lib[i][0])
        aux.append(p2s)
        lst_res.append(aux)
    return lst_res

# Function for comparing extracted pattenrs and existing edges
# input: list of patterns and a graph
# output: size of patterns and edges similar to patterns

def verify_patt_graph(df_patt, big_graph):
    edge_df = pd.DataFrame({attr: big_graph.es[attr] for attr in big_graph.edge_attributes()})
    set_of_values = set(zip(edge_df['source'], edge_df['target']))
    #print(set_of_values)
    patterns_new = []   
    for i in range(len(df_patt)):
        flag = True
        for j in range(len(df_patt.patterns[i])-1):
            s = int(df_patt.patterns[i][j])
            t = int(df_patt.patterns[i][j+1])
            to_val = (s,t)
            #print(to_val)
            if (to_val in set_of_values):
                flag = False
        #print(flag)
        if flag == False:
            #print(df_patt.loc[i])
            patterns_new.append(df_patt.loc[i])
    new_df = pd.DataFrame(patterns_new)
    new_df.reset_index(drop=True, inplace=True)
    return new_df

# Function for extracting data for patterns mining analysis
# input: a dataframe of sequences from diferent traversal technique, a minimal support
# output: a dataframe with the number of pattenrs and the execution time for each traversal technique

def stat_supp_patt_by_traversal_min_supp(lst_p2s, supp_min, big_graph):
    lst_supp_patt = []
    for m in range(len(lst_p2s)):
        lst_aux = []
        patterns = spm(lst_p2s[m][1], supp_min, big_graph)
        lst_aux.append(lst_p2s[m][0])
        lst_aux.append(patterns)
        lst_supp_patt.append(lst_aux)
    return lst_supp_patt

# Function for extracting frequent sequences using PrefixSpan
# input: sequnces of locations, a minimal support (minsup)
# output: patterns appearing at least minsup times

def extract_patt_PrefixSpan(lst_paths_by_lib, big_graph, min_supp):
    lst_res = []
    for i in range(len(lst_paths_by_lib)):
        patts = []
        p2s = recovering_id(lst_paths_by_lib[i][3],big_graph)
        ps = PrefixSpan(p2s)
        results = ps.frequent(min_supp)

        patt_aux = []
        for j in range(len(results)):
            aux = []
            patt = results[j][1]
            supp = results[j][0]
            aux = patt 
            aux.append(supp)
            patt_aux.append(aux)

        patts.append(lst_paths_by_lib[i][0])
        patts.append(patt_aux) 
        lst_res.append(patts)

    return lst_res

# Function for extracting frequent sequences using Spam
# input: sequnces of locations, a minimal support (minsup)
# output: patterns appearing at least minsup times
# Note: TO BE TESTED (we can see the sequences support)

def extract_patt_Spam(lst_paths_by_lib, min_supp):
    list_patt_2_spamAlgo = []
    for list in lst_paths_by_lib[0][3]:
        list_of_lists = [[item] for item in list]
        list_patt_2_spamAlgo.append(list_of_lists)
    algo = SpamAlgo(min_supp)
    algo.spam(list_patt_2_spamAlgo)
    patts = algo.frequent_items
    return patts