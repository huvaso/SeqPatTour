# Recovering libraries

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Basic Libraries
import numpy as np
import pandas as pd

# Libraries for Patricia Tree
import pprint
#from coppredict import preprocessing  as pr
from coppredict.patricia import Patricia
#from coppredict import util


# Function for printing a Patricia trie 
# input: a PT and a space
# output: a representation of PT 

def print_PT(t,s):
    if not isinstance(t,dict) and not isinstance(t,list):
        print("\t"*s+str(t))
    else:
        for key in t:
            print("\t"*s+str(key))
            if not isinstance(t,list):
                print_PT(t[key],s+1)


# Function for building a Patricia trie from list of patterns and their supports
# input: a list of patterns
# output: a Patricia trie

def PT_building(patt_id):
    X_trie = Patricia()
    patt_id.sort(key=len, reverse=True)
    for i in patt_id:
        patt = [str(x) for x in i[0:len(i)-1]]
        supp = i[-1]
        X_trie.add_pattern(patt,supp)
    return X_trie

# Function for merging three different set of patterns (season) in only on PT
# input: three dataset
# output: a dataframe with three new suffix

def merge_lists_with_suffix_three(gr_1, gr_2, gr_3):
    result = []
    suffix_1 = 1
    suffix_2 = 2
    suffix_3 = 3

    def add_suffix(row, suffix):
        return [f"{value}.{suffix}" for value in row]

    # Agregar sufijo .1 a las listas en gr_1
    for row in gr_1:
        result.append(add_suffix(row, suffix_1))

    # Agregar sufijo .2 a las listas en gr_2
    for row in gr_2:
        result.append(add_suffix(row, suffix_2))
        
    # Agregar sufijo .3 a las listas en gr_3
    for row in gr_3:
        result.append(add_suffix(row, suffix_3))

    # Reemplazar sufijos si hay duplicados entre gr_1 y gr_2  y gr_3
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