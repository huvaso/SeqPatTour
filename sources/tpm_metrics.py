# Recovering libraries

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Basic Libraries
import numpy as np
import pandas as pd

# Libraries for Calculating spatial metrics 
from geopy.distance import geodesic as GD

#Libraries for calculating the entropy and the avergage
from collections import Counter
from scipy import stats
import statistics
import geopandas
from shapely.geometry import Point
import math
import collections
import itertools

import csv


# Function for calculating the sum of outdegrees for each sequential pattern
# input: a dataframe with patterns/supports and the corresponding graph
# output: a list with the outdegrees sum for each sequential pattern 

def out_degree_seq(df, gs):
    out_list = []
    aux = 0
    for i in range(0,len(df)):
        aux = 0
        for j in range(len(df.patterns[i])):
            key = int(df.patterns[i][j])
            out1 = gs.vs.find(id=key).outdegree()
            #print(key,out1)
            aux = aux + out1
        #print (aux)
        out_list.append(aux)
    return out_list

# Function for calculatin the mean of the satisfaction for each sequential pattern
# input: a dataframe with patterns/supports and the corresponding graph
# output: a list with the mean of satisfaction for each sequential pattern 

def satisfaction_seq_old(df, gs):
    sat_list = []
    for i in range(0,len(df)):
        aux = 0
        k = 0
        for j in range(len(df.patterns[i])):
            key = int(df.patterns[i][j])
            out1 = gs.vs.find(id=key)["rating"]
            #print(key,out1)
            k = k + 1
            aux = aux + out1
        #print (aux, k)
        sat_list.append(aux/k)
    return sat_list

# Function for calculatin the mean of the satisfaction by the number of comments for each sequential pattern
# input: a dataframe with patterns/supports and the corresponding graph
# output: a list with the mean of satisfaction for each sequential pattern 

def satisfaction_seq(df, gs):
    sat_list = []
    for i in range(0,len(df)):
        sum_rating_avis = 0
        sum_avis = 0
        for j in range(len(df.patterns[i])):
            key = int(df.patterns[i][j])
            rating = gs.vs.find(id=key)["rating"]
            avis = gs.vs.find(id=key)["nbAvis"]
            ## sum (rating*avis) / sum(avis)
            sum_rating_avis += (rating * avis) 
            sum_avis += avis
        sat_list.append(sum_rating_avis/sum_avis)
    return sat_list

# Function for getting the lat and lon of a set or vertices
# input: a dataframe with patterns/supports and the corresponding graph
# output: a list with the lat and lon of vertices 

def getLatLongFromGraph(df,gs):
    all_list_positions = []
    for index,row in df.iterrows():
        tmp_list = [int(i) for i in row['patterns']] #convertir a valores enteros
        list_positions = []
        for value in tmp_list:
            lon_value = gs.vs.find(id=value)["longitude"]
            lat_value = gs.vs.find(id=value)["latitude"]
            pos = (lon_value,lat_value)    
            list_positions.append(pos)    
        all_list_positions.append(list_positions) 
    return all_list_positions

# Function for getting a list of lat log for calculating the route distance in OSM
# input: a dataframe with patterns/supports and the corresponding graph
# output: a list with the lat and lon of vertices 
# Note: This will be used by the osm.ipynb, which take some minutes to process data

def conver_list_ints_osm(lp):
    list_geo_points = []
    for i in range(len(lp)):
        aux1 = []
        for j in range(len(lp[i])):
            aux2 = []
            for k in range(len(lp[i][j])):
                aux2.append(lp[i][j][k])
            aux1 = aux1 + aux2
        list_geo_points.append(aux1)
    return list_geo_points

# Function for saving a list of strings containg the shortest distances provided by OSM
# input: a file
# output: a list with the shortest distances (strings) 
# Note: This save a file used by osm.ipynb, which take some minutes to process data

def save_list_osm(file_name, lp):
    with open(file_name, 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerows(lp)

# Function for reading a file containg the shortest distances provided by OSM
# input: a file
# output: a list with the shortest distances (strings) 
# Note: This read a file provided by osm.ipynb, which take some minutes to process data

def read_list_dist_osm(file_name):
    lp = []
    with open(file_name) as f:
        reader = csv.reader(f)
        for row in reader:
            lp.append(row)
            break
    aux = lp[0]
    return aux[0:len(aux)-1]

# Function for calculating the entropy
# input: a list of lists containing the different types of locations (A, H, R)
# output: a list of entropy values for each list

def entropy_loc(lista):
    list_ent = []
    for i in lista:
        ent = stats.entropy(list(Counter(i).values()), base=2) 
        list_ent.append(ent)
    return list_ent

# Function for getting the different types of locations (restaurant R, hotel H, attraction A)
# input: a dataframe with patterns/supports and the corresponding graph
# output: a list of lists with the different types of locations 

def type_location_seq(df,gs):
    sat_list = []
    for i in range(0,len(df)):
        typeL = []
        for j in range(len(df.patterns[i])):
            key = int(df.patterns[i][j])
            out = gs.vs.find(id=key)["typeR"]
            #print(key,out1)
            typeL.append(out)
        #print (aux, k)
        sat_list.append(typeL)
    return sat_list

# Function for calculating the sum of bird flight distances for each sequential pattern
# input: a dataframe with patterns/supports and the corresponding graph
# output: a dataframe with the sum of distances for each sequential pattern 

def genDf(list_positions,df):
    all_distas = []
    for lpos in list_positions:
        dista = []
        for position in range(0,len(lpos)-1):
            dist = GD(lpos[position],lpos[position+1]).km
            dista.append(dist)
        all_distas.append(dista)
    
    df['distances'] = all_distas
    df['CDP'] = [sum(i) for i in df['distances']]
  
    return df

# Function for calculating the agerage in/out degree for each vertex in all sequences
# input: a dataframe with patterns/supports and the corresponding graph
# output: a list with the in/out degree average for each sequential pattern 

def ratio_in_out (df_patt, gs):
    edge_df = pd.DataFrame({attr: gs.es[attr] for attr in gs.edge_attributes()})
    ratio_in_out = []
    for i in range(0,len(df_patt)):
        aux = df_patt.iloc[i][0]
        for j in range(0,len(aux)-1):
            node_ratio = []
            aux1 = edge_df.loc[edge_df["source"] == int(aux[j])]
            var1 = var2 = 0
            for k in range(0,len(aux1)):
                if aux1.target.iloc[k] == int(aux[j+1]):
                    var1 = var1 + aux1.NbPerMaxDurationDays_inf.iloc[k]
                else:
                    var2 = var2 + aux1.NbPerMaxDurationDays_inf.iloc[k]
            res = var1/var2
            #print(res)
            node_ratio.append(res)
        #media = statistics.mean(node_ratio)
        prod = np.prod(node_ratio)
        ratio_in_out.append(prod)
    return ratio_in_out


# Function for recovering the communes' name for each vertex in all sequences
# input: a dataframe with patterns/supports and the corresponding graph
# output: a list with the name of communes traversed for each sequential pattern
#         and a vector with values fine/coarse if patterns occurs in the Lille commune or not
# NOTICE: the shape (or GEOJSON) can be added as a parameter 

def check(list):
    return all(i == list[0] for i in list)

def comm_location_seq(df_patt,gs):
    #communes_mel = geopandas.read_file('mel_communes/mel_communes.shp')
    communes_mel = geopandas.read_file('mel_communes/mel_communes_coarse.geojson')
    sat_list = []
    granu = []
    for i in range(len(df_patt)):
        typeL = []
        for j in range(len(df_patt.patterns[i])):
            key = int(df_patt.patterns[i][j])
            lox = gs.vs.find(id=int(key))['longitude']
            loy = gs.vs.find(id=int(key))['latitude']
            points = Point(lox, loy)
            aux = communes_mel.contains(points)
            pointsInArea = [ind for ind, val in enumerate(aux) if val == True]
            #for index_i in pointsInArea:
            #    typeL.append(communes_mel.at[index_i,'territoire'])
            typeL.append(communes_mel.at[pointsInArea[0],'territoire'])
        #sat_list.append(list(set(typeL))) ### se convierte a un set para eliminar duplicados
        sat_list.append(typeL)
    return sat_list

# Function for calculating the spatial grannularity (coarse or fine) for each sequence
# input: a dataframe with patterns/supports and the corresponding graph
# output: a list with the spatial grannularity (coarse / fine)

def spatial_gran_patt(df_patt,gs):
    #communes_mel = geopandas.read_file('mel_communes/mel_communes.shp')
    communes_mel = geopandas.read_file('mel_communes/mel_communes_coarse.geojson')
    sat_list = []
    granu = []
    for i in range(len(df_patt)):
        typeL = []
        for j in range(len(df_patt.patterns[i])):
            key = int(df_patt.patterns[i][j])
            lox = gs.vs.find(id=int(key))['longitude']
            loy = gs.vs.find(id=int(key))['latitude']
            points = Point(lox, loy)
            aux = communes_mel.contains(points)
            pointsInArea = [ind for ind, val in enumerate(aux) if val == True]
            typeL.append(communes_mel.at[pointsInArea[0],'territoire'])
        sat_list.append(typeL)
        if (check(typeL) and typeL[0] == "LILLE-LOMME-HELLEMMES"): # ERROR, REVISAR
            granu.append("fine")
        else:
            granu.append("coarse")
    return granu

# Function for recovering the nationality of tourist by pattern
# input: a dataframe with patterns/supports and the corresponding graph
# output: a list with the nationality of tourists for each sequential pattern 

def nac_tourist_seq(df,gs): # I added a new parametter g (complete graph)
    edge_df = pd.DataFrame({attr: gs.es[attr] for attr in gs.edge_attributes()}) # Before was gs instead g
    sat_list = []
    for i in range(0,len(df)):
        typeL = []
        for j in range(len(df.patterns[i])-1):
            #print(df.patterns[i][j])
            source = int(df.patterns[i][j])
            target = int(df.patterns[i][j+1])
            aux = edge_df[(edge_df['source'] == source) & (edge_df['target'] == target)]
            out = aux.groupby(['source','target'], as_index=False)['country'].apply(list)
            #print(key,out1)
            typeL.append(list(dict.fromkeys(out['country'][0])))
        #print (aux, k)
        sat_list.append(typeL)
    return sat_list

# Function for calculating the Gini coeficient of the tourist nationalyty by pattern
# input: a dataframe with patterns/supports and the corresponding graph
# output: a list with the Gini of tourists mationalities for each sequential pattern 
# Note: in this metric, the Gini is calculated taking into account all countries in a path (local)

def gini_nac_old(list_of_values):
    sorted_list = sorted(list_of_values)
    height, area = 0, 0
    for value in sorted_list:
        height += value
        area += height - value / 2.
    fair_area = height * len(list_of_values) / 2.
    return (fair_area - area) / fair_area

def gini_nac(x):
    mad = np.abs(np.subtract.outer(x, x)).mean()
    rmad = mad/np.mean(x)
    g = 0.5 * rmad
    return g

def nationality_vertex_gini(df,gs): 
    edge_df = pd.DataFrame({attr: gs.es[attr] for attr in gs.edge_attributes()})
    sat_list = []
    for i in range(0,len(df)):
        typeL = []
        for j in range(len(df.patterns[i])-1):
            source = int(df.patterns[i][j])
            target = int(df.patterns[i][j+1])
            aux = edge_df[(edge_df['source'] == source) & (edge_df['target'] == target)]
            out = aux.groupby(['source','target'], as_index=False)['country'].apply(list)
            counter = collections.Counter(out['country'][0])
            gini = gini_nac(list(counter.values()))
            typeL.append(gini)
        sat_list.append(typeL)
    return sat_list

def nationality_seq_gini(df,gs): 
    edge_df = pd.DataFrame({attr: gs.es[attr] for attr in gs.edge_attributes()}) 
    gini_patt = []
    for i in range(0,len(df)):
        countries = []
        sum = 0
        typeL = []
        for j in range(len(df.patterns[i])-1):
            source = int(df.patterns[i][j])
            target = int(df.patterns[i][j+1])
            aux = edge_df[(edge_df['source'] == source) & (edge_df['target'] == target)]
            out = aux.groupby(['source','target'], as_index=False)['country'].apply(list)
            counter = collections.Counter(out['country'][0])
            countries.append(counter)
            gini = gini_nac(list(counter.values()))
            typeL.append(gini)
            sum = sum + gini
        j = Counter()
        for i in countries:
            j = j + i
        #print(typeL,len(j))
        gpatt = sum/len(j)
        gini_patt.append(gpatt)
    return gini_patt

# Function for calculating the Gini coeficient of the tourist nationalyty by pattern
# input: a dataframe with patterns/supports and the corresponding graph
# output: a list with the Gini of tourists mationalities for each sequential pattern 
# Note: in this metric, the Gini is calculated taking into account all countries of graph (global)

def nationality_seq_gini_V2(df, gs): 
    edge_df = pd.DataFrame({attr: gs.es[attr] for attr in gs.edge_attributes()}) 
    gini_patt = []
    total_countries = []
    for i in range(0,len(df)):
        sum = 0
        typeL = []
        countries = []
        for j in range(len(df.patterns[i])-1):
            source = int(df.patterns[i][j])
            target = int(df.patterns[i][j+1])
            aux = edge_df[(edge_df['source'] == source) & (edge_df['target'] == target)]
            out = aux.groupby(['source','target'], as_index=False)['country'].apply(list)
            counter = collections.Counter(out['country'][0])
            #print(counter)
            countries.append(counter)
        total_countries.append(countries)
    merged = list(itertools.chain.from_iterable(total_countries))
    lista = []
    for value in merged:
        r = value.elements()
        for i in r:
            #print(i)
            lista.append(i)
    set_countries = set(lista)
    m = collections.Counter(set_countries)
    #print(m)
    for i in range(0,len(df)):
        countries = []
        sum = 0
        typeL = []
        for j in range(len(df.patterns[i])-1):
            source = int(df.patterns[i][j])
            target = int(df.patterns[i][j+1])
            aux = edge_df[(edge_df['source'] == source) & (edge_df['target'] == target)]
            out = aux.groupby(['source','target'], as_index=False)['country'].apply(list)
            counter = collections.Counter(out['country'][0])
            #print(counter+m)
            suma = counter+m
            for ele in suma:
                if suma[ele] == 1:
                   suma[ele] = 0
            countries.append(suma)
            #print(suma.values())
            gini = gini_nac(list(suma.values()))
            typeL.append(gini)
            sum = sum + gini
        j = Counter()
        for i in countries:
            j = j + i
        #print(typeL,len(countries))
        gpatt = sum/len(countries)
        #gpatt = sum
        gini_patt.append(gpatt)
    return gini_patt

# Function for grouping metrics in two spatial grannularities
# input: a dataframe with metrics
# output: a dataframe grouping metrics into two spatial levels 

def grouping_metrics(df_metrics):
    df1 = df_metrics.groupby('granularity')[['supp','CDP_all','PSI','LDI','PPI', 'DDI','NDI_V2']].mean() ## For general approach
    #df1 = df_metrics.groupby('granularity')[['supp','CDP_all','DDI','PPI','NDI_V2']].mean() ## For saisonal approach
    df2 = df_metrics.groupby('granularity')['patterns'].count().tolist()
    df1['count'] = df2
    nb_coun = [len(grouping_country_spatial(df_metrics)[0]),len(grouping_country_spatial(df_metrics)[1])]
    nb_comm = [len(grouping_comm_spatial(df_metrics)[0]),len(grouping_comm_spatial(df_metrics)[1])]
    df1['Nb_Countries'] = nb_coun
    df1['Nb_Comm'] = nb_comm
    df1['Countries'] = grouping_country_spatial(df_metrics)
    df1['Comm'] = grouping_comm_spatial(df_metrics)
    return df1

# Function for getting countries for spatial grannularity
# input: a dataframe with metrics
# output: a list of countries by spatial grannularity 

def grouping_country_spatial(df_metrics):
    grp_country = []
    aux = df_metrics.groupby('granularity')[['Nationality']].sum()
    lst_coa = aux.loc['coarse'].to_list()
    lst_fin = aux.loc['fine'].to_list()
    set_coa = set(element for sublist in lst_coa[0] for element in sublist)
    set_fin = set(element for sublist in lst_fin[0] for element in sublist)
    grp_country.append(list(set_coa))
    grp_country.append(list(set_fin))
    return grp_country

# Function for getting communes for spatial grannularity
# input: a dataframe with metrics
# output: a list of communes by spatial grannularity 

def grouping_comm_spatial(df_metrics):
    grp_comm = []
    aux = df_metrics.groupby('granularity')[['communes']].sum()
    lst_coa = aux.loc['coarse'].to_list()
    lst_fin = aux.loc['fine'].to_list()
    set_coa = set(lst_coa[0])
    set_fin = set(lst_fin[0])
    grp_comm.append(list(set_coa))
    grp_comm.append(list(set_fin))
    return grp_comm

# Function to calculate distance between nodes
def get_distance_between_nodes(df_metrics_summer, node1, node2):
    distance = df_metrics_summer.apply(lambda row: row['CDP_all'] if node1 in row['patterns'] and node2 in row['patterns'] else None, axis=1).dropna()
    if not distance.empty:
        return distance.values[0]
    else:
        return 0 # If pair of IDs not found

def create_dist_matrix_huff(df_metrics, graph):
    # Create a dictionary to map node IDs to indices in the distance matrix
    trajectories = df_metrics['patterns']
    node_to_index = {}
    index = 0
    for trajectory in trajectories:
        for node in trajectory:
            #print(node)
            if node not in node_to_index:
                node_to_index[node] = index
                index += 1
    
    #print(node_to_index)

    weighs = []
    for key in node_to_index.keys():
        #print(key)
        weigh = graph.vs.find(id=int(key))["rating"]
        weighs.append(weigh)

    #print(weighs)

    # Initialize an empty distance matrix
    num_nodes = len(node_to_index)
    #print(num_nodes)
    distance_matrix = np.zeros((num_nodes, num_nodes))
    
    # Iterate through each trajectory to accumulate distances (long trajectories)
    for trajectory in trajectories:
        # Calculate distance between consecutive nodes
        for i in range(len(trajectory) - 1):
            node1 = trajectory[i]
            node2 = trajectory[i + 1]
            distance = get_distance_between_nodes(df_metrics, node1, node2)
            index1 = node_to_index[node1]
            index2 = node_to_index[node2]
            distance_matrix[index1, index2] += distance

    # Fill remaining cells with appropriate distances
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and distance_matrix[i, j] == 0:
                node1 = list(node_to_index.keys())[i]
                node2 = list(node_to_index.keys())[j]
                distance_matrix[i, j] = get_distance_between_nodes(df_metrics, node1, node2)
                
    return distance_matrix, weighs, node_to_index

def huff_attractiveness(distance_matrix, weights):
    num_locations = len(weights)
    huff_matrix = np.zeros((num_locations, num_locations))
    #beta = 1.5
    
    for i in range(num_locations):
        for j in range(num_locations):
            if i != j:
                huff_matrix[i, j] = weights[i] / (np.sum(weights) * distance_matrix[i, j])

    return huff_matrix

def get_keys_from_value(dictionary, search_value):
    return [value for key, value in dictionary.items() if key == search_value]

def find_huff_value(huff_matrix, values_dic, df_metrics):
    trajectories = df_metrics['patterns']
    huff_patterns = []
    for trajectory in trajectories:
        # Calculate Huff between consecutive nodes
        distance = 0
        size_patt = len(trajectory) - 1
        for i in range(len(trajectory) - 1):
            node1 = get_keys_from_value(values_dic,trajectory[i])
            node2 = get_keys_from_value(values_dic,trajectory[i+1])
            #print(node1, '--', node2)
            #print(trajectory[i], '--', trajectory[i+1])
            distance = distance + huff_matrix[node1,node2]
            #print(distance)
        huff_patterns.append(list(distance)[0]/(size_patt)*100) # In percentaje
    return huff_patterns


#### Get metrics main

def getMetrics(df,gs):
    ##list_out = out_degree_seq(df,gs)
    tmp_distances = getLatLongFromGraph(df, gs)
    metrics_df = genDf(tmp_distances, df)
    ##metrics_df["satisfaction_old"] = satisfaction_seq_old(df, gs)
    #aux_walk = read_list_dist_osm('file_geo_dist_walk.csv')
    aux_all = read_list_dist_osm('file_geo_dist_all.csv')
    #aux_bike = read_list_dist_osm('file_geo_dist_bike.csv')
    #metrics_df["CDP_walk"] = list(np.float_(aux_walk)/1000)
    metrics_df["CDP_all"] = list(np.float_(aux_all)/1000)
    #metrics_df["CDP_bike"] = list(np.float_(aux_bike))
    metrics_df["PSI"] = satisfaction_seq(df, gs) ## For saisons this metric not exists
    ##metrics_df["out_degree"] = out_degree_seq(df,gs)
    metrics_df["LDI"] = entropy_loc(type_location_seq(df, gs))
    metrics_df["PPI"] = ratio_in_out(df, gs)
    ##metrics_df["coarce_geo"],metrics_df["granu"] = comm_location_seq(df,gs) # ERROR AQUI
    metrics_df["communes"] = comm_location_seq(df, gs)
    metrics_df["granularity"] = spatial_gran_patt(df, gs)
    lambda_parameter = 1
    metrics_df["DDI"] = np.exp(lambda_parameter*metrics_df['LDI']) / metrics_df["CDP_all"] # We use CDP_all instead a birdh fly
    ##metrics_df["Gini"] = nationality_vertex_gini(df,gs) 
    #metrics_df["NDI"] = nationality_seq_gini(df,gs)
    metrics_df["NDI_V2"] = nationality_seq_gini_V2(df, gs)
    metrics_df["typeR"] = type_location_seq(df, gs)
    metrics_df["Nationality"] = nac_tourist_seq(df, gs)
    metrics_df = metrics_df.drop(['distances'], axis=1)
    
    return metrics_df