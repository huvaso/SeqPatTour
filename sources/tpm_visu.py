# Recovering libraries

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Basic Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Libraries for visualizaing patterns using Plotly
import plotly.graph_objects as go
from plotly import express as px
import plotly.io as pio
import plotly
import requests
import geopandas
import math
import shapely.geometry
from shapely.affinity import rotate as R, translate as T
from shapely.geometry import Point, LineString
from functools import reduce

# Libraries for rescaling data
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

# Librarie for building dataframes from patterns  
from tpm_spm import *

# Libraries for visualizaing patterns using Plotly
from collections import Counter

# Libraries for visualizaing patterns using Kepler
from keplergl import KeplerGl

import ast

# Function for drawing an arrow between two locations
# input: latitud, longitud, marker, color, place
# output: a triangle 
# https://stackoverflow.com/questions/70275332/plotly-how-to-plot-arrow-line-directed-graph

def direction(lat1, lon1, marker = None, color="red", place="center"):
    # default marker is a left pointing arrow, centred as 0,0
    # it may be necessary to adjust sizing
    #marker = None
    if marker is None:
        m = R(shapely.geometry.Polygon([(0, 0), (0, 1), (0.5, 0.5)]), 0)
        #m = shapely.affinity.scale(m, xfact=0.0035, yfact=0.0035)  # size appropriately
        m = shapely.affinity.scale(m, xfact=0.0010, yfact=0.0010)  # size appropriately
        m = T(m, -m.centroid.x, -m.centroid.y)
    else:
        m = marker

    def place_marker(m, x1, x0, y1, y0, place="center"):
        theta = math.atan2(x1 - x0, y1 - y0)
        if place == "center":
            p = dict(
                xoff=(y1 - y0) / 2 + y0,
                yoff=(x1 - x0) / 2 + x0,
            )
        elif place == "end":
            p = dict(xoff=y1, yoff=x1)
        elif place == "beyond":
            H = .05
            p = dict(xoff=y1 + math.cos(theta)*H, yoff=x1 + math.sin(theta)*H)
        
        return T(R(m, theta, use_radians=True), **p)

    return  {
        # list of geometries rotated based on direction and centred on line between pairs of points
        "source": shapely.geometry.MultiPolygon(
            [
                place_marker(m, x1, x0, y1, y0)
                for x0, x1, y0, y1 in zip(lat1, lat1[1:], lon1, lon1[1:])
            ]
        ).__geo_interface__,
        "type": "fill",
        "color": color,
        "below":"traces"
    }

# Function for visualizing a graph in plotly 
# input: a graph
# output: a plot
# https://plotly.com/python/v3/igraph-networkx-comparison/

def visualization(gs):
    labels=list(gs.vs['id'])
    N=len(labels)
    E=[e.tuple for e in gs.es]# list of edges
    layt=gs.layout('kk') #kamada-kawai layout
    Xn=[layt[k][0] for k in range(N)]
    Yn=[layt[k][1] for k in range(N)]
    Xe=[]
    Ye=[]
    for e in E:
        Xe+=[layt[e[0]][0],layt[e[1]][0], None]
        Ye+=[layt[e[0]][1],layt[e[1]][1], None]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=Xe,
                y=Ye,
                mode='lines',
                line= dict(color='rgb(210,210,210)', width=1),
                hoverinfo='none'
                )
    )

    fig.add_trace(
        go.Scatter(x=Xn,
                y=Yn,
                mode='markers',
                name='ntw',
                marker=dict(symbol='circle-dot',
                                            size=5,
                                            color='#6959CD',
                                            line=dict(color='rgb(50,50,50)', width=0.5)
                                            ),
                text=labels,
                hoverinfo='text'
                )
    )

    fig.update_layout(
        showlegend=False,
        autosize=False,
        width=1700,
        height=900 
    )
    fig.show()

# Function for visualizing an hystogram of indices in a random walk
# input: a list of walks
# output: a plot

def plot_hysto(rw_seq):
    data = count_items_seq(rw_seq)
    keys, values = dic2lists(data)
    long_df = px.data.medals_long()
    fig_hysto = px.bar(long_df, x=keys, y=values,
                labels={
                        "x": "Vertices IDs",
                        "y": "Frequency"
                    }
                ).update_layout(template='plotly_white')
    fig_hysto.show()

# Function for visualizing a graph in plotly express and Matbox
# input: a dataframe
# output: a plot (very simple visaulization)

def visu(data_to_viz):
    mapbox_token = requests.get('https://api.mapbox.com/?access_token=pk.eyJ1IjoiaHV2YXNvIiwiYSI6ImNrOTZlb3c5NDBiY24zbHFmZXJnaDFzbGkifQ.TOPkjShqlOiCdMckwNtwuw').text
    px.set_mapbox_access_token(mapbox_token)
    fig = px.scatter_geo(data_to_viz, data_to_viz.latitude, data_to_viz.longitude)
    fig.show()

# Function for visualizing a graph in plotly GO
# input: a set of dataframes (one for each trace)
# output: a figure (to be improved)
# Add an arrow: https://stackoverflow.com/questions/70275332/plotly-how-to-plot-arrow-line-directed-graph

def visualizing_patterns(list_dfs):
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
    fig = go.Figure()
    for i in range(0,len(list_dfs)):
        fig.add_trace(
            go.Scattermapbox(
                mode = "markers+lines",
                lat = list_dfs[i].latitude.to_list(),
                lon = list_dfs[i].longitude.to_list(),
                hovertext = list_dfs[i].nom,
                marker = dict(
                                color = colors[i], 
                                size = 12
                            ),
                line_width=3,
                textposition='bottom right',
                text = list_dfs[i].nom
                #sizemode = data_to_viz.rating # to add new semantic to a graph
            )
    )
    fig.update_layout(
                        margin ={'l':0,'t':0,'b':0,'r':0},
                        mapbox = {
                            'center': {'lon': 3, 'lat': 50.6},
                            #'style': "carto-positron",
                            'style': "carto-darkmatter",
                            'zoom': 10,
                            'layers':[
                                # Note: fig.data is an dictionary created by python to store all data to be plotted
                                # For drawing only a network
                                #direction(fig.data[0].lat, fig.data[0].lon, marker=None, color="red", place="center") 
                                # For drawing multiple networks with one color
                                #direction(t[0].lat, t[0].lon, marker=None, color="red", place="beyond")
                                #for t in zip(fig.data) 
                                # For drawing multiple networks with multiple colors
                                direction(t.lat, t.lon, marker=None, color=c, place="beyond")
                                for t, c in zip(fig.data, colors)
                            ],
                        },
                        # Changing these values for resizong the arrow
                        width=1400,
                        height=600
                    )
    fig.show()

def visualizing_patterns_V2(list_dfs):
    mel_coa = geopandas.read_file('mel_communes/mel_communes_coarse.geojson')
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
    fig = go.Figure()
    for i in range(0,len(list_dfs)):
        fig.add_trace(
            go.Scattermapbox(
                mode = "markers+lines",
                lat = list_dfs[i].latitude.to_list(),
                lon = list_dfs[i].longitude.to_list(),
                hovertext = list_dfs[i].nom,
                marker = dict(
                                color = colors[i], 
                                size = 12
                            ),
                line_width=3,
                textposition='bottom right',
                text = list_dfs[i].nom
                #sizemode = data_to_viz.rating # to add new semantic to a graph
            )
    )
        
    fig.update_layout(
                    margin ={'l':0,'t':0,'b':0,'r':0},
                    mapbox = {
                        'layers': [{
                            'source': mel_coa,
                            'type':'fill', 
                            'below':'traces','color': 'red', 'opacity' : 0.5, "source": 'gj'}],
                    }
                )
    
    fig.update_layout(
                        margin ={'l':0,'t':0,'b':0,'r':0},
                        mapbox = {
                            'center': {'lon': 3, 'lat': 50.6},
                            'style': "carto-positron",
                            #'style': "carto-positron",
                            'zoom': 10,
                            'layers':[
                                # Note: fig.data is an dictionary created by python to store all data to be plotted
                                # For drawing only a network
                                #direction(fig.data[0].lat, fig.data[0].lon, marker=None, color="red", place="center") 
                                # For drawing multiple networks with one color
                                #direction(t[0].lat, t[0].lon, marker=None, color="red", place="beyond")
                                #for t in zip(fig.data) 
                                # For drawing multiple networks with multiple colors
                                direction(t.lat, t.lon, marker=None, color=c, place="beyond")
                                for t, c in zip(fig.data, colors)
                            ],
                        },
                        # Changing these values for resizing the arrow
                        width=1200,
                        height=500
                    )
    fig.show()

# Function for visualizing a graph in plotly GO
# input: a set of dataframes (one for each trace)
# output: a figure (to be improved)
# Add an arrow: https://stackoverflow.com/questions/70275332/plotly-how-to-plot-arrow-line-directed-graph

def visualizing_points(list_dfs):
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
    fig = go.Figure()
    for i in range(0,len(list_dfs)):
        fig.add_trace(
            go.Scattermapbox(
                mode = "markers",
                lat = list_dfs[i].latitude.to_list(),
                lon = list_dfs[i].longitude.to_list(),
                hovertext = list_dfs[i].nom,
                marker = dict(
                                color = colors[i], 
                                size = 12
                            ),
                line_width=3
                #sizemode = data_to_viz.rating # to add new semantic to a graph
            )
    )
    fig.update_layout(
                        margin ={'l':0,'t':0,'b':0,'r':0},
                        mapbox = {
                            'center': {'lon': 3, 'lat': 50.6},
                            'style': "carto-positron",
                            'zoom': 10,
                        },
                        # Changing these values for resizong the arrow
                        width=1200,
                        height=500  
                    )
    fig.show()

# Function for visualizing a graph in plotly GO
# input: a graph in iGraph format
# output: a figure

def getLatLonFromGraph(gs):
    nodes_df = pd.DataFrame({attr: gs.vs[attr] for attr in gs.vertex_attributes()})
    edges_df = pd.DataFrame({attr: gs.es[attr] for attr in gs.edge_attributes()})
    
    df_tmp = edges_df[['source','target']].merge(nodes_df[['id','latitude','longitude']], how='left', left_on='source', right_on='id')
    df_tmp.columns = ['source','target','id','source_lat','source_lon']
    df_tmp = df_tmp.merge (nodes_df[['id','latitude','longitude']], how='left', left_on='target', right_on='id')
    df_tmp = df_tmp.drop(columns=['id_x','id_y'])
    df_tmp.columns = ['source','target','source_lat','source_lon','target_lat','target_long']
    df_tmp = df_tmp.drop_duplicates().reset_index(drop=True)
    return df_tmp

def DrawGraph_v1(gs):
    df_LatLong = getLatLonFromGraph(gs)
    list_srtg = []
    for i in range(0,len(df_LatLong)):
        list_srtg.append([df_LatLong.loc[i,'source_lat'],df_LatLong.loc[i,'source_lon'],df_LatLong.loc[i,'source']])
        list_srtg.append([df_LatLong.loc[i,'target_lat'],df_LatLong.loc[i,'target_long'],df_LatLong.loc[i,'target']])
        
    fig = go.Figure()

    colors = []
    for ind in range(0,len(list_srtg),2):
        source = list_srtg[ind]
        target = list_srtg[ind+1]
    
        fig.add_trace(
            go.Scattermapbox(
                mode = 'markers+lines',
                lat = [source[0],target[0]],
                lon = [source[1],target[1]],
                hovertext = [str(source[2])+'-'+str(target[2])],
                marker = dict(
                        color = 'green', #colors_[ind], 
                        size = 10
                ),
                line_width=2,
                line_color = '#bcbd22' #'blue'
            )
        )
     
        
        fig.update_layout(
                        margin ={'l':0,'t':0,'b':0,'r':0},
                        mapbox = {
                            'center': {'lon': 3, 'lat': 50.6},
                            'style': "carto-positron",
                            'zoom': 9,
                            'layers':[
                                direction(t.lat, t.lon, marker=None, color='blue', place="beyond")
                                for t, c in zip(fig.data, colors) 
                             ],
                        },
                        width=1200,
                        height=500,
                        showlegend=False
          )


    fig.show()

# Function for visualizing a graph in plotly GO
# input: a graph in iGraph format
# output: a figure

def getLatLongFromGraph_patt(df,gs):
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

# Function for visualizing a graph in Kepler
# input: a graph in iGraph format, a list of pattenrs and the dataframe with metrics
# output: the Kepler framework with two layers: data and background

def kepler_vis(df_dist,gs,df_patt):
    mel_coa = geopandas.read_file('mel_communes/mel_communes_coarse.geojson')
    all_list = getLatLongFromGraph_patt(df_patt,gs)
    list_points = []
    for i in all_list:
        list_points.append(LineString(i))
    df_dist["geometry"] = list_points
    gdf = geopandas.GeoDataFrame(df_dist, geometry='geometry')
    gdf = gdf.set_crs('epsg:4326')
    visu_gdf = gdf[["geometry","supp","CDP","PSI","LDI","PPI","DDI"]]
    map_kepler = KeplerGl(height=700)
    map_kepler.add_data(data=visu_gdf, name='Patterns')
    map_kepler.add_data(mel_coa, name='MEL')

    return map_kepler


# Function for visualizing geo-points in geopandas
# input: a graph in iGraph format
# output: the geo-located points

def geo_points_graph(gs):
    nodes_df = pd.DataFrame({attr: gs.vs[attr] for attr in gs.vertex_attributes()})
    #edge_df = pd.DataFrame({attr: gs.es[attr] for attr in gs.edge_attributes()})

    #stations = nodes_df[["id","nom","rating","nbAvis"]]
    stations = nodes_df[["id"]]

    def make_point(row):
        return Point(row.longitude, row.latitude)

    points = nodes_df.apply(make_point, axis=1)
    stations_point = geopandas.GeoDataFrame(stations, geometry=points)

    mel_coa = geopandas.read_file('mel_communes/mel_communes_coarse.geojson')
    #mapa = mel_coa.plot(color='lightgrey', linewidth=0.5, edgecolor='white', figsize=(15,5))
    mapa = mel_coa.plot(color='white', linewidth=0.5, edgecolor='gray', figsize=(15,5))
    stations_point.plot(markersize=10, color='blue', alpha=0.5, ax=mapa)
    mapa.axis('off')

# Function for visualizing a line plot showing the elapsed time by support
# input: a list of data for statistics 
# output: the plot
    
def time_by_supp_patt(lst_supp_patt):
    x = [i[0] for i in lst_supp_patt]
    y1 = [i[4] for i in lst_supp_patt]
    y2 = [i[5] for i in lst_supp_patt]
    y3 = [i[6] for i in lst_supp_patt]
    plt.plot(x, y1, label = "A*")
    plt.plot(x, y2, label = "BFS")
    plt.plot(x, y3, label = "Dijkstra")
    plt.legend()
    #plt.yscale("log")
    plt.xlabel('Support minimal')
    plt.ylabel('Elapsed time (sec)')
    plt.show()

# Function for visualizing a line plot showing the number of patterns by support
# input: a list of data for statistics 
# output: the plot
    
def nb_pat_by_supp_patt(lst_supp_patt):
    x = [i[0] for i in lst_supp_patt]
    y1 = [i[1] for i in lst_supp_patt]
    y2 = [i[2] for i in lst_supp_patt]
    y3 = [i[3] for i in lst_supp_patt]
    plt.plot(x, y1, label = "A*")
    plt.plot(x, y2, label = "BFS")
    plt.plot(x, y3, label = "Dijkstra")
    plt.legend()
    #plt.yscale("log")
    plt.xlabel('Support minimal')
    plt.ylabel('Number of patterns')
    plt.show()

# Function for visualizing a hystogram showing the size of pattens by traversal technique
# input: a list of data for statistics 
# output: the plot
    
def hysto_traversal_nb_patt(list_all_patt_stat):
    values_a, counts_a = zip(*list_all_patt_stat[0][0][2].items())
    values_bfs, counts_bfs = zip(*list_all_patt_stat[1][0][2].items())
    values_dij, counts_dij = zip(*list_all_patt_stat[2][0][2].items())

    lst_plot_histo = []
    lst_plot_histo.append(list(counts_a))
    lst_plot_histo.append(list(counts_bfs))
    lst_plot_histo.append(list(counts_dij))
    arr_t = np.array(lst_plot_histo).T
    arr_t

    x = ['A*', 'BFS', 'Dijkstra']
    #x = ['Size 2', 'Size 3']
    #x = list(values_a)

    X_axis = np.arange(len(x)) 

    plt.bar(X_axis + 0.1, arr_t[1], width=0.2, label = "Size 3")
    plt.bar(X_axis - 0.1, arr_t[0], width=0.2, label = "Size 2")

    plt.legend()
    plt.xticks(X_axis, x) 
    plt.xlabel('Traversal sequences extracton techniques')
    plt.ylabel('Number of pattens')

    plt.show()

# Function for visualizing a line plot showing the elapsed time by support
# input: a list of data for statistics 
# output: the plot
    

def unique(list1):
    return reduce(lambda re, x: re+[x] if x not in re else re, list1, [])

def time_by_supp_patt_by_pathtech(lst_supp_patt):
    labels = unique(([x[0] for x in lst_supp_patt]))
    supports = unique(([x[1] for x in lst_supp_patt]))
    y1, y2, y3, y4, y5, y6 = [], [], [], [], [], []
    for i in range(0, len(labels)*len(supports), 6):
        aux_y1 = lst_supp_patt[i][3]
        y1.append(aux_y1)
        aux_y2 = lst_supp_patt[i+1][3]
        y2.append(aux_y2)
        aux_y3 = lst_supp_patt[i+2][3]
        y3.append(aux_y3)
        aux_y4 = lst_supp_patt[i+3][3]
        y4.append(aux_y4)
        aux_y5 = lst_supp_patt[i+4][3]
        y5.append(aux_y5)
        aux_y6 = lst_supp_patt[i+5][3]
        y6.append(aux_y6)
        #y2 = lst_supp_patt[i][3]
    plt.plot(supports, y1, label = labels[0])
    plt.plot(supports, y2, label = labels[1])
    plt.plot(supports, y3, label = labels[2])
    plt.plot(supports, y4, label = labels[3])
    plt.plot(supports, y5, label = labels[4])
    plt.plot(supports, y6, label = labels[5])
    plt.legend()
    ##plt.yscale("log")
    plt.xlabel('Support minimal')
    plt.ylabel('Elapsed time (sec)')
    plt.show()


# Function for visualizing a line plot showing the elapsed time by support
# input: a list of data for statistics 
# output: the plot
    
def nbpatt_by_supp_patt_by_pathtech(lst_supp_patt):
    labels = unique(([x[0] for x in lst_supp_patt]))
    supports = unique(([x[1] for x in lst_supp_patt]))
    y1, y2, y3, y4, y5, y6 = [], [], [], [], [], []
    for i in range(0, len(labels)*len(supports), 6):
        aux_y1 = lst_supp_patt[i][2]
        y1.append(aux_y1)
        aux_y2 = lst_supp_patt[i+1][2]
        y2.append(aux_y2)
        aux_y3 = lst_supp_patt[i+2][2]
        y3.append(aux_y3)
        aux_y4 = lst_supp_patt[i+3][2]
        y4.append(aux_y4)
        aux_y5 = lst_supp_patt[i+4][2]
        y5.append(aux_y5)
        aux_y6 = lst_supp_patt[i+5][2]
        y6.append(aux_y6)
        #y2 = lst_supp_patt[i][3]
    plt.plot(supports, y1, label = labels[0])
    plt.plot(supports, y2, label = labels[1])
    plt.plot(supports, y3, label = labels[2])
    plt.plot(supports, y4, label = labels[3])
    plt.plot(supports, y5, label = labels[4])
    plt.plot(supports, y6, label = labels[5])
    plt.legend()
    ##plt.yscale("log")
    plt.xlabel('Support minimal')
    plt.ylabel('Number of sequential patterns')
    plt.show()

# Function for visualizing a line/bars plot showing the elapsed time by path extraction technique
# input: a list of data for statistics 
# output: the plot

def time_by_pathtech(lst_paths_by_lib):
    labels = [x[0] for x in lst_paths_by_lib]
    y1 = [x[1] for x in lst_paths_by_lib]
    plt.plot(labels, y1)
    #plt.legend()
    ##plt.yscale("log")
    plt.xlabel('Paths extraction techniques')
    plt.ylabel('Elapsed time (sec)')
    plt.show()

def hysto_traversal_time(lst_paths_by_lib):
    arr_t = [x[1] for x in lst_paths_by_lib]
    x = [i[0] for i in lst_paths_by_lib]
    X_axis = np.arange(len(x)) 

    plt.yscale("log")
    plt.bar(X_axis + 0.1, arr_t, width=0.4)
    plt.xticks(X_axis, x) 
    plt.xlabel('Paths extracton technique')
    plt.ylabel('Elapsed time in secs (log scale)')

    plt.show()

# Function for visualizing a line/bars plot showing the number of paths by path extraction technique
# input: a list of data for statistics 
# output: the plot

def nbpaths_by_pathtech(lst_paths_by_lib):
    labels = [x[0] for x in lst_paths_by_lib]
    y1 = [x[2] for x in lst_paths_by_lib]
    plt.plot(labels, y1)
    #plt.legend()
    ##plt.yscale("log")
    plt.xlabel('Paths extraction techniques')
    plt.ylabel('Number of paths')
    plt.show()

def hysto_traversal_nb_patt(lst_paths_by_lib):
    arr_t = [x[2] for x in lst_paths_by_lib]
    x = [i[0] for i in lst_paths_by_lib]
    X_axis = np.arange(len(x)) 

    #plt.yscale("log")
    plt.bar(X_axis + 0.1, arr_t, width=0.4)
    plt.xticks(X_axis, x) 
    plt.xlabel('Paths extracton technique')
    plt.ylabel('Number of pattens')

    plt.show()

# Function for visualizing a hystogram showing the size of pattens by traversal technique
# input: a list of data for statistics 
# output: the plot

def hysto_traversal_nb_patt_by_path_tech(list_all_patt_stat):
    for i in range(len(list_all_patt_stat)):
        aux = list2dataframe(list_all_patt_stat[i][1]) 
        values, counts = zip(*Counter([len(sublist) for sublist in aux['patterns']]).items())
        lst_plot_histo = []
        lst_plot_histo.append(list(counts))
    arr_t = np.array(lst_plot_histo).T
    arr_t

    x = [i[0] for i in list_all_patt_stat]

    X_axis = np.arange(len(x)) 

    plt.bar(X_axis + 0.1, arr_t[1], width=0.2, label = "Size 3")
    plt.bar(X_axis - 0.1, arr_t[0], width=0.2, label = "Size 2")

    plt.legend()
    plt.xticks(X_axis, x) 
    plt.xlabel('Paths extracton techniques')
    plt.ylabel('Number of pattens')

    plt.show()

# Function for visualizing a hystogram showing the size of pattens extracted by PrefixSpan by traversal technique
# input: a list of data for statistics 
# output: the plot
    
def hysto_traversal_nb_patt_by_path_tech_PrefixSpan(patt_Prefixspan):
    for i in range(len(patt_Prefixspan)):
        aux = list2dataframe(patt_Prefixspan[i][1])
        values, counts = zip(*Counter([len(sublist) for sublist in aux['patterns']]).items())
        lst_plot_histo = []
        lst_plot_histo.append(list(counts))
    arr_t = np.array(lst_plot_histo).T
    arr_t
    
    x = [i[0] for i in patt_Prefixspan]

    X_axis = np.arange(len(x)) 

    plt.bar(X_axis + 0.2, arr_t[2], width=0.2, label = "Size 3")
    plt.bar(X_axis + 0.0, arr_t[1], width=0.2, label = "Size 2")
    plt.bar(X_axis - 0.2, arr_t[0], width=0.2, label = "Size 1")
    

    plt.legend()
    plt.xticks(X_axis, x) 
    plt.xlabel('Paths extracton techniques')
    plt.ylabel('Number of pattens')

    plt.show()


# Function for normalizing the data provided by metrics and grouped by spatial granulartiy
# input: a dataframe with metrics grouped by spatial granulartiy and an option 1: MinMax (default), 2: Z-score, 3: Robust
# output: normalized data in a list
    
def data_normalization(spatialg, scale_opt): 

    titles = ['supp', 'CDP_all', 'PSI', 'LDI', 'PPI', 'NDI_V2', 'count','Nb_Countries', 'Nb_Comm']

    ## Method by default
    scaler = MinMaxScaler()

    spatialg = spatialg.reset_index()
    data = spatialg[titles]
    data.values.tolist()

    if scale_opt == 2:
        scaler = StandardScaler()
    if scale_opt == 3:
        scaler = RobustScaler()

    scaled_data = scaler.fit_transform(data.T)

    #spatial = ['Coarse','Fine']
    df_spatial = pd.DataFrame(scaled_data.T)
    df_spatial.columns = titles
    return df_spatial

# Function for ploting bars from data provided by metrics and grouped by spatial granulartiy
# input: a dataframe with normalized metrics grouped by spatial granulartiy
# output: a plot

def plot_histo_granularity(df_spatial):
    branches = df_spatial.columns

    trace1 = go.Bar(
    x = branches,
    y = df_spatial.loc[0],
    name = 'Coarse',
    marker_color='rgb(55, 83, 109)'
    )
    trace2 = go.Bar(
    x = branches,
    y = df_spatial.loc[1],
    name = 'Fine',
    marker_color='rgb(26, 118, 255)'
    )

    data = [trace1, trace2]
    layout = go.Layout(barmode = 'group')
    fig = go.Figure(data = data, layout = layout).update_layout(xaxis_title="Metrics", yaxis_title="Normalized values", width=1000, height=600)
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    fig.update_layout(
        plot_bgcolor='white'
    )
    fig.show()

# Function for normalizing the data provided by metrics and grouped by spatial granulartiy and seasons
# input: a dataframe with metrics grouped by spatial granulartiy and an option 1: MinMax (default), 2: Z-score, 3: Robust
# output: normalized data in a list
    
def data_normalization_season(spatialg, scale_opt): 

    titles = ['supp', 'CDP_all', 'PSI', 'LDI', 'PPI', 'NDI_V2', 'DDI', 'count','Nb_Countries', 'Nb_Comm']

    ## Method by default
    scaler = MinMaxScaler()

    spatialg = spatialg.reset_index()
    data = spatialg[titles]
    data.values.tolist()

    if scale_opt == 2:
        scaler = StandardScaler()
    if scale_opt == 3:
        scaler = RobustScaler()

    scaled_data = scaler.fit_transform(data.T)

    #spatial = ['Coarse','Fine']
    df_spatial = pd.DataFrame(scaled_data.T)
    df_spatial.columns = titles
    return df_spatial

# Function for ploting bars from data provided by metrics and grouped by spatial granulartiy and for saison
# input: a dataframe with normalized metrics grouped by spatial granulartiy and an option: 1 for coarse and 0 for fine
# output: a plot

def plot_seasonal_hysto(df_spatial, option_c_f): ## 1 for coarse, 0 for fine
  if option_c_f == 1:
    rows = [0,2,4]
  else:
    rows = [1,3,5]
  
  branches = df_spatial.columns

  trace1 = go.Bar(
    x = branches,
    y = df_spatial.loc[rows[0]],
    name = 'Xmas',
    marker_color='rgb(55, 83, 109)'
  )
  trace2 = go.Bar(
    x = branches,
    y = df_spatial.loc[rows[1]],
    name = 'Summer',
    marker_color='rgb(26, 118, 255)'
  )

  trace3 = go.Bar(
    x = branches,
    y = df_spatial.loc[rows[2]],
    name = 'Sping',
    marker_color='rgb(153, 153, 0)'
  )

  data = [trace1, trace2, trace3]
  layout = go.Layout(barmode = 'group')
  fig = go.Figure(data = data, layout = layout).update_layout(xaxis_title="Metrics", yaxis_title="Normalized values", width=1000, height=600)
  fig.update_xaxes(
      mirror=True,
      ticks='outside',
      showline=True,
      linecolor='black',
      gridcolor='lightgrey'
  )
  fig.update_yaxes(
      mirror=True,
      ticks='outside',
      showline=True,
      linecolor='black',
      gridcolor='lightgrey'
  )
  fig.update_layout(
      plot_bgcolor='white'
  )
  fig.show()


# Function for writing a CSV file with data for ploting paths in form of arcs
# input: a dataframe with patterns (list of tuples)
# output: a plot
  
def arcs_paths_kepler_toFile(list_patts):
    list_arcs = []
    for k in list_patts:
        line_r = LineString(k)
        line_gdf_r = geopandas.GeoSeries([line_r])
        line_gdf_r = geopandas.GeoDataFrame(geometry=line_gdf_r)

        for prev, curr in zip(k,k[1:]): 
            temp_k = []
            tmp_str = 'LINESTRING '+ str(prev)+'#' + str(curr).replace(',','')
            temp_k.append(tmp_str.replace('#',','))
            temp_k.append(prev[0])
            temp_k.append(prev[1])
            temp_k.append(curr[0])
            temp_k.append(curr[1])
        list_arcs.append(temp_k)

    aux = pd.DataFrame(list_arcs,columns=['geometry','x1','y1','x2','y2']) 
    aux['coord1'] = list(zip(aux.x1, aux.y1))
    aux['coord2'] = list(zip(aux.x2, aux.y2))
    aux['combined_tuples'] = aux.apply(lambda row: LineString([row['coord1'], row['coord2']]), axis=1)
    aux = aux.drop(columns=['geometry','coord1','coord2'])
    aux = aux.reindex(columns=['combined_tuples','x1','y1','x2','y2'])
    aux.columns = ['geometry', 'x1','y1','x2','y2']
    gdf = geopandas.GeoDataFrame(aux, geometry='geometry')
    gdf = gdf.set_crs('epsg:4326')
    file_path = 'output_arcs_kepler.csv'
    aux.to_csv(file_path, index=False)
    return aux
    #map_kepler = KeplerGl(height=700)
    #map_kepler.add_data(data=aux, name='Patterns')

    #return map_kepler

# Function for ploting a pie of different nationalities par saison 
# input: a dataframe with patterns per saison (list of tuples) and the spatial grannularity (fine, coarse)
# output: a plot

def plot_pye_countries(list_patt, granu):
    countries_xmas = list_patt[list_patt['granularity'] == granu]
    list_natio = countries_xmas['Nationality'].values.tolist()
    flattened_list = [sublist for sublist_list in list_natio for sublist in sublist_list]
    flat_list = [item for sublist in flattened_list for item in sublist]
    frequency_count_xmas_fine = Counter(flat_list)
    df = pd.DataFrame(frequency_count_xmas_fine.items(), columns=['Item', 'Frequency'])
    fig = px.pie(df, values='Frequency', names='Item')
    fig.show()


# Two funcions for ploting a plot showing the elapsed/number of patterns time by support for Latex purposes
# input: a list of data for statistics 
# output: the plot

# Set the text.usetex parameter to True to enable LaTeX rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]
})

plt.rcParams.update({'font.size': 13})

def nbpatt_by_supp_patt_by_pathtech_latex(lst_supp_patt):
    labels = unique(([x[0] for x in lst_supp_patt]))
    supports = unique(([x[1] for x in lst_supp_patt]))
    #y1, y2, y3, y4, y5, y6 = [], [], [], [], [], []
    y1, y2, y3 = [], [], []
    for i in range(0, len(labels)*len(supports), 6):
        aux_y1 = lst_supp_patt[i][2]
        y1.append(aux_y1)
        aux_y2 = lst_supp_patt[i+1][2]
        y2.append(aux_y2)
        aux_y3 = lst_supp_patt[i+2][2]
        y3.append(aux_y3)
    fig, ax = plt.subplots()
    plt.plot(supports, y1, label = "A-star", marker ='o')
    plt.plot(supports, y2, label = "BFS", marker ='o')
    plt.plot(supports, y3, label = "Dijkstra", marker ='o')
    plt.legend()
    ##plt.yscale("log")
    plt.xlabel('Support minimal')
    plt.ylabel('Number of patterns')
    xmin, xmax = ax.get_xlim()
    ax.set(xlim=(xmin-0.25, xmax+0.25), axisbelow=True)
    ax.grid(axis='x')
    ax.invert_xaxis() # To reverse the values in X-axis
    # Save the plot as a PDF file
    plt.savefig('lines_number_patt.pdf', bbox_inches='tight')
    plt.show()

def time_by_supp_patt_by_pathtech_latex(lst_supp_patt):
    labels = unique(([x[0] for x in lst_supp_patt]))
    supports = unique(([x[1] for x in lst_supp_patt]))
    y1, y2, y3 = [], [], []
    #y1, y2, y3, y4, y5, y6 = [], [], [], [], [], []
    for i in range(0, len(labels)*len(supports), 6):
        aux_y1 = lst_supp_patt[i][3]
        y1.append(aux_y1)
        aux_y2 = lst_supp_patt[i+1][3]
        y2.append(aux_y2)
        aux_y3 = lst_supp_patt[i+2][3]
        y3.append(aux_y3)
    fig, ax = plt.subplots()
    plt.plot(supports, y1, label = "A-star", marker ='o')
    plt.plot(supports, y2, label = "BFS", marker ='o')
    plt.plot(supports, y3, label = "Dijkstra", marker ='o')
    plt.legend()
    ##plt.yscale("log")
    plt.xlabel('Support minimal')
    plt.ylabel('Execution time in seconds')
    xmin, xmax = ax.get_xlim()
    ax.set(xlim=(xmin-0.25, xmax+0.25), axisbelow=True)
    ax.grid(axis='x')
    ax.invert_xaxis() # To reverse the values in X-axis
    # Save the plot as a PDF file
    plt.savefig('lines_time_patt.pdf', bbox_inches='tight')
    plt.show()


def list2dataframe_sankey(list_patt):
    patterns = []
    supp = []
    for i in list_patt:
        patterns.append([str(x) for x in i[0:len(i)-1]])
        supp.append(i[len(i)-1])
    df = pd.DataFrame({'patterns': patterns,'supp': supp})
    return df

## Create a list of colors by feature: if feature = dataset (seasons), only 7 colors are built. If feature = supp, various coulours are built

def create_colors(df_patts, feature, palete):
    legend = []
    sc = len(set(df_patts[feature])) # Number of different colors formed by CMAP = unique support values
    cmap = plt.cm.get_cmap(palete, sc)
    hex_colors = []
    set_fea = set(df_patts[feature]) # we obtain a set of unique support values
    set_fea = sorted(set_fea, reverse=True)
    for i in range(cmap.N):
        rgba = cmap(i)
        hex_colors.append(mpl.colors.rgb2hex(rgba)) # transforming RGB into hex color values
    color_fea = []
    for id, val in enumerate(set_fea): # associate a different color for a different support (only unique support values)
        aux_leg = []
        aux = []
        #print(hex_colors[id], ' -- ', val)
        aux.append(val)
        aux.append(hex_colors[id])
        color_fea.append(aux)
        aux_leg.append(str(hex_colors[id]))
        aux_leg.append(str(val))
        legend.append(aux_leg)
    #print('Number of different colors = ',len(color_fea))
    hex_color_fea = []
    for i in range(len(df_patts[feature])):
        for j in range(len(color_fea)):
            if df_patts[feature][i] == color_fea[j][0]:
                hex_color_fea.append(color_fea[j][1])
                #print(df_patts['patterns'][i], ' ------ ', color_fea[j][1])
    if feature == 'supp':
        legend_entries = legend
    else:
        legend_entries = [["#fbb4ae", "Spring"],["#b3cde3", "Summer + Spring"],["#decbe4", "Summer"],["#fed9a6", "Chritsmas + Spring"],
                          ["#e5d8bd", "Chritsmas + Summer + Spring"], ["#fddaec", "Chritsmas + Summer"],["#f2f2f2", "Chritsmas"]]
    return hex_color_fea, legend_entries

# Creating a dictionary before Sankey visualization

def create_dic_sankey_simple(df_patts_conc, nodes, colors, feat_lines):
    patterns = list(df_patts_conc['patterns'])
    supp = list(df_patts_conc[feat_lines])
    
    #print('Size patterns ---->', len(patterns))

    items = [item for sublist in patterns for item in sublist]

    items_label = list(set(items))

    items_uniq = enumerate(items_label)

    #print('Size items uniq ---->', len(items_label))

    colors_nodes = nodes
    colors_link = colors
    #print('Size nodes color ---->', len(colors_nodes))
    #print('Size link color ---->', len(colors_link))

    #print(colors_link)

    patt_colors = {}
    patt_dict = {}
    for k,v in items_uniq: #### ERROR
        #print('--->',k, '--',v)
        patt_dict[v] = k
        patt_colors[v] = [colors_nodes[k],colors_link[k]]

    #for i in range(len(items_label)-1):
    #    patt_colors[i] = [colors_nodes[i],colors_link[i]]

    #print('patt colors \n' , patt_colors)

    patt = []

    cont = 0
    for seq in patterns:
        #print(j)
        #print(seq)
        #print('len seq : ', len(seq))
        for i in range(len(seq)-1):
            #patt.append(pd.Series([patt_dict[seq[i]],patt_dict[seq[i+1]], colors_nodes[cont],patt_colors[seq[i+1]][1]], 
            #                    index=['index_source', 'index_target', 'color_node','color_link']))
            patt.append(pd.Series([patt_dict[seq[i]],patt_dict[seq[i+1]], colors_nodes[cont], colors_link[cont], supp[cont]], 
                                index=['index_source', 'index_target', 'color_node','color_link', feat_lines]))
        cont = cont + 1

    #print(patt)
    result_dic = pd.DataFrame(patt)
    result_dic.columns=['index_source', 'index_target', 'color_node', 'color_link', feat_lines]
    result_dic = result_dic.drop_duplicates() # We do so because some links are duplicated
    return result_dic, items_label, colors_nodes, patterns


def create_dic_sankey(df_patts_conc, nodes, colors, feat_lines):
    patterns = list(df_patts_conc['patterns'])
    supp = list(df_patts_conc[feat_lines])
    items = [item for sublist in patterns for item in sublist]

    items_label = list(set(items))

    items_uniq = enumerate(items_label)

    #print('Number of supports = ',len(supp))

    colors_nodes = nodes
    colors_link = colors

    #print(colors_link)

    patt_colors = {}
    patt_dict = {}
    #patt_sup = {}
    for k,v in items_uniq:
        patt_dict[v] = k
        #patt_sup[v] = k
        patt_colors[v] = [colors_nodes[k],colors_link[k]]

    #print('patt dict \n' , patt_dict)
    #print('patt colors \n' , patt_colors)

    #print(supp)
    patt = []

    #print('Number of patterns = ',len(patterns))
    #j=0
    for seq in patterns:
        #print(j)
        #print(seq)
        #print('len seq : ', len(seq))
        cont = 0
        for i in range(len(seq)-1):
            #patt.append(pd.Series([patt_dict[seq[i]],patt_dict[seq[i+1]],
            #                       patt_colors[seq[i]][0],patt_colors[seq[i+1]][1]], 
            #                          index=['index_source', 'index_target', 'color_node',"color_link"]))
            #patt.append(pd.Series([patt_dict[seq[i]],patt_dict[seq[i+1]], patt_colors[seq[i]][0],patt_colors[seq[i+1]][1]], 
            #                    index=['index_source', 'index_target', 'color_node','color_link']))
            #print(supp)
            #print(patt_sup[seq[i]])
            #print(seq)
            #print(supp[cont])
            patt.append(pd.Series([patt_dict[seq[i]],
                                patt_dict[seq[i+1]], 
                                patt_colors[seq[i]][0],
                                patt_colors[seq[i+1]][1], 
                                supp[cont]
                                ],
                                index=['index_source', 'index_target', 'color_node','color_link', feat_lines]
                                )
                        )
            cont = cont +1
        #j = j+1

    #print(patt)
    result = pd.DataFrame(patt)
    result.columns=['index_source', 'index_target', 'color_node', 'color_link', feat_lines]
    result = result.drop_duplicates() # We do so because some links are duplicated
    return result

def visu_sankey_old(result, pattern_label_places, colors_nodes, patterns):
    fig = go.Figure(data=[go.Sankey(
        #arrangement = "snap",
        node = dict(
        #pad = 15,
        #thickness = 15,
        line = dict(color = "black", width = 0.5),
        label = pattern_label_places,
        color = colors_nodes,
        ),
        link = dict(
        #arrowlen=15,
        source = result["index_source"],
        target = result["index_target"],  
        value = [4 for i in range(0,len(patterns))],
        color = result["color_link"],
    ))])

    legend = []
    legend_entries = [
        ["#fbb4ae", "Spring"],
        ["#b3cde3", "Summer + Spring"],
        ["#decbe4", "Summer"],
        ["#fed9a6", "Chritsmas + Spring"],
        ["#e5d8bd", "Chritsmas + Summer + Spring"],
        ["#fddaec", "Chritsmas + Summer"],
        ["#f2f2f2", "Chritsmas"]
    ]
    for entry in legend_entries:
        legend.append(
            go.Scatter(
                mode="markers",
                x=[None],
                y=[None],
                marker=dict(size=10, color=entry[0], symbol="square"),
                name=entry[1],
            )
        )

    traces = [fig] + legend
    layout = go.Layout(
        showlegend=True,
        plot_bgcolor="rgba(0,0,0,0)",
    )

    fig.update_layout(
        #hovermode='x',
        #paper_bgcolor='#51504f',
        width=1300,
        height=1200,
        #margin={'t':50,'b':20}
    )

    #fig_leg = go.Figure(data=traces, layout=layout)
    #fig_leg.update_xaxes(visible=False)
    #fig_leg.update_yaxes(visible=False)
    #fig_leg.show()
    fig.show()
    fig.write_html("file.html")

## Visualize a Sankey with a legend (no documentation for adding a legenx, except https://stackoverflow.com/questions/58852056/how-to-show-a-legend-in-plotly-python-sankey)

def visu_sankey(result, pattern_label_places, colors_nodes, patterns, legend_entries, lines_feature, out_file_name):

    if lines_feature == "simple": # Plot weight of line as a constant value (4) or using the support
        flag_value = [4 for i in range(0,len(patterns))]
    else:
        flag_value = result[lines_feature]
 
    fig = go.Sankey(
        #arrangement = "snap",
        node = dict(
        #pad = 15,
        #thickness = 15,
        line = dict(color = "black", width = 0.5),
        label = pattern_label_places,
        color = colors_nodes,
        ),
        link = dict(
        #arrowlen=15,
        source = result["index_source"],
        target = result["index_target"],
        value = flag_value,
        color = result["color_link"],
    ))

    legend = []
    #legend_entries = legend_entries
    for entry in legend_entries:
        legend.append(
            go.Scatter(
                mode="markers",
                x=[None],
                y=[None],
                marker=dict(size=10, color=entry[0], symbol="square"),
                name=entry[1],
            )
        )

    traces = [fig] + legend
    layout = go.Layout(
        showlegend=True,
        plot_bgcolor="rgba(0,0,0,0)",
    )

    fig_leg = go.Figure(data=traces, layout=layout)
    fig_leg.update_xaxes(visible=False)
    fig_leg.update_yaxes(visible=False)
    fig_leg.update_layout(
        #hovermode='x',
        #paper_bgcolor='#51504f',
        width=1300,
        height=1200,
        #margin={'t':50,'b':20}
    )
    fig_leg.show()
    #pio.write_image(fig_leg, out_file_name,scale=6, width=1300, height=1200)
    #fig.show()
    #fig.write_html("file.html")

def concatenate_unique(df_test, feat_lines):
    concatenated_dict = {}
    for index, row in df_test.iterrows():
        sum_support = 0
        key = str(row['patterns'])  # Convert list of integers to string for dictionary key
        integer_supp = row[feat_lines]  # Convert integer value to intenger for adding supports
        id_seasons = str(row['dataset'])  # Convert integer value to string for concatenation
        list_supp_season = []
        if key in concatenated_dict:
            concatenated_dict[key][0] += id_seasons
            concatenated_dict[key][1] = concatenated_dict[key][1] + integer_supp
        else:
            sum_support = sum_support + integer_supp
            list_supp_season.append(id_seasons)
            list_supp_season.append(sum_support)
            concatenated_dict[key] = list_supp_season
    df_patts_conc = pd.DataFrame(list(concatenated_dict.items()), columns=['patterns', 'aux'])
    df_patts_conc['patterns'] = df_patts_conc['patterns'].apply(lambda x: [int(i) for i in ast.literal_eval(x)])
    df_patts_conc['dataset'] = df_patts_conc['aux'].apply(lambda x: x[0] if len(x) > 0 else None)
    df_patts_conc[feat_lines] = df_patts_conc['aux'].apply(lambda x: x[1] if len(x) > 1 else None)
    df_patts_conc.drop(columns=['aux'], inplace=True)
    return df_patts_conc

def concatenate_unique_seasons(df_test):
    concatenated_dict = {}
    for index, row in df_test.iterrows():
        integers_str = str(row['patterns'])  # Convert list of integers to string for dictionary key
        integer_value = str(row['dataset'])  # Convert integer value to string for concatenation
        if integers_str in concatenated_dict:
            concatenated_dict[integers_str] += integer_value  # Concatenate integer values if key already exists
        else:
            concatenated_dict[integers_str] = integer_value  # Otherwise, add new key-value pair
    df_patts_conc = pd.DataFrame(list(concatenated_dict.items()), columns=['patterns', 'dataset'])
    df_patts_conc['patterns'] = df_patts_conc['patterns'].apply(lambda x: [int(i) for i in ast.literal_eval(x)])

    return df_patts_conc

def getting_names_vertices(pattern_label, vertices):
    pattern_label_places = []
    for i in range(len(pattern_label)):
        #print(pattern_label[i])
        condition = vertices['id'] == int(pattern_label[i])
        aux = vertices.loc[condition, 'name']
        #print(aux)
        pattern_label_places.append(aux)

    return pattern_label_places

