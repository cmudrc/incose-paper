"""
Network Visual Data Generation Script
This script generates network data (node list and edge list) for visualizations based on different impact scenarios.
The visualizations are realized by Gephi.
"""

import os
import sys
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.optimize import curve_fit

from datasets import load_dataset



def net_data_generate(T_points_list, node_list, edge_list, t_unit):

    node_list['agent_id'] = node_list['agent_id'].astype(int)
    nodes = node_list['agent_id'].tolist()
    nodes.append(0)

    edge_list['n1'] = edge_list['n1'].astype(int)
    edge_list['n2'] = edge_list['n2'].astype(int)
    edge_list['t_step'] = edge_list['t_step'].astype(int)

    if t_unit == 'day':
        edge_list['t_step_day'] = edge_list['t_step'].apply(lambda x: math.ceil(x / (3600*24)))
        T_list = edge_list['t_step_day']
        T_list = list(set(T_list))
        T_list = sorted(T_list)
    else:
        T_list = edge_list['t_step']
        T_list = list(set(T_list))
        T_list = sorted(T_list)
    
        if t_unit == '2h':
            pass
        elif t_unit == 'hour_out_home':
            filtered_list = [item for item in T_list if item in T_points_list]
            T_list = filtered_list

    total_T = len(T_list)
    if total_T < 5:
        T_vals = T_list
    else:
        indices = [0, total_T // 4, total_T //2, 3 * total_T //4, total_T -1]
        T_vals = [T_list[j] for j in indices]

    net_edge_list = []

    i = 1

    for T in T_vals:
        if t_unit == 'day':
            sub_edge_list = edge_list[edge_list['t_step_day'] == T]
        else:
            sub_edge_list = edge_list[edge_list['t_step'] == T]
        sub_edge_list = sub_edge_list.reset_index(drop=True)

        net_edge_list.append(sub_edge_list)
  

    return nodes, net_edge_list, T_vals




def hous_size_5_diff_impacts_level15(T_points_list):
    """
    Population = 40, 
    Seed = 12, 
    Impact Type: Control (no impact), 15% Random Impact, 15% Critical Impact, 
    Household Size = 5
    """
    col_names = ['n1', 'n2', 'amount', 'resource_type', 't_step']

    repository_name = "cmudrc/incose-paper-data"

    # No impact
    node_file_path = "3_1_2_0/agents.csv"
    node_list = load_dataset(repository_name, data_files=node_file_path)
    node_list = node_list['train'].to_pandas()
    edge_file_path = "3_1_2_0/transactions.csv"
    edge_list = load_dataset(repository_name, data_files=edge_file_path)
    edge_list = edge_list['train'].to_pandas()
    edge_list.columns = col_names
    edge_list = edge_list.iloc[1:].reset_index(drop=True)

    nodes_no, net_edge_list_no, T_vals_no = net_data_generate(T_points_list, node_list, edge_list, 'day')

    nodes_df = pd.DataFrame(nodes_no, columns=['id'])
    nodes_df.to_csv('nodes_list_NoImpact.csv', index=False)


    for i in range(len(T_vals_no)):
        edges_df_file_name = f'edge_list_NoImpact_{T_vals_no[i]}.csv'
        edges = net_edge_list_no[i]
        edges = edges[['n1','n2']]
        edges.columns = ['Source', 'Target']



        edges.to_csv(edges_df_file_name, index=False)


    
    # Random impact
    node_file_path = "3_1_2_5/agents.csv"
    node_list = load_dataset(repository_name, data_files=node_file_path)
    node_list = node_list['train'].to_pandas()
    edge_file_path = "3_1_2_5/transactions.csv"
    edge_list = load_dataset(repository_name, data_files=edge_file_path)
    edge_list = edge_list['train'].to_pandas()
    edge_list.columns = col_names
    edge_list = edge_list.iloc[1:].reset_index(drop=True)

    nodes_random, net_edge_list_random, T_vals_random = net_data_generate(T_points_list, node_list, edge_list, 'day')

    nodes_df = pd.DataFrame(nodes_random, columns=['id'])
    nodes_df.to_csv('nodes_list_RandomImpact15.csv', index=False)


    for i in range(len(T_vals_random)):
        edges_df_file_name = f'edge_list_RandomImpact15_{T_vals_random[i]}.csv'
        edges = net_edge_list_random[i]
        edges = edges[['n1','n2']]
        edges.columns = ['Source', 'Target']

        edges.to_csv(edges_df_file_name, index=False)
    
    # Critical impact
    node_file_path = "3_1_2_2/agents.csv"
    node_list = load_dataset(repository_name, data_files=node_file_path)
    node_list = node_list['train'].to_pandas()
    edge_file_path = "3_1_2_2/transactions.csv"
    edge_list = load_dataset(repository_name, data_files=edge_file_path)
    edge_list = edge_list['train'].to_pandas()
    edge_list.columns = col_names
    edge_list = edge_list.iloc[1:].reset_index(drop=True)

    nodes_critical, net_edge_list_critical, T_vals_critical = net_data_generate(T_points_list, node_list, edge_list, 'day')

    nodes_df = pd.DataFrame(nodes_critical, columns=['id'])
    nodes_df.to_csv('nodes_list_CriticalImpact15.csv', index=False)


    for i in range(len(T_vals_critical)):
        edges_df_file_name = f'edge_list_CriticalImpact15_{T_vals_critical[i]}.csv'
        edges = net_edge_list_critical[i]
        edges = edges[['n1','n2']]
        edges.columns = ['Source', 'Target']

        edges.to_csv(edges_df_file_name, index=False)



if __name__ == "__main__":

    T_points_list = []
    for k in range(3000):
        for i in range(1, 4):
            T_points_list.append((i + 13 * k) * 7200)

    
    hous_size_5_diff_impacts_level15(T_points_list)
