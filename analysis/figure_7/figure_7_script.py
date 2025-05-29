"""
Fitted values of parameter a for different household sizes and impact levels
"""


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.optimize import curve_fit
from datasets import load_dataset


# Define a logarithmic decay function
def log_decay(x, a, b):
    return a * np.log(x) + b

# Define the polynomial function of degree 2
def poly_fit(x, a, b, c):
    return a * x**2 + b * x + c

# Define the power-law function
def power_law(x, a, b):
    return a * np.power(x, b)


# Define the linear function
def linear_fit(x, m, c):
    return m * x + c


# Function to calculate R-squared
def calculate_r_squared(y, y_pred):
    residual_sum_of_squares = np.sum((y - y_pred) ** 2)
    total_sum_of_squares = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)
    return r_squared


def calculate_giant_component_size(G):
    # Return the size of the largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    return len(largest_cc)
    

def net_robust_measure(T_points_list, node_list, edge_list, t_unit):

    # t_unit = 
    # '2h': aggregate network based on original t_step column, time-stamp = 2h
    # 'day': aggregate network with time-stamp = 1 day
    # 'hour_out_home': aggregate network with time-stamp = 2h, but sample time step with rule: 2h, 4h, 6h, 28h, 30h, 32h, ..., (2 + 26*K)h, (4 + 26*K), (6+26*K)
    
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
        indices = [0, 1, 2, 3, 4]
    else:
        indices = [0, total_T // 4, total_T //2, 3 * total_T //4, total_T -1]
    
    nets = {}

    i = 1

    # Generate networks
    for T in T_list:
        if t_unit == 'day':
            sub_edge_list = edge_list[edge_list['t_step_day'] == T]
        else:
            sub_edge_list = edge_list[edge_list['t_step'] == T]
        sub_edge_list = sub_edge_list.reset_index(drop=True)
        edges = list(zip(sub_edge_list['n1'], sub_edge_list['n2']))
        
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        nets[f'G_{i}'] = G

        i += 1

    # Robustness calculation
    LCC_t = []
    for i in range(1, total_T+1):
        G_name = f'G_{i}'
        G = nets[G_name]
        lcc = calculate_giant_component_size(G)
        LCC_t.append(lcc)

    LCC_t = np.array(LCC_t)
    T_list = np.array(T_list)

    return LCC_t, T_list

def log_decay_parameters(LCCs, t_list):

    A = []
    B = []

    for i in range(3):
        x = t_list[i]
        y = LCCs[i]

        params, _ = curve_fit(log_decay, x, y, maxfev = 5000)
        a, b = params

        A.append(a)
        B.append(b)

    A_mean = np.mean(A)
    B_mean = np.mean(B)
    A_std = np.std(A)
    B_std = np.std(B)
        
    return abs(A_mean), abs(B_mean), A_std, B_std
    
  

def critical_impacts(T_points_list, repository_name):
    # Population = 40, seed = 12, impact_type = no, random, critical, impact_level = 5, household size = 5
    col_names = ['n1', 'n2', 'amount', 'resource_type', 't_step']

    impact_level = [5, 15, 25]

    # Household size = 1
    A_m_list1 = []
    B_m_list1 = []
    A_std_list1 = []
    B_std_list1 = []
    
    for j in range(3):
        # j is index for impact level 5, 15, 25
        LCC_list = []
        Time_list = []
        for i in range(3):
            # i is index for seed
            node_file_path = f'3_0_{i}_{j+1}/agents.csv'
            node_list = load_dataset(repository_name, data_files=node_file_path)
            node_list = node_list['train'].to_pandas()
            edge_file_path = f'3_0_{i}_{j+1}/transactions.csv'
            edge_list = load_dataset(repository_name, data_files=edge_file_path)
            edge_list = edge_list['train'].to_pandas()
            edge_list.columns = col_names
            edge_list = edge_list.iloc[1:].reset_index(drop=True)


            LCC_t, T_list = net_robust_measure(T_points_list, node_list, edge_list, 'day')

            LCC_list.append(T_list)
            Time_list.append(LCC_t)

        A_m, B_m, A_std, B_std = log_decay_parameters(LCC_list, Time_list)
        A_m_list1.append(A_m)
        B_m_list1.append(B_m) 
        A_std_list1.append(A_std)
        B_std_list1.append(B_std)

    # Household size = 5
    A_m_list5 = []
    B_m_list5 = []
    A_std_list5 = []
    B_std_list5 = []
    
    for j in range(3):
        # j is index for impact level 5, 15, 25
        LCC_list = []
        Time_list = []
        for i in range(3):
            # i is index for seed
            node_file_path = f'3_1_{i}_{j+1}/agents.csv'
            node_list = load_dataset(repository_name, data_files=node_file_path)
            node_list = node_list['train'].to_pandas()
            edge_file_path = f'3_1_{i}_{j+1}/transactions.csv'
            edge_list = load_dataset(repository_name, data_files=edge_file_path)
            edge_list = edge_list['train'].to_pandas()
            edge_list.columns = col_names
            edge_list = edge_list.iloc[1:].reset_index(drop=True)

            LCC_t, T_list = net_robust_measure(T_points_list, node_list, edge_list, 'day')

            LCC_list.append(T_list)
            Time_list.append(LCC_t)

        A_m, B_m, A_std, B_std = log_decay_parameters(LCC_list, Time_list)
        A_m_list5.append(A_m)
        B_m_list5.append(B_m)
        A_std_list5.append(A_std)
        B_std_list5.append(B_std)

    # Household size = 10
    A_m_list10 = []
    B_m_list10 = []
    A_std_list10 = []
    B_std_list10 = []
    
    for j in range(3):
        # j is index for impact level 5, 15, 25
        LCC_list = []
        Time_list = []
        for i in range(3):
            # i is index for seed
            node_file_path = f'3_2_{i}_{j+1}/agents.csv'
            node_list = load_dataset(repository_name, data_files=node_file_path)
            node_list = node_list['train'].to_pandas()
            edge_file_path = f'3_2_{i}_{j+1}/transactions.csv'
            edge_list = load_dataset(repository_name, data_files=edge_file_path)
            edge_list = edge_list['train'].to_pandas()
            edge_list.columns = col_names
            edge_list = edge_list.iloc[1:].reset_index(drop=True)

            LCC_t, T_list = net_robust_measure(T_points_list, node_list, edge_list, 'day')

            LCC_list.append(T_list)
            Time_list.append(LCC_t)

        A_m, B_m, A_std, B_std = log_decay_parameters(LCC_list, Time_list)
        A_m_list10.append(A_m)
        B_m_list10.append(B_m)
        A_std_list10.append(A_std)
        B_std_list10.append(B_std)  

    
    # Colors for plotting
    colors = ['red', 'green', 'blue']
    a_scale = [0, 10, 20, 30, 40]
    b_scale = [0, 20, 40, 60, 80, 100, 120]
    a_scale = [0, 10, 20, 30]
    b_scale = [10, 20, 40, 60, 80, 90]

    # plt.rcParams['legend.fontsize'] = 14 

    legend_point_label = ['Household Size = 1', 'Household Size = 5', 'Household Size = 10']
    
    fig = plt.figure(figsize=(10,6))
    plt.rcParams.update({'font.size': 18})
    plt.xticks(impact_level)
    # Household size = 1
    plt.plot(impact_level, A_m_list1, marker = 'o', linestyle = '-', color = 'red', label = 'Household Size = 1')
    # plt.errorbar(impact_level, A_m_list1, A_std_list1, fmt='o',capsize=5, color = 'red')
    # Household size = 5
    plt.plot(impact_level, A_m_list5, marker = 'o', linestyle = '-', color = 'green', label = 'Household Size = 5')
    # plt.errorbar(impact_level, A_m_list5, A_std_list5, fmt='o',capsize=5, color = 'green')
    # Household size = 10
    plt.plot(impact_level, A_m_list10, marker = 'o', linestyle = '-', color = 'blue', label = 'Household Size = 10')
    # plt.errorbar(impact_level, A_m_list10, A_std_list10, fmt='o',capsize=5, color = 'blue')

    # plt.yscale('log')
    plt.yticks(a_scale)

    plt.xlabel('Critical Impact Level')
    plt.ylabel('Value of Parameter a')
    plt.legend()
    plt.grid(True)
    plt.close(fig)

    fig_filename = f'Critical_Impact_Diff_HHSize_param_a_update_noErrorbar.png'
    fig.savefig(fig_filename)

    fig = plt.figure(figsize=(10,6))
    plt.rcParams.update({'font.size': 18})
    plt.xticks(impact_level)

    # Household size = 1
    plt.plot(impact_level, B_m_list1, marker = 'o', linestyle = '-', color = 'red', label = 'Household Size = 1')
    # plt.errorbar(impact_level, B_m_list1, B_std_list1, fmt='o',capsize=5, color = 'red')
    # Household size = 5
    plt.plot(impact_level, B_m_list5, marker = 'o', linestyle = '-', color = 'green', label = 'Household Size = 5')
    # plt.errorbar(impact_level, B_m_list5, B_std_list5, fmt='o',capsize=5, color = 'green')
    # Household size = 10
    plt.plot(impact_level, B_m_list10, marker = 'o', linestyle = '-', color = 'blue', label = 'Household Size = 10')
    # plt.errorbar(impact_level, B_m_list10, B_std_list10, fmt='o',capsize=5, color = 'blue')

    # plt.yscale('log')
    plt.yticks(b_scale)

    plt.xlabel('Critical Impact Level')
    plt.ylabel('Value of Parameter b')
    plt.legend()
    plt.grid(True)
    plt.close(fig)

    fig_filename = f'Critical_Impact_Diff_HHSize_param_b_update_noErrorbar.png'
    fig.savefig(fig_filename)



def random_impacts(T_points_list, repository_name):
    # Population = 40, seed = 12, impact_type = no, random, critical, impact_level = 5, household size = 5
    col_names = ['n1', 'n2', 'amount', 'resource_type', 't_step']

    impact_level = [5, 15, 25]

    # Household size = 1
    A_m_list1 = []
    B_m_list1 = []
    A_std_list1 = []
    B_std_list1 = []
    
    for j in range(3):
        # j is index for impact level 5, 15, 25
        LCC_list = []
        Time_list = []
        for i in range(3):
            # i is index for seed
            node_file_path = f'3_0_{i}_{j+4}/agents.csv'
            node_list = load_dataset(repository_name, data_files=node_file_path)
            node_list = node_list['train'].to_pandas()
            edge_file_path = f'3_0_{i}_{j+4}/transactions.csv'
            edge_list = load_dataset(repository_name, data_files=edge_file_path)
            edge_list = edge_list['train'].to_pandas()
            edge_list.columns = col_names
            edge_list = edge_list.iloc[1:].reset_index(drop=True)

            LCC_t, T_list = net_robust_measure(T_points_list, node_list, edge_list, 'day')

            LCC_list.append(T_list)
            Time_list.append(LCC_t)

        A_m, B_m, A_std, B_std = log_decay_parameters(LCC_list, Time_list)
        A_m_list1.append(A_m)
        B_m_list1.append(B_m)
        A_std_list1.append(A_std)
        B_std_list1.append(B_std)

    # Household size = 5
    A_m_list5 = []
    B_m_list5 = []
    A_std_list5 = []
    B_std_list5 = []
    
    for j in range(3):
        # j is index for impact level 5, 15, 25
        LCC_list = []
        Time_list = []
        for i in range(3):
            # i is index for seed
            node_file_path = f'3_1_{i}_{j+4}/agents.csv'
            node_list = load_dataset(repository_name, data_files=node_file_path)
            node_list = node_list['train'].to_pandas()
            edge_file_path = f'3_1_{i}_{j+4}/transactions.csv'
            edge_list = load_dataset(repository_name, data_files=edge_file_path)
            edge_list = edge_list['train'].to_pandas()
            edge_list.columns = col_names
            edge_list = edge_list.iloc[1:].reset_index(drop=True)

            LCC_t, T_list = net_robust_measure(T_points_list, node_list, edge_list, 'day')

            LCC_list.append(T_list)
            Time_list.append(LCC_t)

        A_m, B_m, A_std, B_std = log_decay_parameters(LCC_list, Time_list)
        A_m_list5.append(A_m)
        B_m_list5.append(B_m)
        A_std_list5.append(A_std)
        B_std_list5.append(B_std)  

    # Household size = 10
    A_m_list10 = []
    B_m_list10 = []
    A_std_list10 = []
    B_std_list10 = []
    
    for j in range(3):
        # j is index for impact level 5, 15, 25
        LCC_list = []
        Time_list = []
        for i in range(3):
            # i is index for seed
            node_file_path = f'3_2_{i}_{j+4}/agents.csv'
            node_list = load_dataset(repository_name, data_files=node_file_path)
            node_list = node_list['train'].to_pandas()
            edge_file_path = f'3_2_{i}_{j+4}/transactions.csv'
            edge_list = load_dataset(repository_name, data_files=edge_file_path)
            edge_list = edge_list['train'].to_pandas()
            edge_list.columns = col_names
            edge_list = edge_list.iloc[1:].reset_index(drop=True)

            LCC_t, T_list = net_robust_measure(T_points_list, node_list, edge_list, 'day')

            LCC_list.append(T_list)
            Time_list.append(LCC_t)

        A_m, B_m, A_std, B_std = log_decay_parameters(LCC_list, Time_list)
        A_m_list10.append(A_m)
        B_m_list10.append(B_m)
        A_std_list10.append(A_std)
        B_std_list10.append(B_std)  

    
    # Colors for plotting
    colors = ['red', 'green', 'blue']
    a_scale = [0, 10, 20, 30, 40]
    b_scale = [0, 20, 40, 60, 80, 100, 120]
    a_scale = [0, 10, 20, 30]
    b_scale = [10, 20, 40, 60, 80, 90]

    # plt.rcParams['legend.fontsize'] = 14 

    legend_point_label = ['Household Size = 1', 'Household Size = 5', 'Household Size = 10']
    
    fig = plt.figure(figsize=(10,6))
    plt.rcParams.update({'font.size': 18})
    plt.xticks(impact_level)

    # Household size = 1
    plt.plot(impact_level, A_m_list1, marker = 'o', linestyle = '-', color = 'red', label = 'Household Size = 1')
    # plt.errorbar(impact_level, A_m_list1, A_std_list1, fmt='o',capsize=5, color = 'red')
    # Household size = 5
    plt.plot(impact_level, A_m_list5, marker = 'o', linestyle = '-', color = 'green', label = 'Household Size = 5')
    # plt.errorbar(impact_level, A_m_list5, A_std_list5, fmt='o',capsize=5, color = 'green')
    # Household size = 10
    plt.plot(impact_level, A_m_list10, marker = 'o', linestyle = '-', color = 'blue', label = 'Household Size = 10')
    # plt.errorbar(impact_level, A_m_list10, A_std_list10, fmt='o',capsize=5, color = 'blue')

    # plt.yscale('log')
    plt.yticks(a_scale)

    plt.xlabel('Random Impact Level')
    plt.ylabel('Value of Parameter a')
    plt.legend()
    plt.grid(True)
    plt.close(fig)

    fig_filename = f'Random_Impact_Diff_HHSize_param_a_update_noErrorbar.png'
    fig.savefig(fig_filename)

    fig = plt.figure(figsize=(10,6))
    plt.rcParams.update({'font.size': 18})
    plt.xticks(impact_level)

    # Household size = 1
    plt.plot(impact_level, B_m_list1, marker = 'o', linestyle = '-', color = 'red', label = 'Household Size = 1')
    # plt.errorbar(impact_level, B_m_list1, B_std_list1, fmt='o',capsize=5, color = 'red')
    # Household size = 5
    plt.plot(impact_level, B_m_list5, marker = 'o', linestyle = '-', color = 'green', label = 'Household Size = 5')
    # plt.errorbar(impact_level, B_m_list5, B_std_list5, fmt='o',capsize=5, color = 'green')
    # Household size = 10
    plt.plot(impact_level, B_m_list10, marker = 'o', linestyle = '-', color = 'blue', label = 'Household Size = 10')
    # plt.errorbar(impact_level, B_m_list10, B_std_list10, fmt='o',capsize=5, color = 'blue')

    # plt.yscale('log')
    plt.yticks(b_scale)

    plt.xlabel('Random Impact Level')
    plt.ylabel('Value of Parameter b')
    plt.legend()
    plt.grid(True)
    plt.close(fig)

    fig_filename = f'Random_Impact_Diff_HHSize_param_b_update_noErrorbar.png'
    fig.savefig(fig_filename)



if __name__ == "__main__":

    repository_name = "cmudrc/incose-paper-data"

    T_points_list = []
    for k in range(3000):
        for i in range(1, 4):
            T_points_list.append((i + 13 * k) * 7200)

    
    critical_impacts(T_points_list, repository_name)
    random_impacts(T_points_list, repository_name)
