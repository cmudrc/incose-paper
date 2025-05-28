"""
Robustness (LCC) measurements throughout the community life cycle under varying impact scenarios and household sizes
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
  


def hous_size_5_diff_impacts_level15(T_points_list, repository_name):
    # Population = 40, seed = 37, 107, 12, impact_type = no, random, critical, impact_level = 5, household size = 5
    col_names = ['n1', 'n2', 'amount', 'resource_type', 't_step']

    x_list = []
    y_list = []

    # No impact
    for i in range(3):
        node_file_path = f'3_1_{i}_0/agents.csv'
        node_list = load_dataset(repository_name, data_files=node_file_path)
        node_list = node_list['train'].to_pandas()
        edge_file_path = f'3_1_{i}_0/transactions.csv'
        edge_list = load_dataset(repository_name, data_files=edge_file_path)
        edge_list = edge_list['train'].to_pandas()
        edge_list.columns = col_names
        edge_list = edge_list.iloc[1:].reset_index(drop=True)

        LCC_t, T_list = net_robust_measure(T_points_list, node_list, edge_list, 'day')

        x_list.append(T_list)
        y_list.append(LCC_t)
    
    # Random impact
    for i in range(3):
        node_file_path = f'3_1_{i}_5/agents.csv'
        node_list = load_dataset(repository_name, data_files=node_file_path)
        node_list = node_list['train'].to_pandas()
        edge_file_path = f'3_1_{i}_5/transactions.csv'
        edge_list = load_dataset(repository_name, data_files=edge_file_path)
        edge_list = edge_list['train'].to_pandas()
        edge_list.columns = col_names
        edge_list = edge_list.iloc[1:].reset_index(drop=True)

        LCC_t, T_list = net_robust_measure(T_points_list, node_list, edge_list, 'day')

        x_list.append(T_list)
        y_list.append(LCC_t)
    
    # Critical impact
    for i in range(3):
        node_file_path = f'3_1_{i}_2/agents.csv'
        node_list = load_dataset(repository_name, data_files=node_file_path)
        node_list = node_list['train'].to_pandas()
        edge_file_path = f'3_1_{i}_2/transactions.csv'
        edge_list = load_dataset(repository_name, data_files=edge_file_path)
        edge_list = edge_list['train'].to_pandas()
        edge_list.columns = col_names
        edge_list = edge_list.iloc[1:].reset_index(drop=True)

        LCC_t, T_list = net_robust_measure(T_points_list, node_list, edge_list, 'day')

        x_list.append(T_list)
        y_list.append(LCC_t)

    # Colors for plotting
    colors = ['red', 'green', 'blue']

    # plt.rcParams['legend.fontsize'] = 14 

    legend_point_label = ['Control', '15% Random Impact', '15% Critical Impact']
    
    seed_list = [37, 107, 12]

    for j in range(3):

        # Initialize the plot
        fig = plt.figure(figsize=(10, 6))
        plt.rcParams.update({'font.size': 18})

        for i in range(3):

            x = x_list[j + 3 *i]
            y = y_list[j + 3* i]

            # initial_guess = [max(y), 0.1, min(y)]

            # bounds = ([0, 1e-10, -np.inf], [np.inf, np.inf, np.inf])  # Bounds for a, b, c (b must be positive)
            params, _ = curve_fit(log_decay, x, y, maxfev = 5000)
            # params, _ = curve_fit(power_law, x, y, maxfev = 5000)
            # params, _ = curve_fit(linear_fit, x, y, maxfev = 5000)

            # Extract the fitted parameters
            a, b= params
            # a, b = params
            
            # Print the fitted curve equation
            print(f'Fitted curve for seed {seed_list[j]} dataset {i+1}: y = {a:.4f} * log(x) + {b:.4f}')
            # print(f'Fitted curve for dataset {i+1}: y = {a:.4f} * x^2 + {b:.4f} * x + {c:.4f}')
            # print(f'Fitted power-law curve for dataset {i+1}: y = {a:.4f} * x^{b:.4f}')
            # print(f'Fitted linear curve for dataset {i+1}: y = {a:.4f} * x + {b:.4f}')
        

            x_fit = np.linspace(min(x), max(x), 100)
            y_fit = log_decay(x_fit, *params)
            # y_fit = power_law(x_fit, *params)
            # y_fit = linear_fit(x_fit, *params)

            # Calculate R-squared
            # Calculate fitted y values for the original x values (for R-squared calculation)
            y_pred = log_decay(x, *params)
            # y_pred = power_law(x, *params)
            # y_pred = linear_fit(x, *params)
            r_squared = calculate_r_squared(y, y_pred)
            r_squared = round(r_squared, 3)
            print(f'R-squared: {r_squared}')
            print("")


            # Plot the original data points
            plt.scatter(x, y, label= legend_point_label[i], color=colors[i])
            # plt.errorbar(x, y, yerr=y_std, fmt='o', capsize=5, label= legend_point_label[i], color=colors[i])
            
            # Plot the fitted curve
            plt.plot(x_fit, y_fit, color=colors[i], label=f'Fitted Curve, R-Squared = {r_squared}')


        plt.xlabel('Time (days)')
        plt.ylabel('Robustness (LCC)')
        # plt.title('Logarithmic Decay Curve Fitting')
        plt.legend()
        plt.grid(True)
        plt.close(fig)
        # plt.show()

        seed_num = seed_list[j]

        fig_filename = f'hous_size_5_diff_impacts_level15_{seed_num}_updateLogFit.png'
        fig.savefig(fig_filename)



def hous_size_1_diff_impacts_level15(T_points_list, repository_name):
        # Population = 40, seed = 37, 107, 12, impact_type = no, random, critical, impact_level = 5, household size = 5
    col_names = ['n1', 'n2', 'amount', 'resource_type', 't_step']

    x_list = []
    y_list = []

    # No impact
    for i in range(3):
        node_file_path = f'3_0_{i}_0/agents.csv'
        node_list = load_dataset(repository_name, data_files=node_file_path)
        node_list = node_list['train'].to_pandas()
        edge_file_path = f'3_0_{i}_0/transactions.csv'
        edge_list = load_dataset(repository_name, data_files=edge_file_path)
        edge_list = edge_list['train'].to_pandas()
        edge_list.columns = col_names
        edge_list = edge_list.iloc[1:].reset_index(drop=True)

        LCC_t, T_list = net_robust_measure(T_points_list, node_list, edge_list, 'day')

        x_list.append(T_list)
        y_list.append(LCC_t)
    
    # Random impact
    for i in range(3):
        node_file_path = f'3_0_{i}_5/agents.csv'
        node_list = load_dataset(repository_name, data_files=node_file_path)
        node_list = node_list['train'].to_pandas()
        edge_file_path = f'3_0_{i}_5/transactions.csv'
        edge_list = load_dataset(repository_name, data_files=edge_file_path)
        edge_list = edge_list['train'].to_pandas()
        edge_list.columns = col_names
        edge_list = edge_list.iloc[1:].reset_index(drop=True)

        LCC_t, T_list = net_robust_measure(T_points_list, node_list, edge_list, 'day')

        x_list.append(T_list)
        y_list.append(LCC_t)
    
    # Critical impact
    for i in range(3):
        node_file_path = f'3_0_{i}_2/agents.csv'
        node_list = load_dataset(repository_name, data_files=node_file_path)
        node_list = node_list['train'].to_pandas()
        edge_file_path = f'3_0_{i}_2/transactions.csv'
        edge_list = load_dataset(repository_name, data_files=edge_file_path)
        edge_list = edge_list['train'].to_pandas()
        edge_list.columns = col_names
        edge_list = edge_list.iloc[1:].reset_index(drop=True)

        LCC_t, T_list = net_robust_measure(T_points_list, node_list, edge_list, 'day')

        x_list.append(T_list)
        y_list.append(LCC_t)

    # Colors for plotting
    colors = ['red', 'green', 'blue']

    # plt.rcParams['legend.fontsize'] = 14 

    legend_point_label = ['Control', '15% Random Impact', '15% Critical Impact']
    
    seed_list = [37, 107, 12]

    for j in range(3):

        # Initialize the plot
        fig = plt.figure(figsize=(10, 6))
        plt.rcParams.update({'font.size': 18})

        for i in range(3):

            x = x_list[j + 3 *i]
            y = y_list[j + 3* i]

            # initial_guess = [max(y), 0.1, min(y)]

            # bounds = ([0, 1e-10, -np.inf], [np.inf, np.inf, np.inf])  # Bounds for a, b, c (b must be positive)
            params, _ = curve_fit(log_decay, x, y, maxfev = 5000)
            # params, _ = curve_fit(power_law, x, y, maxfev = 5000)
            # params, _ = curve_fit(linear_fit, x, y, maxfev = 5000)

            # Extract the fitted parameters
            a, b= params
            # a, b = params
            
            # Print the fitted curve equation
            print(f'Fitted curve for seed {seed_list[j]} dataset {i+1}: y = {a:.4f} * log(x) + {b:.4f}')
            # print(f'Fitted curve for dataset {i+1}: y = {a:.4f} * x^2 + {b:.4f} * x + {c:.4f}')
            # print(f'Fitted power-law curve for dataset {i+1}: y = {a:.4f} * x^{b:.4f}')
            # print(f'Fitted linear curve for dataset {i+1}: y = {a:.4f} * x + {b:.4f}')
        

            x_fit = np.linspace(min(x), max(x), 100)
            y_fit = log_decay(x_fit, *params)
            # y_fit = power_law(x_fit, *params)
            # y_fit = linear_fit(x_fit, *params)

            # Calculate R-squared
            # Calculate fitted y values for the original x values (for R-squared calculation)
            y_pred = log_decay(x, *params)
            # y_pred = power_law(x, *params)
            # y_pred = linear_fit(x, *params)
            r_squared = calculate_r_squared(y, y_pred)
            r_squared = round(r_squared, 3)
            print(f'R-squared: {r_squared}')
            print("")


            # Plot the original data points
            plt.scatter(x, y, label= legend_point_label[i], color=colors[i])
            # plt.errorbar(x, y, yerr=y_std, fmt='o', capsize=5, label= legend_point_label[i], color=colors[i])
            
            # Plot the fitted curve
            plt.plot(x_fit, y_fit, color=colors[i], label=f'Fitted Curve, R-Squared = {r_squared}')


        plt.xlabel('Time (days)')
        plt.ylabel('Robustness (LCC)')
        # plt.title('Logarithmic Decay Curve Fitting')
        plt.legend()
        plt.grid(True)
        plt.close(fig)
        # plt.show()

        seed_num = seed_list[j]

        fig_filename = f'hous_size_1_diff_impacts_level15_{seed_num}_updateLogFit.png'
        fig.savefig(fig_filename)



def hous_size_10_diff_impacts_level15(T_points_list, repository_name):
    # Population = 40, seed = 37, 107, 12, impact_type = no, random, critical, impact_level = 5, household size = 5
    col_names = ['n1', 'n2', 'amount', 'resource_type', 't_step']

    x_list = []
    y_list = []

    # No impact
    for i in range(3):
        node_file_path = f'3_2_{i}_0/agents.csv'
        node_list = load_dataset(repository_name, data_files=node_file_path)
        node_list = node_list['train'].to_pandas()
        edge_file_path = f'3_2_{i}_0/transactions.csv'
        edge_list = load_dataset(repository_name, data_files=edge_file_path)
        edge_list = edge_list['train'].to_pandas()
        edge_list.columns = col_names
        edge_list = edge_list.iloc[1:].reset_index(drop=True)

        LCC_t, T_list = net_robust_measure(T_points_list, node_list, edge_list, 'day')

        x_list.append(T_list)
        y_list.append(LCC_t)
    
    # Random impact
    for i in range(3):
        node_file_path = f'3_2_{i}_5/agents.csv'
        node_list = load_dataset(repository_name, data_files=node_file_path)
        node_list = node_list['train'].to_pandas()
        edge_file_path = f'3_2_{i}_5/transactions.csv'
        edge_list = load_dataset(repository_name, data_files=edge_file_path)
        edge_list = edge_list['train'].to_pandas()
        edge_list.columns = col_names
        edge_list = edge_list.iloc[1:].reset_index(drop=True)

        LCC_t, T_list = net_robust_measure(T_points_list, node_list, edge_list, 'day')

        x_list.append(T_list)
        y_list.append(LCC_t)
    
    # Critical impact
    for i in range(3):
        node_file_path = f'3_2_{i}_2/agents.csv'
        node_list = load_dataset(repository_name, data_files=node_file_path)
        node_list = node_list['train'].to_pandas()
        edge_file_path = f'3_2_{i}_2/transactions.csv'
        edge_list = load_dataset(repository_name, data_files=edge_file_path)
        edge_list = edge_list['train'].to_pandas()
        edge_list.columns = col_names
        edge_list = edge_list.iloc[1:].reset_index(drop=True)

        LCC_t, T_list = net_robust_measure(T_points_list, node_list, edge_list, 'day')

        x_list.append(T_list)
        y_list.append(LCC_t)

    # Colors for plotting
    colors = ['red', 'green', 'blue']

    # plt.rcParams['legend.fontsize'] = 14 

    legend_point_label = ['Control', '15% Random Impact', '15% Critical Impact']
    
    seed_list = [37, 107, 12]

    for j in range(3):

        # Initialize the plot
        fig = plt.figure(figsize=(10, 6))
        plt.rcParams.update({'font.size': 18})

        for i in range(3):

            x = x_list[j + 3 *i]
            y = y_list[j + 3* i]

            # initial_guess = [max(y), 0.1, min(y)]

            # bounds = ([0, 1e-10, -np.inf], [np.inf, np.inf, np.inf])  # Bounds for a, b, c (b must be positive)
            params, _ = curve_fit(log_decay, x, y, maxfev = 5000)
            # params, _ = curve_fit(power_law, x, y, maxfev = 5000)
            # params, _ = curve_fit(linear_fit, x, y, maxfev = 5000)

            # Extract the fitted parameters
            a, b= params
            # a, b = params
            
            # Print the fitted curve equation
            print(f'Fitted curve for seed {seed_list[j]} dataset {i+1}: y = {a:.4f} * log(x) + {b:.4f}')
            # print(f'Fitted curve for dataset {i+1}: y = {a:.4f} * x^2 + {b:.4f} * x + {c:.4f}')
            # print(f'Fitted power-law curve for dataset {i+1}: y = {a:.4f} * x^{b:.4f}')
            # print(f'Fitted linear curve for dataset {i+1}: y = {a:.4f} * x + {b:.4f}')
        

            x_fit = np.linspace(min(x), max(x), 100)
            y_fit = log_decay(x_fit, *params)
            # y_fit = power_law(x_fit, *params)
            # y_fit = linear_fit(x_fit, *params)

            # Calculate R-squared
            # Calculate fitted y values for the original x values (for R-squared calculation)
            y_pred = log_decay(x, *params)
            # y_pred = power_law(x, *params)
            # y_pred = linear_fit(x, *params)
            r_squared = calculate_r_squared(y, y_pred)
            r_squared = round(r_squared, 3)
            print(f'R-squared: {r_squared}')
            print("")


            # Plot the original data points
            plt.scatter(x, y, label= legend_point_label[i], color=colors[i])
            # plt.errorbar(x, y, yerr=y_std, fmt='o', capsize=5, label= legend_point_label[i], color=colors[i])
            
            # Plot the fitted curve
            plt.plot(x_fit, y_fit, color=colors[i], label=f'Fitted Curve, R-Squared = {r_squared}')


        plt.xlabel('Time (days)')
        plt.ylabel('Robustness (LCC)')
        # plt.title('Logarithmic Decay Curve Fitting')
        plt.legend()
        plt.grid(True)
        plt.close(fig)
        # plt.show()

        seed_num = seed_list[j]

        fig_filename = f'hous_size_10_diff_impacts_level15_{seed_num}_updateLogFit.png'
        fig.savefig(fig_filename)



if __name__ == "__main__":

    repository_name = "cmudrc/incose-paper-data"

    T_points_list = []
    for k in range(3000):
        for i in range(1, 4):
            T_points_list.append((i + 13 * k) * 7200)

    
    hous_size_1_diff_impacts_level15(T_points_list, repository_name)
    hous_size_5_diff_impacts_level15(T_points_list, repository_name)
    hous_size_10_diff_impacts_level15(T_points_list, repository_name)
