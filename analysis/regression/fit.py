import sys
import os
import shutil
import requests
import numpy as np
import statsmodels.api as sm
import piperabm as pa

# Append the root path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from analysis.info import *


def average_tables(tables):
    """
    Compute the element-wise average of multiple 2D tables.

    Args:
    tables (list of list of list): A list where each element is a 2D table (list of lists).

    Returns:
    list of list: A 2D table representing the element-wise average of the input tables.
    """
    if not tables:
        return []  # Return an empty list if there are no tables

    # Get the dimensions of the tables
    num_rows = len(tables[0])
    num_cols = len(tables[0][0])

    # Initialize a table to store the sums of corresponding elements
    sum_table = [[0 for _ in range(num_cols)] for _ in range(num_rows)]

    # Sum up all elements from each table
    for table in tables:
        for i in range(num_rows):
            for j in range(num_cols):
                sum_table[i][j] += table[i][j]

    # Compute the average by dividing each element by the number of tables
    num_tables = len(tables)
    average_table = [[sum_table[i][j] / num_tables for j in range(num_cols)] for i in range(num_rows)]

    return average_table


def create_table(
        path,
        impact: list = None
    ):
    tables = []
    ticks = [household_sizes, populations]
    labels = ['Household Sizes', 'Populations']

    # Local
    path_here = os.path.dirname(os.path.realpath(__file__))  # We are now here!
    path_root = os.path.dirname(os.path.dirname(path_here))  # Project root diretory
    path_main_local = os.path.join(path_root, 'main')  # Main path
    #print("Result: ", path_result_local)

    for seed in seeds:
        table = []
        for population in populations:
            row = []
            for household_size in household_sizes:
                setup = {
                    'population': population,
                    'household_size': household_size,
                    'seed': seed,
                    'impact': impact
                }
                name = setup_to_name(setup)
                path_scenario_local = os.path.join(path_main_local, 'result', name)
                if os.path.exists(path_scenario_local):  # Local access
                    access = 'local'
                    path = path_main_local
                else:  # Online access
                    access = 'online'
                    folder_path = os.path.join(path, 'result')
                    os.makedirs(folder_path, exist_ok=True)
                    path_scenario_online = os.path.join(folder_path, name)
                    os.makedirs(path_scenario_online, exist_ok=True)
                    # Download measurement.json from f"{name}/" to path_scenario_online
                    base_url = "https://huggingface.co/datasets/cmudrc/incose-paper-data/resolve/main"
                    url = f"{base_url}/{name}/measurement.json"
                    local_path = os.path.join(path_scenario_online, "measurement.json")

                    response = requests.get(url)
                    if response.status_code == 200:
                        with open(local_path, 'wb') as f:
                            f.write(response.content)
                        print(f"Downloaded measurement.json to {local_path}")
                    else:
                        raise RuntimeError(f"Failed to download {url}: {response.status_code}")
                    path = path_here
                measurement = pa.Measurement(
                    path=path,
                    name=name
                )
                measurement.load()
                row.append(measurement.accessibility.average())
            table.append(row)
        tables.append(table)

    # Remove folder
    if access == 'online':
        shutil.rmtree(folder_path)

    #print(tables)
    result = average_tables(tables)
    return result, ticks, labels


def add(data, table, ticks, labels, impact):
    for i, row in enumerate(table):
        for j, accessibility in enumerate(row):
            household_size = ticks[0][j]
            population = ticks[1][i]
            if impact is None:
                random_impact = 0
                critical_impact = 0
            else:
                if impact[0] == 'critical':
                    random_impact = 0
                    critical_impact = impact[1]
                elif impact[0] == 'random':
                    random_impact = impact[1]
                    critical_impact = 0
                else:
                    raise ValueError
            result = [accessibility, population, household_size, critical_impact, random_impact]
            data.append(result)
    return data
        

def main():
    # Report
    print(">>> Running regression model...\n")

    path = os.path.dirname(os.path.realpath(__file__))
    data = []
    for impact in impacts:
        #save(path=path, impact=impact)
        table, ticks, labels = create_table(path, impact)
        data = add(data, table, ticks, labels, impact)
    data = np.array(data)
    X = data[:, 1:]  # all rows, all columns except the first
    y = data[:, 0]   # all rows, first column

    # Add a constant (intercept) to the model
    X = sm.add_constant(X)

    # Create and fit the linear regression model
    model = sm.OLS(y, X).fit()

    # Print the coefficients, intercept, and R-squared value
    print('Coefficients: \n', model.params[1:])  # Coefficients for each feature
    print('Intercept: \n', model.params[0])      # Intercept
    print('R^2: \n', model.rsquared)             # R-squared value
    print(model.summary())  # Detailed summary of the regression model

    # Report
    print(">>> Regression ran successfully.\n")


if __name__ == "__main__":
    main()