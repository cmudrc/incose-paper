import os
import numpy as np
import csv
import piperabm as pa

from info import *


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

def extract_data(path):
    data = []
    for population in populations:
        for household_size in household_sizes:
            for impact in impacts:
                for seed in seeds:
                    setup = {
                        'population': population,
                        'household_size': household_size,
                        'seed': seed,
                        'impact': impact
                    }
                    name = setup_to_name(setup)
                    measurement = pa.Measurement(
                        path=path,
                        name=name
                    )
                    measurement.load()
                    entry = []
                    avg = measurement.accessibility.average()
                    #if isinstance(avg, float):
                    #    pass
                    #elif isinstance(avg, complex):
                    #    avg = float(avg.real)
                    entry.append(avg) # y
                    entry.append(population) # X_1
                    entry.append(household_size) # X_2
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
                    entry.append(random_impact) # X_3
                    entry.append(critical_impact) # X_4
                    data.append(entry)
    header = [
        'Accessibility', # y
        'Population', # X_1
        'Household Size', # X_2
        'Random Impact', # X_3
        'Critical Impact' # X_4
    ]
    return data, header

def to_csv(path, data, header):
    filepath = os.path.join(path, 'result.csv')
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in data:
            writer.writerow(row)

def main():
    # Report
    print(">>> saving the extracted results...\n")

    path = os.path.dirname(os.path.realpath(__file__))
    data, header = extract_data(path)
    to_csv(path, data, header)
    
    # Report
    print(">>> extracted results have been saved successfully.\n")


if __name__ == "__main__":
    main()