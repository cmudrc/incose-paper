import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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


def create_table(
        path,
        impact: list = None
    ):
    tables = []
    ticks = [household_sizes, populations]
    labels = ['Household Sizes', 'Populations']
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
                measurement = pa.Measurement(
                    path=path,
                    name=name
                )
                measurement.load()
                avg = measurement.accessibility.average()
                if isinstance(avg, float):
                    pass
                elif isinstance(avg, complex):
                    avg = float(avg.real)
                row.append(avg)
            table.append(row)
        tables.append(table)
    #print(tables)
    result = average_tables(tables)
    return result, ticks, labels


def create_plot(
        path,
        impact: list = None
    ):
    result, ticks, labels = create_table(
        path=path,
        impact=impact
    )
    result = np.array(result)

    colors = ["red", "orange", "green"]  # Red to green
    n_bins = 100  # Increase this to make the transition smoother
    cmap_name = "my_custom_map"
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(result, interpolation='nearest', cmap=cm)
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(ticks[0])))
    ax.set_xticklabels(ticks[0])
    ax.set_yticks(np.arange(len(ticks[1])))
    ax.set_yticklabels(ticks[1])

    #ax.grid(which='both', color='gray', linestyle='-', linewidth=0.5)

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            ax.text(j, i, f"{result[i, j]:.2f}", ha='center', va='center', color='black')
    title = f"Average Accessibility"
    if impact is not None:
        title += f" (impact: {impact[0]}, percent: {str(impact[1])})"
    ax.set_title(title)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])

    return fig


def show(
        path,
        impact: list = None
    ):
    fig = create_plot(
        path=path,
        impact=impact
    )
    plt.show()


def save(
        path,
        impact: list = None,
    ):
    fig = create_plot(
        path=path,
        impact=impact
    )
    if impact is None:
        filename = 'base'
    else:
        filename = impact[0] + '_' + str(impact[1])
    filename += '.' + 'png'
    filepath = os.path.join(path, 'plots')
    os.makedirs(filepath, exist_ok=True)
    filepath = os.path.join(filepath, filename)
    fig.savefig(filepath)


def main():
    # Report
    print(">>> visualizing all results...\n")

    path = os.path.dirname(os.path.realpath(__file__))
    for impact in impacts:
        save(path=path, impact=impact)

    # Report
    print(">>> all results have been visualized successfully.\n")


if __name__ == "__main__":
    main()

