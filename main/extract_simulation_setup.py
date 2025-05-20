"""
Extract simulation setup names and save them to setups.csv
"""

import os
import csv

from info import *


def main():
    path = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(path, 'result')

    # Save model setups to file
    headers = ['name', 'population', 'household_size', 'seed', 'impact_type', 'impact_level']
    filename = os.path.join(filepath, 'setups.csv')
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
    for name in names:
        setup = name_to_setup(name)
        if setup['impact'] is None:
            impact_info = ['none', 'none']
        else:
            impact_info = setup['impact']
        row = [name, setup['population'], setup['household_size'], setup['seed'], impact_info[0], impact_info[1]]
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)


if __name__ == "__main__":
    main()