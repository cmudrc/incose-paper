"""
Extract society info and save them into agents.csv
"""

import os
import csv

import piperabm as pa

from info import *


def main():
    headers = ['agent_id', 'socioeconomic_status', 'household_id']
    path = os.path.dirname(os.path.realpath(__file__))
    result_path = os.path.join(path, 'result')
    for name in names:
        filepath = os.path.join(result_path, name)
        filename = os.path.join(filepath, 'agents.csv')
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
        model = pa.Model(path=path, name=name)
        model.load_final()
        for agent_id in model.society.agents:
            row = [
                agent_id,
                model.society.get_socioeconomic_status(id=agent_id),
                model.society.get_home_id(id=agent_id)
            ]
            with open(filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)


if __name__ == "__main__":
    main()
