"""
Setup information for the study.
"""

import numpy as np
from copy import deepcopy


populations = [10]  # Agents population
household_sizes = [5]  # Agents/Homes (household size)
impact_types = ["critical", "random"]
impact_levels = [5, 15]  # % of the elements removed
repetitions = 1  # How many times to repeat each experiment
steps = 500  # Number of steps tp run the simulation
impacts = [None] + [[impact_type, level] for impact_type in impact_types for level in impact_levels]


# General
grid_size = [150, 100]  # [x_size, y_size] grid size in meters
grid_num = [6, 6]  # [x_num, y_num] number of grids
initial_imperfection = 0  # % pf random imperfections in the grid
infrastructure_coeff_usage = 0.3
neighbor_radius = max(grid_size) * max(grid_num)  # Ensure everyone is neighbor
max_time_outside = 8 * 60 * 60  # Lnegth of activity per day in seconds
activity_cycle = 24 * 60 * 60  # Length of day in seconds
gini_index = 0.45  # Inequality
average_income = 3000 / (30 * 24 * 60 * 60) #  $/seconds
average_balance = 0  # Average initial balance of agents in $
average_resources = {"food": 20,"water": 15, "energy": 10}  # Average initial amount of resources in kg
transportation_resource_rates = {"food": 4 / (24 * 3600),"water": 3 / (24 * 3600), "energy": 2 / (24 * 3600)} # kg/s
idle_resource_rates = {"food": 2 / (24 * 3600),"water": 1.5 / (24 * 3600), "energy": 1 / (24 * 3600)} # kg/s
speed = 5 * ((1000) / (60 * 60)) # km/hour to m/s
prices = {"food": 10,"water": 10, "energy": 10} # $/kg
market_pos = [0, 0]  # Market node position [x, y] in meters
market_resource_factor = 10  # Market initial resources is 10 times the total agents avergae


# Setup
step_size = 4 * 3600  # Step size in seconds


# Tools
master_seed = 1  # Seed for reproducibility
np.random.seed(master_seed)
seeds = [int(np.random.randint(low=0, high=np.iinfo(np.int8).max, dtype=np.int8)) for _ in range(repetitions)]
np.random.seed(None)

def create_names(impacted: bool):
    """
    Create list of names
    """
    names = []
    for p in range(len(populations)):
        for h in range(len(household_sizes)):
            for s in range(len(seeds)):
                for i in range(len(impacts)):
                    create = False
                    if impacted is True:
                        if impacts[i] != None:
                            create = True
                    else:
                        if impacts[i] == None:
                            create = True
                    if create is True:
                        name = str(p) + "_" + str(h) + "_" + str(s) + "_" + str(i)
                        names.append(name)
    return names

names_impacted = create_names(impacted=True)  # Impacted models names
names_unimpacted = create_names(impacted=False)  # Unimpacted models names
names = names_impacted + names_unimpacted  # All models names

def name_to_setup(name: str):
    """
    Convert name to setup
    """
    name_splitted = name.split(sep="_")
    return {
        "population": populations[int(name_splitted[0])],
        "household_size": household_sizes[int(name_splitted[1])],
        "seed": seeds[int(name_splitted[2])],
        "impact": impacts[int(name_splitted[3])]
    }

def setup_to_name(setup):
    """
    Convert setup to name
    """
    p = populations.index(setup["population"])
    h = household_sizes.index(setup["household_size"])
    s = seeds.index(setup["seed"])
    i = impacts.index(setup["impact"])
    return str(p) + "_" + str(h) + "_" + str(s) + "_" + str(i)

def unimpacted_to_impacteds(unimpacted_name):
    """
    Convert unimpacted name to corresponding impacted_names
    """
    impacteds = []
    unimpacted_setup = name_to_setup(unimpacted_name)
    for impact in impacts:
        if impact is not None:
            impacted_setup = deepcopy(unimpacted_setup)
            impacted_setup["impact"] = impact
            impacteds.append(impacted_setup)
    return impacteds

def calculate_homes_num(population, household_size):
    """
    Calculate number of homes
    """
    return int(np.floor(population / household_size))


if __name__ == "__main__":
    size = len(populations) * len(household_sizes) * len(impacts) * len(seeds)
    print(f"Number of simulations: {size}") # Number of distinct models
