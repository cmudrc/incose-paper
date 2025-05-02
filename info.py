import numpy as np
from copy import deepcopy


populations = [10, 20, 30, 40] # Agents
household_sizes = [1, 5, 10] # Agents/Homes
impact_types = ['critical', 'random']
impact_levels = [5, 15, 25]
repetitions = 3 # How many times to repeat each experiment
steps = 1000
'''
populations = [10, 15] # Agents
household_sizes = [1, 5] # Agents/Homes
impact_types = ['critical', 'random']
impact_levels = [5, 15]
repetitions = 2 # How many times to repeat each experiment
steps = 10
'''
impacts = [None] + [[impact_type, level] for impact_type in impact_types for level in impact_levels]


# General
grid_size = [150, 100] # [x, y] in meters
grid_num = [6, 6] # [x, y]
initial_imperfection = 0 # %
infrastructure_coeff_usage = 0.3
neighbor_radius = max(grid_size) * max(grid_num) # Ensure everyone is neighbor
max_time_outside = 8 * 3600 # seconds
activity_cycle = 24 * 3600 # seconds
gini_index = 0.45 # inequality
average_income = 3000 / (30 * 24 * 3600) # $/s
average_balance = 0 # $
average_resources = {'food': 20,'water': 15, 'energy': 10} # kg
transportation_resource_rates = {'food': 4 / (24 * 3600),'water': 3 / (24 * 3600), 'energy': 2 / (24 * 3600)} # kg/s
idle_resource_rates = {'food': 2 / (24 * 3600),'water': 1.5 / (24 * 3600), 'energy': 1 / (24 * 3600)} # kg/s
speed = 5 * ((1000) / (60 * 60)) # km/hour to m/s
prices = {'food': 10,'water': 10, 'energy': 10} # $/kg
market_pos = [0, 0] # [x, y] in meters
market_resource_factor = 10 # market average times agents average


# Setup
step_size = 2 * 3600 # seconds


# Tools
master_seed = 1
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
                        name = str(p) + '_' + str(h) + '_' + str(s) + '_' + str(i)
                        names.append(name)
    return names

names_impacted = create_names(impacted=True)
names_unimpacted = create_names(impacted=False)
names = names_impacted + names_unimpacted

def name_to_setup(name: str):
    """
    Convert name to setup
    """
    name_splitted = name.split(sep='_')
    return {
        'population': populations[int(name_splitted[0])],
        'household_size': household_sizes[int(name_splitted[1])],
        'seed': seeds[int(name_splitted[2])],
        'impact': impacts[int(name_splitted[3])]
    }

def setup_to_name(setup):
    """
    Convert setup to name
    """
    p = populations.index(setup['population'])
    h = household_sizes.index(setup['household_size'])
    s = seeds.index(setup['seed'])
    i = impacts.index(setup['impact'])
    return str(p) + '_' + str(h) + '_' + str(s) + '_' + str(i)

def unimpacted_to_impacteds(unimpacted_name):
    """
    Convert unimpacted name to corresponding impacted_names
    """
    impacteds = []
    unimpacted_setup = name_to_setup(unimpacted_name)
    for impact in impacts:
        if impact is not None:
            impacted_setup = deepcopy(unimpacted_setup)
            impacted_setup['impact'] = impact
            impacteds.append(impacted_setup)
    return impacteds

def calculate_homes_num(population, household_size):
    """
    Calculate number of homes
    """
    return int(np.floor(population / household_size))


if __name__ == "__main__":
    size = len(populations) * len(household_sizes) * len(impacts) * len(seeds)
    print(size) # Number of distinct models
