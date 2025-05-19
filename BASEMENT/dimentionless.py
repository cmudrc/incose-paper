from info import *


# Total edges length
total_edges_length = (((grid_num[0] - 1) * grid_size[0]) * grid_num[1] + ((grid_num[1] - 1) * grid_size[1]) * grid_num[0]) * (1 - (initial_imperfection / 100))
# Model equivalent diameter
equivalent_diameter = (((grid_num[0] - 1) * grid_size[0]) ** 2 + ((grid_num[1] - 1) * grid_size[1]) ** 2) ** 0.5
# Average agent resource value
average_resource_value = prices['food'] * average_resources['food'] + prices['water'] * average_resources['water'] + prices['energy'] * average_resources['energy']
# Average transportation fuel value
transportation_consumption_value = prices['food'] * transportation_resource_rates['food'] + prices['water'] * transportation_resource_rates['water'] + prices['energy'] * transportation_resource_rates['energy']
# Average consumption fuel value
idle_consumption_value = prices['food'] * idle_resource_rates['food'] + prices['water'] * idle_resource_rates['water'] + prices['energy'] * idle_resource_rates['energy']
# Simualtion length
#simulation_length = steps * step_size


# transport / idle fuel ratio
c_1 = transportation_consumption_value / idle_consumption_value

#equivalent_diameter_cruise_time = equivalent_diameter / speed
#equivalent_diameter_cruise_fuel = (transportation_consumption_value + idle_consumption_value) * equivalent_diameter_cruise_time
#c_2 =  average_resource_value / equivalent_diameter_cruise_fuel




pi_1 = gini_index
pi_2 = steps
pi_3 = total_edges_length / equivalent_diameter
pi_4 = average_resource_value / (average_income * step_size)
pi_5 = transportation_consumption_value / idle_consumption_value
pi_6 = speed * step_size / equivalent_diameter

print(pi_1, pi_2, pi_3, pi_4, pi_5, pi_6)