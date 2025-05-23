import sys
import os

# Append the root path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from analysis.access import access_setups, access_scenario_agents, access_scenario_transactions


# Setups
df = access_setups()
print(df.head())

# Agents
scenario_name = "0_0_0_0"
df = access_scenario_agents(scenario_name)
print(df.head())

# Transactions
scenario_name = "0_0_0_0"
df = access_scenario_transactions(scenario_name)
print(df.head())