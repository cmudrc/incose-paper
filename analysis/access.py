"""
For easier access to simulation data either locally or online
    - Online: https://huggingface.co/datasets/cmudrc/incose-paper-data
    - Local: main/result
"""
import os
import requests
import pandas as pd
from datasets import load_dataset

# For online access
repository_name = "cmudrc/incose-paper-data"

# For local access
path_analysis = os.path.dirname(os.path.realpath(__file__))  # We are now here!
path_root = os.path.dirname(path_analysis)  # Project root diretory
path_main = os.path.join(path_root, 'main')  # main path
path_result = os.path.join(path_main, 'result')  # result path


def access_scenario(scenario_name: str, file_name: str):
    """
    Try to access the scenarios data as dataframe locally or online.
    -> Suitable for .csv files

    if available:
        return dataframe and access type (='local')
    if not available:
        access it from the online repository,
        return dataframe and access type (='online')
        REMEMBER TO DELETE THE FOLDER AFTER USE
    """
    path_file = os.path.join(path_result, scenario_name, file_name)
    if os.path.exists(path_file):  # Local access
        return pd.read_csv(path_file)
    else:  # Online access
        path_file = f"{scenario_name}/{file_name}"
        dataset = load_dataset(
            repository_name,
            data_files=path_file,
        )
        return dataset['train'].to_pandas()
    
def access_scenario_agents(scenario_name: str):
    """
    Return the agents.csv file as dataframe for a given scenario
    """
    return access_scenario(scenario_name=scenario_name, file_name="agents.csv")

def access_scenario_transactions(scenario_name: str):
    """
    Return the transactions.csv file as dataframe for a given scenario
    """
    return access_scenario(scenario_name=scenario_name, file_name="transactions.csv")

def access_setups():
    """
    Access setups data
    """
    path_file = os.path.join(path_result, "setups.csv")
    if os.path.exists(path_file):  # Local access
        return pd.read_csv(path_file)
    else:  # Online access
        path_file = f"setups.csv"
        dataset = load_dataset(
            repository_name,
            data_files=path_file,
        )
        return dataset['train'].to_pandas()
    
def access_json(path_here: str, scenario_name: str, file_name: str):
    """
    Try to access the scenario files locally or online.
    -> Suitable for .json files

    if available:
        return path and access type (='local')
    if not available:
        download it from the online repository,
        return path and access type (='online')
        REMEMBER TO DELETE THE FOLDER AFTER USE
    """
    path_scenario_local = os.path.join(path_main, 'result', scenario_name)
    
    # Local access
    if not os.path.exists(path_scenario_local):
        access = 'local'
        path = path_main

    # Online access
    else:
        access = 'online'
        path = path_here

        # Create the folders if they do not exist
        folder_path = os.path.join(path_here, 'result')
        os.makedirs(folder_path, exist_ok=True)
        path_scenario_online = os.path.join(folder_path, scenario_name)
        os.makedirs(path_scenario_online, exist_ok=True)

        # Download measurement.json from f"{name}/" to path_scenario_online
        base_url = f"https://huggingface.co/datasets/{repository_name}/resolve/main"
        url = f"{base_url}/{scenario_name}/{file_name}"
        local_path = os.path.join(path_scenario_online, file_name)
        response = requests.get(url)
        if response.status_code == 200:
            with open(local_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {file_name} to {local_path}")
        else:
            raise RuntimeError(f"Failed to download {url}: {response.status_code}")
    
    return path, access


if __name__ == "__main__":
    # Scenario
    df = access_scenario_agents(scenario_name="0_0_0_0")
    print(df.head())

    # Setups
    df = access_setups()
    print(df.head())