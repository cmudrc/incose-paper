"""
For easier access to simulation data either locally or online
    - Online: https://huggingface.co/datasets/cmudrc/incose-paper-data
    - Local: main/result
"""
import os
import pandas as pd
from datasets import load_dataset

# For online access
repo_name = "cmudrc/incose-paper-data"

# For local access
path_analysis = os.path.dirname(os.path.realpath(__file__))  # We are now here!
path_root = os.path.dirname(path_analysis)  # Project root diretory
path_result = path_main = os.path.join(os.path.join(path_root, 'main'), 'result')  # result path


def access_scenario(scenario: str, file: str):
    """
    Access scenario data
    """
    path_file = os.path.join(path_result, scenario, file)
    if os.path.exists(path_file):
        return pd.read_csv(path_file)
    else:
        path_file = f"{scenario}/{file}"
        dataset = load_dataset(
            repo_name,
            data_files=path_file,
        )
        return dataset['train'].to_pandas()

def access_setups():
    """
    Access setups data
    """
    path_file = os.path.join(path_result, "setups.csv")
    if os.path.exists(path_file):
        return pd.read_csv(path_file)
    else:
        path_file = f"setups.csv"
        dataset = load_dataset(
            repo_name,
            data_files=path_file,
        )
        return dataset['train'].to_pandas()


if __name__ == "__main__":
    df = access_scenario("0_0_0_0", "agents.csv")
    print(df.head())
    df = access_setups()
    print(df.head())