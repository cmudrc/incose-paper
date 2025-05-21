import os


path_analysis = os.path.dirname(os.path.realpath(__file__))  # We are now here!
path_root = os.path.dirname(path_analysis)  # Project root diretory
path_result = path_main = os.path.join(os.path.join(path_root, 'main'), 'result')  # result path
print("Result: ", path_result)

scenario_name = '0_1_2_3'  # Example name
path_scenario = os.path.join(path_result, scenario_name)
print("Scenario: ", path_scenario)