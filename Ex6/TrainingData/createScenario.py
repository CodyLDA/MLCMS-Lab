import json
import numpy as np


def add_pedestrians(coord, i):
    '''
    Modify the coordinates of the pedestrian in the scenario to generate a new one.
    :param coord: coordinates samples randomly
    :param i: samples number
    :return: json file of the new scanrio
    '''
    path_to_scenario = "./ScenarioFiles/SimpleScenario.scenario"

    with open(path_to_scenario, 'r') as myfile:
        data = myfile.read()
        # parse file
        obj = json.loads(data)
        obj["name"] += str(int(i/2))
        ## Get x and y coordinates for pedestrian
        obj["scenario"]["topography"]["dynamicElements"][0]["position"]["x"]= 1 + 14 * float(coord[0])
        obj["scenario"]["topography"]["dynamicElements"][0]["position"]["y"] = 1 + 14 * float(coord[1])
    path_to_scenario = path_to_scenario.replace(".scenario", "_"+str(int(i/2))+".scenario")


    # Save to a new json file
    with open(path_to_scenario, 'w') as outfile:
        json.dump(obj, outfile, indent=4)


# generate random number
coords =np.random.uniform(size=4000)

for coord in range(0, len(coords), 2):
    add_pedestrians([coords[coord], coords[coord+1]], coord)
