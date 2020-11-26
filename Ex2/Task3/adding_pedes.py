import json
import copy
import numpy as np

path_to_scenario = r"C:\Users\leduc\Documents\Uni\Informatik\MLCMS\MLCMS-Lab\Ex2\Task4\VadereProject\scenarios\SRI_Template.scenario"
# read file
with open(path_to_scenario, 'r') as myfile:
    data = myfile.read()

# parse file
obj = json.loads(data)
obj["name"] = obj["name"]+"_newer"

drawing_area = obj["scenario"]['topography']['attributes']['bounds']
drawing_area['width'], drawing_area['height'] = 50, 50
width, height = drawing_area['width'], drawing_area['height']

pedestrian_template = copy.deepcopy(obj['scenario']['topography']['dynamicElements'][0])
obj['scenario']['topography']['dynamicElements'] = []

print(pedestrian_template)

for i in range(1000):
    new_pedes = copy.deepcopy(pedestrian_template)
    new_pedes["attributes"]["id"] = i
    new_pedes["position"]["x"] = np.random.rand() * width
    new_pedes["position"]["y"] = np.random.rand() * height
    obj['scenario']['topography']['dynamicElements'].append(new_pedes)

#new_pedestrian_coord_x, new_pedestrian_coord_y = np.random.rand() * width, np.random.rand() * height
#new_pedestrian = copy.deepcopy(obj["scenario"]["topography"]["dynamicElements"][0])
#new_pedestrian["attributes"]["id"] = 10
#new_pedestrian["position"]["x"] = new_pedestrian_coord_x
#new_pedestrian["position"]["y"] = new_pedestrian_coord_y
#obj["scenario"]["topography"]["dynamicElements"].append(new_pedestrian)

with open(path_to_scenario.replace(".scenario","_newer.scenario"), 'w') as outfile:
    json.dump(obj, outfile, indent=4)

if __name__ == '__main__':
    pass
