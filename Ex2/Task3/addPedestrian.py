import json
import copy

path_to_scenario = r".\scenarios\RiMEA6.scenario"
# read file
with open(path_to_scenario, 'r') as myfile:
    data = myfile.read()

# parse file
obj = json.loads(data)
obj["name"] = obj["name"]+"_new"

new_pedestrian_coord_y = obj["scenario"]["topography"]["sources"][0]["shape"]["y"] + obj["scenario"]["topography"]["sources"][0]["shape"]["height"]/2
new_pedestrian_coord_x = obj["scenario"]["topography"]["targets"][0]["shape"]["x"] + obj["scenario"]["topography"]["targets"][0]["shape"]["width"]/2
new_pedestrian = copy.deepcopy(obj["scenario"]["topography"]["dynamicElements"][0])
new_pedestrian["attributes"]["id"] = 9
new_pedestrian["position"]["x"] = new_pedestrian_coord_x
new_pedestrian["position"]["y"] = new_pedestrian_coord_y
obj["scenario"]["topography"]["dynamicElements"].append(new_pedestrian)

with open(path_to_scenario.replace(".scenario","_new.scenario"), 'w') as outfile:
    json.dump(obj, outfile, indent=4)
