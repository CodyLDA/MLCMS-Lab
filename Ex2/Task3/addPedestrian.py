import json
import copy
import numpy as np


def generate_default_pedestrian():
    """
    Generates an initialized pedestrian object to be added to the scenario json
    :return: an initialized pedestrian
    """
    false = False
    null = None
    return {
        "attributes": {
            "id": 8,
            "radius": 0.2,
            "densityDependentSpeed": false,
            "speedDistributionMean": 1.34,
            "speedDistributionStandardDeviation": 0.26,
            "minimumSpeed": 0.5,
            "maximumSpeed": 2.2,
            "acceleration": 2.0,
            "footstepHistorySize": 4,
            "searchRadius": 1.0,
            "walkingDirectionCalculation": "BY_TARGET_CENTER",
            "walkingDirectionSameIfAngleLessOrEqual": 45.0
        },
        "source": null,
        "targetIds": [1],
        "nextTargetListIndex": 0,
        "isCurrentTargetAnAgent": false,
        "position": {
            "x": 5.7,
            "y": 1.2
        },
        "velocity": {
            "x": 0.0,
            "y": 0.0
        },
        "freeFlowSpeed": 1.6239471913829229,
        "followers": [],
        "idAsTarget": -1,
        "isChild": false,
        "isLikelyInjured": false,
        "psychologyStatus": {
            "mostImportantStimulus": null,
            "threatMemory": {
                "allThreats": [],
                "latestThreatUnhandled": false
            },
            "selfCategory": "TARGET_ORIENTED",
            "groupMembership": "OUT_GROUP",
            "knowledgeBase": {
                "knowledge": []
            }
        },
        "groupIds": [],
        "groupSizes": [],
        "trajectory": {
            "footSteps": []
        },
        "modelPedestrianMap": null,
        "type": "PEDESTRIAN"
    }

# Path to scenario to be modified
path_to_scenario = "/content/drive/MyDrive/RiMEA6.scenario"
# read file
with open(path_to_scenario, 'r') as myfile:
    data = myfile.read()

# parse file
obj = json.loads(data)
obj["name"] = obj["name"]+"_new"

# calculate new x and y coordinates
new_pedestrian_coord_y = obj["scenario"]["topography"]["sources"][0]["shape"]["y"] + obj["scenario"]["topography"]["sources"][0]["shape"]["height"]/2
new_pedestrian_coord_x = obj["scenario"]["topography"]["targets"][0]["shape"]["x"] + obj["scenario"]["topography"]["targets"][0]["shape"]["width"]/2
new_pedestrian = generate_default_pedestrian()

# Find already used pedestrian Ids to avoid confusions
ids = []
for dynamicattr in obj["scenario"]["topography"]["dynamicElements"]:
    if dynamicattr["type"] == "PEDESTRIAN":
        ids.append(dynamicattr["attributes"]["id"])
if len(ids) > 0:
  max_id = np.max(ids)
else:
  max_id = 0
# Find target id  
targetid = obj["scenario"]["topography"]["sources"][0]["targetIds"]
# Set new pedestrian's target id
new_pedestrian["targetIds"] = targetid
# Set new pedestrian's id
new_pedestrian["attributes"]["id"] = int(max_id + 1)
# Set new pedestrian's x and y coordinates
new_pedestrian["position"]["x"] = new_pedestrian_coord_x
new_pedestrian["position"]["y"] = new_pedestrian_coord_y
# Add pedestrian to the scenario's dictionary
obj["scenario"]["topography"]["dynamicElements"].append(new_pedestrian)

# Save to a new json file
with open(path_to_scenario.replace(".scenario","_new.scenario"), 'w') as outfile:
    json.dump(obj, outfile, indent=4)

