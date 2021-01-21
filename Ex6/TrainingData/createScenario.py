import json
import copy
import numpy as np

def generate_default_pedestrian():
    """
    Generates an initialized pedestrian object to be added to the scenario json

    :returns:
        return: an initialized pedestrian as a dictionary
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


def add_pedestrians(coord, i):
    path_to_scenario = "./ScenarioFiles/SimpleScenario.scenario"

    with open(path_to_scenario, 'r') as myfile:
        data = myfile.read()
        # parse file
        obj = json.loads(data)
        obj["name"] += str(int(i/2))
        obj["scenario"]["topography"]["dynamicElements"][0]["position"]["x"]= 1 + 4 * float(coord[0])
        obj["scenario"]["topography"]["dynamicElements"][0]["position"]["y"] = 1 + 4 * float(coord[1])
    path_to_scenario = path_to_scenario.replace(".scenario", "_"+str(int(i/2))+".scenario")


    # Save to a new json file
    with open(path_to_scenario, 'w') as outfile:
        json.dump(obj, outfile, indent=4)

# generate random number
coords =np.random.uniform(size=4000)
print(len(coords))
for coord in range(0, len(coords), 2):
    if coords[coord] > 1 or coords[coord+1] > 1:
        print("HERE")
    add_pedestrians([coords[coord], coords[coord+1]], coord)