import os
import glob

# Check all of the scenarios in the directory
scenarios = glob.glob("ScenarioFiles/*.scenario")
doneScenarios = glob.glob("OutputFiles/*")
# Go through the scenarios and run the simulation
for scenario in scenarios:
    done = False
    # Check whether scenario has already be simulated
    for doneScenario in doneScenarios:
        doneScenario = doneScenario.split('_')[0].replace("OutputFiles", "")
        scenarioName = scenario.split('.')[0].replace("ScenarioFiles", "")
        scenarioName = scenarioName.replace("_", "")
        if doneScenario == scenarioName:
            done = True
            break
    # If not done, run the simulation
    if not done:
        outputPath = "OutputFiles"
        unknown_dir = os.system("java -jar C:/Users/cyrin/Downloads/vadere/VadereSimulator/target/vadere-console.jar scenario-run --scenario-file \""+scenario+ "\" "+ "--output-dir=\""+ outputPath+ "\"")
