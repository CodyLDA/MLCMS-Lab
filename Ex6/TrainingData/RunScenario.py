import os
import glob

scenarios = glob.glob("ScenarioFiles/*.scenario")
doneScenarios = glob.glob("OutputFiles/*")
for scenario in scenarios:
    done = False
    for doneScenario in doneScenarios:
        doneScenario = doneScenario.split('_')[0].replace("OutputFiles", "")
        scenarioName = scenario.split('.')[0].replace("ScenarioFiles", "")
        scenarioName = scenarioName.replace("_", "")
        #print(doneScenario)
        #print(scenarioName)
        if doneScenario == scenarioName:
            print("HERE")
            done = True
            break
    if not done:
        outputPath = "OutputFiles"
        unknown_dir = os.system("java -jar C:/Users/cyrin/Downloads/vadere/VadereSimulator/target/vadere-console.jar scenario-run --scenario-file \""+scenario+ "\" "+ "--output-dir=\""+ outputPath+ "\"")
