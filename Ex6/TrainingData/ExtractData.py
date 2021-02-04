import glob
import numpy
from numpy import save

# Go through the output files from the simulated data
outputs = glob.glob("OutputFiles/*")
for output in outputs:
    # Read the output file
    simFile = output+"\\postvis.traj"
    data = numpy.loadtxt(simFile, skiprows=1).reshape(-1,8)
    if data.size == 0:
        pass
    else:
        init = [data[0,3], data[0,4]]
        pedesPath = []
        pedesPath.append(init)
        for i in range(data.shape[0]):
            pedesPath.append([data[i,5], data[i,6]])
        arr = numpy.array(pedesPath)
        output = output.replace("OutputFiles\\", "")
        # Save the trajectory array
        path = "TrajArr/"+output+".npy"
        save(path, arr)