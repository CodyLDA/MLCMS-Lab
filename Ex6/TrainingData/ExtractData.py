import glob
import numpy
from numpy import save

outputs = glob.glob("OutputFiles/*")
for output in outputs:
    print(output)
    simFile = output+"\\postvis.traj"
    data = numpy.loadtxt(simFile, skiprows=1).reshape(-1,8)
    print(data)
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
        path = "TrajArr/"+output+".npy"
        print(path)
        save(path, arr)