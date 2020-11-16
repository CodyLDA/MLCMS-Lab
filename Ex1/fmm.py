import numpy as np
import skfmm


def fmm(initial, obstacles, width, height):
    phi = np.ones((width, height))
    phi[initial[0], initial[1]] = -1
    dist_mat = skfmm.distance(phi)
    dist_map = {}
    for i in range(width):
        for j in range(height):
            dist_map[str(i)+str(j)] = dist_mat[i,j]
    for obst in obstacles:
        dist_map[str(obst[0]) + str(obst[1])] = float('inf')
    # print(dist_map)
    return dist_map
