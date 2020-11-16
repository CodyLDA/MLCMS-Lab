import numpy as np
import skfmm


def fmm(initial, obstacles, width, height):
    phi = np.ones((width, height))
    phi[initial[0], initial[1]] = -1
    dist_mat = skfmm.distance(phi)
    for obs in obstacles:
        dist_mat[[obs[0], obs[1]]] = float('inf')
    dist_map = {}
    for i in range(width):
        for j in range(height):
            dist_map[str(i)+str(j)] = dist_mat[i,j]

    return dist_map
