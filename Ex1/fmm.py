import numpy as np
import skfmm

from collections import defaultdict
import numpy as np


class Graph_fmm:
  def __init__(self, width, height, obstacles):
    # Generate the graph with the nodes and the edges 

    self.nodes = set()
    self.edges = defaultdict(list)
    self.distances = {}
    self.obs = []

    for obst in obstacles:
        self.obs.append(str(obst[0]) +','+ str(obst[1]))

    for i in range(width):
        for j in range(height):
            node_name = str(i)+','+str(j)
            if node_name not in self.obs:
                self.add_node(node_name)
                if i+1 < width:
                    if str(i+1)+','+str(j) not in self.obs:
                        self.add_edge(node_name, str(i+1)+','+str(j))
                    if j+1 < height:
                        if str(i + 1) +','+ str(j + 1) not in self.obs:
                            self.add_edge(node_name, str(i + 1) +','+ str(j + 1))
                    if j-1 >= 0:
                        if str(i + 1) +','+ str(j - 1) not in self.obs:
                            self.add_edge(node_name, str(i + 1) +','+ str(j - 1))

                if i-1 >= 0:
                    if str(i-1)+','+str(j) not in self.obs:
                        self.add_edge(node_name, str(i-1)+','+str(j))
                    if j+1 < height:
                        if str(i - 1) +','+ str(j + 1) not in self.obs:
                            self.add_edge(node_name, str(i - 1) +','+ str(j + 1))

                    if j-1 >= 0:
                        if str(i - 1) +','+ str(j - 1) not in self.obs:
                            self.add_edge(node_name, str(i - 1) +','+ str(j - 1))

                if j+1 < height:
                    if str(i) +','+ str(j + 1) not in self.obs:
                        self.add_edge(node_name, str(i) +','+ str(j + 1))

                if j - 1 >= 0:
                    if str(i) +','+ str(j - 1) not in self.obs:
                        self.add_edge(node_name, str(i) +','+ str(j - 1))

  def add_node(self, value):
    self.nodes.add(value)

  def add_edge(self, from_node, to_node):
    self.edges[from_node].append(to_node)
    self.edges[to_node].append(from_node)



def _fmm(initial, width, height, obstacles):
     # Calculate the distance using the FMM algorithm
     # Initialize distance array
     phi = np.ones((width, height))
     X, Y = np.meshgrid(np.linspace(0,width-1, width), np.linspace(0,height-1, height))
     
     # Mask the cells containing obstacles
     for obst in obstacles:
         mask = np.logical_and(X == obst[0], Y == obst[1])
         phi  = np.ma.MaskedArray(phi, mask)
     # Initialize the zero contour of phi
     phi[initial[0], initial[1]] = -1
     # Compute the distance
     dist_mat = skfmm.distance(phi)
     dist_map = {}
     # Save in a dictionary
     for i in range(width):
         for j in range(height):
             dist_map[str(i)+','+str(j)] = dist_mat[i,j]
     return dist_map

def fmm(initial, obstacles, width, height):
  # generate graph
  graph = Graph_fmm(width, height, obstacles)
  # Transform to int in case of string for adaptability reasons
  initial = [int(initial[0]), int(initial[1])]

  # Initialize search algorithm
  visited = {str(initial[0])+','+str(initial[1]): 0}

  nodes = set(graph.nodes)
  # Create distance map
  dist_map = _fmm(initial, width, height, obstacles)

  while nodes:
    min_node = None
    for node in nodes:
      if node in visited:
        if min_node is None:
          min_node = node
        elif visited[node] < visited[min_node]:
          min_node = node

    if min_node is None:
      break

    nodes.remove(min_node)
    current_weight = visited[min_node]

    for edge in graph.edges[min_node]:

      # Use FMM distance for the weights
      weight = current_weight + dist_map[edge]
      if edge not in visited or weight < visited[edge]:
        visited[edge] = weight

  return visited
