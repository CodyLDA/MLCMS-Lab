from collections import defaultdict
import numpy as np

class Graph:
  def __init__(self, width, height, obstacles):
    self.nodes = set()
    self.edges = defaultdict(list)
    self.distances = {}
    self.obs = []

    for obst in obstacles:
        self.obs.append(str(obst[0]) + str(obst[1]))

    for i in range(width):
        for j in range(height):
            node_name = str(i)+str(j)
            if node_name not in self.obs:
                self.add_node(node_name)
                if i+1 < width:
                    if str(i+1)+str(j) not in self.obs:
                        self.add_edge(node_name, str(i+1)+str(j), 1)
                    if j+1 < height:
                        if str(i + 1) + str(j + 1) not in self.obs:
                            self.add_edge(node_name, str(i + 1) + str(j + 1), np.sqrt(2))
                    if j-1 >= 0:
                        if str(i + 1) + str(j - 1) not in self.obs:
                            self.add_edge(node_name, str(i + 1) + str(j - 1), np.sqrt(2))

                if i-1 >= 0:
                    if str(i-1)+str(j) not in self.obs:
                        self.add_edge(node_name, str(i-1)+str(j), 1)
                    if j+1 < height:
                        if str(i - 1) + str(j + 1) not in self.obs:
                            self.add_edge(node_name, str(i - 1) + str(j + 1), np.sqrt(2))

                    if j-1 >= 0:
                        if str(i - 1) + str(j - 1) not in self.obs:
                            self.add_edge(node_name, str(i - 1) + str(j - 1), np.sqrt(2))

                if j+1 < height:
                    if str(i) + str(j + 1) not in self.obs:
                        self.add_edge(node_name, str(i) + str(j + 1), 1)

                if j - 1 >= 0:
                    if str(i) + str(j - 1) not in self.obs:
                        self.add_edge(node_name, str(i) + str(j - 1), 1)

  def add_node(self, value):
    self.nodes.add(value)

  def add_edge(self, from_node, to_node, distance):
    self.edges[from_node].append(to_node)
    self.edges[to_node].append(from_node)
    self.distances[(from_node, to_node)] = distance


def dijsktra(graph, initial):
  visited = {initial: 0}
  path = {}

  nodes = set(graph.nodes)

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
      weight = current_weight + graph.distances[(min_node, edge)]
      if edge not in visited or weight < visited[edge]:
        visited[edge] = weight
        path[edge] = min_node

  return visited, path