import math
from itertools import product
from dijsktra import *
from fmm import *


class Component:
    """ used as Base class for the visualization """

    def __init__(self, position, drawConfig):
        if isinstance(position, tuple) and len(position) == 2:
            self.current_x = position[0]
            self.current_y = position[1]
            self.drawConfig = drawConfig

    def label(self):
        """ the label of the class seen in visualization """
        pass

    def color(self):
        pass

    def getDrawInfo(self):
        """ collect all the important information for graphics """
        return self.current_x, self.current_y, self.color(), self.label()


class Pedestrian(Component):

    def __init__(self, position, drawConfig, target, id):
        super().__init__(position, drawConfig)
        self.trajectory = []  # used for visualizing the path a pedestrian followed
        self.target = target  # if there are more than 1 targets in the scenario
        self.id = id  # id of the pedestrian, used as a unique identifier for each pedestrian
        self.at_goal = False # set to true if pedestrian has reach the target (by being in one of the cells surrounding it)
        self.nextStepTime = 0
        self.plannedStep = (0, 0)

    def label(self):
        return 'P'

    def color(self):
        return 'red'

    def __str__(self):
        return f'Current_pos: {self.current_x} | {self.current_y} with target: {self.target}\n'

    def __repr__(self):
        return self.__str__()


class Target(Component):

    def __init__(self, position, drawConfig, distanceMap):
        super().__init__(position, drawConfig)
        self.distanceMap = distanceMap # Saves the distance of each cell to the target
        self.target_env = [] # Saves the coordinates/graph nodes of the cells surrounding the target

    def label(self):
        return 'T'

    def color(self):
        return 'yellow'


class Obstacle(Component):

    def __init__(self, position, drawConfig):
        super().__init__(position, drawConfig)

    def label(self):
        return 'O'

    def color(self):
        return 'blue'


class Automaton:
    """ the actual cellular automaton """

    PEDES = ('red', 'P')
    TARGET = ('yellow', 'T')
    OBSTACLE = ('blue', 'O')
    EMPTY = ('white', '')
    DIJKSTRA = 'dijkstra'
    FMM = 'fmm'

    def __init__(self, grid_size, pedestrians, targets, obstables, used_algo, dmax):
        if isinstance(grid_size, tuple) and len(grid_size) == 2:
            self.width, self.height = grid_size[0], grid_size[1]

            if len(targets) == 1:  # if only one target given, this target is set for all pedestrians
                targets = len(pedestrians) * targets
            # creating the objects for visualization
            self.pedestrians = self.createPedestrians(pedestrians, targets)
            self.obstacles = self.createObstacles(obstables)
            # creating the graph that contains all the nodes (cells without obstacles) and the edges between them
            self.board = Graph(self.width, self.height, obstables)
            # saves the obstacles as an attribute
            self.obst = obstables
            # the algorithm chosen by the user (Dijkstra/FMM)
            self.algo = used_algo
            # saves the distance of each cell to the target
            self.distanceMaps = self.calculateDistanceMaps(targets)
            self.step_num = 0
            # saves the current predestrian coordinates on the grid
            self.pedes_coord = {}
            # initialize the dictionary with the coordinates chosen by the user
            for ped in self.pedestrians:
                #self.pedes_coord[ped.id] = str(ped.current_x) +','+ str(ped.current_y)
                self.pedes_coord[ped.id] = ped
            self.stepCounter = 0
            self.dmax = dmax

    def createPedestrians(self, pedestrians, targets):
        index = [i for i in range(len(pedestrians))]
        return [Pedestrian(position, Automaton.PEDES, target, id) for position, target, id in zip(pedestrians, targets, index)]

    def createObstacles(self, obstacles):
        return [Obstacle(position, Automaton.OBSTACLE) for position in obstacles]

    def calculateDistanceMaps(self, targets):
        different_target = []
        distance_maps = dict()
        for target in targets:
            if target not in different_target:
                different_target.append(target)
                # Generate distance map with the chosen algorithm
                if self.algo == Automaton.DIJKSTRA:
                    distancemap, _ = dijsktra(self.board, str(target[0]) +','+ str(target[1]))
                elif self.algo == Automaton.FMM:
                    distancemap = fmm(target, self.obst, self.width, self.height)
                # Create target object with the distancemap attribute
                distance_maps[target] = Target((target[0], target[1]), Automaton.TARGET, distancemap)
                # Save the target neighbors in the target_env attribute
                distance_maps[target].target_env = self.calculateTargetNeighbors(target)
                
        return distance_maps

    def calculateTargetNeighbors(self, target):
        # Find the neighbouring cells to the target and save them in a list. This is used later on to check
        # if a pedestrian has reached the target
        fields = list(self.board.nodes)
        target_env = []
        for i, j in product([0, 1, -1], repeat=2):
            if str(target[0] + i) +','+ str(target[1] + j) in fields:
                target_env.append(str(target[0] + i) +','+ str(target[1] + j))
        return target_env

    def calculateDistance(self, target, obstacles):
        # DEPRECATED FUNCTION, not in use anymore
        # Used to generate the rudimentary cost function, where the distance is set to 'inf' if an obstacle is in the cell
        def single_distance(x, y):
            return math.sqrt((x - target[0]) ** 2 + (y - target[1]) ** 2)

        distancemap = np.array([single_distance(i, j) for i in range(self.width) for j in range(self.height)]).reshape(
            (self.width, self.height))
        for obst in obstacles:
            distancemap[obst[0]][obst[1]] = float('inf')
        return Target((target[0], target[1]), Automaton.TARGET, distancemap)

    def step(self):
        # Perform a step
        # Minimum allowed distance between Pedestrians
        dmax = self.dmax
        # Retrieve a list of the valid board coordinates (in bound and do not contain an obstacles)
        fields = list(self.board.nodes)
        for pedes in self.pedestrians:
            if not pedes.at_goal and pedes.nextStepTime == self.stepCounter:
                if pedes.nextStepTime > 0:
                    pedes.trajectory.append((pedes.current_x, pedes.current_y))
                    pedes.current_x += pedes.plannedStep[0]
                    pedes.current_y += pedes.plannedStep[1]
                    pedes.plannedStep = (0, 0)
                # initialize next cell coordinates and best distance
                smallest = ()
                best_distance = float('inf')
                
                # having a look at all the 9 neighbors of the current pedestrian
                for i, j in product([0, 1, -1], repeat=2):
                    if str(pedes.current_x + i)+','+str(pedes.current_y + j) in fields:  # staying in bound
                        # Calculate distance of cells to the pedestrian, used to check the distance between the pedestrians
                        if self.algo == Automaton.DIJKSTRA:
                            distance_map, _ = dijsktra(self.board, str(pedes.current_x + i) +','+ str(pedes.current_y + j))
                        elif self.algo == Automaton.FMM:
                            distance_map = fmm([pedes.current_x + i, pedes.current_y + j], self.obst, self.width,
                                               self.height)

                        dist = 0 # initialize the cost related to the distance between pedestrians
                        for k, other_pedes in list(self.pedes_coord.items()):
                          
                            if int(k) == int(pedes.id): # if the current pedestrian skip
                                pass
                            # if the cell is occupied set cost to inf
                            elif (pedes.current_x + i, pedes.current_y + j) == (other_pedes.current_x, other_pedes.current_y):
                                dist += float('inf')
                            elif (pedes.current_x + i, pedes.current_y + j) == (other_pedes.current_x + other_pedes.plannedStep[0], other_pedes.current_y + other_pedes.plannedStep[1]):
                                dist += float('inf')
                            elif distance_map[str(other_pedes.current_x) + "," + str(other_pedes.current_y)] < dmax:
                                # set the cost to the provided formula
                                dist += np.exp(1/(distance_map[str(other_pedes.current_x) + "," + str(other_pedes.current_y)]**2 - dmax**2))

                        # find the distance to the goal and add the cost for the distance to the pedestrians
                        distance = self.distanceMaps[pedes.target].distanceMap[str(pedes.current_x + i)+','+str(pedes.current_y + j)]

                        distance += dist
                        if distance < best_distance:
                            best_distance = distance
                            smallest = (i, j)


                pedes.plannedStep = (smallest[0], smallest[1])
                if abs(smallest[0]) + abs(smallest[1]) == 2:
                    pedes.nextStepTime += 10
                else:
                    pedes.nextStepTime += 7
                # update the pedestrian's coordinates
                self.pedes_coord[pedes.id] = pedes
                # If the predestrian is in the neighborhood of the target set the attribute at_goal to True
                if str(pedes.current_x)+','+str(pedes.current_y) in self.distanceMaps[pedes.target].target_env:
                    pedes.at_goal = True
                    pedes.plannedStep = (0, 0)
                    print("GOAL REACHED")
        self.stepCounter += 1
