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
        self.id = id
        self.at_goal = False
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
        self.distanceMap = distanceMap
        self.target_env = []

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

    def __init__(self, grid_size, pedestrians, targets, obstables, used_algo):
        if isinstance(grid_size, tuple) and len(grid_size) == 2:
            self.width, self.height = grid_size[0], grid_size[1]

            if len(targets) == 1:  # if only one target given, this target is set for all pedestrians
                targets = len(pedestrians) * targets
            # creating the objects for visualization
            self.pedestrians = self.createPedestrians(pedestrians, targets)
            self.obstacles = self.createObstacles(obstables)
            self.board = Graph(self.width, self.height, obstables)
            self.obst = obstables
            self.algo = used_algo
            self.distanceMaps = self.calculateDistanceMaps(targets)
            self.step_num = 0
            self.pedes_coord = {}
            for ped in self.pedestrians:
                #self.pedes_coord[ped.id] = str(ped.current_x) +','+ str(ped.current_y)
                self.pedes_coord[ped.id] = ped
            print(self.pedes_coord)
            self.stepCounter = 0

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
                # Uncomment to use fmm
                # distancemap = fmm(target, self.obst, self.width, self.height)
                # Using dijsktra
                # distancemap, _ = dijsktra(self.board, str(target[0])+str(target[1]))
                if self.algo == Automaton.DIJKSTRA:
                    distancemap, _ = dijsktra(self.board, str(target[0]) +','+ str(target[1]))
                elif self.algo == Automaton.FMM:
                    distancemap = fmm(target, self.obst, self.width, self.height)

                distance_maps[target] = Target((target[0], target[1]), Automaton.TARGET, distancemap)
                distance_maps[target].target_env = self.calculateTargetNeighbors(target)
                # distance_maps[target] = self.calculateDistance(target, obstacles)
        return distance_maps

    def calculateTargetNeighbors(self, target):
        fields = list(self.board.nodes)
        target_env = []
        for i, j in product([0, 1, -1], repeat=2):
            if str(target[0] + i) +','+ str(target[1] + j) in fields:
                target_env.append(str(target[0] + i) +','+ str(target[1] + j))
        return target_env

    def calculateDistance(self, target, obstacles):
        def single_distance(x, y):
            return math.sqrt((x - target[0]) ** 2 + (y - target[1]) ** 2)

        distancemap = np.array([single_distance(i, j) for i in range(self.width) for j in range(self.height)]).reshape(
            (self.width, self.height))
        for obst in obstacles:
            distancemap[obst[0]][obst[1]] = float('inf')
        return Target((target[0], target[1]), Automaton.TARGET, distancemap)

    def step(self):
        dmax = 1
        for pedes in self.pedestrians:
            if not pedes.at_goal and pedes.nextStepTime == self.stepCounter:
                if pedes.nextStepTime > 0:
                    pedes.trajectory.append((pedes.current_x, pedes.current_y))
                    pedes.current_x += pedes.plannedStep[0]
                    pedes.current_y += pedes.plannedStep[1]
                    pedes.plannedStep = (0, 0)
                smallest = ()
                best_distance = float('inf')
                fields = list(self.board.nodes)
                # print(fields)
                # print(pedes.id)
                # having a look at all the 9 neighbors of the current pedestrian
                for i, j in product([0, 1, -1], repeat=2):
                    if str(pedes.current_x + i)+','+str(pedes.current_y + j) in fields:  # staying in bound
                        # Uncomment to use fmm
                        # distance_map = fmm([pedes.current_x + i, pedes.current_y + j], self.obst, self.width, self.height)
                        # Using dijsktra
                        # distance_map, _ = dijsktra(self.board, str(pedes.current_x + i)+str(pedes.current_y + j))
                        # print(distance_map)
                        if self.algo == Automaton.DIJKSTRA:
                            distance_map, _ = dijsktra(self.board, str(pedes.current_x + i) +','+ str(pedes.current_y + j))
                        elif self.algo == Automaton.FMM:
                            distance_map = fmm([pedes.current_x + i, pedes.current_y + j], self.obst, self.width,
                                               self.height)

                        dist = 0
                        for k, other_pedes in list(self.pedes_coord.items()):
                            #other_pedes_x = int(self.pedes_coord[k].split(',')[0])
                            #other_pedes_y = int(self.pedes_coord[k].split(',')[1])
                            # print(distance_map[self.pedes_coord[k]])
                            if int(k) == int(pedes.id):
                                pass
                            #elif str(pedes.current_x + i)+','+str(pedes.current_y + j) == self.pedes_coord[k]:
                            elif (pedes.current_x + i, pedes.current_y + j) == (other_pedes.current_x, other_pedes.current_y):
                                dist += float('inf')
                            elif (pedes.current_x + i, pedes.current_y + j) == (other_pedes.current_x + other_pedes.plannedStep[0], other_pedes.current_y + other_pedes.plannedStep[1]):
                                dist += float('inf')
                            elif distance_map[str(other_pedes.current_x) + "," + str(other_pedes.current_y)] < dmax:
                                dist += np.exp(1/(distance_map[self.pedes_coord[k]]**2 - dmax**2))
                        distance = self.distanceMaps[pedes.target].distanceMap[str(pedes.current_x + i)+','+str(pedes.current_y + j)]
                        #print("printing coordinates")
                        #print(str(pedes.current_x + i)+','+str(pedes.current_y + j))
                        #print("Printing distance")
                        #1print(distance)
                        distance += dist
                        if distance < best_distance:
                            best_distance = distance
                            smallest = (i, j)
                #pedes.trajectory.append((pedes.current_x, pedes.current_y))
                #pedes.current_x += smallest[0]
                #pedes.current_y += smallest[1]
                pedes.plannedStep = (smallest[0], smallest[1])
                if abs(smallest[0]) + abs(smallest[1]) == 2:
                    pedes.nextStepTime += 10
                else:
                    pedes.nextStepTime += 7
                #self.pedes_coord[pedes.id] = str(pedes.current_x)+','+str(pedes.current_y)
                self.pedes_coord[pedes.id] = pedes
                # print(self.pedes_coord)
                # print(self.distanceMaps[pedes.target].target_env)
                if str(pedes.current_x)+','+str(pedes.current_y) in self.distanceMaps[pedes.target].target_env:
                    pedes.at_goal = True
                    pedes.plannedStep = (0, 0)
                    print("GOAL REACHED")
        self.stepCounter += 1
