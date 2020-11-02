import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from itertools import chain


class Pedestrian:

    def __init__(self, position, target):
        if isinstance(position, tuple) and len(position) == 2:
            self.current_x = position[0]
            self.current_y = position[1]

            self.trajectory = []
            self.target = target

    def __str__(self):
        return f'Current_pos: {self.current_x} | {self.current_y} with target: {self.target}\n'

    def __repr__(self):
        return self.__str__()


class Automaton:

    def __init__(self, grid_size, pedestrians, targets, obstables):
        if isinstance(grid_size, tuple) and len(grid_size) == 2:
            self.height = grid_size[0]
            self.width = grid_size[1]

            self.distanceMaps = self.calculateDistanceMaps(targets, obstables)

            if len(targets) == 1:
                targets = len(pedestrians) * targets
            self.pedestrians = self.createPedestrians(pedestrians, targets)

            self.PEDES_COLOR = 'red'
            self.TARGET_COLOR = 'yellow'
            self.OBSTACLE_COLOR = 'blue'

    def createPedestrians(self, pedestrians, targets):
        return [Pedestrian(position, target) for position, target in zip(pedestrians, targets)]

    def calculateDistanceMaps(self, targets, obstacles):
        different_target = []
        distance_maps = dict()
        for target in targets:
            if target not in different_target:
                different_target.append(target)
                distance_maps[target] = self.calculateDistance(target)
        return distance_maps

    def calculateDistance(self, target):
        def single_distance(x, y):
            return math.sqrt((x - target[0]) ** 2 + (y - target[1]) ** 2)
        return np.array([single_distance(i, j) for i in range(self.height) for j in range(self.width)])

    def generateGraphic(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for rect, letter_descr in self.getFigures():
            ax.add_patch(rect)
            rect_coor_x, rect_coor_y = rect.get_xy()
            descr_coor_x = rect_coor_x + rect.get_width() / 2
            descr_coor_y = rect_coor_y + rect.get_height() / 2
            ax.annotate(letter_descr, (descr_coor_x, descr_coor_y), color='black', weight='bold', ha='center', va='center')

        ax.add_patch(Rectangle((0, 0), self.width, self.height, fill=None, alpha=1))
        plt.xlim([-0.5, self.width + 0.5])
        plt.ylim([-0.5, self.height + 0.5])
        plt.axis('off')
        plt.show()

    def getFigures(self):
        def draw_rect(x, y, color):
            return Rectangle((y, self.height - 1 - x), 1, 1, color=color)
        pedes_generator = ((draw_rect(pedes.current_x, pedes.current_y, self.PEDES_COLOR), 'P') for pedes in self.pedestrians)
        target_generator = ((draw_rect(target_x, target_y, self.TARGET_COLOR), 'T') for target_x, target_y in self.distanceMaps)
        return chain(pedes_generator, target_generator)


if __name__ == '__main__':
    width = 50
    height = 40
    tt = Automaton(grid_size=(height, width), pedestrians=[(0, 0), (1, 3), (2, 2)], targets=[(3, 3)], obstables=[])
    print(tt.distanceMaps)
    print(tt.pedestrians)

    tt.generateGraphic()

    #https://stackoverflow.com/questions/11545062/matplotlib-autoscale-axes-to-include-annotations