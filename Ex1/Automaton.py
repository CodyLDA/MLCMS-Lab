import numpy as np
import math
from matplotlib.patches import Rectangle
from itertools import chain, product

import tkinter as tk
import tkinter.ttk as ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import matplotlib

matplotlib.use('TkAgg')


class Component:
    """ used as Base class for the visualization """

    def __init__(self, position):
        if isinstance(position, tuple) and len(position) == 2:
            self.current_x = position[0]
            self.current_y = position[1]

    def label(self):
        """ the label of the class seen in visualization """
        pass

    def color(self):
        pass

    def getDrawInfo(self):
        """ collect all the important information for graphics """
        return self.current_x, self.current_y, self.color(), self.label()


class Pedestrian(Component):

    def __init__(self, position, target):
        super().__init__(position)
        self.trajectory = []  # used for visualizing the path a pedestrian followed
        self.target = target  # if there are more than 1 targets in the scenario

    def label(self):
        return 'P'

    def color(self):
        return 'red'

    def __str__(self):
        return f'Current_pos: {self.current_x} | {self.current_y} with target: {self.target}\n'

    def __repr__(self):
        return self.__str__()


class Target(Component):

    def __init__(self, position, distanceMap):
        super().__init__(position)
        self.distanceMap = distanceMap

    def label(self):
        return 'T'

    def color(self):
        return 'yellow'


class Obstacle(Component):

    def __init__(self, position):
        super().__init__(position)

    def label(self):
        return 'O'

    def color(self):
        return 'blue'


class Automaton:
    """ the actual cellular automaton """

    def __init__(self, grid_size, pedestrians, targets, obstables):
        if isinstance(grid_size, tuple) and len(grid_size) == 2:
            self.width, self.height = grid_size[0], grid_size[1]

            self.distanceMaps = self.calculateDistanceMaps(targets, obstables)

            if len(targets) == 1:  # if only one target given, this target is set for all pedestrians
                targets = len(pedestrians) * targets
            # creating the objects for visualization
            self.pedestrians = self.createPedestrians(pedestrians, targets)
            self.obstacles = self.createObstacles(obstables)

            self.step_num = 0
            self.graphics = self.generateGraphic()

    def createPedestrians(self, pedestrians, targets):
        return [Pedestrian(position, target) for position, target in zip(pedestrians, targets)]

    def createObstacles(self, obstacles):
        return [Obstacle(position) for position in obstacles]

    def calculateDistanceMaps(self, targets, obstacles):
        different_target = []
        distance_maps = dict()
        for target in targets:
            if target not in different_target:
                different_target.append(target)
                distance_maps[target] = self.calculateDistance(target, obstacles)
        return distance_maps

    def calculateDistance(self, target, obstacles):
        def single_distance(x, y):
            return math.sqrt((x - target[0]) ** 2 + (y - target[1]) ** 2)

        distancemap = np.array([single_distance(i, j) for i in range(self.width) for j in range(self.height)]).reshape(
            (self.width, self.height))
        for obst in obstacles:
            distancemap[obst[0]][obst[1]] = float('inf')
        return Target((target[0], target[1]), distancemap)

    def generateGraphic(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for rect, letter_descr in self.getFigures():
            ax.add_patch(rect)
            # add the annotation label to the rectangle
            rect_coor_x, rect_coor_y = rect.get_xy()
            descr_coor_x = rect_coor_x + rect.get_width() / 2
            descr_coor_y = rect_coor_y + rect.get_height() / 2
            ax.annotate(letter_descr, (descr_coor_x, descr_coor_y),
                        color='black', weight='bold', ha='center', va='center',
                        fontsize=int(52.816 * (self.height * self.width) ** -0.302))

        ax.add_patch(Rectangle((0, 0), self.width, self.height, fill=None, alpha=1))
        plt.xlim([-0.5, self.width + 0.5])
        plt.ylim([-0.5, self.height + 0.5])
        ax.set_aspect('equal')
        plt.axis('off')

        plt.title(self.step_num)

        # plt.show()

        self.step_num += 1

        return fig

    def getFigures(self):
        """ create the rectangles to be drawn """

        def draw_rect(obj):
            x, y, color, label = obj.getDrawInfo()
            return Rectangle((x, y), 1, 1, color=color), label

        pedes_generator = (draw_rect(pedes) for pedes in self.pedestrians)
        target_generator = (draw_rect(target) for target in self.distanceMaps.values())
        obstacles_generator = (draw_rect(obstacles) for obstacles in self.obstacles)
        return chain(pedes_generator, target_generator, obstacles_generator)

    def step(self):
        for pedes in self.pedestrians:
            smallest = ()
            best_distance = float('inf')
            # having a look at all the 9 neighbors of the current pedestrian
            for i, j in product([0, 1, -1], repeat=2):
                if 0 <= pedes.current_x + i < self.width and 0 <= pedes.current_y + j < self.height:  # staying in bound
                    distance = self.distanceMaps[pedes.target].distanceMap[pedes.current_x + i][pedes.current_y + j]
                    if distance < best_distance:
                        best_distance = distance
                        smallest = (i, j)
            pedes.current_x += smallest[0]
            pedes.current_y += smallest[1]
        self.graphics = self.generateGraphic()


class Application(tk.Frame):
    def __init__(self, master=None, automaton=None):
        tk.Frame.__init__(self, master)
        self.automaton = automaton
        self.createWidgets()

    def createWidgets(self):
        fig = self.automaton.graphics
        self.canvas = FigureCanvasTkAgg(fig, master=root)
        self.canvas.get_tk_widget().grid(row=0, column=1)

        self.plotbutton = tk.Button(master=root, text="plot", command=lambda: self.on_click())
        self.plotbutton.grid(row=0, column=0)

        self.quitbutton = tk.Button(master=root, text="exit", command=lambda: self.quit())
        self.quitbutton.grid(row=0, column=2)

    def on_click(self):
        self.automaton.step()
        self.canvas.figure = self.automaton.graphics
        self.canvas.draw()

    def quit(self):
        root.quit()
        root.destroy()


if __name__ == '__main__':
    width = 7
    height = 5
    tt = Automaton(grid_size=(width, height), pedestrians=[(0, 0), (1, 3), (2, 2)], targets=[(3, 3)],
                   obstables=[(3, 4)])

    root = tk.Tk()
    app = Application(master=root, automaton=tt)
    app.mainloop()
