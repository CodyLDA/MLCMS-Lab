import numpy as np
import math
from matplotlib.patches import Rectangle
from itertools import chain, product
from dijsktra import *
from fmm import *
import tkinter as tk
import matplotlib.pyplot as plt


# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# import matplotlib

# matplotlib.use('TkAgg')


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

    def __init__(self, grid_size, pedestrians, targets, obstables):
        if isinstance(grid_size, tuple) and len(grid_size) == 2:
            self.width, self.height = grid_size[0], grid_size[1]



            if len(targets) == 1:  # if only one target given, this target is set for all pedestrians
                targets = len(pedestrians) * targets
            # creating the objects for visualization
            self.pedestrians = self.createPedestrians(pedestrians, targets)
            self.obstacles = self.createObstacles(obstables)
            self.board = Graph(self.width, self.height, obstables)
            self.obst = obstables
            self.distanceMaps = self.calculateDistanceMaps(targets)
            self.step_num = 0
            self.pedes_coord = {}
            for ped in self.pedestrians:
                self.pedes_coord[ped.id] = str(ped.current_x) + str(ped.current_y)
            print(self.pedes_coord)
            self.graphics = self.generateGraphic()

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
                distancemap, _ = dijsktra(self.board, str(target[0])+str(target[1]))
                distance_maps[target] = Target((target[0], target[1]), Automaton.TARGET, distancemap)
                distance_maps[target].target_env = self.calculateTargetNeighbors(target)
                # distance_maps[target] = self.calculateDistance(target, obstacles)
        return distance_maps

    def calculateTargetNeighbors(self, target):
        fields = list(self.board.nodes)
        target_env = []
        for i, j in product([0, 1, -1], repeat=2):
            if str(target[0] + i) + str(target[1] + j) in fields:
                target_env.append(str(target[0] + i) + str(target[1] + j))
        return target_env

    def calculateDistance(self, target, obstacles):
        def single_distance(x, y):
            return math.sqrt((x - target[0]) ** 2 + (y - target[1]) ** 2)

        distancemap = np.array([single_distance(i, j) for i in range(self.width) for j in range(self.height)]).reshape(
            (self.width, self.height))
        for obst in obstacles:
            distancemap[obst[0]][obst[1]] = float('inf')
        return Target((target[0], target[1]), Automaton.TARGET, distancemap)

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
        dmax = 1
        for pedes in self.pedestrians:
            if not pedes.at_goal:
                smallest = ()
                best_distance = float('inf')
                fields = list(self.board.nodes)
                # print(fields)
                # print(pedes.id)
                # having a look at all the 9 neighbors of the current pedestrian
                for i, j in product([0, 1, -1], repeat=2):
                    if str(pedes.current_x + i)+str(pedes.current_y + j) in fields:  # staying in bound
                        # Uncomment to use fmm
                        # distance_map = fmm([pedes.current_x + i, pedes.current_y + j], self.obst, self.width, self.height)
                        # Using dijsktra
                        distance_map, _ = dijsktra(self.board, str(pedes.current_x + i)+str(pedes.current_y + j))
                        # print(distance_map)
                        dist = 0
                        for k in list(self.pedes_coord.keys()):
                            # print(distance_map[self.pedes_coord[k]])
                            if int(k) == int(pedes.id):
                                pass
                            elif str(pedes.current_x + i)+str(pedes.current_y + j) == self.pedes_coord[k]:
                                dist += float('inf')
                            elif distance_map[self.pedes_coord[k]] < dmax:
                                dist += np.exp(1/(distance_map[self.pedes_coord[k]]**2 - dmax**2))
                        distance = self.distanceMaps[pedes.target].distanceMap[str(pedes.current_x + i)+str(pedes.current_y + j)]
                        print(distance)
                        distance += dist
                        if distance < best_distance:
                            best_distance = distance
                            smallest = (i, j)
                pedes.current_x += smallest[0]
                pedes.current_y += smallest[1]
                self.pedes_coord[pedes.id] = str(pedes.current_x)+str(pedes.current_y)
                # print(self.pedes_coord)
                # print(self.distanceMaps[pedes.target].target_env)
                if str(pedes.current_x)+str(pedes.current_y) in self.distanceMaps[pedes.target].target_env:
                    pedes.at_goal = True
            # self.graphics = self.generateGraphic()


class Application(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.master.protocol("WM_DELETE_WINDOW", self.quit)
        self.createWidgets()

        self.PEDES = ('red', 'P')
        self.TARGET = ('yellow', 'T')
        self.OBSTACLE = ('blue', 'O')
        self.EMPTY = ('white', '')
        self.cellState = {self.PEDES: [], self.TARGET: [], self.OBSTACLE: []}

    def createWidgets(self):
        self.lbl_width = tk.Label(self.master, text="Width")
        self.lbl_width.grid(row=0)

        self.txt_width = tk.Entry(self.master, width=20)
        self.txt_width.grid(column=1, row=0)

        self.lbl_height = tk.Label(self.master, text="Height")
        self.lbl_height.grid(row=1)

        self.txt_height = tk.Entry(self.master, width=20)
        self.txt_height.grid(column=1, row=1)

        self.btn_submit = tk.Button(self.master, text="Use grid size", command=self.generatePlainGrid)
        self.btn_submit.grid(column=1, row=2)

        self.btn_start = tk.Button(self.master, text="Start simulation", command=self.startSimulation)
        self.btn_start.grid(column=0, row=3)

        self.btn_start = tk.Button(self.master, text="Step simulation", command=self.stepSimulation)
        self.btn_start.grid(column=1, row=3)

        self.fig = plt.figure()
        # self.plottingArea = FigureCanvasTkAgg(self.fig, master=self.master)
        # self.plottingArea.get_tk_widget().grid(row=3, column=2)
        self.canvas_width = 1000
        self.canvas_height = 800
        self.plottingArea = tk.Canvas(self.master, bg='white', height=self.canvas_height, width=self.canvas_width)
        self.plottingArea.grid(row=3, column=2)
        self.plottingArea.bind("<Button-1>", self.settingCell)

        # self.fig.canvas.callbacks.connect("button_press_event", self.settingCell)

        # self.plotbutton = tk.Button(master=self.master, text="step", command=self.on_click)
        # self.plotbutton.grid(row=0, column=0)

        # self.quitbutton = tk.Button(master=root, text="exit", command=self.quit)
        # self.quitbutton.grid(row=0, column=2)

    def settingCell(self, event):
        cell_id_x = int(event.x / self.cell_width)
        cell_id_y = int(event.y / self.cell_height)

        cell_pos_x = cell_id_x * self.cell_width
        cell_pos_y = cell_id_y * self.cell_height

        self.drawCell(cell_pos_x, cell_pos_y, self.getNextCellState(cell_id_x, cell_id_y))

    def getNextCellState(self, x, y):
        if (x, y) in self.cellState[self.PEDES]:
            self.cellState[self.PEDES].remove((x, y))
            self.cellState[self.TARGET].append((x, y))
            return self.TARGET
        elif (x, y) in self.cellState[self.TARGET]:
            self.cellState[self.TARGET].remove((x, y))
            self.cellState[self.OBSTACLE].append((x, y))
            return self.OBSTACLE
        elif (x, y) in self.cellState[self.OBSTACLE]:
            self.cellState[self.OBSTACLE].remove((x, y))
            return self.EMPTY
        else:
            self.cellState[self.PEDES].append((x, y))
            return self.PEDES
        # return self.TARGET if (x, y) in self.cellState[self.PEDES] else self.OBSTACLE if (x, y) in self.cellState[self.TARGET] \
        #    else self.EMPTY if (x, y) in self.cellState[self.OBSTACLE] else self.PEDES

    def startSimulation(self):
        if len(self.cellState[self.TARGET]) <= 1:
            self.automaton = Automaton(grid_size=(self.width, self.height), pedestrians=self.cellState[self.PEDES],
                                       targets=self.cellState[self.TARGET], obstables=self.cellState[self.OBSTACLE])

    def stepSimulation(self):
        self.automaton.step()
        self.generatePlainGrid()
        self.drawerClasses(self.automaton.pedestrians)
        self.drawerClasses(self.automaton.distanceMaps.values())
        self.drawerClasses(self.automaton.obstacles)

    def drawerClasses(self, liste):
        for list_obj in liste:
            cell_pos_x = list_obj.current_x * self.cell_width
            cell_pos_y = list_obj.current_y * self.cell_height
            self.drawCell(cell_pos_x, cell_pos_y, list_obj.drawConfig)

    def drawCell(self, cell_pos_x, cell_pos_y, class_obj):
        cell_bottom_right_x = cell_pos_x + self.cell_width
        cell_bottom_right_y = cell_pos_y + self.cell_height
        color = class_obj[0]
        text = class_obj[1]

        self.plottingArea.create_rectangle(cell_pos_x, cell_pos_y,
                                           cell_bottom_right_x, cell_bottom_right_y, fill=color)
        self.plottingArea.create_text((cell_pos_x + cell_bottom_right_x) / 2, (cell_pos_y + cell_bottom_right_y) / 2,
                                      text=text, fill='black',
                                      font=('Helvetica', str(int(85.935 * (self.width * self.height) ** -0.353))))

    def generatePlainGrid(self):
        self.plottingArea.delete('all')

        self.width = int(self.txt_width.get())
        self.height = int(self.txt_height.get())

        self.cell_width = self.canvas_width / self.width
        self.cell_height = self.canvas_height / self.height
        for i in range(self.width):
            self.plottingArea.create_line(self.cell_width * i, 0, self.cell_width * i, self.canvas_height)
        for i in range(self.height):
            self.plottingArea.create_line(0, self.cell_height * i, self.canvas_width, self.cell_height * i)

        self.cellState = {self.PEDES: [], self.TARGET: [], self.OBSTACLE: []}
        # ax = self.fig.add_subplot(111)

        # self.width = int(self.txt_width.get())
        # self.height = int(self.txt_height.get())

        # largeRect = Rectangle((0, 0), self.width, self.height, fill=None, alpha=1)
        # ax.add_patch(largeRect)
        # for i in range(self.width):
        #    plt.plot([i, i], [0, self.height], color='black')
        # for i in range(self.height):
        #    plt.plot([0, self.width], [i, i], color='black')
        # plt.xlim([-0.5, self.width + 0.5])
        # plt.ylim([-0.5, self.height + 0.5])
        # ax.set_aspect('equal')
        # plt.axis('off')

        # self.plottingArea.figure = self.fig
        # self.plottingArea.draw()

    def placeholder(self):
        width = 7
        height = 5
        tt = Automaton(grid_size=(width, height), pedestrians=[(0, 0), (1, 3), (2, 2)], targets=[(3, 3)],
                       obstables=[(3, 4)])

        fig = self.automaton.graphics
        # self.plottingArea = FigureCanvasTkAgg(fig, master=root)
        # self.plottingArea.get_tk_widget().grid(row=0, column=1)

    def on_click(self):
        self.automaton.step()
        self.plottingArea.figure = self.automaton.graphics
        self.plottingArea.draw()

    def quit(self):
        root.quit()
        root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()

# class Application(tk.Frame):
#     def __init__(self, master=None, automaton=None):
#         tk.Frame.__init__(self, master)
#         self.automaton = automaton
#         self.createWidgets()
#
#     def createWidgets(self):
#         fig = self.automaton.graphics
#         self.canvas = FigureCanvasTkAgg(fig, master=root)
#         self.canvas.get_tk_widget().grid(row=0, column=1)
#
#         self.plotbutton = tk.Button(master=root, text="plot", command=lambda: self.on_click())
#         self.plotbutton.grid(row=0, column=0)
#
#         self.quitbutton = tk.Button(master=root, text="exit", command=lambda: self.quit())
#         self.quitbutton.grid(row=0, column=2)
#
#     def on_click(self):
#         self.automaton.step()
#         self.canvas.figure = self.automaton.graphics
#         self.canvas.draw()
#
#     def quit(self):
#         root.quit()
#         root.destroy()
#
#
# if __name__ == '__main__':
#     width = 7
#     height = 5
#     tt = Automaton(grid_size=(width, height), pedestrians=[(0, 0), (1, 3), (2, 2)], targets=[(3, 3)],
#                    obstables=[(3, 4)])
#
#     root = tk.Tk()
#     app = Application(master=root, automaton=tt)
#     app.mainloop()
