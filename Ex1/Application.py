import tkinter as tk
from Automaton import *
from time import sleep


class Application(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.master.protocol("WM_DELETE_WINDOW", self.quit)

        self.PEDES = ('red', 'P')
        self.TARGET = ('yellow', 'T')
        self.OBSTACLE = ('blue', 'O')
        self.EMPTY = ('white', '')
        self.cellState = {self.PEDES: [], self.TARGET: [], self.OBSTACLE: []}
        self.automaton = None
        self.lock_canvas = False
        self.running = False
        self.width, self.height = 0, 0
        self.dmax = 1

        self.createWidgets()

    def createWidgets(self):
        self.createInputFrame()

        self.canvas_width = 1000
        self.canvas_height = 800
        self.plottingArea = tk.Canvas(self.master, bg='white', height=self.canvas_height, width=self.canvas_width)
        self.plottingArea.bind("<Button-1>", self.settingCell)
        self.plottingArea.bind("<Configure>", self.scale)

        # placement on screen
        self.plottingArea.pack(side='left', fill='both', expand=True)
        self.inputFrame.pack(side='right', fill='both', expand=False)

    def createInputFrame(self):
        self.inputFrame = tk.Frame(self.master)

        self.lbl_width = tk.Label(self.inputFrame, text="Width:")
        self.lbl_width.grid(row=0, padx=(10, 20))

        self.txt_width = tk.Entry(self.inputFrame, width=20)
        self.txt_width.insert(0, '15')
        self.txt_width.grid(column=1, row=0, padx=(0, 10))

        self.lbl_height = tk.Label(self.inputFrame, text="Height:")
        self.lbl_height.grid(row=1, padx=(10, 20))

        self.txt_height = tk.Entry(self.inputFrame, width=20)
        self.txt_height.insert(0, '15')
        self.txt_height.grid(column=1, row=1, padx=(0, 10))

        self.btn_submit = tk.Button(self.inputFrame, text="Use grid size", command=self.submit_gridSize)
        self.btn_submit.grid(column=1, row=2, pady=(15, 0))

        self.btn_start = tk.Button(self.inputFrame, text="Start simulation", command=self.startSimulation)
        self.btn_start.grid(column=0, row=3, padx=(10, 10), pady=(20, 10))

        self.lbl_failedStart = tk.Label(self.inputFrame, text='')
        self.lbl_failedStart.grid(column=1, row=3, padx=(10, 10), pady=(20, 10))

        self.btn_step = tk.Button(self.inputFrame, text="Step simulation", command=self.stepSimulation)
        self.btn_step.grid(column=0, row=4, padx=(10, 10), pady=(10, 10))

        self.lbl_failedStep = tk.Label(self.inputFrame, text='')
        self.lbl_failedStep.grid(column=1, row=4, padx=(10, 10), pady=(10, 10))

        self.btn_reset = tk.Button(self.inputFrame, text="Reset to init", command=self.resetToInit)
        self.btn_reset.grid(column=0, row=5, padx=(10, 10), pady=(10, 10))

        self.btn_reset_all = tk.Button(self.inputFrame, text="Clear everything", command=self.clearApp)
        self.btn_reset_all.grid(column=1, row=5, padx=(10, 10), pady=(10, 10))

        self.lbl_simulateNumberStep = tk.Label(self.inputFrame, text="Simulate for ... steps:")
        self.lbl_simulateNumberStep.grid(column=0, row=6, padx=(10, 20), pady=(20, 0))

        self.txt_simulateNumberStep = tk.Entry(self.inputFrame, width=20)
        self.txt_simulateNumberStep.insert(0, '20')
        self.txt_simulateNumberStep.grid(column=1, row=6, padx=(0, 10), pady=(20, 0))

        self.createSimuBtn()
        self.createSwitchFrame()

        self.createAlgoSwitchFrame()
        self.createScenarioSwitchFrame()
        self.createDmaxField()

    def createSimuBtn(self):
        self.simubtnFrame = tk.Frame(self.inputFrame)

        self.btn_simulateNumberStep = tk.Button(self.simubtnFrame, text='Simulate', command=self.stepNumberOfSteps)
        self.btn_stopSimulation = tk.Button(self.simubtnFrame, text='Stop', command=self.stopSimulation)

        self.btn_simulateNumberStep.pack(side='left', padx=(0, 10))
        self.btn_stopSimulation.pack(side='left', padx=(10, 0))

        self.simubtnFrame.grid(column=1, row=7, padx=(0, 10), pady=(15, 0))

    def createDmaxField(self):
        self.dmaxFrame = tk.Frame(self.inputFrame)

        self.lbl_dmax = tk.Label(self.dmaxFrame, text="Avoidance r_max:")
        self.lbl_dmax.grid(row=0, padx=(10, 20))

        self.txt_dmax = tk.Entry(self.dmaxFrame, width=20)
        self.txt_dmax.insert(0, '1')
        self.txt_dmax.grid(column=1, row=0, padx=(0, 20))

        self.btn_submit_dmax = tk.Button(self.dmaxFrame, text="Set r_max", command=self.submit_dmax)
        self.btn_submit_dmax.grid(column=2, row=0, pady=(0, 0))

        self.dmaxFrame.grid(column=0, row=11, padx=(0, 10), pady=(30, 0))

    def submit_dmax(self):
        self.resetToInit()
        self.dmax = float(self.txt_dmax.get())

    def createSwitchFrame(self):
        self.switchFrame = tk.Frame(self.inputFrame)

        self.str_off = 'off'
        self.str_pedes = 'pedes'
        self.str_target = 'target'
        self.str_obst = 'obst'
        self.str_empty = 'empty'
        self.switch_var = tk.StringVar(value=self.str_off)
        self.btn_off = tk.Radiobutton(self.switchFrame, text='Off', variable=self.switch_var, indicatoron=False, value=self.str_off)
        self.btn_pedes = tk.Radiobutton(self.switchFrame, text='Pedestrian', variable=self.switch_var, indicatoron=False, value=self.str_pedes)
        self.btn_target = tk.Radiobutton(self.switchFrame, text='Target', variable=self.switch_var, indicatoron=False, value=self.str_target)
        self.btn_obst = tk.Radiobutton(self.switchFrame, text='Obstacle', variable=self.switch_var, indicatoron=False, value=self.str_obst)
        self.btn_empty = tk.Radiobutton(self.switchFrame, text='Empty', variable=self.switch_var, indicatoron=False,
                                        value=self.str_empty)

        self.btn_off.pack(side='left', padx=(2, 2))
        self.btn_pedes.pack(side='left', padx=(2, 2))
        self.btn_target.pack(side='left', padx=(2, 2))
        self.btn_obst.pack(side='left', padx=(2, 2))
        self.btn_empty.pack(side='left', padx=(2, 2))

        self.switchFrame.grid(column=0, row=8, padx=(20, 10), pady=(50, 0))

    def createAlgoSwitchFrame(self):
        self.switchAlgoFrame = tk.Frame(self.inputFrame)

        self.str_dijkstra = Automaton.DIJKSTRA
        self.str_fmm = Automaton.FMM
        self.switch_algo = tk.StringVar(value=self.str_dijkstra)
        self.btn_dijkstra = tk.Radiobutton(self.switchAlgoFrame, text='Dijkstra', variable=self.switch_algo, indicatoron=False, value=self.str_dijkstra, command=self.changeAlgo)
        self.btn_fmm = tk.Radiobutton(self.switchAlgoFrame, text='FMM', variable=self.switch_algo, indicatoron=False, value=self.str_fmm, command=self.changeAlgo)

        self.btn_dijkstra.pack(side='left', padx=(2, 2))
        self.btn_fmm.pack(side='left', padx=(2, 2))

        self.switchAlgoFrame.grid(column=0, row=9, padx=(20, 10), pady=(50, 0))

    def createScenarioSwitchFrame(self):
        self.switchScenarioFrame = tk.Frame(self.inputFrame)

        self.scene_chicken = 'chicken'
        self.scene_circle_small = 'circle_small'
        self.scene_circle = 'circle'
        self.scene_single = 'single'
        self.scene_corridor = 'corridor'
        self.scene_bottle = 'bottle'
        self.scene_bottle2 = 'bottle2'
        self.switch_scene = tk.StringVar(value='')
        self.btn_chicken = tk.Radiobutton(self.switchScenarioFrame, text='Chicken', variable=self.switch_scene, indicatoron=False, value=self.scene_chicken, command=self.changeScene)
        self.btn_circle_small = tk.Radiobutton(self.switchScenarioFrame, text='Circle Small', variable=self.switch_scene,
                                         indicatoron=False, value=self.scene_circle_small, command=self.changeScene)
        self.btn_circle = tk.Radiobutton(self.switchScenarioFrame, text='Circle Large', variable=self.switch_scene, indicatoron=False, value=self.scene_circle, command=self.changeScene)
        self.btn_single = tk.Radiobutton(self.switchScenarioFrame, text='Single', variable=self.switch_scene, indicatoron=False, value=self.scene_single, command=self.changeScene)
        self.btn_corridor = tk.Radiobutton(self.switchScenarioFrame, text='Corridor', variable=self.switch_scene, indicatoron=False, value=self.scene_corridor, command=self.changeScene)
        self.btn_bottle = tk.Radiobutton(self.switchScenarioFrame, text='Bottle 1', variable=self.switch_scene, indicatoron=False, value=self.scene_bottle, command=self.changeScene)
        self.btn_bottle2 = tk.Radiobutton(self.switchScenarioFrame, text='Bottle 2', variable=self.switch_scene, indicatoron=False, value=self.scene_bottle2, command=self.changeScene)

        self.scene_label = tk.Label(self.switchScenarioFrame, text='Choose Scenario:')
        self.scene_label.grid(column=0, row=0, padx=(0, 3), pady=(0, 8))
        self.btn_chicken.grid(column=1, row=1, padx=(0, 3), pady=(0, 3))
        self.btn_circle_small.grid(column=2, row=1, padx=(0, 3), pady=(0, 3))
        self.btn_circle.grid(column=3, row=1, padx=(0, 3), pady=(0, 3))
        self.btn_single.grid(column=1, row=2, padx=(0, 3), pady=(3, 3))
        self.btn_corridor.grid(column=2, row=2, padx=(0, 3), pady=(3, 3))
        self.btn_bottle.grid(column=3, row=2, padx=(0, 3), pady=(3, 3))
        self.btn_bottle2.grid(column=4, row=2, padx=(0, 3), pady=(3, 3))

        self.switchScenarioFrame.grid(column=0, row=10, padx=(20, 10), pady=(50, 0))

    def changeScene(self):
        self.setScenario(self.switch_scene.get())
        self.resetToInit()
        self.startSimulation()

    def changeAlgo(self):
        self.resetToInit()
        self.startSimulation()

    def scale(self, event):
        self.plottingArea.delete('all')
        self.canvas_width, self.canvas_height = event.width, event.height
        self.generatePlainGrid()
        if self.lock_canvas:
            self.drawAutomaton()
        else:
            self.drawObjects()

    def resetToInit(self):
        self.lock_canvas = False
        self.automaton = None
        self.generatePlainGrid()
        self.drawObjects()

    def clearApp(self):
        self.lock_canvas = False
        self.cellState = {self.PEDES: [], self.TARGET: [], self.OBSTACLE: []}
        self.automaton = None
        self.generatePlainGrid()

    def settingCell(self, event):
        if self.lock_canvas:
            return

        cell_id_x = int(event.x / self.cell_width)
        cell_id_y = int(event.y / self.cell_height)

        cell_pos_x = cell_id_x * self.cell_width
        cell_pos_y = cell_id_y * self.cell_height

        def objectSwitch(category):
            self.removeObjectFromCell(cell_id_x, cell_id_y)
            self.cellState[category].append((cell_id_x, cell_id_y))

        cell_label = None
        setting_status = self.switch_var.get()
        if setting_status == self.str_off:
            cell_label = self.getNextCellState(cell_id_x, cell_id_y)
        elif setting_status == self.str_pedes:
            cell_label = self.PEDES
            objectSwitch(self.PEDES)
        elif setting_status == self.str_obst:
            cell_label = self.OBSTACLE
            objectSwitch(self.OBSTACLE)
        elif setting_status == self.str_target:
            cell_label = self.TARGET
            objectSwitch(self.TARGET)
        elif setting_status == self.str_empty:
            cell_label = self.EMPTY
            self.removeObjectFromCell(cell_id_x, cell_id_y)

        self.drawCell(cell_pos_x, cell_pos_y, cell_label)

    def removeObjectFromCell(self, x, y):
        def removeObject(x, y, category):
            if (x, y) in self.cellState[category]:
                self.cellState[category].remove((x, y))

        removeObject(x, y, self.PEDES)
        removeObject(x, y, self.TARGET)
        removeObject(x, y, self.OBSTACLE)

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

    def startSimulation(self):
        if len(self.cellState[self.PEDES]) < 1:
            self.lbl_failedStart['text'] = 'Please enter at least 1 pedestrian!'
            return
        if len(self.cellState[self.TARGET]) != 1:
            self.lbl_failedStart['text'] = 'Please enter exactly 1 target!'
            return
        if self.automaton:
            return
        self.lbl_failedStart.configure(text='')
        self.lock_canvas = True
        self.automaton = Automaton(grid_size=(self.width, self.height), pedestrians=self.cellState[self.PEDES],
                                   targets=self.cellState[self.TARGET], obstables=self.cellState[self.OBSTACLE], used_algo=self.switch_algo.get(), dmax=self.dmax)

    def stepSimulation(self):
        if self.automaton:
            self.lbl_failedStep['text'] = ""
            for i in range(7):
                self.automaton.step()
            self.generatePlainGrid()
            self.drawAutomaton()
        else:
            self.lbl_failedStep['text'] = "Please click Start simulation first!"

    def stopSimulation(self):
        if self.running:
            self.running = False
            self.lbl_failedStep['text'] = "Stopping ..."
            self.update()

    def stepNumberOfSteps(self):
        if self.automaton:
            self.running = True
            number_steps = int(self.txt_simulateNumberStep.get())
            for i in range(number_steps):
                if self.running:
                    self.stepSimulation()
                    self.update()
                    sleep(0.5)
            self.lbl_failedStep['text'] = ''
        else:
            self.stepSimulation()

    def drawAutomaton(self):
        self.drawTrajectory(self.automaton.pedestrians)
        self.drawerClasses(self.automaton.pedestrians)
        self.drawerClasses(self.automaton.distanceMaps.values())
        self.drawerClasses(self.automaton.obstacles)

    def drawTrajectory(self, pedestrians):
        for pedes in pedestrians:
            if len(pedes.trajectory) > 0:
                cell_pos_x, cell_pos_y = self.calcCellCoor(pedes.trajectory[0][0], pedes.trajectory[0][1])
                self.drawCell(cell_pos_x, cell_pos_y, ('grey', 'x'))
            for x, y in pedes.trajectory[1:]:
                cell_pos_x, cell_pos_y = self.calcCellCoor(x, y)
                self.drawCell(cell_pos_x, cell_pos_y, ('grey', ''))

    def calcCellCoor(self, cell_x, cell_y):
        return cell_x * self.cell_width, cell_y * self.cell_height

    def drawerClasses(self, liste):
        for list_obj in liste:
            #cell_pos_x = list_obj.current_x * self.cell_width
            #cell_pos_y = list_obj.current_y * self.cell_height
            cell_pos_x, cell_pos_y = self.calcCellCoor(list_obj.current_x, list_obj.current_y)
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

    def drawObjects(self):
        for key, cell_list in self.cellState.items():
            for x_coor, y_coor in cell_list:
                #cell_pos_x = x_coor * self.cell_width
                #cell_pos_y = y_coor * self.cell_height
                cell_pos_x, cell_pos_y = self.calcCellCoor(x_coor, y_coor)
                self.drawCell(cell_pos_x, cell_pos_y, key)

    def submit_gridSize(self):
        if (self.width, self.height) == (int(self.txt_width.get()), int(self.txt_height.get())):
            return
        self.clearApp()

    def generatePlainGrid(self):
        self.plottingArea.delete('all')

        self.width = int(self.txt_width.get())
        self.height = int(self.txt_height.get())

        self.cell_width = self.canvas_width / self.width
        self.cell_height = self.canvas_height / self.height
        for i in range(self.width):
            self.plottingArea.create_text(self.cell_width * i + self.cell_width / 2,
                                          self.cell_height / 2, text=i + 1)
            self.plottingArea.create_line(self.cell_width * i, 0, self.cell_width * i, self.canvas_height)
        for i in range(self.height):
            self.plottingArea.create_text(self.cell_width / 2, self.cell_height * i + self.cell_height / 2,
                                          text=i + 1)
            self.plottingArea.create_line(0, self.cell_height * i, self.canvas_width, self.cell_height * i)

    def changingSceneGridSize(self, size):
        self.width, self.height = size, size
        self.txt_width.delete(0, 'end')
        self.txt_width.insert(0, str(size))
        self.txt_height.delete(0, 'end')
        self.txt_height.insert(0, str(size))

    def setScenario(self, scene):
        if scene == self.scene_circle_small:
            self.changingSceneGridSize(20)
            self.cellState = {('red', 'P'): [(19, 9), (1, 9), (10, 0), (10, 18), (3, 16), (5, 2)], ('yellow', 'T'): [(10, 9)], ('blue', 'O'): []}
        elif scene == self.scene_circle:
            self.changingSceneGridSize(50)
            self.cellState = {('red', 'P'): [(4, 24), (24, 44), (24, 4), (44, 24), (10, 10), (16, 42)], ('yellow', 'T'): [(24, 24)], ('blue', 'O'): []}
        elif scene == self.scene_chicken:
            self.changingSceneGridSize(15)
            self.cellState = {('red', 'P'): [(1, 7)], ('yellow', 'T'): [(8, 7)], ('blue', 'O'): [(3, 3), (4, 3), (5, 3), (6, 3), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (7, 11), (6, 11), (5, 11), (4, 11), (3, 11)]}
        elif scene == self.scene_single:
            self.changingSceneGridSize(50)
            self.cellState = {('red', 'P'): [(5, 25)], ('yellow', 'T'): [(25, 25)], ('blue', 'O'): []}
        elif scene == self.scene_corridor:
            self.changingSceneGridSize(20)
            self.cellState = {('red', 'P'): [(1, 14), (2, 15), (1, 17), (4, 15), (3, 13), (6, 14), (6, 16), (0, 16)], ('yellow', 'T'): [(16, 2)], ('blue', 'O'): [(1, 12), (0, 12), (3, 12), (2, 12), (4, 12), (5, 12), (6, 12), (8, 12), (7, 12), (9, 12), (10, 12), (11, 12), (12, 12), (13, 12), (13, 11), (13, 10), (13, 9), (13, 8), (13, 7), (13, 6), (13, 5), (13, 4), (13, 3), (0, 18), (1, 18), (2, 18), (3, 18), (4, 18), (5, 18), (6, 18), (7, 18), (8, 18), (9, 18), (10, 18), (11, 18), (12, 18), (13, 18), (15, 18), (14, 18), (16, 18), (17, 18), (18, 18), (19, 18), (19, 17), (19, 16), (19, 15), (19, 14), (19, 13), (19, 12), (19, 11), (19, 10), (19, 9), (19, 8), (19, 7), (19, 6), (19, 5), (19, 4), (19, 3)]}
        elif scene == self.scene_bottle:
            self.changingSceneGridSize(15)
            self.cellState = {('red', 'P'): [(10, 14), (11, 1), (5, 1), (1, 1), (6, 12), (8, 13), (0, 12), (1, 6)], ('yellow', 'T'): [(13, 7)], ('blue', 'O'): [(12, 6), (11, 6), (10, 5), (9, 4), (8, 4), (7, 4), (6, 4), (5, 3), (4, 3), (3, 3), (12, 8), (11, 8), (10, 9), (9, 10), (8, 10), (7, 10), (6, 10), (5, 11), (4, 11), (3, 11), (13, 8), (13, 6), (14, 6), (14, 7), (14, 8), (11, 5), (10, 4), (6, 3), (6, 11), (10, 10), (11, 9)]}
        elif scene == self.scene_bottle2:
            self.changingSceneGridSize(15)
            self.cellState = {('red', 'P'): [(0, 3), (0, 4), (0, 5), (0, 6), (0, 8), (0, 7), (0, 9), (0, 10), (0, 11)], ('yellow', 'T'): [(13, 7)], ('blue', 'O'): [(12, 6), (11, 6), (10, 5), (9, 4), (8, 4), (7, 4), (6, 4), (5, 3), (4, 3), (3, 3), (12, 8), (11, 8), (10, 9), (9, 10), (8, 10), (7, 10), (6, 10), (5, 11), (4, 11), (3, 11), (13, 8), (13, 6), (14, 6), (14, 7), (14, 8), (11, 5), (10, 4), (6, 3), (6, 11), (10, 10), (11, 9)]}

    def quit(self):
        root.quit()
        root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    app = Application(master=root)
    root.title("CELLULAR AUTOMATON BASED CROWD SIMULATION")
    app.mainloop()