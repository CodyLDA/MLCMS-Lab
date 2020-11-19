# Based on:
# An Introduction to Tkinter
# Fredrik Lundh
# http://www.pythonware.com/library/tkinter/introduction/

# Copyright 1999 by Fredrik Lundh

# Modifications 2013 by Nina Amenta

# get all of the functions in the tkinter module
from tkinter import *


def drawPlot(dataList):
    # The window is an object of type tk
    root = Tk()
    root.title('Simple Plot')

    # A canvas object is something you can draw on
    # we put it into the root window
    canvas = Canvas(root, width=450, height=300, bg='white')
    # figures out how the canvas sits in the window
    canvas.pack()

    # draw x and y axes
    canvas.create_line(100, 250, 400, 250, width=2)
    canvas.create_line(100, 250, 100, 50, width=2)

    # markings on x axis
    for i in range(11):
        x = 100 + (i * 30)
        canvas.create_line(x, 250, x, 245, width=2)
        canvas.create_text(x, 254, text='%d' % (30 * i), anchor=N)

    # markings on y axis
    for i in range(6):
        y = 250 - (i * 40)
        canvas.create_line(100, y, 105, y, width=2)
        canvas.create_text(96, y, text='%5.1f' % (40 * i), anchor=E)

    # rescale the input data so it matches the axes
    ##    scaled = []
    ##    for (x,y) in dataList:
    ##        scaled.append((100 + 3*x, 250 - (4*y)/5))

    # draw the wiggly line
    canvas.create_line(dataList, fill='black')

    # and some dots at the corner points
    for (xs, ys) in dataList:
        canvas.create_oval(xs - 6, ys - 6, xs + 6, ys + 6, width=1,
                           outline='black', fill='SkyBlue2')

    # display window and wait for it to close
    root.mainloop()


def main():
    # detect if this is being run by itself or because it was imported
    if __name__ != "__main__":
        return
    # some meaningless x-y data points to plot
    # the input data is in the range x = 0 to 100, and y = 0 to 250.
    originalData = [(12, 56), (20, 94), (33, 98), (45, 120), (61, 180),
                    (75, 160), (98, 223)]

    # rescale the data to lie in the graph range x = 100 to 400, y = 250 to 50
    # remember y is zero at the top of the window.
    scaledDataList = []
    for (x, y) in originalData:
        scaledDataList.append((100 + 3 * x, 250 - (4 * y) / 5))

    drawPlot(scaledDataList)


main()