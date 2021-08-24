import pygame, sys
import math as m
import typing
from queue import Queue

"""
    Future imports for path planning visualization

    from random import randrange
    from pygame.locals import *
    from ctypes import c_int, WINFUNCTYPE, windll
    from ctypes.wintypes import HWND, LPCSTR, UINT


def MessageBox(title, text, style):
    return windll.user32.MessageBoxW(0, text, title, style)

def found():
    MessageBox("A* finished. Found the shortest path!", "Try again?", 0)
    resetPath()
"""

class Coordinate:
    def __init__(self, x=None, y=None, dist=0):
        self.x = x
        self.y = y
        self.dist = dist

    def get_coordinates(self) -> [int, int]:
        return self.x, self. y

    def get_x(self) -> int:
        return self.x

    def get_y(self) -> int:
        return self.y

    def get_distance(self) -> int:
        return self.dist

    def set_distance(self, dist: int) -> None:
        self.dist = dist


class PriorityQueue:
    def __init__(self):
        self.q = []

    def insert(self, x: (float, Coordinate)) -> None:
        self.q.append(x)
        self.sort_queue()

    def get(self) -> (float, Coordinate):
        return self.q.pop(0)

    def sort_queue(self) -> None:
        self.q.sort(key=lambda x: x[0])

    def search_queue(self, c: Coordinate):
        for idx, node in enumerate(self.q):
            if (node[1].x, node[1].y) == (c.x, c.y):
                return idx
        return None

    def update_queue(self, node: (int, Coordinate)) -> None:
        idx = self.search_queue(node[1])
        if idx:
            self.q.pop(idx)
        self.q.append((node[0], node[1]))
        self.sort_queue()

    def print_queue(self) -> None:
        print([(node[0], (node[1].x, node[1].y)) for node in self.q])


# class to hold all information about the world provided by world*.dat
class World:
    def __init__(self):
        self.path = []
        self.start = Coordinate()
        self.end = Coordinate()
        self.k = 0

    def reset_world(self) -> None:
        self.path = []
        self.start = Coordinate()
        self.end = Coordinate()
        self.k = 0


    def initialize_world(self, world_dat_file: typing.IO) -> None:
        with open(world_dat_file, "r") as data_file:

            lines = [line.strip(" \n").split() for line in data_file.readlines()]

            for i in range(0, len(lines)):
                for j in range(0, len(lines[i])):
                    lines[i][j] = int(lines[i][j])

            self.max_x = len(lines[0])-1
            self.min_x = 0
            self.max_y = len(lines) - 1
            self.min_y = 0
            self.world_map = lines

        while True:
            self.start.x = int(input("Please provide an x coordinate for your starting point. Choose between 0-%d:  " % (len(self.world_map[0])-1)))
            self.start.y = int(input("Please provide an y coordinate for your starting point. Choose between 0-%d:  " % (len(self.world_map)-1)))

            if(self.world_map[self.start.y][self.start.x] == 1):
                print("- - - - - - - - - -")
                print("Invalid xy-coordinate pair. Obstacle exists in this location x = %d, y = %d." % (self.start.x, self.start.y))
                print("- - - - - - - - - -")
                continue

            break

        print()
        print("- - - - - - - - - -")
        print()

        # Ask user for ending coordinates
        # If user provides invalid coordinates, keep asking until coordinates are valid
        # start and end cannot be the same
        while True:
            self.end.x = int(input("Please provide an x coordinate for your ending point. Choose between 0-%d:  " % (len(self.world_map[0])-1)))
            self.end.y = int(input("Please provide an y coordinate for your ending point. Choose between 0-%d:  " % (len(self.world_map)-1)))

            if(self.world_map[self.end.y][self.end.x] == 1):
                print("- - - - - - - - - -")
                print("Invalid xy-coordinate pair. Obstacle exists in this location x = %d, y = %d." % (self.end.x, self.end.y))
                print("- - - - - - - - - -")
                continue

            if(self.start.x==self.end.x and self.start.y==self.end.y):
                print("- - - - - - - - - -")
                print("Invalid xy-coordinate pair. Start and end coordinates must not be the same.")
                print("- - - - - - - - - -")
                continue

            break

        print()
        print("- - - - - - - - - -")
        print()


        # Initialize start and end in the world map
        self.world_map[self.start.y][self.start.x] = 2
        self.world_map[self.end.y][self.end.x] = 2


    # Print the current state of the world for debugging
    def show_world(self):
        print()
        print("Final World Map")
        print("- - - - - - - - - - - - - - -")
        for line in self.world_map:
            for i in range(len(line)):
                if i < len(line) - 1:
                    print("%-3d  " % line[i], end='')
                else:
                    print("%-3d" % line[i])

    # check to see if a coordinate exists within the world
    def within_range(self, x: int, y: int) -> bool:
        if self.min_x <= x and x <= self.max_x and self.min_y <= y and y <= self.max_y:
            return True
        return False

    # check to see if the current coordinate is the start coordinate
    def is_start(self, x: int, y: int) -> bool:
        if x == self.start.x and y == self.start.y:
            return True
        return False

    # check to see if the current coordinate is the end coordinate
    def is_end(self, x: int, y: int) -> bool:
        if x == self.end.x and y == self.end.y:
            return True
        return False

    # Find the path from the start goal to the end goal with the current world map
    def calculate_path(self) -> None:
        x, y = self.end.x, self.end.y
        self.path.append((x, y))

        checknode = (x, y)
        counter = 0

        while (True):
            if checknode == (x, y):
                counter += 1
                if counter == 50:
                    break

            if self.is_start(x, y):
                break

            if (self.within_range(x, y - 1) and (self.world_map[y - 1][x] == self.world_map[y][x] - 1)) or self.is_start(x,y-1):
                y = y - 1
                self.path.append((x, y))
                self.k += 1

            elif (self.within_range(x + 1, y) and (self.world_map[y][x + 1] == self.world_map[y][x] - 1)) or self.is_start(x+1, y):
                x = x + 1
                self.path.append((x, y))
                self.k += 1
            elif (self.within_range(x - 1, y) and (self.world_map[y][x - 1] == self.world_map[y][x] - 1)) or self.is_start(x-1, y):
                x = x - 1
                self.path.append((x, y))
                self.k += 1
            elif (self.within_range(x, y + 1) and (self.world_map[y + 1][x] == self.world_map[y][x] - 1)) or self.is_start(x, y+1):
                y = y + 1
                self.path.append((x, y))
                self.k += 1

    # display the resulting path
    def show_path(self) -> None:
        print("The number of steps from (%d, %d) to (%d, %d) is %d steps" % (self.start.x, self.start.y, self.end.x, self.end.y, self.k))
        print("The path is: ", end='')
        for i in range(len(self.path)):
            if i < len(self.path) - 1:
                print("(%d, %d), " % (self.path[i][0], self.path[i][1]), end='')
            else:
                print("(%d, %d)" % (self.path[i][0], self.path[i][1]))


class WaveFront:
    def __init__(self):
        pass

    def find_path(self, w: World) -> None:
        q = Queue()
        q.put((w.start.x, w.start.y))

        # Use BFS to find the path from the start to the end
        # BFS starts with the end coordinates and searches for path to the start
        while (True):
            coordinate = q.get()

            # If the start goal is reached, calculate path and display results
            if w.is_end(coordinate[0], coordinate[1]):
                w.calculate_path()
                w.show_path()
                w.show_world()
                break

            x1, y1 = coordinate[0], coordinate[1] - 1
            x2, y2 = coordinate[0] + 1, coordinate[1]
            x3, y3 = coordinate[0], coordinate[1] + 1
            x4, y4 = coordinate[0] - 1, coordinate[1]

            if w.within_range(x1, y1) and (w.world_map[y1][x1] == 0 or w.is_end(x1, y1)):
                w.world_map[y1][x1] = w.world_map[coordinate[1]][coordinate[0]] + 1
                q.put((x1, y1))

            if w.within_range(x2, y2) and (w.world_map[y2][x2] == 0 or w.is_end(x2, y2)):
                w.world_map[y2][x2] = w.world_map[coordinate[1]][coordinate[0]] + 1
                q.put((x2, y2))

            if w.within_range(x3, y3) and (w.world_map[y3][x3] == 0 or w.is_end(x3, y3)):
                w.world_map[y3][x3] = w.world_map[coordinate[1]][coordinate[0]] + 1
                q.put((x3, y3))

            if w.within_range(x4, y4) and (w.world_map[y4][x4] == 0 or w.is_end(x4, y4)):
                w.world_map[y4][x4] = w.world_map[coordinate[1]][coordinate[0]] + 1
                q.put((x4, y4))


class AStar:

    def __init__(self, heuristic="Euclidean"):
        self.heuristic = heuristic

    def find_path(self, w: World) -> None:
        q = PriorityQueue()
        q.insert((0, w.start))

        # Use BFS to find the path from the start to the end
        # BFS starts with the end coordinates and searches for path to the start
        while (True):
            dist, coordinate = q.get()


            # If the start goal is reached, calculate path and display results
            if w.is_end(coordinate.x, coordinate.y):
                w.calculate_path()
                w.show_path()
                w.show_world()
                break

            c1 = Coordinate(coordinate.x, coordinate.y - 1, coordinate.dist + 1)
            c2 = Coordinate(coordinate.x + 1, coordinate.y, coordinate.dist + 1)
            c3 = Coordinate(coordinate.x, coordinate.y + 1, coordinate.dist + 1)
            c4 = Coordinate(coordinate.x - 1, coordinate.y, coordinate.dist + 1)

            if w.within_range(c1.x, c1.y) and (w.world_map[c1.y][c1.x] == 0 or w.is_end(c1.x, c1.y)):
                w.world_map[c1.y][c1.x] = w.world_map[coordinate.y][coordinate.x] + 1
                cost = self.g(c1) + self.h(c1, w.end)
                q.update_queue((cost, c1))

            if w.within_range(c2.x, c2.y) and (w.world_map[c2.y][c2.x] == 0 or w.is_end(c2.x, c2.y)):
                w.world_map[c2.y][c2.x] = w.world_map[coordinate.y][coordinate.x] + 1
                cost = self.g(c2) + self.h(c2, w.end)
                q.update_queue((cost, c2))

            if w.within_range(c3.x, c3.y) and (w.world_map[c3.y][c3.x] == 0 or w.is_end(c3.x, c3.y)):
                w.world_map[c3.y][c3.x] = w.world_map[coordinate.y][coordinate.x] + 1
                cost = self.g(c3) + self.h(c3, w.end)
                q.update_queue((cost, c3))

            if w.within_range(c4.x, c4.y) and (w.world_map[c4.y][c4.x] == 0 or w.is_end(c4.x, c4.y)):
                w.world_map[c4.y][c4.x] = w.world_map[coordinate.y][coordinate.x] + 1
                cost = self.g(c4) + self.h(c4, w.end)
                q.update_queue((cost, c4))

    def set_heuristic(self, heuristic: str) -> None:
        self.heuristic = heuristic

    def get_heuristic(self) -> str:
        return self.heuristic

    # Actual distance from starting node to current node
    def g(self, c: Coordinate) -> int:
        return c.dist

    # Heuristic function
    def h(self, p1: Coordinate, p2: Coordinate, D=1, D2=1) -> float:
        #Manhattan Distance
        if self.heuristic == "Manhattan":
            dx = abs(p2.x - p1.x)
            dy = abs(p2.y - p1.y)
            return D * (dx + dy)

        #Diagonal Distance
        elif self.heuristic == "Diagonal":
            dx = abs(p2.x - p1.x)
            dy = abs(p2.y - p1.y)
            return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)

        #Euclidean Distance
        else:
            dx = (p2.x - p1.x)**2
            dy = (p2.y - p1.y)**2
            return D * m.sqrt(dx + dy)



if __name__=="__main__":

    grid = World()
    grid.initialize_world(r"world1.dat")

    #algorithm = WaveFront()
    algorithm = AStar()
    algorithm.find_path(grid)
