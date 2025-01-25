import matplotlib.pyplot as plt
import pygame
import numpy as np


class Maze:
    def __init__(self, width, height,type):
        """
        :param width: int
        :param height: int
        :param type: str
        simple: no swich
        swich: setp on swich to get to the goal
        """
        self.width = width
        self.height = height
        self.maze = np.zeros((height, width), dtype=int)
        self.type = type
        self.start = np.random.randint(0, height), np.random.randint(0, width)
        self.goal = np.random.randint(0, height), np.random.randint(0, width)
        while self.goal == self.start:
            self.goal = np.random.randint(0, height), np.random.randint(0, width)
        if type == "simple": 
            self.swich = None
        elif type == "swich":
            # make sure the start, goal and swich are not on the same spot
            self.swich = np.random.randint(0, height), np.random.randint(0, width)
            while self.swich == self.start or self.swich == self.goal:
                self.swich = np.random.randint(0, height), np.random.randint(0, width)
        self.player = self.start
        self.path = []
        self.generate()

    def generate(self):
        """
        Generate a maze and make sure path among start, goal and swich is available
        """
        valid = False
        while not valid:
            # -1 represent wall, 0 represent empty space, 1 represent start, 2 represent goal, 3 represent swich
            self.maze = np.zeros((self.height, self.width), dtype=int)
            self.maze[self.start] = 1
            self.maze[self.goal] = 2
            if self.type == "swich":
                self.maze[self.swich] = 3  
            # generate walls
            for i in range(1, self.height - 1):
                for j in range(1, self.width - 1):
                    if np.random.random() < 0.3:
                        self.maze[i, j] = -1
            # make sure path is available
            if self.type == "simple":
                path = self.find_path(self.start, self.goal)
            elif self.type == "swich":
                path1 = self.find_path(self.start, self.swich)
                path2 = self.find_path(self.swich, self.goal)
                if path1 and path2:
                    path = path1 + path2
                else:
                    path = None
            if path:
                valid = True
        self.path = path
    def find_path(self, start, goal):
        """
        Find a path between start and goal with A* ,return None if no path is available
        :param start: tuple
        :param goal: tuple
        :return: list
        """
        g = np.zeros_like(self.maze)
        h = np.zeros_like(self.maze)
        f = np.zeros_like(self.maze)
        open_list = []
        close_list = []
        parent = {}
        open_list.append(start)
        while open_list:
            current = open_list[0]
            for i in range(1, len(open_list)):
                if f[open_list[i]] < f[current]:
                    current = open_list[i]
            open_list.remove(current)
            close_list.append(current)
            if current == goal:
                path = []
                while current in parent:
                    path.append(current)
                    current = parent[current]
                return path
            for i, j in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = current[0] + i, current[1] + j
                if neighbor[0] < 0 or neighbor[0] >= self.height or neighbor[1] < 0 or neighbor[1] >= self.width:
                    continue
                if self.maze[neighbor] == -1:
                    continue
                if neighbor in close_list:
                    continue
                if neighbor not in open_list:
                    open_list.append(neighbor)
                    parent[neighbor] = current
                    g[neighbor] = g[current] + 1
                    h[neighbor] = abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
                    f[neighbor] = g[neighbor] + h[neighbor]
                else:
                    if g[neighbor] > g[current] + 1:
                        g[neighbor] = g[current] + 1
                        f[neighbor] = g[neighbor] + h[neighbor]
                        parent[neighbor] = current
        return None
       

    def regenerate(self):
        """
        Regenerate the maze
        """
        self.start = np.random.randint(0, self.height), np.random.randint(0, self.width)
        self.goal = np.random.randint(0, self.height), np.random.randint(0, self.width)
        while self.goal == self.start:
            self.goal = np.random.randint(0, self.height), np.random.randint(0, self.width)
        if self.type == "swich":
            self.swich = np.random.randint(0, self.height), np.random.randint(0, self.width)
            while self.swich == self.start or self.swich == self.goal:
                self.swich = np.random.randint(0, self.height), np.random.randint(0, self.width)
        self.player = self.start
        self.generate()

    def show(self, path=False):
        """
        Show the maze
        """
        plt.imshow(self.maze, cmap="viridis", interpolation="nearest")
        if path:
            maze_with_path = np.copy(self.maze)
            for i, j in self.path:
                maze_with_path[i, j] = 4
            plt.imshow(maze_with_path, cmap="viridis", interpolation="nearest")
        plt.show()

        
if __name__ == "__main__":
    maze = Maze(20, 20,"swich")
    # plot the maze
    plt.imshow(maze.maze, cmap="viridis", interpolation="nearest")
    # plt path with red
    plt.figure()
    maze_with_path = np.copy(maze.maze)
    for i, j in maze.path:
        maze_with_path[i, j] = 4
    plt.imshow(maze_with_path, cmap="viridis", interpolation="nearest")
    plt.show()