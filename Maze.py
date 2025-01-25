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
            self.maze = np.zeros((self.height, self.width), dtype=int)
            # generate walls
            for i in range(1, self.height - 1):
                for j in range(1, self.width - 1):
                    if np.random.random() < 0.3:
                        self.maze[i, j] = -1                    # wall is -1

            # generate start, goal and swich
            self.maze[self.start] = 1                           # start is 1
            self.maze[self.goal] = 2                            # goal is 2
            if self.type == "swich":
                self.maze[self.swich] = 3                       # swich is 3
                self.maze[self.goal] = 0                        # Hide the goal
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
       
    def move(self, direction):
        """
        Move the player to the given direction and return reward
        :param direction: str
        """
        # calculate new position
        if direction == "up":
            new_position = self.player[0] - 1, self.player[1]
        elif direction == "down":
            new_position = self.player[0] + 1, self.player[1]
        elif direction == "left":
            new_position = self.player[0], self.player[1] - 1
        elif direction == "right":
            new_position = self.player[0], self.player[1] + 1
        # check if new position is valid
        if new_position[0] < 0 or new_position[0] >= self.height or new_position[1] < 0 or new_position[1] >= self.width:
            return -1
        if self.maze[new_position] == -1:
            return -1
        if self.type == "simple":
            self.player = new_position
        elif self.type == "swich":
            if new_position == self.swich:
                self.player = new_position
                self.maze[self.goal] = 2
                return -0.1
            if new_position == self.goal:
                self.player = new_position
                return 2
            else:
                self.player = new_position
                return -0.1            
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

    def show(self, path=False, player=True):
        """
        Show the maze
        """
        # 2D Grid Plot
        # Black: wall, white: empty space, blue: start, red: goal, green: swich, yellow: path
        plt.figure()
        Grid = np.zeros((self.height, self.width, 3))
        Grid[self.maze == -1] = [0, 0, 0]   # wall is black
        Grid[self.maze == 0] = [1, 1, 1]     # empty space is white
        Grid[self.maze == 1] = [0, 0, 1]     # start is blue
        Grid[self.maze == 2] = [1, 0, 0]     # goal is red
        Grid[self.maze == 3] = [0, 1, 0]     # swich is green

        if path:
            for i, j in self.path:
                Grid[i, j] = [1, 1, 0]  # path is yellow
        if player:
            plt.scatter(self.player[1], self.player[0], color="orange", s=100)
        plt.imshow(Grid)
        plt.show()

class Gameplayer:
    """A window to play the game"""
    def __init__(self, maze):
        self.maze = maze
        self.width = maze.width
        self.height = maze.height
        self.size = 40
        self.screen = pygame.display.set_mode((self.width * self.size, self.height * self.size))
        pygame.display.set_caption("Maze Game")
        self.clock = pygame.time.Clock()
        self.running = True

    def run(self,terminate = False):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        reward = self.maze.move("up")
                    elif event.key == pygame.K_DOWN:
                        reward = self.maze.move("down")
                    elif event.key == pygame.K_LEFT:
                        reward = self.maze.move("left")
                    elif event.key == pygame.K_RIGHT:
                        reward = self.maze.move("right")
                    if reward == 2:
                        print("You win!")
                        self.maze.regenerate()
                    if reward == -1:
                        if terminate:
                            print("You lose!")
                            self.maze.regenerate()
                    if event.key == pygame.K_r:
                        self.maze.regenerate()
                    if event.key == pygame.K_q:
                        self.running = False
            self.screen.fill((255, 255, 255))
            for i in range(self.height):
                for j in range(self.width):
                    if self.maze.maze[i, j] == -1:
                        pygame.draw.rect(self.screen, (0, 0, 0), (j * self.size, i * self.size, self.size, self.size))
                    if self.maze.maze[i, j] == 1:
                        pygame.draw.rect(self.screen, (0, 0, 255), (j * self.size, i * self.size, self.size, self.size))
                    if self.maze.maze[i, j] == 2:
                        pygame.draw.rect(self.screen, (255, 0, 0), (j * self.size, i * self.size, self.size, self.size))
                    if self.maze.maze[i, j] == 3:
                        pygame.draw.rect(self.screen, (0, 255, 0), (j * self.size, i * self.size, self.size, self.size))
            pygame.draw.rect(self.screen, (255, 165, 0), (self.maze.player[1] * self.size, self.maze.player[0] * self.size, self.size, self.size))
            pygame.display.flip()
            self.clock.tick(10)
        pygame.quit()

        
if __name__ == "__main__":
    maze = Maze(20, 20,"swich")
    game = Gameplayer(maze)
    game.run()
