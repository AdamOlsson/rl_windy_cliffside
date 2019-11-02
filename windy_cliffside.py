#import numpy as np

class WindyCliffside():
    def __init__(self):

        self.state_space = (4,12)
        self.action_space = 4

        self.action_list = [(1,0), (-1,0), (0,1), (0,-1)]

        #self.grid = np.zeros(self.state_space)

        self.initial_state = (3,0)
        self.terminate_state = (3,11)
        self.reset_state = [(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7),(3,8),(3,9),(3,10)]

        #self.north_wind = [0,0,0,1,1,1,2,2,1,0] # how many cells respective column will push agent north

        self.pos = None


    def step(self, action):

        pos0 = min(max(self.pos[0] + action[0], 0), self.state_space[0]-1)
        pos1 = min(max(self.pos[1] + action[1], 0), self.state_space[1]-1)

        self.pos = (pos0, pos1)

        if self.pos == self.terminate_state:
            game_over = True
            reward = 0
        elif self.pos in self.reset_state:
            game_over = False
            reward = -100
            self.reset()
        else:
            game_over = False
            reward = -1

        info = []
        
        return self.pos, reward, game_over, info


    def reset(self):
        self.pos = self.initial_state

        return self.pos

    def get_actions(self):
        return [(-1,0),(1,0),(0,-1),(0,1)]