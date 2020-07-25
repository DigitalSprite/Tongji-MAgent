import math
import time

import matplotlib.pyplot as plt
import numpy as np

import magent
from models.tf_model import DeepQNetwork
from renderer.server import BaseServer

def generate_map(env, map_size, handles):
    width = height = map_size
    init_num = map_size * map_size * 0.04

    gap = 3
    leftID, rightID = 0, 1

    # add left square of agents
    n = init_num
    side = int(math.sqrt(n)) * 2
    pos = []
    for x in range(width//2 - gap - side, width//2 - gap - side + side, 2):
        for y in range((height - side)//2, (height - side)//2 + side, 2):
            pos.append([x, y, 0])
    env.add_agents(handles[leftID], method="custom", pos=pos)

    # add right square of agents
    n = init_num
    side = int(math.sqrt(n)) * 2
    pos = []
    for x in range(width//2 + gap, width//2 + gap + side, 2):
        for y in range((height - side)//2, (height - side)//2 + side, 2):
            pos.append([x, y, 0])
    env.add_agents(handles[rightID], method="custom", pos=pos)


class Againstserver(BaseServer):
    def __init__(self, path="data/battle_model", total_step=500):
        # some parameter
        map_size = 125
        eps = 0.00

        # init the game
        env = magent.GridWorld("battle", map_size=map_size)

        handles = env.get_handles()
        models = []
        models.append(DeepQNetwork(env, handles[0], 'trusty-battle-game-l', use_conv=True))
        models.append(DeepQNetwork(env, handles[1], 'battle', use_conv=True))

        # load model
        models[0].load(path, 0, 'trusty-battle-game-l')
        models[1].load(path, 0, 'battle')

        # init environment
        env.reset()
        generate_map(env, map_size, handles)

        # save to member variable
        self.env = env
        self.handles = handles
        self.eps = eps
        self.models = models
        self.map_size = map_size
        self.total_step = total_step
        self.done = False
        self.total_handles = [self.env.get_num(self.handles[0]), self.env.get_num(self.handles[1])]
        print(env.get_view2attack(handles[0]))
        plt.show()

    def get_info(self):
        return (self.map_size, self.map_size), self.env._get_groups_info(), {'wall': self.env._get_walls_info()}

    def step(self):
        handles = self.handles
        models = self.models
        env = self.env

        obs = [env.get_observation(handle) for handle in handles]
        ids = [env.get_agent_id(handle) for handle in handles]

        for i in range(len(handles)):
            acts = models[i].infer_action(obs[i], ids[i], 'e_greedy', eps=self.eps)
            env.set_action(handles[i], acts)

        done = env.step()
        env.clear_dead()

        return done

    def get_data(self, frame_id, x_range, y_range):
        start = time.time()
        if self.done:
            return None
        self.done = self.step()
        pos, event = self.env._get_render_info(x_range, y_range)
        print(" fps ", 1 / (time.time() - start))
        return pos, event

    def add_agents(self, x, y, g):
        pass

    def get_map_size(self):
        return self.map_size, self.map_size

    def get_banners(self, frame_id, resolution):
        return []

    def get_status(self, frame_id):
        if self.done:
            return None
        else:
            return True

    def keydown(self, frame_id, key, mouse_x, mouse_y):
        return False

    def mousedown(self, frame_id, pressed, mouse_x, mouse_y):
        return False

    def get_endscreen(self, frame_id):
        return []
