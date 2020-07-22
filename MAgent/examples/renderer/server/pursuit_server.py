import math
import time

import matplotlib.pyplot as plt
import numpy as np

import magent
from models.tf_model import DeepQNetwork
from renderer.server import BaseServer


def load_config(map_size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})

    predator = cfg.register_agent_type(
        "predator",
        {
            'width': 2, 'length': 2, 'hp': 1, 'speed': 1,
            'view_range': gw.CircleRange(5), 'attack_range': gw.CircleRange(2),
            'attack_penalty': -0.2
        })

    prey = cfg.register_agent_type(
        "prey",
        {
            'width': 1, 'length': 1, 'hp': 1, 'speed': 1.5,
            'view_range': gw.CircleRange(4), 'attack_range': gw.CircleRange(0)
        })

    predator_group  = cfg.add_group(predator)
    prey_group = cfg.add_group(prey)

    a = gw.AgentSymbol(predator_group, index='any')
    b = gw.AgentSymbol(prey_group, index='any')

    cfg.add_reward_rule(gw.Event(a, 'attack', b), receiver=[a, b], value=[1, -1])

    return cfg


def generate_map(env, map_size, handles):
    env.add_walls(method="random", n=map_size * map_size * 0.03)
    env.add_agents(handles[0], method="random", n=map_size * map_size * 0.0125)
    env.add_agents(handles[1], method="random", n=map_size * map_size * 0.025)


class PursuitServer(BaseServer):
    def __init__(self, path="data/pursuit_model", total_step=500):
        # some parameter
        map_size = 1000
        eps = 0.00

        # init the game
        env = magent.GridWorld(load_config(map_size))

        handles = env.get_handles()
        models = []
        models.append(DeepQNetwork(env, handles[0], 'predator', use_conv=True))
        models.append(DeepQNetwork(env, handles[1], 'prey', use_conv=True))

        # load model
        models[0].load(path, 423, 'predator')
        models[1].load(path, 423, 'prey')

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
