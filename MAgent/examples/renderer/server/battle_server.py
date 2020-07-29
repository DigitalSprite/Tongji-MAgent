import math
import time

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('..')

import magent
from models.tf_model import DeepQNetwork
from renderer.server import BaseServer
import tensorflow as tf

import numpy as np
import utils


class BattleServer(BaseServer):
    def __init__(self, path="data/battle_model_3_players", total_step=1000, add_counter=10, add_interval=50):
        # some parameter
        map_size = 125
        eps = 0.00

        # init the game
        env = magent.GridWorld(utils.load_config(map_size))

        handles = env.get_handles()
        models = []
        models.append(DeepQNetwork(env, handles[0], 'trusty-battle-game-l1', use_conv=True))
        # models.append(DeepQNetwork(env, handles[1], 'trusty-battle-game-l2', use_conv=True))
        models.append(DeepQNetwork(env, handles[1], 'trusty-battle-game-r', use_conv=True))
        

        # load model
        # tf.reset_default_graph()
        models[0].load(path, 1, 'trusty-battle-game-l1')
        # models[1].load(path, 1, 'trusty-battle-game-l2')
        # tf.reset_default_graph()
        models[2].load(path, 1, 'trusty-battle-game-r')

        # init environment
        env.reset()
        utils.generate_map(env, map_size, handles)

        # save to member variable
        self.env = env
        self.handles = handles
        self.eps = eps
        self.models = models
        self.map_size = map_size
        self.total_step = total_step
        self.add_interval = add_interval
        self.add_counter = add_counter
        self.done = False
        self.total_handles = [self.env.get_num(self.handles[0]), self.env.get_num(self.handles[1])]

    def get_info(self):
        return (self.map_size, self.map_size), self.env._get_groups_info(), {'wall': self.env._get_walls_info()}

    def step(self):
        handles = self.handles
        models = self.models
        env = self.env

        obs = [env.get_observation(handle) for handle in handles]
        ids = [env.get_agent_id(handle) for handle in handles]

        counter = []
        for i in range(len(handles)):
            acts = models[i].infer_action(obs[i], ids[i], 'e_greedy', eps=self.eps)
            env.set_action(handles[i], acts)
            counter.append(np.zeros(shape=env.get_action_space(handles[i])))
            for j in acts:
                counter[-1][j] += 1

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

    def get_map_size(self):
        return self.map_size, self.map_size

    def get_banners(self, frame_id, resolution):
        red_total = 'Total: {}'.format(0)
        red = 'Total:{}   Left:{}'.format(self.total_handles[0], self.env.get_num(self.handles[0])), (200, 0, 0)
        vs = ' vs ', (0, 0, 0)
        blue = 'Total:{}   Left:{}'.format(self.total_handles[1], self.env.get_num(self.handles[1])), (0, 0, 200)
        result = [(red, vs, blue)]

        tmp = '{} chance(s) remained'.format(
            max(0, self.add_counter)), (0, 0, 0)
        result.append((tmp,))

        tmp = '{} / {} steps'.format(frame_id, self.total_step), (0, 0, 0)
        result.append((tmp,))
        if frame_id % self.add_interval == 0 and frame_id < self.total_step and self.add_counter > 0:
            tmp = 'Please press your left mouse button to add agents', (0, 0, 0)
            result.append((tmp,))
        return result

    def get_status(self, frame_id):
        # if frame_id % self.add_interval == 0 and self.add_counter > 0:
        #     return False
        # elif frame_id >= self.total_step or self.done:
        #     return None
        # else:
        #     return True
        return True

    def keydown(self, frame_id, key, mouse_x, mouse_y):
        return False

    def mousedown(self, frame_id, pressed, mouse_x, mouse_y):
        if frame_id % self.add_interval == 0 and frame_id < self.total_step and pressed[0] \
                and self.add_counter > 0 and not self.done:
            self.add_counter -= 1
            count_before = self.env.get_num(self.handles[0])
            count_before = self.env.get_num(self.handles[1])
            pos = []
            for i in range(-5, 5):
                for j in range(-5, 5):
                    pos.append((mouse_x + i, mouse_y + j))
            
            self.env.add_agents(self.handles[0], method="custom", pos=pos)
            self.total_handles[0] += self.env.get_num(self.handles[0]) - count_before

            pos = []
            x = self.map_size - 10
            y = self.map_size - 10
            for i in range(-5, 0):
                for j in range(-5, 5):
                    pos.append((x + i, y + j))
            
            self.env.add_agents(self.handles[1], method="custom", pos=pos)
            self.total_handles[1] += self.env.get_num(self.handles[1]) - count_before
            return True
        return False

    def get_endscreen(self, frame_id):
        if frame_id == self.total_step or self.done:
            if self.env.get_num(self.handles[0]) > self.env.get_num(self.handles[1]):
                return [(("You", (200, 0, 0)), (" win! :)", (0, 0, 0)))]
            else:
                return [(("You", (200, 0, 0)), (" lose. :(", (0, 0, 0)))]
        else:
            return []
