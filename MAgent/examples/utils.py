import math
import numpy as np

import sys
sys.path.append('..')

import magent
from models.tf_model import DeepQNetwork
from renderer.server import BaseServer
import tensorflow as tf

import numpy as np

def load_config(map_size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})
    cfg.set({"minimap_mode": True})

    cfg.set({"embedding_size": 10})

    small = cfg.register_agent_type(
        "small",
        {'width': 1, 'length': 1, 'hp': 10, 'speed': 2,
         'view_range': gw.CircleRange(6), 'attack_range': gw.CircleRange(1.5),
         'damage': 2, 'step_recover': 0.1,
         'step_reward': -0.001, 'kill_reward': 100, 'dead_penalty': -0.05, 'attack_penalty': -1,
         })

    big = cfg.register_agent_type(
        "big",
        {'width': 1, 'length': 1, 'hp': 10, 'speed': 2,
         'view_range': gw.CircleRange(6), 'attack_range': gw.CircleRange(1.5),
         'damage': 4, 'step_recover': 0.1,
         'step_reward': -0.001, 'kill_reward': 100, 'dead_penalty': -0.05, 'attack_penalty': -1,
         })

    g0 = cfg.add_group(small)
    g1 = cfg.add_group(big)
    g2 = cfg.add_group(small)
    

    a = gw.AgentSymbol(g0, index='any')
    b = gw.AgentSymbol(g1, index='any')
    c = gw.AgentSymbol(g2, index='any')

    cfg.add_reward_rule(gw.Event(a, 'attack', b), receiver=a, value=2)
    cfg.add_reward_rule(gw.Event(b, 'attack', a), receiver=b, value=2)
    cfg.add_reward_rule(gw.Event(c, 'attack', b), receiver=c, value=2)
    cfg.add_reward_rule(gw.Event(b, 'attack', c), receiver=b, value=2)

    return cfg


def generate_map(env, map_size, handles):
    width = map_size
    height = map_size

    env.init_terrain(width, height)

    init_num = 20

    # gap = 10
    # leftID, rightID = 0, 1

    # left
    # pos = []
    # for y in range(10, 45):
    #     pos.append((width / 2 - 5, y))
    #     pos.append((width / 2 - 4, y))
    # for y in range(50, height // 2 + 25):
    #     pos.append((width / 2 - 5, y))
    #     pos.append((width / 2 - 4, y))

    # for y in range(height // 2 - 25, height - 50):
    #     pos.append((width / 2 + 5, y))
    #     pos.append((width / 2 + 4, y))
    # for y in range(height - 45, height - 10):
    #     pos.append((width / 2 + 5, y))
    #     pos.append((width / 2 + 4, y))
    # env.add_walls(pos=pos, method="custom")

    
    # xs = [i for i in range(10)]
    # ys = [i for i in range(10)]
    # env.add_terrain('water', xs, ys)

    # n = init_num
    # side = int(math.sqrt(n)) * 2
    # pos = []
    border = 40
    x_num = 20
    y_num = 25
    x = np.random.randint(10, 10 + border - x_num)
    y = np.random.randint(10, map_size - y_num * 2 - 50)
    add_agents(env, x, y, handles[0], map_size, x_num, y_num, agents_border=border, random=False)

    add_agents(env, x, y + y_num + 30, handles[2], map_size, x_num, y_num, agents_border=border, random=False)

    x1 = np.random.randint(map_size - border - 10, map_size - x_num - 10)
    y1 = np.random.randint(10, map_size - y_num- 10)
    add_agents(env, x1, y1, handles[1], map_size, x_num, y_num, agents_border=border, random=False)


def add_agents(env, x, y, handle, map_size, x_num, y_num, agents_border=40, random=False):
    if random:
        env.add_agents(handle, method="random", n = x_num*y_num)
    else:
        pos = []
        for i in range(0, x_num):
            for j in range(0, y_num):
                pos.append((x + i, y + j))
        
        env.add_agents(handle, method='custom', pos=pos)