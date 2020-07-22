import math
import numpy as np

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

    a = gw.AgentSymbol(g0, index='any')
    b = gw.AgentSymbol(g1, index='any')

    cfg.add_reward_rule(gw.Event(a, 'attack', b), receiver=a, value=2)
    cfg.add_reward_rule(gw.Event(b, 'attack', a), receiver=b, value=2)

    return cfg


def generate_map(env, map_size, handles):
    width = map_size
    height = map_size

    init_num = 20

    gap = 3
    leftID, rightID = 0, 1

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
    xs = [i for i in range(10)]
    ys = [i for i in range(10)]
    env.add_terrain('water', xs, ys)

    n = init_num
    side = int(math.sqrt(n)) * 2
    pos = []
    for x in range(width // 2 - gap - side, width // 2 - gap - side + side, 2):
        for y in range((height - side) // 2, (height - side) // 2 + side, 2):
            pos.append([x, y, 0])
    env.add_agents(handles[leftID], method="custom", pos=pos)

    # right
    n = init_num
    side = int(math.sqrt(n)) * 2
    pos = []
    for x in range(width // 2 + gap, width // 2 + gap + side, 4):
        for y in range((height - side) // 2, (height - side) // 2 + side, 2):
            pos.append([x, y, 0])
    env.add_agents(handles[rightID], method="custom", pos=pos)


def add_agents(env, x, y, handles, map_size, random=False):
    if random:
        x = np.random.randint(0, map_size - 1)
        y = np.random.randint(0, map_size - 1)
    pos = []
    for i in range(-5, 5):
        for j in range(-5, 5):
            pos.append((x + i, y + j))
    env.add_agents(handles[0], method="custom", pos=pos)

    pos = []
    x = np.random.randint(0, map_size - 1)
    y = np.random.randint(0, map_size - 1)
    for i in range(-5, 0):
        for j in range(-5, 5):
            pos.append((x + i, y + j))
    env.add_agents(handles[1], method="custom", pos=pos)