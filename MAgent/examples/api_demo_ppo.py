import numpy as np

import sys
sys.path.append('..')

import magent
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Dropout, Flatten, concatenate
from keras import backend as K
from keras.optimizers import Adam

LOSS_CLIPPING = 0.2 # Only implemented clipping for the surrogate loss, paper said it was best
EPOCHS = 50
NOISE = 1.0 # Exploration noise
LR = 0.0001 # Lower lr stabilises training greatly
ENTROPY_LOSS = 0.005
NUM_ACTIONS = 21

DUMMY_ACTION, DUMMY_VALUE = np.zeros((1, NUM_ACTIONS)), np.zeros((1, 1))

def proximal_policy_optimization_loss(advantage, old_prediction):
    def loss(y_true, y_pred):
        prob = K.sum(y_true * y_pred, axis=-1)
        old_prob = K.sum(y_true * old_prediction, axis=-1)
        r = prob/(old_prob + 1e-10)
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10)))
    return loss


def proximal_policy_optimization_loss_continuous(advantage, old_prediction):
    def loss(y_true, y_pred):
        var = K.square(NOISE)
        pi = 3.1415926
        denom = K.sqrt(2 * pi * var)
        prob_num = K.exp(- K.square(y_true - y_pred) / (2 * var))
        old_prob_num = K.exp(- K.square(y_true - old_prediction) / (2 * var))

        prob = prob_num/denom
        old_prob = old_prob_num/denom
        r = prob/(old_prob + 1e-10)

        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage))
    return loss

class PPO():
    def __init__(self):
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        
        
    def build_actor(self):
        advantage = Input(shape=(1,), name='advantage')
        old_prediction = Input(shape=(21,), name='old_prediction')
        view_input = Input(shape=(13,13,7,), name='input_view')
        h_conv1 = Conv2D(filters=32, kernel_size=3, activation='relu', name='conv1')(view_input)
        h_conv2 = Conv2D(filters=32, kernel_size=3, activation='relu', name='conv2')(h_conv1)
        flatten_view = Flatten(name='view_flatten')(h_conv2)
        h_view = Dense(256, activation='relu', name='hidden_view')(flatten_view)
        feature_input = Input(shape=(34,), name='input_feature')
        h_feature = Dense(256, activation='relu', name='hidden_feature')(feature_input)
        concat = concatenate([h_view, h_feature])
        action = Dense(21, activation='softmax', name='output')(concat)
        # model = Model(inputs=[view_input,feature_input], outputs=[action])
        model = Model(inputs=[view_input,feature_input, advantage, old_prediction], outputs=[action])
        model.compile(optimizer=Adam(lr=LR),
                      loss=[proximal_policy_optimization_loss(advantage, old_prediction)])
        model.summary()
        return model

    def build_critic(self):
        view_input = Input(shape=(13,13,7,), name='input_view')
        h_conv1 = Conv2D(filters=32, kernel_size=3, activation='relu', name='conv1')(view_input)
        h_conv2 = Conv2D(filters=32, kernel_size=3, activation='relu', name='conv2')(h_conv1)
        flatten_view = Flatten(name='view_flatten')(h_conv2)
        h_view = Dense(256, activation='relu', name='hidden_view')(flatten_view)
        feature_input = Input(shape=(34,), name='input_feature')
        h_feature = Dense(256, activation='relu', name='hidden_feature')(feature_input)
        concat = concatenate([h_view, h_feature])
        out_value = Dense(1, activation='softmax', name='value_of_state')(concat)

        model = Model(inputs=[view_input, feature_input], outputs=[out_value])
        model.compile(optimizer=Adam(lr=LR), loss='mse')
        model.summary()
        return model
    
    def get_action(self, observation, batch_size=64):
        view, feature = observation[0], observation[1]
        n = len(view)
        ret = []
        for i in range(0, n, batch_size):
            action = []
            beg, end = i, i+batch_size
            t_DUMMY_VALUE = np.array([DUMMY_VALUE[0] for _ in range(batch_size)])
            t_DUMMY_ACTION = np.array([DUMMY_ACTION[0] for _ in range(batch_size)])
            p = self.actor.predict([view[beg:end], feature[beg:end], t_DUMMY_VALUE, t_DUMMY_ACTION])
            for j in range(len(p)):
                action.append(np.argmax(p[j]))
            ret.append(action)
        action = np.argmax(p[0])
        action_matrix = np.zeros(21)
        action_matrix[action] = 1
        ret = np.concatenate(ret).astype(np.int32)
        return ret


tf.reset_default_graph()
map_size = 100
env = magent.GridWorld('battle', map_size=map_size)
handles = env.get_handles()
env.reset()
env.add_agents(handles[0], method="random", n=map_size * map_size * 0.02)
env.add_agents(handles[1], method="random", n=map_size * map_size * 0.02)
model1 = PPO()
model2 = PPO()
done = False
step_ct = 0
while not done:
    # take actions for deers
    obs_1 = env.get_observation(handles[0])
    ids_1 = env.get_agent_id(handles[0])
    acts_1 = model1.get_action(obs_1)
    env.set_action(handles[0], acts_1)

    # take actions for tigers
    obs_2  = env.get_observation(handles[1])
    ids_2  = env.get_agent_id(handles[1])
    acts_2 = model2.get_action(obs_2)
    env.set_action(handles[1], acts_2)

    # simulate one step
    done = env.step()

    # render
    env.render()

    # get reward
    reward = [sum(env.get_reward(handles[0])), sum(env.get_reward(handles[1]))]

    # clear dead agents
    env.clear_dead()

    # print info
    if step_ct % 10 == 0:
        print("step: %d\t handles0' reward: %d\t handles1' reward: %d" %
                (step_ct, reward[0], reward[1]))

    step_ct += 1
    if step_ct > 250:
        break