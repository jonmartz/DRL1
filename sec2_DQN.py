import gym
from collections import deque
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta
import tensorflow as tf
import datetime

TOTAL_EPISODES = 500
T = 500
LR = 0.001
GAMMA = 0.90
MIN_EPSILON = 0.01
EPSILON_DECAY_RATE = 0.9995
epsilon = 1.0  # moving epsilon

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
batch_size = 1  # size actually defined by experience_replay size
C = 4
deque_size = 2000
train_start = batch_size * 10

render = False


def build_model(_state_size, _action_size, _learning_rate, option):
    my_model = Sequential()
    if option:
        print("3 Layers")
        my_model.add(Dense(200, input_dim=_state_size, activation='relu'))
        my_model.add(Dense(200, activation='relu'))
        # my_model.add(Dense(4, activation='relu'))
    else:
        print("5 Layers")
        my_model.add(Dense(24, input_dim=_state_size, activation='relu'))
        my_model.add(Dense(24, activation='relu'))
        my_model.add(Dense(24, activation='relu'))
        my_model.add(Dense(24, activation='relu'))
        # my_model.add(Dense(18, activation='relu'))

    my_model.add(Dense(action_size, activation='sigmoid'))
    my_model.compile(loss='mse', optimizer=Adam(lr=_learning_rate))

    my_model.summary()

    return my_model


def sample_batch(_memory, _batch_size, _action_size, _target_model, _gamma):
    if len(_memory) < train_start:
        return
    minibatch = random.sample(_memory, min(len(_memory), _batch_size))
    # Initialize
    state_batch = np.zeros((_batch_size, 4))
    next_state_batch = np.zeros((_batch_size, 4))
    action_batch, reward_batch, done_batch = [], [], []

    for i in range(_batch_size):
        state_batch[i] = minibatch[i][0]
        action_batch.append(minibatch[i][1])
        reward_batch.append(minibatch[i][2])
        next_state_batch[i] = minibatch[i][3]
        done_batch.append(minibatch[i][4])

    # do batch prediction to save speed
    target = np.zeros((_batch_size, _action_size))
    target_next = _target_model.predict(next_state_batch)

    # print("reward_batch:{}".format(reward_batch))

    for i in range(_batch_size):
        # correction on the Q value for the action used
        if done_batch[i]:
            target[i][action_batch[i]] = reward_batch[i]  # for terminal transition
        else:
            # Q_max = max_a' Q_target(s', a')
            target[i][action_batch[i]] = reward_batch[i] + _gamma * (np.amax(target_next[i]))  # for non-terminal transition

    # return state_batch, target


def sample_action(_epsilon, _action_size, _model, _state):
    _target = _model.predict(_state)
    if np.random.random() <= _epsilon:
        _action = random.randrange(_action_size)
    else:
        _action = np.argmax(_target)
    return _action, _target


losses = []
experience_replay = deque(maxlen=deque_size)

model = build_model(state_size, action_size, LR, 1)
target_model = build_model(state_size, action_size, LR, 1)

for e in range(TOTAL_EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    ttl_reward = 0
    for step in range(env._max_episode_steps):     # env max steps = 500

        action, target = sample_action(epsilon, action_size, model, state)

        if render:
            env.render()

        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        ttl_reward += reward
        norm_reward = ttl_reward/500
        # print("experience_replay:{}".format((state, action, reward, next_state, done)))
        experience_replay.append((state, action, norm_reward, next_state, done))

        state = next_state

        if done:
            break

        sample_batch(experience_replay, batch_size, action_size, target_model, GAMMA)

        history = model.fit(state, target, batch_size=batch_size, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        losses.append(loss)

        # eps decay
        if len(experience_replay) > train_start:
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY_RATE

        if step % C == 0:
            target_model.set_weights(model.get_weights())

    print("episode:{}/{}, steps:{}, e:{:.2}".format(e, TOTAL_EPISODES, step, epsilon))
