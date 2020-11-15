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
LR = 0.01
GAMMA = 0.90
MIN_EPSILON = 0.01
EPSILON_DECAY_RATE = 0.9995
epsilon = 1.0  # moving epsilon

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
MAX_STEPS = env._max_episode_steps

batch_size = 12  # size actually defined by experience_replay size
C = 10  # set target weights every C
deque_size = 1000
MIN_REPLAY_MEMORY_SIZE = 100    # fill replay memory and than use it

render = False

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


def build_model(_state_size, _action_size, _learning_rate, option):
    my_model = Sequential()
    if option:
        print("3 Layers")
        my_model.add(Dense(26, input_dim=_state_size, activation='relu'))
        my_model.add(Dense(38, activation='relu'))
    else:
        print("5 Layers")
        my_model.add(Dense(24, input_dim=_state_size, activation='relu'))
        my_model.add(Dense(24, activation='relu'))
        my_model.add(Dense(24, activation='relu'))
        my_model.add(Dense(24, activation='relu'))

    my_model.add(Dense(action_size, activation='sigmoid'))
    # my_model.compile(loss='mse', optimizer=RMSprop(lr=_learning_rate))
    my_model.compile(loss='mse', optimizer='RMSprop')   # checking

    my_model.summary()

    return my_model


def train_agent(_model, _target_model, _memory, _terminal_state, _step, _batch_size, _gamma):
        # training only if memory is up to
        if len(_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        sample_batch = random.sample(_memory, _batch_size)
        # print("sample_batch[0]:\n\tstate{}\n\taction:{}\n\treward:{}\n\tn_state:{}\n\tdont:{}\n"
        #       .format(sample_batch[0][0], sample_batch[0][1],
        #               sample_batch[0][2], sample_batch[0][3],
        #               sample_batch[0][4]))

        # minibatch->current states , then get Q values from NN model
        current_states = np.array([transition[0] for transition in sample_batch])
        current_qs_list = _model.predict(current_states)

        # same for next state
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in sample_batch])
        future_qs_list = _target_model.predict(new_current_states)

        X = []
        y = []

        for index, (current_state, i_action, i_reward, new_current_state, i_done) in enumerate(sample_batch):

            if not i_done:
                new_q = i_reward + _gamma * np.max(future_qs_list[index])   # not a terminal state
            else:
                new_q = i_reward                                            # terminal state

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[i_action] = new_q

            # add to our training data
            X.append(current_state)
            y.append(current_qs)

        # fit as one batch, tensorboard log only on terminal state
        # tensorboard still need solving
        _model.fit(np.array(X), np.array(y),
                   batch_size=_batch_size, verbose=0, shuffle=False,
                   callbacks=[tensorboard] if _terminal_state else None)


def old_sample_batch(_memory, _batch_size, _action_size, _target_model, _gamma):
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
    print("state_batch, target_batch:{},{}".format(state_batch, target))
    # return state_batch, target
    return 0, 0


def sample_action(_epsilon, _action_size, _model, _state):
    q_action = _model.predict(np.array(_state).reshape(-1, *_state.shape))[0]    # Get action from Q table
    if np.random.random() <= _epsilon:
        _action = random.randrange(_action_size)
    else:
        _action = np.argmax(q_action)
    return _action


losses = []
experience_replay = deque(maxlen=deque_size)

model = build_model(state_size, action_size, LR, 1)
target_model = build_model(state_size, action_size, LR, 1)

for e in range(TOTAL_EPISODES):
    state = env.reset()
    # state = np.reshape(state, [1, state_size])
    done = False
    ttl_reward = 0
    for step in range(MAX_STEPS):     # env max steps = 500

        action = sample_action(epsilon, action_size, model, state)

        if render:
            env.render()

        next_state, reward, done, _ = env.step(action)
        # next_state = np.reshape(next_state, [1, state_size])

        ttl_reward += reward
        norm_reward = ttl_reward/500
        # print("experience_replay:{}".format((state, action, reward, next_state, done)))
        experience_replay.append((state, action, norm_reward, next_state, done))
        train_agent(model, target_model, experience_replay, done, step, batch_size, GAMMA)
        state = next_state

        if step % C == 0:
            target_model.set_weights(model.get_weights())

        if done:
            break

        # eps decay
        if len(experience_replay) > MIN_REPLAY_MEMORY_SIZE:
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY_RATE

    print("episode:{}/{}, steps:{}, e:{:.2}".format(e, TOTAL_EPISODES, step, epsilon))
