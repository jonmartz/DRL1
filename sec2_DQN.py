import gym
from collections import deque
import numpy as np
import random
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta
from tensorflow.keras.backend import clear_session
import tensorflow as tf
import datetime
from log import Log


TOTAL_EPISODES = 1_500
LR = 0.001
GAMMA = 0.9
MIN_EPSILON = 0.001
EPSILON_DECAY_RATE = 0.9995
epsilon = 1.0  # moving epsilon

env_name = 'CartPole-v1'
env = gym.make(env_name)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
MAX_STEPS = env._max_episode_steps

batch_size = 64  # size actually defined by experience_replay size
C = 80  # set target weights every C
deque_size = 500
MIN_REPLAY_MEMORY_SIZE = 140    # fill replay memory and than use it
OPTIMIZER = "RMSprop"

render = False

# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
tensorboard = None


# layers3=30,22
def build_model(_state_size, _action_size, _learning_rate, _layers, _optimizer):
    LR = float(_learning_rate)
    my_model = Sequential()
    _option = len(_layers) + 1
    print("_layers[0],[1]:{},{}".format(_layers[0], _layers[1]))
    if _option == 3:
        print("3 Layers")
        my_model.add(Dense(_layers[0], input_dim=_state_size, activation='relu'))
        my_model.add(Dense(_layers[1], activation='relu'))
    elif _option == 5:
        print("5 Layers")
        my_model.add(Dense(24, input_dim=_state_size, activation='relu'))
        my_model.add(Dense(24, activation='relu'))
        my_model.add(Dense(24, activation='relu'))
        my_model.add(Dense(24, activation='relu'))
    else:
        print("Check you model def.")
        return

    my_model.add(Dense(action_size, activation='linear'))
    print('_learning_rate:{}'.format(type(LR)))
    if _optimizer == "RMSprop":
        my_model.compile(loss='mse', optimizer=RMSprop(lr=LR))
    elif _optimizer == "SGD":
        my_model.compile(loss='mse', optimizer=SGD(lr=LR))
    elif _optimizer == "Adam":
        my_model.compile(loss='mse', optimizer=Adam(lr=LR))

    # my_model.compile(loss='mse', optimizer='RMSprop')   # checking
    my_model.summary()

    return my_model


def train_agent(_model, _target_model, _memory, _terminal_state, _step, _batch_size, _gamma):
    # training only if memory is up to
    if len(_memory) < MIN_REPLAY_MEMORY_SIZE:
        return 0.0

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
    history = _model.fit(np.array(X), np.array(y),
               batch_size=_batch_size, verbose=0, shuffle=False)
    loss = history.history['loss']
    return loss[0]


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


def train(_params, _save=False):
    time_started = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = "saved_model/" + time_started
    logger = Log(model_dir, "log.log")
    logger.write("{} \t\t Training params:\n".format(time_started))
    for item in _params:
        logger.write("{}: {}\n".format(item, _params[item]))
    logger.write("\n\t\t---START TRAINING---\n")

    train_state_size = _params['state_size']
    train_action_size = _params['action_size']
    train_MAX_STEPS = _params['MAX_STEPS']
    train_TOTAL_EPISODES = _params['TOTAL_EPISODES']
    train_LR = _params['LR']
    train_GAMMA = _params['GAMMA']
    train_MIN_EPSILON = _params['MIN_EPSILON']
    train_EPSILON_DECAY_RATE = _params['EPSILON_DECAY_RATE']
    train_epsilon = _params['epsilon']
    train_batch_size = _params['batch_size']
    train_C = _params['C']
    train_deque_size = _params['deque_size']
    train_MIN_REPLAY_MEMORY_SIZE = _params['MIN_REPLAY_MEMORY_SIZE']
    train_OPTIMIZER = _params['OPTIMIZER']
    layers = _params['layers']

    losses = []
    experience_replay = deque(maxlen=train_deque_size)
    reward_arr = deque(maxlen=100)

    model = build_model(train_state_size, train_action_size, train_LR, layers, train_OPTIMIZER)
    target_model = build_model(train_state_size, train_action_size, train_LR, layers, train_OPTIMIZER)

    for e in range(train_TOTAL_EPISODES):
        state = env.reset()
        # state = np.reshape(state, [1, state_size])
        done = False
        ttl_reward = 0
        los_ttl = 0
        for step in range(train_MAX_STEPS):     # env max steps = 500

            action = sample_action(train_epsilon, train_action_size, model, state)

            if render:
                env.render()

            next_state, reward, done, _ = env.step(action)
            # next_state = np.reshape(next_state, [1, state_size])

            ttl_reward += reward
            norm_reward = ttl_reward/500
            # print("experience_replay:{}".format((state, action, reward, next_state, done)))
            experience_replay.append((state, action, norm_reward, next_state, done))
            loss = train_agent(model, target_model, experience_replay, done, step, train_batch_size, train_GAMMA)
            los_ttl += loss
            # losses.append()loss
            state = next_state

            if step % train_C == 0:
                target_model.set_weights(model.get_weights())
                # print("Set target weights")

            if done:
                break

            # eps decay
            if len(experience_replay) > train_MIN_REPLAY_MEMORY_SIZE:
                if train_epsilon > train_MIN_EPSILON:
                    train_epsilon *= train_EPSILON_DECAY_RATE
        # end step loop
        reward_arr.append(step)
        logger.write("episode: {:4}/{} steps: {:3} eps: {:.2} avg100: {:.4} loss: {:.4}\n".format(
            e, train_TOTAL_EPISODES, step, train_epsilon, np.average(reward_arr), los_ttl))
    # end episode loop
    if _save is True:
        save_training_res(model, target_model, model_dir, _params)
    # loss_file =
    logger.close()
    #end


def save_training_res(_model, _target_model, _folder, _params):
    model_dir = _folder + "/model/"
    target_model_dir = _folder + "/target_model/"

    _model.save(model_dir)
    _target_model.save(target_model_dir)
    clear_session()


def initialize_parameters_from_global():
    _params = {'env_name': env_name, 'state_size': state_size, 'action_size': action_size,
             'MAX_STEPS': MAX_STEPS, 'TOTAL_EPISODES': TOTAL_EPISODES, 'LR': LR,
             'GAMMA': GAMMA, 'MIN_EPSILON': MIN_EPSILON, 'EPSILON_DECAY_RATE': EPSILON_DECAY_RATE,
             'epsilon': epsilon, 'batch_size': batch_size, 'C': C, 'deque_size': deque_size,
             'MIN_REPLAY_MEMORY_SIZE': MIN_REPLAY_MEMORY_SIZE, 'OPTIMIZER': OPTIMIZER,
             }
    return _params


def main():
    params = initialize_parameters_from_global()
    params['layers'] = (16, 12)
    params['TOTAL_EPISODES'] = 1_500

    params['batch_size'] = 128
    train(params, True)     # save = True

    params['batch_size'] = 32
    train(params, True)

    params['batch_size'] = 16
    train(params, True)

if __name__ == '__main__':
    main()
