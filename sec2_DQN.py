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
from tensorflow.keras.callbacks import TensorBoard
# import time

tf.compat.v1.disable_eager_execution() # uncomment if needed
if tf.executing_eagerly():
    print('Executing eagerly')


TOTAL_EPISODES = 1_500
LR = 0.001
GAMMA = 0.97
MIN_EPSILON = 0.001
EPSILON_DECAY_RATE = 0.9995
epsilon = 1.0  # moving epsilon

env_name = 'CartPole-v1'
env = gym.make(env_name)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
MAX_STEPS = env._max_episode_steps

batch_size = 16  # size actually defined by experience_replay size
C = 80  # set target weights every C
deque_size = 500
MIN_REPLAY_MEMORY_SIZE = 140    # fill replay memory and than use it
OPTIMIZER = "RMSprop"

render = False

fqs_max = 0


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.model = None
        self.TB_graph = tf.compat.v1.Graph()
        with self.TB_graph.as_default():
            self.writer = tf.summary.create_file_writer(self.log_dir, flush_millis=5000)
            self.writer.set_as_default()
            self.all_summary_ops = tf.compat.v1.summary.all_v2_summary_ops()
        self.TB_sess = tf.compat.v1.InteractiveSession(graph=self.TB_graph)
        self.TB_sess.run(self.writer.init())

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        self.model = model
        self._train_dir = self.log_dir + '\\train'

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # added for performance?
    def on_train_batch_end(self, _, __):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        for name, value in logs.items():
            self.TB_sess.run(self.all_summary_ops)
            if self.model is not None:
                name = f'{name}_{self.model.name}'
            self.TB_sess.run(tf.summary.scalar(name, value, step=index))
        self.model = None


# layers3=30,22
def build_model(_state_size, _action_size, _learning_rate, _layers, _optimizer):
    LR = float(_learning_rate)
    my_model = Sequential()
    _option = len(_layers) + 1
    # print("_layers[0],[1]:{},{}".format(_layers[0], _layers[1]))
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
    if _optimizer == "RMSprop":
        my_model.compile(loss='mse', optimizer=RMSprop(lr=LR))
    elif _optimizer == "SGD":
        my_model.compile(loss='mse', optimizer=SGD(lr=LR))
    elif _optimizer == "Adam":
        my_model.compile(loss='mse', optimizer=Adam(lr=LR))

    # my_model.compile(loss='mse', optimizer='RMSprop')   # checking
    my_model.summary()

    return my_model


def train_agent(_model, _target_model, _memory, _terminal_state, _step, _batch_size, _gamma, _tensorboard):
    global fqs_max

    """
    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf
    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right
        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """


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
    current_states = np.array([data[0] for data in sample_batch])
    current_qs_list = _model.predict(current_states)    # fast model

    # same for next state
    # When using target network, query it, otherwise main network should be queried
    new_current_states = np.array([data[3] for data in sample_batch])
    future_qs_list = _target_model.predict(new_current_states)  # stable model

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
    if _terminal_state:
        history = _model.fit(np.array(X), np.array(y),
                             batch_size=_batch_size, verbose=0, shuffle=False,
                             callbacks=[_tensorboard])
    else:
        history = _model.fit(np.array(X), np.array(y),
                             batch_size=_batch_size, verbose=0, shuffle=False)
    loss = history.history['loss']
    return loss[0]


def sample_action(_epsilon, _action_size, _model, _state):
    q_action = _model.predict(np.array(_state).reshape(-1, *_state.shape))[0]    # Get action from Q table
    if np.random.random() <= _epsilon:
        _action = random.randrange(_action_size)
    else:
        _action = np.argmax(q_action)
    return _action


def train(_params, _save=False):
    time_started = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format("train", time_started))

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
        tensorboard.step = e
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
            loss = train_agent(model, target_model, experience_replay, done,
                               step, train_batch_size, train_GAMMA, tensorboard)
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
        tensorboard.update_stats(reward_steps=step,
                                 reward_avg100=np.average(reward_arr),
                                 epsilon=train_epsilon)

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
    # clear_session()


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
    params['layers'] = (64, 64)
    params['TOTAL_EPISODES'] = 1_500
    params['batch_size'] = 64
    params['GAMMA'] = 0.999
    params['EPSILON_DECAY_RATE'] = 0.99992
    params['deque_size'] = 100_000
    params['C'] = 200

    params['LR'] = 0.005
    train(params, True)     # save = True

    params['LR'] = 0.008
    train(params, True)


if __name__ == '__main__':
    main()
