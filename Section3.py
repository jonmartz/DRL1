import random
from collections import deque
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta
from tensorflow.keras.callbacks import TensorBoard
import tensorflow.keras.backend as K
from log import Log                     # logger attached as Log.py
import gym


class SumTree:                      # Added for section 3
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class PER_Memory:  # stored as ( s, a, r, s_ ) in SumTree
    def __init__(self, capacity, **keys):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        try:    # load default values if not given others
            self.e = 0.01 if keys['e'] is None else keys['e']
            self.a = 0.8 if keys['a'] is None else keys['a']
            self.beta = 0.3 if keys['beta'] is None else keys['beta']
            self.beta_inc_per_sampling = 0.0005 if keys['beta_sample_inc'] is None else keys['beta_sample_inc']
            self.gamma = 0.95 if keys['gamma'] is None else keys['gamma']
        except KeyError:
            pass

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def store(self, model, target_model, state, action, reward, next_state, done):
        # Calculate TD-Error for Prioritized Experience Replay
        td_error = reward + \
                   self.gamma * np.argmax(target_model.predict(next_state.reshape(1, 4))) - \
                   np.argmax(model.predict(state.reshape(1, 4)))
        # Save TD-Error into Memory
        p = self._get_priority(td_error)
        self.tree.add(p, (state, action, reward, next_state, done))

    def sample_batch(self, n):
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_inc_per_sampling])
        state_batch = []
        next_state_batch = []
        action_batch = []
        reward_batch = []
        done_batch = []
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)

            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            done_batch.append(data[4])
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()            # always scale downwards

        return ((state_batch, action_batch, reward_batch, next_state_batch, done_batch), idxs, is_weight)

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)


# custom Tensorboard class
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


def custom_loss(y_true, y_pred):
    # make sure that loss is calculated as described (Algo. line #15)
    # calculating squared difference between target and predicted values
    loss = K.square(y_pred - y_true)  # (batch_size, 2)
    return loss


class NeuralNetwork:
    """
    DQN NeuralNetwork
    DQN agent that explores the game and learn how to play the game by
    learning how to predict the expected long-term return, the Q value given
    a state-action pair.
    """

    def __init__(self, _env_st_size, _env_ac_size, _lr, _layers, _optimizer, _initializer):
        self.q_net = self._build_dqn_model(_env_st_size, _env_ac_size,
                                           _lr, _layers, _optimizer, _initializer, "Q_Net")
        self.target_q_net = self._build_dqn_model(_env_st_size, _env_ac_size,
                                                  _lr, _layers, _optimizer, _initializer, "Q_Target_Net")

    def load_model(self, _path_to_model):
        target_model_dir = _path_to_model + "/target_model/"
        self.target_q_net = load_model(target_model_dir, compile=False)
        # this compile solve load_model but for custom loss function
        self.target_q_net.compile(loss=custom_loss, optimizer=RMSprop(), metrics=['mae'])


    @staticmethod
    def _build_dqn_model(_env_st_size, _env_ac_size,
                         _learning_rate, _layers, _optimizer, _initializer,
                         _name=None
                         ):
        """
        Builds a deep neural net which predicts the Q values for all possible
        actions given a state. The input should have the shape of the state, and
        the output should have the same shape as the action space since we want
        1 Q value per possible action.
        :return: Q network
        """

        state_size = _env_st_size
        action_size = _env_ac_size

        n_network = Sequential(name=_name if (_name is not None) else None)
        # build with no. of layers given
        n_network.add(Dense(_layers[0],
                            input_dim=state_size,
                            activation='relu',
                            kernel_initializer=_initializer))
        for l in range(1, len(_layers)):
            n_network.add(Dense(_layers[l],
                                activation='relu',
                                kernel_initializer=_initializer))

        #output layer with fixed (action_size) output size.
        n_network.add(Dense(action_size, activation='linear', kernel_initializer=_initializer))

        if _optimizer is "RMSprop":
            n_network.compile(loss=custom_loss, optimizer=RMSprop(lr=_learning_rate), metrics=['mae'])
        elif _optimizer is "SGD":
            n_network.compile(loss=custom_loss, optimizer=SGD(lr=_learning_rate), metrics=['mae'])
        elif _optimizer is "Adam":
            n_network.compile(loss=custom_loss, optimizer=Adam(lr=_learning_rate), metrics=['mae'])
        elif _optimizer is "Adadelta":
            n_network.compile(loss=custom_loss, optimizer=Adadelta(lr=_learning_rate), metrics=['mae'])

        return n_network

    def greedy_e_policy(self, _state, _eps):
        """
        Similar to policy but with some randomness to encourage exploration.

        :param state: the game state
        :return: action
        """
        if np.random.random() <= _eps:
            action = np.random.randint(2)
        else:
            action = self.policy(_state)
        return action

    def policy(self, _state):
        """
        Takes a state from the game environment and returns an action that
        has the highest Q value and should be taken as the next step.

        :param _state: the current game environment state
        :return: an action
        """
        state_np = _state.reshape(1, 4)
        action_q = self.q_net.predict(state_np).reshape(2, )
        action = np.argmax(action_q)
        return action

    def update_target_network(self):
        """
        Updates the current target_q_net with the q_net which brings all the
        training in the q_net to the target_q_net.

        :return: None
        """
        self.target_q_net.set_weights(self.q_net.get_weights())

    def train(self, _batch, idxs, is_weight, _gamma):
        """
        Trains the underlying network with a batch of gameplay experiences to
        help it better predict the Q values.

        :param _batch: a batch of gameplay experiences
        :return: training loss
        """
        if _batch is 0:
            return 0.0
        state_batch = np.array(_batch[0])
        action_batch = _batch[1]
        reward_batch = _batch[2]
        next_state_batch = np.array(_batch[3])
        done_batch = _batch[4]

        current_q = self.q_net.predict(state_batch)
        target_q = np.copy(current_q)
        next_q = self.target_q_net.predict(next_state_batch)
        max_next_q = np.amax(next_q, axis=1)
        batch_size = state_batch.shape[0]
        for i in range(batch_size):   # proccess the minibatch
            if not done_batch[i]:
                target_q_val = reward_batch[i] + _gamma * max_next_q[i]     # algo. line 13
            else:
                target_q_val = reward_batch[i]                              # algo. line 14

            target_q[i][action_batch[i]] = target_q_val                     # y_j = ...

        # Gradient Update. Pay attention at the sample weight as proposed by the PER Paper
        training_history = self.q_net.fit(x=state_batch,                    # algo. line 15
                                          y=target_q,
                                          batch_size=batch_size,
                                          epochs=1,
                                          verbose=0,
                                          sample_weight=np.array(is_weight)
                                          )
        loss = training_history.history['loss']
        return loss[0]

    def test(self, _env, _no_of_episodes, _render=False):
        """
        Evaluates the performance of the current DQN agent by using it to play a
        few episodes of the game and then calculates the average reward it gets.
        The higher the average reward is the better the DQN agent performs.

        :param env: the game environment
        :param agent: the DQN agent
        :return: average reward across episodes
        """
        total_reward = 0.0
        episodes_to_play = _no_of_episodes
        for i in range(episodes_to_play):
            state = _env.reset()
            if _render:
                _env.render()
            done = False
            episode_reward = 0.0
            while not done:
                action = self.policy(state)
                next_state, reward, done, _ = _env.step(action)
                episode_reward += reward
                state = next_state
            total_reward += episode_reward
        episode_reward_avg = total_reward / episodes_to_play
        return episode_reward_avg

    def save_model(self, _folder):
        model_dir = _folder + "/model/"
        target_model_dir = _folder + "/target_model/"
        self.q_net.save(model_dir)
        self.target_q_net.save(target_model_dir)


def norm_state(_state):
    """
    Normalize 4 state vars.
    These norm values were calculated during first training jobs
    by observing max values over long runs.

    :param _state: array(4, ) contains
    :return: normalized state
    """
    _state[0] /= 2.4
    _state[1] /= 3.4
    _state[2] /= 0.22
    _state[3] /= 3.4
    return _state


def initialize_parameters(_env):
    _params = {
        'env_name': _env.unwrapped.spec.id, 'state_size': _env.observation_space.shape[0],
        'action_size': _env.action_space.n, 'max_steps': _env._max_episode_steps,
        'ttl_episode': 1_200, 'learning_rate': 0.0003,
        'gamma': 0.95, 'min_eps': 0.0001, 'eps_decay_rate': 0.99,
        'epsilon': 1, 'batch_size': 128, 'C': 1_000, 'exp_replay_size': 1_000_000,
        'optimizer': 'RMSprop', 'initializer': "he_uniform",
        'layers': (8, 16), 'P': 8, 'PER_e': 0.01, 'PER_alpha': 0.2,
        'PER_beta': 0.2, 'PER_beta_sample_inc': 0.08
    }
    return _params


def train_agent(env, params, logger, save=False):
    try:
        time_started = logger.time_started
        model_dir = logger.model_dir
        logger.write("\n\t\t---START TRAINING---\n")
        tensorboard = ModifiedTensorBoard(log_dir="tensorboard/{}-{}".format("train", time_started))

        avg_100_ep = deque(maxlen=100)

        memory_buffer = PER_Memory(params['exp_replay_size'], e=params['PER_e'],
                                   a=params['PER_alpha'], beta=params['PER_beta'],
                                   beta_sample_inc=params['PER_beta_sample_inc'],
                                   gamma=params['gamma'])
        agent = NeuralNetwork(params['state_size'], params['action_size'],
                              params['learning_rate'], params['layers'],
                              params['optimizer'], params['initializer'])

        agent.q_net.summary()                       # print network structure
        agent.target_q_net.summary()

        C_cnt = 1                                   # to apply algorithm line 16
        PER_cnt = 1
        C = params['C']
        P = params['P']
        epsilon = params['epsilon']
        gamma = params['gamma']
        for episode in range(1, params['ttl_episode'] + 1):
            state = env.reset()
            state = norm_state(state)
            reward = 0
            for step in range(1, params['max_steps'] + 1):
                tensorboard.step = episode                          # TensorBoard graph update
                action = agent.greedy_e_policy(state, epsilon)      # algorithm line 5
                next_state, st_reward, done, _ = env.step(action)
                next_state = norm_state(next_state)
                reward = reward if not done else -10                # NEW reward policy
                memory_buffer.store(agent.q_net, agent.target_q_net, state, action, reward, next_state, done)
                state = next_state

                if C_cnt % C == 0:                  # algo. line 16
                    agent.update_target_network()
                C_cnt += 1
                if PER_cnt % P == 0:
                    (minibatch, idxs, is_weight) = memory_buffer.sample_batch(
                        params['batch_size'])  # batch with index and weights
                    loss = agent.train(minibatch, idxs, is_weight, gamma)
                PER_cnt += 1

                if done:                            # on terminal step start over
                    break
            # end step loop

            if epsilon > params['min_eps']:
                epsilon *= params['eps_decay_rate']
            avg_100_ep.append(step)
            avg_reward = np.average(avg_100_ep)
            logger.write('Episode {}/{}, reward:{:3}, avg: {:.4}'
                         ' loss: {:.6}, eps: {:.4}\n'.format(episode, params['ttl_episode'],
                                                             step, avg_reward, loss, epsilon))
            tensorboard.update_stats(loss=loss,
                                     reward_steps=step,
                                     reward_avg100=avg_reward
                                     )
        # end episode loop
    finally:
        if save is True:
            agent.save_model(model_dir)

        env.close()
    return agent


def test_agent(agent, env, params, logger, episodes_to_play):
    logger.write("--- Running test with this model ---")
    performance = agent.test(env, episodes_to_play, True)
    logger.write("Testing training model:\n "
                 "Performance: {}, for {} episodes.\n".
                 format(performance, episodes_to_play))


def main_load(_path_to_model):
    env = gym.make('CartPole-v1')
    params = initialize_parameters(env)
    logger = Log("log.log", params)
    # build NN with saved params and than load
    agent = NeuralNetwork(params['state_size'], params['action_size'],
                           params['learning_rate'], params['layers'],
                           params['optimizer'], params['initializer'])
    agent.load_model(_path_to_model)
    test_agent(agent, env, params, logger, 200)


def main_train():
    env = gym.make('CartPole-v1')
    params = initialize_parameters(env)

    logger = Log("log.log", params)
    agent = train_agent(env, params, logger, True)  # save=True, save model
    test_agent(agent, env, params, logger, 200)

    logger.close()


if __name__ == '__main__':
    model_name = None
    if model_name is None:
        main_train()
    else:
        main_load(model_name)

