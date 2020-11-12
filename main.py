import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import itertools


def fit_model(n_episodes, max_n_steps, alpha, gamma, verbose=False, save_q_values_at=None):
    """
    Fit a q-learning model.
    :param n_episodes: to run
    :param max_n_steps: per episode
    :param alpha: learning rate
    :param gamma: discount parameter
    :param verbose: true to print total rewards and #steps per episode
    :return: the lists: rewards_per_episode, steps_per_episode, epsilons
    """
    q_values = np.zeros([env.observation_space.n, env.action_space.n])
    rewards_per_episode, steps_per_episode, epsilons = [], [], []
    epsilon = epsilon_max

    total_steps = 0
    for episode in range(n_episodes):
        state, rewards, steps, done = env.reset(), 0, 0, False

        if epsilon > epsilon_min:
            epsilon *= (1 - epsilon_decay_rate)
        epsilons.append(epsilon)

        while not done and steps < max_n_steps:
            steps += 1
            total_steps += 1
            if save_q_values_at is not None and total_steps in save_q_values_at:
                save_q_values(q_values, total_steps)
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore action space
            else:
                action = np.argmax(q_values[state])  # Exploit learned values

            next_state, reward, done, info = env.step(action)
            rewards += reward

            if done:
                target = reward
                if reward == 0:
                    steps = 100
            else:
                target = reward + gamma * np.max(q_values[next_state])
            q_values[state, action] = (1 - alpha) * q_values[state, action] + alpha * target

            state = next_state
        if verbose:
            print('episode %d/%d reward=%s steps=%d' % (episode + 1, n_episodes, rewards, steps))
        rewards_per_episode.append(rewards)
        steps_per_episode.append(steps)

    return rewards_per_episode, steps_per_episode, epsilons


def get_best_params(verbose=True):
    """
    Find the best parameters, according to the success rate metric.
    :param verbose: true to print performance per combination of parameters
    :return: best_alpha, best_gamma
    """
    # alphas = np.linspace(0.05, 0.95, 19)
    # gammas = np.linspace(0.05, 0.95, 19)
    alphas = np.linspace(0.05, 0.09, 5)
    gammas = np.linspace(0.9, 0.99, 10)

    best_alpha, best_gamma, best_success_rate = 0, 0, 0
    i = 0
    for alpha, gamma in itertools.product(alphas, gammas):
        i += 1
        success_rates = []
        for iteration in range(n_iterations_for_tuning):
            rewards_per_episode, steps_per_episode, _ = fit_model(n_episodes, max_n_steps, alpha, gamma)
            success_rates.append(np.mean(rewards_per_episode))
        success_rate = np.mean(success_rates)
        best_string = ''
        if success_rate > best_success_rate:
            best_alpha, best_gamma, best_success_rate = alpha, gamma, success_rate
            best_string = ' NEW BEST'
        if verbose:
            print('%d/%d alpha=%.2f gamma=%.2f success_rate=%.4f%s' %
                  (i, len(alphas) * len(gammas), alpha, gamma, success_rate, best_string))
    return best_alpha, best_gamma


def save_q_values(q_values, steps):
    """
    Plot the q-values as a colormap
    :param q_values: the q-value table
    :param steps: for the plot title
    """
    fig, ax = plt.subplots(figsize=(3.5, 10.5))
    ax.matshow(q_values, cmap=plt.cm.Blues)
    for i in range(q_values.shape[1]):
        for j in range(q_values.shape[0]):
            ax.text(i, j, '%.4f' % q_values[j, i], va='center', ha='center')
    plt.xlabel('action')
    plt.ylabel('state')
    plt.title('Q-values at step=%d' % steps)
    plt.savefig('q-values %d steps.png' % steps, bbox_inches='tight')
    plt.show()


env = gym.make('FrozenLake-v0')

n_episodes = 5000
max_n_steps = 100
n_iterations_for_tuning = 3

epsilon_min, epsilon_max, epsilon_decay_rate = 0.0, 0.8, 0.002
# best_alpha, best_gamma = get_best_params()
best_alpha, best_gamma = 0.07, 0.99  # best ones found in past execution

print('\ntesting best hyper-parameters (alpha=%.2f, gamma=%.2f)...' % (best_alpha, best_gamma))
rewards_per_episode, steps_per_episode, epsilons = fit_model(
    n_episodes, max_n_steps, best_alpha, best_gamma, save_q_values_at=[500, 2000])
success_rate = np.mean(rewards_per_episode)
print('\nsuccess rate = %.6f' % success_rate)

plt.plot(range(len(epsilons)), epsilons)
plt.title('epsilon')
plt.show()

plt.scatter(range(len(rewards_per_episode)), rewards_per_episode)
plt.xlabel('episode')
plt.ylabel('reward')
plt.title('Rewards')
plt.savefig('rewards.png', bbox_inches='tight')
plt.show()

x = range(0, n_episodes, max_n_steps)
y = np.mean(np.array(steps_per_episode).reshape(-1, max_n_steps), axis=1)
plt.plot(x, y)
plt.xlabel('episode')
plt.ylabel('#steps')
plt.title('Steps')
plt.savefig('steps.png', bbox_inches='tight')
plt.show()

env.close()
print('\ndone')
