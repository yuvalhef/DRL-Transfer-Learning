import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import time


env = gym.make('CartPole-v1')

np.random.seed(1)


class PolicyNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='cartpole_policy'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):

            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            self.W1 = tf.get_variable("W1", [self.state_size, 12], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b1 = tf.get_variable("b1", [12], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [12, self.action_size], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b2 = tf.get_variable("b2", [self.action_size], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.elu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


class ValueEstimator:
    def __init__(self, state_size, learning_rate, name='value_network'):
        self.state_size = state_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):

            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state_v")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards_v")

            self.W1 = tf.get_variable("W1", [self.state_size, 12], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b1 = tf.get_variable("b1", [12], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [12, 1], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b2 = tf.get_variable("b2", [1], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.elu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # Softmax probability distribution over actions
            self.value_estimate = tf.squeeze(self.output)
            self.loss = tf.squared_difference(self.value_estimate, self.R_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)


# Define hyperparameters
state_size = 6
action_size = 6
actions_dict = [0, 0, 0, 1, 1, 1]

max_episodes = 5000
max_steps = 501
discount_factor = 0.99
learning_rate = 0.0007

render = False

# Initialize the policy network
tf.reset_default_graph()
policy = PolicyNetwork(state_size, action_size, learning_rate)
value = ValueEstimator(state_size, 0.006)

saver = tf.train.Saver()
checkpoint_path = "models/cartpole_expert.ckpt"


# Setup TensorBoard Writer
writer = tf.summary.FileWriter('cartpole_expert/output')

start = time.time()

# Start training the agent
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    solved = False
    episode_rewards = np.zeros(max_episodes)
    average_rewards = 0.0
    rewards_stats = []
    losses_v = []
    losses_e = []
    avg_losses_e = []
    avg_losses_v = []
    episode_steps = []

    for episode in range(max_episodes):
        state = env.reset()
        state = np.append(state, [0] * (state_size - len(state)))
        state = state.reshape([1, state_size])
        episode_transitions = []

        for step in range(max_steps):
            actions_distribution = sess.run(policy.actions_distribution, {policy.state: state})
            action_idx = np.random.choice(np.arange(action_size), p=actions_distribution)
            action = actions_dict[action_idx]
            next_state, reward, done, _ = env.step(action)
            next_state = np.append(next_state, [0] * (state_size - len(next_state)))
            next_state = next_state.reshape([1, state_size])

            if render:
                env.render()

            action_one_hot = np.zeros(action_size)
            action_one_hot[action_idx] = 1
            episode_rewards[episode] += reward
            action = action_one_hot

            # Calculate TD Target
            value_next = sess.run(value.value_estimate, {value.state: next_state})
            state_value = sess.run(value.value_estimate, {value.state: state})
            if done:
                td_target = reward
                td_error = td_target - state_value
            else:
                td_target = reward + discount_factor * value_next
                td_error = td_target - state_value

            # Update the value estimator
            feed_dict_v = {value.state: state, value.R_t: td_target}
            _, loss_value = sess.run([value.optimizer, value.loss], feed_dict_v)
            losses_v.append(loss_value)

            # Update the policy estimator
            # using the td error as our advantage estimate
            feed_dict = {policy.state: state, policy.R_t: td_error, policy.action: action}
            _, loss = sess.run([policy.optimizer, policy.loss], feed_dict)
            losses_e.append(loss)

            if done:
                if episode > 98:
                    # Check if solved
                    average_rewards = np.mean(episode_rewards[(episode - 99):episode+1])
                    rewards_stats.append(average_rewards)
                    if average_rewards > 475:
                        print(' Solved at episode: ' + str(episode))
                        solved = True
                        saver.save(sess, checkpoint_path)
                print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode], round(average_rewards, 2)))
                break
            state = next_state

        if solved:
            break

        avg_losses_e.append(np.mean(np.array(losses_e)))
        avg_losses_v.append(np.mean(np.array(losses_v)))
        losses_e = []
        losses_v = []
        episode_steps.append(step)

end = time.time()

print('Time to convergence: '+str(end - start))


def plot_reward(dir, start, rewards, name):
    df = pd.DataFrame()
    df[name] = rewards
    df['episode'] = list(range(start, len(rewards)+start))
    df.to_csv(dir + '/' + name + '.csv')
    ax =df.plot(x='episode', y=name)
    ax.set_xlabel("Episode")
    ax.set_ylabel(name)
    ax.legend().remove()
    ax = plt.savefig(dir+'/'+name+'.jpg')


def plot_loss(dir, name, losses):
    df = pd.DataFrame(losses)
    ax = df.plot()
    df.to_csv(log_dir+"/Loss_" + name + '_network.csv')
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss - "+name+' network')
    ax.legend().remove()
    ax = plt.savefig(dir+'/Loss_'+name+'.jpg')


def plot_steps(dir, steps):
    df = pd.DataFrame(steps)
    ax = df.plot()
    df.to_csv(dir+"/steps.csv")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.legend().remove()
    ax = plt.savefig(dir+"/steps.jpg")


log_dir = 'cartpole_expert'
os.makedirs(log_dir, exist_ok=True)

plot_reward(log_dir, 100, rewards_stats, 'Average 100 Episode Rewards')
plot_reward(log_dir, 0, episode_rewards[:episode], 'Episode Rewards')
plot_loss(log_dir, 'value', losses_v)
plot_loss(log_dir, 'policy', losses_e)
plot_steps(log_dir, episode_steps)
print('max episode: {}'.format(episode))

vloss_summary = tf.Summary()
eloss_summary = tf.Summary()
total_rewards_summary = tf.Summary()
avg_rewards_summary = tf.Summary()
episode_steps_summary = tf.Summary()

for i in range(len(avg_losses_e)):
    eloss_summary.value.add(tag='Policy Loss', simple_value=avg_losses_e[i])
    writer.add_summary(eloss_summary, i)
writer.flush()

for i in range(len(rewards_stats)):
    avg_rewards_summary.value.add(tag='Average 100 Episodes rewards', simple_value=rewards_stats[i])
    writer.add_summary(avg_rewards_summary, i)
writer.flush()

for i in range(len(avg_losses_v)):
    vloss_summary.value.add(tag='Value Loss', simple_value=avg_losses_v[i])
    writer.add_summary(vloss_summary, i)
writer.flush()

for i in range(episode):
    total_rewards_summary.value.add(tag='Episode Rewards', simple_value=episode_rewards[i])
    writer.add_summary(total_rewards_summary, i)
writer.flush()

for i in range(len(episode_steps)):
    episode_steps_summary.value.add(tag='Episode Steps', simple_value=episode_steps[i])
    writer.add_summary(episode_steps_summary, i)
writer.flush()
