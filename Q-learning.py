import numpy as np
import gym

env = gym.make("FrozenLake-v1", is_slippery=False)

state_size = env.observation_space.n
action_size = env.action_space.n
q_table = np.zeros((state_size, action_size))

total_episodes = 1000
max_steps = 100
learning_rate = 0.8
gamma = 0.95
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.005

for episode in range(total_episodes):
    state = env.reset()[0]
    done = False

    for step in range(max_steps):
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state, :])

        new_state, reward, done, truncated, info = env.step(action)
        q_table[state, action] = q_table[state, action] + learning_rate * (reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action])
        state = new_state

        if done:
            break

    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

print("Trained Q-Table:")
print(q_table)

# Test the agent
state = env.reset()[0]
env.render()
done = False
while not done:
    action = np.argmax(q_table[state, :])
    state, reward, done, truncated, info = env.step(action)
    env.render()

print("Reward:", reward)
