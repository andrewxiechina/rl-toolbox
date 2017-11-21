import numpy as np
import pandas as pd
import tensorflow as tf
import gym
from rl.agents import  DQN
np.random.seed(1)
tf.set_random_seed(1)
from rl.memories import ReplayMemory

# Deep Q Network off-policy


if __name__ == "__main__":
    N_EPISODES = 10000
    MAX_STEPS = 200

    env = gym.make("FrozenLake-v0")
    RL = DQN(4, 1, env)
    step = 0
    total_rewards = []
    for episode in range(N_EPISODES):

        observation = env.reset()
        observation = np.array([observation])
        # Preprocess-> state

        total_reward = 0  # Reward tracker for reporting
        for _ in range(MAX_STEPS):

            # RL choose action based on observation
            action = RL.choose_action(observation)
            observation_, reward, done, _ = env.step(action)
            observation_ = np.array([observation_])
            RL.store_transition(observation, action, reward, observation_, done)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            if step % 300 == 0:
                RL.update_network_value()
            # Reporting functions
            total_reward += reward

            # Update target every x steps
            # Before episode ends
            observation = observation_
            if done:
                total_rewards.append(total_reward)
                if episode % 100 == 0:
                    print(episode, ": ", np.mean(total_rewards[:100]))
                    total_rewards = []
                break
            step += 1

    # end of game
    print('Training Finished')
    env.close()