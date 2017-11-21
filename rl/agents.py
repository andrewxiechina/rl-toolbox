import numpy as np
import pandas as pd
import tensorflow as tf
from rl.memories import ReplayMemory
from rl.policies import EpsilonAnneal

memory = ReplayMemory(capacity=500)
policy = EpsilonAnneal(initial_epsilon=1.0, final_epsilon=0.1, timesteps=10000, start_timestep=0)
class DQN:
    def __init__(
            self,
            n_actions,
            n_features,
            env,
            learning_rate=0.1,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            memory=memory,
            policy = policy
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.memory = memory
        # total learning step
        self.policy = policy
        self.env = env

        # consist of [target_net, evaluate_net]
        self._build_net()

        self.sess = tf.Session()
        self.step = 0

        self.sess.run(tf.global_variables_initializer())
        self.learn_step_counter = 0


    def _build_model(self, n=64):
        Sequential = tf.keras.models.Sequential
        Dense = tf.keras.layers.Dense
        model = Sequential()
        model.add(Dense(n, input_shape=(self.n_features,), activation="relu"))
        model.add(Dense(n, activation="relu"))
        model.add(Dense(n, activation="relu"))
        model.add(Dense(n, activation="relu"))
        model.add(Dense(self.n_actions))
        return model
    def _build_net(self):
        self.Q = self._build_model()
        self.Q_ = self._build_model()
        self.Q.compile(optimizer='rmsprop',
                loss='mse')

    def store_transition(self, s, a, r, s_, d):
        self.memory.store(s,a,r,s_,d)
    def choose_action(self, observation):
        epsilon = self.policy(timestep=self.step)
        if np.random.random() < epsilon:
            return self.env.action_space.sample()

        q = self.Q.predict(observation[np.newaxis, :], batch_size=1)
        return np.argmax(q)
    def update_network_value(self):
        w = self.Q.get_weights()
        self.Q_.set_weights(w)
    def learn(self):
        # sample batch memory from all memory
        s, a, r, s_, d = self.memory.sample(self.batch_size)
        # q_ is calculated from Q_ network (fixed), q is from Q
        q = self.Q.predict(s, batch_size=self.batch_size)
        q_ = self.Q_.predict(s, batch_size=self.batch_size)
        # Copy q, so that if action is not chozen, no update will be done
        y = q.copy()
        # Update sate, action pair with new reward value
        y[np.arange(self.batch_size), a.astype(int)] = r + self.gamma * np.max(q_, axis=1)

        # train eval network
        log = self.Q.train_on_batch(s, y)
        print(log)
        self.step += 1
