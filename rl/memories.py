import numpy as np
class ReplayMemory(object):
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.state = None
        self.action = None
        self.reward = None
        self.state_next = None
        self.done = None
        self.counter = 0
    def store(self, state, action, reward, state_next, done):
        if self.counter == 0:
            self.state = np.empty((self.capacity,)+state.shape,dtype=np.float32)
            self.state_next = np.empty((self.capacity,) + state.shape, dtype=np.float32)
            self.action = np.empty((self.capacity,1), dtype=np.int32)
            self.reward = np.empty((self.capacity, 1), dtype=np.int32)
            self.done = np.empty((self.capacity, 1), dtype=np.int8)
        index = self.counter % self.capacity
        self.state[index] = state
        self.action[index] = action
        self.reward[index] = reward
        self.state_next[index] = state_next
        self.done[index] = done

        self.counter += 1
    def _retrieve(self, indices):
        return self.state[indices], self.action[indices], self.reward[indices], self.state_next[indices], self.done[indices]
    def sample(self, batch_size):
        '''Sample n transitions from memory, n = batch size.
        If the memory is not full, get only from from index below
        current counter.
        '''
        if self.counter >= self.capacity:
            limit = self.capacity
        else:
            limit = self.counter
        indices = np.random.choice(limit,batch_size, replace=False)
        return self._retrieve(indices)