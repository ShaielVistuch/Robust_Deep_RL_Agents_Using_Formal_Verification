import random
from collections import namedtuple, deque
class ReplayMemory(object):
    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        return self.Transition, random.sample(self.memory, batch_size)

    def deque_value(self):
        return self.memory

    def __len__(self):
        return len(self.memory)

    def remove(self):
        self.memory.popleft()


