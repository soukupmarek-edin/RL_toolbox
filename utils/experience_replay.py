import numpy as np
import torch
from collections import namedtuple


Transition = namedtuple("Transition", ("states", "actions", "next_states", "rewards", "done"))


class ReplayBuffer:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = None
        self.writes = 0

    def init_memory(self, transition):
        for t in transition:
            assert t.ndim == 1

        self.memory = Transition(*[np.zeros([self.capacity, t.size], dtype=t.dtype) for t in transition])

    def push(self, *args):
        if not self.memory:
            self.init_memory(Transition(*args))

        position = (self.writes) % self.capacity
        for i, data in enumerate(args):
            self.memory[i][position, :] = data

        self.writes += 1

    def sample(self, batch_size, device="cpu"):
        samples = np.random.randint(0, high=len(self), size=batch_size)
        batch = Transition(*[torch.from_numpy(np.take(d, samples, axis=0)).to(device) for d in self.memory])

        return batch

    def __len__(self):
        return min(self.writes, self.capacity)