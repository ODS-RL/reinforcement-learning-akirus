import random
from collections import deque
class ReplayMemory(object):
    def __init__(self, memory_size: int, batch_size: int) -> None:
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = deque([], maxlen=memory_size)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)
    
class Buffer(object):
    def __init__(self) -> None:
        self.buffer = list()
    def add(self, *args):
        self.buffer.append((args))

    def take(self):
        return self.buffer
    
    def reset(self):
        self.buffer = list()

    def __len__(self):
        return len(self.buffer)
    
class ExtendableBuffer():
    def __init__(self) -> None:
        self.buffer:list[list] = list()
    
    def extend(self, *args):
        if len(self.buffer) == 0:
            for arg in args:
                self.buffer.append(list())
        assert len(self.buffer) == len(args), "Buffer sizes do not match"
        for i, arg in enumerate(args):
            self.buffer[i].extend(arg)

    def take(self):
        return self.buffer
    
    def reset(self):
        self.buffer = list()

    def __len__(self):
        return len(self.buffer[0])