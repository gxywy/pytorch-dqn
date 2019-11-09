import torch
import numpy as np

N_HIGH = 84
N_WEIGHT = 84
N_CHANNEL = 4

class ReplayMemory():
    def __init__(self, memory_size):
        self.memory_counter = 0
        self.memory_size = memory_size
        self.state_memory = torch.FloatTensor(self.memory_size, N_CHANNEL, N_HIGH, N_WEIGHT)
        self.action_memory = torch.LongTensor(self.memory_size)
        self.reward_memory = torch.FloatTensor(self.memory_size)
        self.state__memory = torch.FloatTensor(self.memory_size, N_CHANNEL, N_HIGH, N_WEIGHT)

    def store(self, s, a, r, s_):
        index = self.memory_counter % self.memory_size
        self.state_memory[index] = s
        self.action_memory[index] = torch.LongTensor([a.tolist()])
        self.reward_memory[index] = torch.FloatTensor([r])
        self.state__memory[index] = s_
        self.memory_counter += 1
    
    def sample(self, size):
        sample_index = np.random.choice(self.memory_size, size)
        state_sample = torch.FloatTensor(size, N_CHANNEL, N_HIGH, N_WEIGHT).cuda()
        action_sample = torch.LongTensor(size, 1).cuda()
        reward_sample = torch.FloatTensor(size, 1).cuda()
        state__sample = torch.FloatTensor(size, N_CHANNEL, N_HIGH, N_WEIGHT).cuda()
        for index in range(sample_index.size):
            state_sample[index] = self.state_memory[sample_index[index]]
            action_sample[index] = self.action_memory[sample_index[index]]
            reward_sample[index] = self.reward_memory[sample_index[index]]
            state__sample[index] = self.state__memory[sample_index[index]]
        return state_sample, action_sample, reward_sample, state__sample