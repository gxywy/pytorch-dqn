import random
from collections import deque

import cv2
import gym
import numpy as np
import torch
from PIL import Image

cv2.ocl.setUseOpenCL(False)
N_CHANNEL = 4
N_HIGH = 84
N_WEIGHT = 84

class MyEnv():
    def __init__(self, env_name, skip, use_cuda=True):
        self.env = gym.make(env_name)
        self.frame_skip = skip
        self.frame_cache = deque([], maxlen=self.frame_skip)  ## 11.08 use deque instead of 4d-tensor (important)
        self.use_cuda = use_cuda
        self.action_space_number = self.env.action_space.n
        self.reset()

    def step_skip(self, action):
        totoal_reward = 0
        done = None
        for _ in range(self.frame_skip):
            s_, r, done, info = self.env.step(action)
            totoal_reward += r
            if done:
                break
        self.__append_frame(s_)
        return self.__get_state(), totoal_reward, done, info
    
    def render(self):
        self.env.render()

    def reset(self):
        s = self.env.reset()
        for _ in range(self.frame_skip):
            self.__append_frame(s)
        return self.__get_state()

    def noop_reset(self, noop_max):
        assert self.env.unwrapped.get_action_meanings()[0] == 'NOOP'
        self.reset()
        noops = random.randint(1, noop_max + 1)
        
        for _ in range(noops):
            s_, r, done, info = self.env.step(0)
            if done:
                self.reset()
        return self.__get_state()

    def __process_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)   ## 11.05 use cv2 instead of torchvision
        frame = cv2.resize(frame, (N_HIGH, N_WEIGHT), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]

    def __append_frame(self, frame):
        self.frame_cache.append(self.__process_frame(frame))
    
    def __get_state(self):  ## 11.08 add this func
        state = np.concatenate(self.frame_cache, axis=2)  # get (84 * 84 * 4) state from deque
        state = np.array(state)
        state = state.transpose((2, 0, 1))  # from (84 * 84 * 4) to (4 * 84 * 84)
        state = torch.FloatTensor(state)
        if self.use_cuda:
            return state.unsqueeze(0).cuda()
        else:
            return state.unsqueeze(0)
