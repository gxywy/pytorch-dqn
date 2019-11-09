import random

import gym
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm

from env import MyEnv
from memory import ReplayMemory
from model import Net

# 超参数
BATCH_SIZE = 32              # mini-batch大小
REPLAY_MEMORY_SIZE = 100000  # batch总大小
EPISODE = 600                # episode个数
LR = 0.00025                 # 学习率
FINAL_EPSILON = 0.1          # 最终e贪婪
INITIAL_EPSILON = 1          # 起始e贪婪
GAMMA = 0.99                 # 奖励递减参数
TARGET_REPLACE_FREQ = 1000   # Q 目标网络的更新频率
NOOP_MAX = 30                # 初始化时无动作最大数

# 环境
ENV_NAME = 'PongNoFrameskip-v4'
env = MyEnv(ENV_NAME, skip=4)   ## 10.28 add my env
N_ACTIONS = env.action_space_number

class DQN():
    def __init__(self):
        self.evaluate_net = Net(N_ACTIONS)
        self.target_net = Net(N_ACTIONS)
        self.optimizer = torch.optim.RMSprop(self.evaluate_net.parameters(), lr=LR, alpha=0.95, eps=0.01) ## 10.24 fix alpha and eps
        self.loss_func = torch.nn.MSELoss()
        self.replay_memory = ReplayMemory(REPLAY_MEMORY_SIZE)
        self.learn_step_counter = 0
        self.evaluate_net.cuda()
        self.target_net.cuda()
    
    def select_action(self, s, epsilon):
        if random.random() > epsilon:
            q_eval = self.evaluate_net.forward(s)
            action = q_eval[0].max(0)[1].cpu().data.numpy() ## 10.21 to cpu
        else:
            action = np.asarray(random.randrange(N_ACTIONS))
        return action

    def store_transition(self, s, a, r, s_):
        self.replay_memory.store(s, a, r, s_)

    def learn(self, ):
        if self.learn_step_counter % TARGET_REPLACE_FREQ == 0:
            self.target_net.load_state_dict(self.evaluate_net.state_dict())
        self.learn_step_counter += 1

        s_s, a_s, r_s, s__s = self.replay_memory.sample(BATCH_SIZE)

        q_eval = self.evaluate_net(s_s).gather(1, a_s)
        q_next = self.target_net(s__s).detach()
        q_target = r_s + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss

def main():
    dqn = DQN()
    writer = SummaryWriter()

    step_counter = 1
    learning_flag = True
    EPSILON = INITIAL_EPSILON

    for episode in tqdm(range(EPISODE)):
        s = env.noop_reset(NOOP_MAX)
        episode_reward = 0
        while True:
            env.render()
            a = dqn.select_action(s, EPSILON)
            s_, r, done, info = env.step_skip(a)

            dqn.store_transition(s, a, r, s_)
            episode_reward += r
            
            if dqn.replay_memory.memory_counter > REPLAY_MEMORY_SIZE:
                EPSILON = FINAL_EPSILON
                if learning_flag:
                    print("Start learning...")
                    learning_flag = False
                loss = dqn.learn()
                writer.add_scalar('Train/loss', loss, step_counter)
            else:
                EPSILON -= (INITIAL_EPSILON - FINAL_EPSILON)/REPLAY_MEMORY_SIZE
            if done:
                break
            s = s_
            step_counter += 1
        writer.add_scalar('Train/reward', episode_reward, episode + 1)

    torch.save(dqn.evaluate_net.state_dict(), ENV_NAME + '.pkl')
    writer.close()

if __name__ == '__main__':
    main()
