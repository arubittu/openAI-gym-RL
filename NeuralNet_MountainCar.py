# -*- coding: utf-8 -*-


# pytorch neural network
import gym
import numpy as np
import math
import time
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class net(nn.Module):
    
  def __init__(self):
    super(net, self).__init__()

    self.fc1 = nn.Linear(9, 64)
    self.out = nn.Linear(64, 3)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.out(x)
    return x


class sarsaAgent():
    
    def __init__(self):
        self.env = gym.make('MountainCar-v0')
        self.epsilon_T3 = 0.1
        self.learning_rate_T3 = 0.001


        # neural network
        self.net = net() 
        self.net.to(device)
        self.net.train()
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = self.learning_rate_T3, weight_decay=0)
        #
        self.weights_T3 = self.net.state_dict()
        
    
        self.lamda = 0.4 # for eligibility trace
        self.discount = 1.0
        self.train_num_episodes = 5000
        self.test_num_episodes = 100
        self.upper_bounds = [self.env.observation_space.high[0], self.env.observation_space.high[1]]
        self.lower_bounds = [self.env.observation_space.low[0], self.env.observation_space.low[1]]
        np.random.seed(0)
   
    def get_simpler_features(self,obs):
        pos, vel = obs
        vel = 2*(vel + 0.07)/0.14 - 1
        obs = 2*(pos + 1.2)/1.8 - 1
        feature_vector = torch.tensor([1 , pos, vel, pos*vel, pos**2, vel**2, pos*vel**2, vel*pos**2, (pos*vel)**2])
        feature_vector = feature_vector.float() 
        return feature_vector.to(device)

    def choose_simple_action(self,state,weights,epsilon):
        if np.random.random() < epsilon:
            action = np.random.randint(0,high = 3)
            return action
        else:
            return torch.argmax(self.net(state)).item() 
       
    def sarsa_update(self, state, action, reward, new_state, new_action, learning_rate, weights):

        # Neural network update rule
        Q_new = self.net(new_state)[new_action]
        Q_old = self.net(state)[action]
        target = reward + self.discount * Q_new
        loss = self.loss(Q_old,target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.weights_T3 = self.net.state_dict() 
        return self.net.state_dict()


    def train(self, task='T3'):
        
        if task='T3':
            get_features = self.get_simpler_features
            weights = self.weights_T3
            epsilon = self.epsilon_T3
            learning_rate = self.learning_rate_T3
            choose_action = self.choose_simple_action

        reward_list = []
        plt.clf()
        plt.cla()
        
        #self.net.load_state_dict(torch.load('model.pth'))
        
        for e in range(self.train_num_episodes):
            # lr and epsilon decay scheduler
            epsilon = self.epsilon_T3 * (1 - e/self.train_num_episodes)                  
            learning_rate = self.learning_rate_T3 * (1 - e/self.train_num_episodes)    

            #if task =='T3':
            #    self.eligiblity_trace = np.zeros(9) # eligibility trace for T3 , initialized to 0 every episode
            #else:
            #    self.eligiblity_trace = np.zeros((24,24)) 

            current_state = get_features(self.env.reset()) 
            done = False
            t = 0
            new_action = choose_action(current_state, weights, epsilon)

            while not done:
                
                action = new_action
                obs, reward, done, _ = self.env.step(action)
                new_state = get_features(obs)
                new_action = choose_action(new_state, weights, epsilon)
                weights = self.sarsa_update(current_state, action, reward, new_state, new_action, learning_rate,
                                            weights)
                current_state = new_state

                if done:
                    reward_list.append(-t)
                    break
                t += 1

        self.save_data(task)
        reward_list=[np.mean(reward_list[i-100:i]) for i in range(100,len(reward_list))]
        plt.plot(reward_list)
        plt.savefig(task + '.jpg')

 

    def load_data(self, task):
        if task == 'T3':
          return torch.load('model.pth')
        else:
          return np.load(task + '.npy')
 
    def save_data(self, task):
        if (task == 'T1'):
            with open(task + '.npy', 'wb') as f:
                np.save(f, self.weights_T1)
            f.close()
        elif (task == 'T2'):
            with open(task + '.npy', 'wb') as f:
                np.save(f, self.weights_T2)
            f.close()
        else:
            torch.save(self.net.state_dict(),'model.pth')



    def test(self, task='T3'):
        if (task == 'T1'):
            get_features = self.get_table_features
            choose_action = self.choose_action
        elif (task == 'T2'):
            get_features = self.get_better_features
            choose_action = self.choose_action
        else:
            get_features = self.get_simpler_features
            choose_action = self.choose_simple_action
        if task =='T3':
          self.net.load_state_dict(torch.load('model.pth'))
          weights = self.load_data(task) 
        else:
          weights = self.load_data(task)
        reward_list = []
        self.net.eval()
        for e in range(self.test_num_episodes):
            current_state = get_features(self.env.reset())
            done = False
            t = 0
            while not done:
                
                action = choose_action(current_state, weights, 0)
                obs, reward, done, _ = self.env.step(action)
                new_state = get_features(obs)
                current_state = new_state
                if done:
                    reward_list.append(-1.0 * t)
                    break
                t += 1
        
        return float(np.mean(reward_list))

#torch.autograd.set_detect_anomaly(False)
#torch.autograd.profiler.profile(False)
#torch.autograd.profiler.emit_nvtx(False)

task= 'T3' #args['task']
#train= 1 #int(args['train'])
agent = sarsaAgent()
agent.env.seed(0)
np.random.seed(0)
agent.env.action_space.seed(0)
#if(train):
agent.train(task)
#else:
print(agent.test(task))

