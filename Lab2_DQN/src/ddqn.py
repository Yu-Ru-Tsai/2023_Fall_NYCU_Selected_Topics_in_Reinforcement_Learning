import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from base_agent import DQNBaseAgent
from models.atari_model import AtariNetDQN
import gym
import random
import torch.nn.functional as F
from gym.wrappers import FrameStack
import cv2

class AtariDDQNAgent(DQNBaseAgent):
	def __init__(self, config):
		super(AtariDDQNAgent, self).__init__(config)
		### TODO ###
  		# initialize env
		self.env = gym.make(config["env_id"], render_mode="rgb_array")
		self.env = gym.wrappers.ResizeObservation(self.env, (84, 84))
		self.env = gym.wrappers.GrayScaleObservation(self.env)
		self.env = gym.wrappers.FrameStack(self.env, 4)
		# initialize env

		### TODO ###
		# initialize test_env
		self.test_env = gym.make(config["env_id"], render_mode="rgb_array")
		self.test_env = gym.wrappers.ResizeObservation(self.test_env, (84, 84))
		self.test_env = gym.wrappers.GrayScaleObservation(self.test_env)
		self.test_env = gym.wrappers.FrameStack(self.test_env, 4)
  
		# initialize behavior network and target network
		self.behavior_net = AtariNetDQN(self.env.action_space.n)
		self.behavior_net.to(self.device)
		self.target_net = AtariNetDQN(self.env.action_space.n)
		self.target_net.to(self.device)
		self.target_net.load_state_dict(self.behavior_net.state_dict())
  
		# initialize optimizer
		self.lr = config["learning_rate"]
		self.optim = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr, eps=1.5e-4)
		
	def decide_agent_actions(self, observation, epsilon=0.0, action_space=None):
		### TODO ###
		# get action from behavior net, with epsilon-greedy selection 
  		# observation shape: (4, 210, 160, 3)

		if random.random() < epsilon:   # pick random action
			action = action_space.sample()
		else:                  
			with torch.no_grad():		
				observation = np.array(observation)	
				vec = self.behavior_net.forward(torch.from_numpy(observation).unsqueeze(0).to(self.device))
				action = torch.argmax(vec).cpu()
  
		return action
	
	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)
		### TODO ###
		# calculate the loss and update the behavior network
		# 1. get Q(s,a) from behavior net
		# 2. get max_a Q(s',a) from target net
		# 3. calculate Q_target = r + gamma * max_a Q(s',a)
		# 4. calculate loss between Q(s,a) and Q_target
		# 5. update behavior net
		# state: [32, 4, 210, 160, 3], action: [32, 1], next_state: [32, 4, 210, 160, 3]

		action = action.type(torch.long)
		q_value = self.behavior_net(state).gather(1, action) # 32 x 1
  
		with torch.no_grad():
			q_next = self.behavior_net(next_state)
			action_index = q_next.max(dim=1)[1].view(-1, 1)
			q_next = self.target_net(next_state).gather(dim=1, index=action_index.long())
			q_target = reward + self.gamma * q_next * (1 - done)

		loss = F.mse_loss(q_value, q_target)
		self.writer.add_scalar('DQN/Loss', loss.item(), self.total_time_step)
		self.optim.zero_grad()
		loss.backward()
		self.optim.step()
		self.soft_update(self.behavior_net, self.target_net)
		
  
	def update(self):
		if self.total_time_step % self.update_freq == 0:
			self.update_behavior_network()
		
	def soft_update(self, behavior_model, target_model, tau=1e-3):
		for target_param, behavior_param in zip(target_model.parameters(), behavior_model.parameters()):
			target_param.data.copy_(tau * behavior_param.data + (1.0 - tau) * target_param.data)
  
	def resize(self, observation):
		gray_observation = cv2.cvtColor(np.array(observation), cv2.COLOR_RGB2GRAY)
		resized_observation = cv2.resize(gray_observation, (84, 84))
		return resized_observation
		
	   