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

class AtariDQNAgent(DQNBaseAgent):
	def __init__(self, config):
		super(AtariDQNAgent, self).__init__(config)
		### TODO ###
		self.env = gym.make(config["env_id"], render_mode="rgb_array")
		self.env = gym.wrappers.ResizeObservation(self.env, (84, 84))
		self.env = gym.wrappers.GrayScaleObservation(self.env)
		self.env = gym.wrappers.FrameStack(self.env, 4)
		# initialize env
		# self.env = ???

		### TODO ###
		# initialize test_env
		# self.test_env = ???
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
		else:                   # pick argmax(Q(s,a)), given state predict action
			with torch.no_grad():		
				observation = np.array(observation)	
				vec = self.behavior_net.forward(torch.from_numpy(observation).unsqueeze(0).to(self.device))
				action = torch.argmax(vec).cpu()
  
		return action

		# return NotImplementedError
	
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
  
		# q_value = ???
		# with torch.no_grad():
			# q_next = ???

			# if episode terminates at next_state, then q_target = reward
			# q_target = ???

		# criterion = ?
		# loss = criterion(q_value, q_target)

		action = action.type(torch.long)
		q_value = self.behavior_net(state).gather(1, action) # 32 x 1

		with torch.no_grad():
			q_target = self.target_net(next_state).detach().max(1)[0].unsqueeze(1)
			q_target = reward + self.gamma * q_target * (1 - done)

		loss = F.mse_loss(q_value, q_target)
		self.writer.add_scalar('DQN/Loss', loss.item(), self.total_time_step)
		self.optim.zero_grad()
		loss.backward()
		self.optim.step()
  
	def resize(self, observation):
		gray_observation = cv2.cvtColor(np.array(observation), cv2.COLOR_RGB2GRAY)
		resized_observation = cv2.resize(gray_observation, (84, 84))
		return resized_observation
		
	   