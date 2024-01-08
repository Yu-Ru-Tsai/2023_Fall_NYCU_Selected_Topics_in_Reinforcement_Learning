import argparse
from collections import deque
import itertools
import random
import time
import cv2
from matplotlib.pylab import f

# import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from racecar_gym.env import RaceEnv
import gymnasium as gym

class CarRacingEnvironment:
	def __init__(self, N_frame=4, test=False, scenario='austria_competition', render_mode='rgb_array_birds_eye'):
		env = RaceEnv(scenario=scenario,
            render_mode=render_mode,
            reset_when_collision=True if 'austria' in scenario else False)
		
		self.test = test
		self.env = env
		if scenario == 'austria_competition':
			low = [-0.15, -1] 
			high = [0.15, 1]     	
			self.action_space = gym.spaces.box.Box(low=np.array(low), high=np.array(high), shape=(2,), dtype=np.float32)
			# self.action_space = gym.spaces.box.Box(low=-1, high=1., shape=(2,), dtype=float)
		else:
			self.action_space = self.env.action_space
		# self.action_space = self.env.action_space
		self.observation_space = self.env.observation_space
		self.ep_len = 0
		self.frames = deque(maxlen=N_frame)
		self._last_stored_progress = None
		self._last_stored_checkpoint = None

	def step(self, action, test=False):
		self.test = test
		obs, reward, terminates, truncates, info = self.env.step(action)
		obs = np.transpose(obs, (1, 2, 0))
		original_reward = reward
		original_terminates = terminates
		self.ep_len += 1
		wall_collision = info['wall_collision']
		progress = info['progress']
		velocity = info['velocity']
		lap = info['lap']
		pose = info['pose']
		checkpoint = info['checkpoint']
		speed = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
		reward += speed * 0.1
		reward += abs(pose[5]) * 0.22
		if self._last_stored_progress is None:
			self._last_stored_progress = progress
		if progress >= self._last_stored_progress:
			reward += (progress - self._last_stored_progress)
		else:
			reward += (self._last_stored_progress - progress)
			
		reward -= 0.001 # time penalty #

		if self._last_stored_checkpoint is None:
			self._last_stored_checkpoint = checkpoint
		if checkpoint - self._last_stored_checkpoint > 0:
			reward *= 1.5
   
		if wall_collision:
			# terminates = True
			reward *= 0.7
   
		self._last_stored_progress = progress
		self._last_stored_checkpoint = checkpoint  
		obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
  
		# # test 15 :[-0.1, 0.1]
		# resize image
		obs = cv2.resize(obs, (64, 64), interpolation=cv2.INTER_AREA) 
  
  		# save image for debugging
		# filename = "images/image" + str(self.ep_len) + ".jpg"
		# cv2.imwrite(filename, obs)

		# frame stacking
		self.frames.append(obs)
		obs = np.stack(self.frames, axis=0)

		if self.test:
			# enable this line to recover the original reward
			reward = original_reward
			# enable this line to recover the original terminates signal, disable this to accerlate evaluation
			# terminates = original_terminates
		# print(f"reward: {reward}")
		return obs, reward, terminates, truncates, info
	
	def reset(self, train=True):
		obs, info = self.env.reset(train)
		self.ep_len = 0
		# convert to grayscale obs = 128 * 128 * 3
		obs = np.transpose(obs, (1, 2, 0)) # 128 x 128 x 3
		obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY) 
		obs = cv2.resize(obs, (64, 64), interpolation=cv2.INTER_AREA)

		# frame stacking
		for _ in range(self.frames.maxlen):
			self.frames.append(obs)
		obs = np.stack(self.frames, axis=0)

		return obs, info
	
	def render(self):
		self.env.render()
	
	def close(self):
		self.env.close()