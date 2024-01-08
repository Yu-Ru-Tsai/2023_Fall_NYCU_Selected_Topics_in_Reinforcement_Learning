import torch
import torch.nn as nn
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from replay_buffer.gae_replay_buffer import GaeSampleMemory
from base_agent import PPOBaseAgent
from models.atari_model import AtariNet
import gym


class AtariPPOAgent(PPOBaseAgent):
	def __init__(self, config):
		super(AtariPPOAgent, self).__init__(config)
		### TODO ###
		# initialize env
		# self.env = ???
		self.env = gym.make(config["env_id"], render_mode="rgb_array")
		self.env = gym.wrappers.ResizeObservation(self.env, (84, 84))
		self.env = gym.wrappers.GrayScaleObservation(self.env)
		self.env = gym.wrappers.FrameStack(self.env, 4)
		# self.env.seed(config["seed"])
		# self.env.action_space.seed(config["seed"])
		# self.env.observation_space.seed(config["seed"])
        
		### TODO ###
		# initialize test_env
		# self.test_env = ???
		self.test_env = gym.make(config["env_id"], render_mode="rgb_array")
		self.test_env = gym.wrappers.ResizeObservation(self.test_env, (84, 84))
		self.test_env = gym.wrappers.GrayScaleObservation(self.test_env)
		self.test_env = gym.wrappers.FrameStack(self.test_env, 4)
		# self.test_env.seed(config["seed"])
		# self.test_env.action_space.seed(config["seed"])
		# self.test_env.observation_space.seed(config["seed"])

		self.net = AtariNet(self.env.action_space.n)
		self.net.to(self.device)
		self.lr = config["learning_rate"]
		self.update_count = config["update_ppo_epoch"]
		self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)
		
	def decide_agent_actions(self, observation, eval=False):
		### TODO ###
		# add batch dimension in observation
		# get action, value, logp from net
		
		# if eval:
		# 	with torch.no_grad():
		# 		???, ???, ???, _ = self.net(observation, eval=True)
		# else:
		# 	???, ???, ???, _ = self.net(observation)
		observation = np.array(observation)
		observation = torch.tensor(observation, device=self.device, dtype=torch.float32).unsqueeze(0) # (1 x 4 x 84 x 84)

        # Get action, value, logp from net
		if eval:  # action:(1), logp:(1), value:()
			with torch.no_grad():
				action, logp, value, _ = self.net(observation, eval=True)
		else:
			action, logp, value, _ = self.net(observation)

		return [action.item()], value.item(), logp.item()
		
		# return NotImplementedError

	
	def update(self):
		loss_counter = 0.0001
		total_surrogate_loss = 0
		total_v_loss = 0
		total_entropy = 0
		total_loss = 0

		batches = self.gae_replay_buffer.extract_batch(self.discount_factor_gamma, self.discount_factor_lambda)
		sample_count = len(batches["action"])
		batch_index = np.random.permutation(sample_count)
		
		observation_batch = {}
		for key in batches["observation"]:
			observation_batch[key] = batches["observation"][key][batch_index]
		action_batch = batches["action"][batch_index]
		return_batch = batches["return"][batch_index]
		adv_batch = batches["adv"][batch_index]
		v_batch = batches["value"][batch_index]
		logp_pi_batch = batches["logp_pi"][batch_index]

		for _ in range(self.update_count):
			for start in range(0, sample_count, self.batch_size):
				ob_train_batch = {}
				for key in observation_batch:
					ob_train_batch[key] = observation_batch[key][start:start + self.batch_size]
				ac_train_batch = action_batch[start:start + self.batch_size]
				return_train_batch = return_batch[start:start + self.batch_size]
				adv_train_batch = adv_batch[start:start + self.batch_size]
				v_train_batch = v_batch[start:start + self.batch_size]
				logp_pi_train_batch = logp_pi_batch[start:start + self.batch_size]

				ob_train_batch = torch.from_numpy(ob_train_batch["observation_2d"])
				ob_train_batch = ob_train_batch.to(self.device, dtype=torch.float32) # (128 x 4 x 84 x 84)   
				ac_train_batch = torch.from_numpy(ac_train_batch)
				ac_train_batch = ac_train_batch.to(self.device, dtype=torch.long) # (128 x 1)
				adv_train_batch = torch.from_numpy(adv_train_batch)
				adv_train_batch = adv_train_batch.to(self.device, dtype=torch.float32) # (128)
				logp_pi_train_batch = torch.from_numpy(logp_pi_train_batch)
				logp_pi_train_batch = logp_pi_train_batch.to(self.device, dtype=torch.float32) # (128)
				return_train_batch = torch.from_numpy(return_train_batch)
				return_train_batch = return_train_batch.to(self.device, dtype=torch.float32) # (128)

				### TODO ###
				# calculate loss and update network
				# ???, ???, ???, ??? = self.net(...)

				# calculate policy loss
				# ratio = ???
				# surrogate_loss = ???

				# calculate value loss
				# value_criterion = nn.MSELoss()
				# v_loss = value_criterion(...)
				
				# calculate total loss
				# loss = surrogate_loss + self.value_coefficient * v_loss - self.entropy_coefficient * entropy

				# update network
				# self.optim.zero_grad()
				# loss.backward()
				# nn.utils.clip_grad_norm_(self.net.parameters(), self.max_gradient_norm)
				# self.optim.step()

				# total_surrogate_loss += surrogate_loss.item()
				# total_v_loss += v_loss.item()
				# total_entropy += entropy.item()
				# total_loss += loss.item()
				# loss_counter += 1

				# Calculate loss and update network
				_, logp, value, entropy = self.net(ob_train_batch, False, torch.squeeze(ac_train_batch)) # (128), (128), (128), (128)
    
				# Calculate policy loss
				ratio = torch.exp(logp - logp_pi_train_batch)
				surrogate_loss1 = ratio * adv_train_batch
				surrogate_loss2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv_train_batch
				surrogate_loss = -torch.min(surrogate_loss1, surrogate_loss2).mean()

				# Calculate value loss
				value_criterion = nn.MSELoss()
				v_loss = value_criterion(value, return_train_batch)

				# Calculate total loss
				entropy = entropy.mean()
				loss = surrogate_loss + self.value_coefficient * v_loss - self.entropy_coefficient * entropy

				# Update network
				self.optim.zero_grad()
				loss.backward()
				torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_gradient_norm)
				self.optim.step()

				# Aggregate loss values
				total_surrogate_loss += surrogate_loss.item()
				total_v_loss += v_loss.item()
				total_entropy += entropy.item()
				total_loss += loss.item()
				loss_counter += 1

		self.writer.add_scalar('PPO/Loss', total_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Surrogate Loss', total_surrogate_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Value Loss', total_v_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Entropy', total_entropy / loss_counter, self.total_time_step)
		print(f"Loss: {total_loss / loss_counter}\
			\tSurrogate Loss: {total_surrogate_loss / loss_counter}\
			\tValue Loss: {total_v_loss / loss_counter}\
			\tEntropy: {total_entropy / loss_counter}\
			")
	



