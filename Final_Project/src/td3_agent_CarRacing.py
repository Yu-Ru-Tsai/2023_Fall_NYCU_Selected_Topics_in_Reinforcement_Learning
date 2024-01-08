import torch
import torch.nn as nn
import numpy as np
from base_agent import TD3BaseAgent
from models.CarRacing_model import ActorNetSimple, CriticNetSimple
import random
from base_agent import OUNoiseGenerator, GaussianNoise
from racecar_gym.env import RaceEnv
from environment_wrapper.CarRacingEnv import CarRacingEnvironment

class CarRacingTD3Agent(TD3BaseAgent):
	def __init__(self, config):
		super(CarRacingTD3Agent, self).__init__(config)
		# initialize environment
		# scenario='austria_competition', # e.g., 'austria_competition', 'circle_cw_competition_collisionStop'
		
		self.env = CarRacingEnvironment(N_frame=4, test=False, scenario=config['scenario'], render_mode='rgb_array_birds_eye')
		# behavior network
		self.actor_net = ActorNetSimple(self.env.observation_space.shape[1], self.env.action_space.shape[0], 4) # 3 x 128 x 128
		self.critic_net1 = CriticNetSimple(self.env.observation_space.shape[1], self.env.action_space.shape[0], 4)
		self.critic_net2 = CriticNetSimple(self.env.observation_space.shape[1], self.env.action_space.shape[0], 4)
		self.actor_net.to(self.device)
		self.critic_net1.to(self.device)
		self.critic_net2.to(self.device)
		# target network
		self.target_actor_net = ActorNetSimple(self.env.observation_space.shape[1], self.env.action_space.shape[0], 4)
		self.target_critic_net1 = CriticNetSimple(self.env.observation_space.shape[1], self.env.action_space.shape[0], 4)
		self.target_critic_net2 = CriticNetSimple(self.env.observation_space.shape[1], self.env.action_space.shape[0], 4)
		self.target_actor_net.to(self.device)
		self.target_critic_net1.to(self.device)
		self.target_critic_net2.to(self.device)
		self.target_actor_net.load_state_dict(self.actor_net.state_dict())
		self.target_critic_net1.load_state_dict(self.critic_net1.state_dict())
		self.target_critic_net2.load_state_dict(self.critic_net2.state_dict())
		
		# set optimizer
		self.lra = config["lra"]
		self.lrc = config["lrc"]
		
		self.actor_opt = torch.optim.Adam(self.actor_net.parameters(), lr=self.lra)
		self.critic_opt1 = torch.optim.Adam(self.critic_net1.parameters(), lr=self.lrc)
		self.critic_opt2 = torch.optim.Adam(self.critic_net2.parameters(), lr=self.lrc)

		# choose Gaussian noise or OU noise
		noise_mean = np.full(self.env.action_space.shape[0], 0.0, np.float32)
		noise_std = np.full(self.env.action_space.shape[0], 1.0, np.float32)
		self.noise = OUNoiseGenerator(noise_mean, noise_std)
		# self.noise = GaussianNoise(self.env.action_space.shape[0], 0.0, 1.0)
		
	
	def decide_agent_actions(self, state, sigma=0.0, brake_rate=0.015):
		### TODO ###
		# based on the behavior (actor) network and exploration noise
		
		with torch.no_grad():
			state = torch.FloatTensor(state).unsqueeze(0).to(self.device) # 1 x 4 x 128 x 128
			action = self.actor_net(state).cpu().detach().numpy() + sigma * self.noise.generate()
		return action.squeeze()		

	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)
		# state:torch.Size([32, 4, 128, 128]), action: torch.Size([32, 2])
		### TODO ###
		### TD3 ###
		
		# Clipped Double Q-Learning for Actor-Critic
		with torch.no_grad():
			noise = self.noise.generate()
			a_next = self.target_actor_net(next_state) + torch.tensor(noise).to(self.device)
			q_next1 = self.target_critic_net1(next_state, a_next.float())
			q_next2 = self.target_critic_net2(next_state, a_next.float())
			q_next = torch.min(q_next1, q_next2)
			target_q = reward + self.gamma * (1 - done) * q_next

    	# Update Critic
		q_value1 = self.critic_net1(state, action)
		q_value2 = self.critic_net2(state, action)

		criterion = nn.MSELoss()
		critic_loss1 = criterion(q_value1, target_q)
		critic_loss2 = criterion(q_value2, target_q)

		self.critic_opt1.zero_grad()
		critic_loss1.backward()
		self.critic_opt1.step()

		self.critic_opt2.zero_grad()
		critic_loss2.backward()
		self.critic_opt2.step()

		# Delayed Actor(Policy) Updates
		if self.total_time_step % self.update_freq == 0:
			# actor loss
			actor_actions = self.actor_net(state)
			q_values = self.critic_net1(state, actor_actions.float())

			# Calculate the mean Q-value
			mean_q_value = q_values.mean().float()

			# The objective is to maximize the mean Q-value, which is equivalent to minimizing its negative
			actor_loss = -mean_q_value

			# Optimize the actor
			self.actor_net.zero_grad()
			actor_loss.backward()
			self.actor_opt.step()