
import argparse
import random
from gym.wrappers import FrameStack
import gym 
import numpy as np 
import torch
from dqn_agent_atari import AtariDQNAgent
from ddqn import AtariDDQNAgent
from Dueling_DQN import AtariDuelingDQNAgent

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=205000)
parser.add_argument('--path', type=str, default='./log/DQN/model_396118_688.pth')
args = parser.parse_args()

if __name__ == '__main__':
    # my hyperparameters, you can change it as you like
    config = {
		"gpu": True,
		"training_steps": 1e8,
		"gamma": 0.99,
		"batch_size": 32,
		"eps_min": 0.1,
		"warmup_steps": 20000,
		"eps_decay": 1000000,
		"eval_epsilon": 0.01,
		"replay_buffer_capacity": 100000,
		"logdir": 'log/DQN/',
		"update_freq": 4,
		"update_target_freq": 10000,
		"learning_rate": 0.0000625,
        "eval_interval": 50,
        "eval_episode": 1,
		"env_id": 'ALE/MsPacman-v5',
	}
    
    agent = AtariDQNAgent(config)
    # agent = AtariDuelingDQNAgent(config)
    agent.test_env = gym.make("ALE/MsPacman-v5", render_mode="human")
    agent.test_env = FrameStack(agent.test_env, num_stack=4)
    # max_reward = 1000
    # max_seed = 0
    seed = int(args.seed)
    # for i in range(1000):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    agent.test_env.seed(seed)
    
    agent.load(args.path)
    reward = agent.evaluate()
    #     if reward >= max_reward:
    #         max_reward = reward
    #         max_seed = i
    # print(f"seed: {max_seed}")
    # print(f"reward: {max_reward}")
