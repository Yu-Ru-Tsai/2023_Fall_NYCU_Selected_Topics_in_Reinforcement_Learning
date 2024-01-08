
import argparse
import random
import numpy as np 
import torch
from ppo_agent_atari import AtariPPOAgent

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='./log/Enduro/model_97497822_1626.pth')
args = parser.parse_args()

if __name__ == '__main__':
    # my hyperparameters, you can change it as you like
    config = {
		"gpu": True,
		"training_steps": 1e8,
		"update_sample_count": 10000,
		"discount_factor_gamma": 0.99,
		"discount_factor_lambda": 0.95,
		"clip_epsilon": 0.2,
		"max_gradient_norm": 0.5,
		"batch_size": 128,
		"logdir": 'log/test/',
		"update_ppo_epoch": 3,
		"learning_rate": 2.5e-4,
		"value_coefficient": 0.5,
		"entropy_coefficient": 0.01,
		"horizon": 128,
		"env_id": 'ALE/Enduro-v5',
		"eval_interval": 50,
		"eval_episode": 1,
		"seed": 1,
		}
    
    agent = AtariPPOAgent(config)
    agent.load(args.path)
    agent.test_env.seed(588)
    reward = agent.evaluate() 
 
 