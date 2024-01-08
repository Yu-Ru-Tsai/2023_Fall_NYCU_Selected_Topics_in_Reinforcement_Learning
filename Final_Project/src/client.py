import argparse
import json
import numpy as np
import requests
from torch import rand
from td3_agent_CarRacing import CarRacingTD3Agent
import cv2
from collections import deque
import gymnasium as gym


parser = argparse.ArgumentParser()
parser.add_argument('--url', type=str, default='http://140.113.135.85:6006', help='The url of the server.')
parser.add_argument('--size', type=int, default=64)
args = parser.parse_args()
frames = deque(maxlen=4)

def connect(agent, url: str = 'http://localhost:5000', first_call = 1):
    while True:
        # Get the observation
        response = requests.get(f'{url}')
        if json.loads(response.text).get('error'):
            print(json.loads(response.text)['error'])
            break
        obs = json.loads(response.text)['observation']
        obs = np.array(obs).astype(np.uint8)
        
        # obs 
        obs = np.transpose(obs, (1, 2, 0))
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        obs = cv2.resize(obs, (args.size, args.size), interpolation=cv2.INTER_AREA)

        if first_call == 1:
            for _ in range(4):
                frames.append(obs)
            first_call = 0
            obs = np.stack(frames, axis=0)
        else:
            frames.append(obs)
            obs = np.stack(frames, axis=0)
   
        # Decide an action based on the observation (Replace this with your RL agent logic)
        action_to_take = agent.decide_agent_actions(obs)  # Replace with actual action

        # Send an action and receive new observation, reward, and done status
        response = requests.post(f'{url}', json={'action': action_to_take.tolist()})
        if json.loads(response.text).get('error'):
            print(json.loads(response.text)['error'])
            break

        result = json.loads(response.text)
        terminal = result['terminal']

        if terminal:
            print('Episode finished.')
            return


if __name__ == '__main__':
    
    # Initialize the RL Agent
    config = {
        "gpu": True,
        "training_steps": 1e8,
        "gamma": 0.99,
        "tau": 0.005,
        "batch_size": 32,
        "warmup_steps": 1000,
        "total_episode": 100000,
        "lra": 4.5e-5,
        "lrc": 4.5e-5,
        "replay_buffer_capacity": 5000,
        "logdir": 'log/evaluate/test/',
        "update_freq": 2,
        "eval_interval": 1,
        "eval_episode": 1,
        "scenario": 'austria_competition', # circle_cw_competition_collisionStop or austria_competition 
    }
    # test 21: model_7899817_0.454: 0.55, model_6083128_0.454: 0.55, model_8129236_0.454: 0.4, model_8096321_0.454
    load_path = './log/austria_competition/test21/model_8096321_0.454.pth' 

    rand_agent = CarRacingTD3Agent(config)
    rand_agent.load(load_path)
    connect(rand_agent, url=args.url, first_call=1)
