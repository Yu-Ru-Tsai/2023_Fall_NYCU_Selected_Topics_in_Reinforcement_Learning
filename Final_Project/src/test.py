from racecar_gym.env import RaceEnv

env = RaceEnv(
scenario='austria_competition', # e.g., 'austria_competition', 'circle_cw_competition_collisionStop'
render_mode='human',
reset_when_collision=True, # Only work for 'austria_competition' and 'austria_competition_collisionStop'
)

total_reward = 0
obs, info = env.reset()
terminated = False
for i in range(100):
    action = env.action_space.sample()
    obs, rewards, terminated, truncated, states = env.step(action)
    # print(f"action: {action.shape}, obs: {obs.shape}, states: {states.keys()}")
    total_reward += rewards
    if terminated or truncated:
        observation = env.reset()
print(f"total_reward: {total_reward}")
env.close()
    # done = terminated or truncated
    
