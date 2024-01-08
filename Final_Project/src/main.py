from td3_agent_CarRacing import CarRacingTD3Agent

if __name__ == '__main__':
	# my hyperparameters, you can change it as you like
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
		"logdir": 'log/austria_competition/test24/', # test21: 64x64, test22-23: 32x32
		"update_freq": 2,
		"eval_interval": 10,
		"eval_episode": 5,
		"scenario": 'austria_competition',    # circle_cw_competition_collisionStop or austria_competition 
	}
	agent = CarRacingTD3Agent(config)
	agent.train()

