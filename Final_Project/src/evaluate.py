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
		"logdir": 'log/evaluate/test/',
		"update_freq": 2,
		"eval_interval": 1,
		"eval_episode": 1,
		"scenario": 'austria_competition', # circle_cw_competition_collisionStop or austria_competition 
	} # 50017 test 1
	##### circle_cw_competition_collisionStop test1  model_830065_1.65, model_813919_2.05,model_845930_1.97, model_920336_2.09, model_1074701_2.01####
	##### circle_cw_competition_collisionStop test2 model_1398507_2.14, model_1403374_2.13, ####
	# load_path = './log/CarRacing/austria_competition2/model_2068770_0.pth'
	# load_path = './log/CarRacing/test1/model_387672_0.pth'
	# load_path = './log/circle_cw_competition_collisionStop/test1/model_1024244_2.01.pth'
	load_path = './log/austria_competition/test1/model_330112_0.139.pth'
	agent = CarRacingTD3Agent(config)
	agent.load_and_evaluate(load_path)