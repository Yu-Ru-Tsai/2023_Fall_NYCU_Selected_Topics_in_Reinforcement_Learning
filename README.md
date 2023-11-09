# 2023_Fall_NYCU_Selected_Topics_in_Reinforcement_Learning

# Lab1 2048

In this lab, you will learn temporal difference learning (TD) algorithm by solving the 2048 game using an 𝑛-tuple network.

## Experimental Results:

(1) A plot shows scores (mean) of at least 100k training episodes

## Bonus

(1) Describe the implementation and the usage of 𝑛-tuple network.

(2) Explain the mechanism of TD(0).

(3) Describe your implementation in detail including action selection and TDbackup diagram

# Lab2 Deep Q-Network (DQN)

In this lab, you will learn and implement the Deep Q-Network algorithm by solving MsPacman-v5.

## Experimental Results:

(1) Screenshot of Tensorboard training curve and testing results on DQN.

## Experimental Results and Discussion of bonus parts (bonus)

(1) Screenshot of Tensorboard training curve and testing results on Enduro-v5

(2) Screenshot of Tensorboard training curve and testing results on DDQN, and discuss the difference between DQN and DDQN.

(3) Screenshot of Tensorboard training curve and testing results on Dueling DQN and discuss the difference between DQN and Dueling DQN.

(4) Screenshot of Tensorboard training curve and testing results on DQN with parallelized rollout, and discuss the difference between DQN and DQN with parallelized rollout.

# Lab 3: Proximal Policy Optimization (PPO)

In this lab, you will learn and implement the Proximal Policy Optimization algorithm by solving Enduro-v5.

## Experimental Results

(1) Screenshot of Tensorboard training curve and testing results on PPO.

## Answer the questions of bonus parts (bonus)

(1) PPO is an on-policy or an off-policy algorithm? Why? 

(2) Explain how PPO ensures that policy updates at each step are not too large to avoid destabilization.

(3) Why is GAE-lambda used to estimate advantages in PPO instead of just one-step advantages? How does it contribute to improving the policy learning process?

(4) Please explain what the lambda parameter represents in GAE-lambda, and how adjusting the lambda parameter affects the training process and performance of PPO?

# Lab 4: Twin Delayed DDPG (TD3)

In this lab, you will learn and implement the Twin Delayed DDPG (TD3) algorithm by solving CarRacing-v2.

## Experimental Results

(1) Screenshot of Tensorboard training curve and testing results on TD3.

## Answer the questions of bonus parts (bonus)

(1) Impact of Twin Q-Networks:
■ Compare the performance of using twin Q-networks and single Q-networks in TD3, and explain.

(2) Target Policy Smoothing:
■ Compare the impact of enabling and disabling target policy smoothing in TD3. Attempt to disable the component of target policy smoothing in TD3 and observe whether performance decreases, and explain.

(3) Delayed Policy Update Mechanism:
■ Study the effects of using a single delayed update or adding more delayed update steps in TD3. Increase or decrease the number of delayed update steps and compare the results, and explain.

(4) Action Noise Injection:
■ Compare the effects of adding different levels of action noise (exploration noise) in TD3. Try increasing or decreasing the magnitude of action noise in TD3 and observe how it affects algorithm convergence speed and stability, and explain.

(5) Reward Function Design:
■ Design a different reward function that works better than the original one, and explain why it’s effective.
