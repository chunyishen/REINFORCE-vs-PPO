import numpy as np
import gym
from agent import PolicyGradientAgent
import matplotlib.pyplot as plt
from utils import plot_learning_curve, plot_avg_learning_curve
from gym import wrappers
import torch
import time
if __name__ == '__main__':
    
    
    env = gym.make('CartPole-v0')
    env.seed(0)
    torch.manual_seed(0)
    agent = PolicyGradientAgent(ALPHA=0.0003, input_dims=env.observation_space.shape, GAMMA=0.99,
                                n_actions=env.action_space.n, layer1_size=128, layer2_size=128)
    score_history = []
    score = 0
    num_episodes = 600
    #env = wrappers.Monitor(env, "tmp/lunar-lander",
    #                        video_callable=lambda episode_id: True, force=True)
    max_score = 0
    start = time.time()
    for i in range(num_episodes):
        
        done = False
        score = 0
        n_steps = 0
        observation = env.reset()
        while not done and n_steps < 240 :
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            agent.store_rewards(reward)
            observation = observation_
            score += reward
        if score >= max_score:
            max_score = score
            agent.save_model()
        score_history.append(score)
        agent.learn()
        #agent.save_checkpoint()
    end = time.time()
    print(end - start)
    
    figure_file = 'CartPole_REINFORCe'
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
    figure_file = 'CartPole_REINFORCE_avg600'
    x = [i+1 for i in range(len(score_history))]
    plot_avg_learning_curve(x, score_history, figure_file)

