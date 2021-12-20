import gym
import numpy as np
from agent import Agent
from utils import plot_learning_curve, plot_avg_learning_curve
import torch
import time
if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env.seed(0)
    torch.manual_seed(0)

    N = 20
    batch_size = 5
    n_epochs = 4
    
    agent = Agent(n_actions=env.action_space.n, 
                    input_dims=env.observation_space.shape)
    n_games = 300

    

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0
    start = time.time()
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        current_game_step = 0
        while not done and current_game_step < 500:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            current_game_step += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', current_game_step, 'learning_steps', learn_iters)

    figure_file = 'CartPole_ppo'
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
    figure_file = 'CartPole_ppo_avg100'
    x = [i+1 for i in range(len(score_history))]
    plot_avg_learning_curve(x, score_history, figure_file)
    