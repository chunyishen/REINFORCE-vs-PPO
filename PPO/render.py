import gym
import numpy as np
from agent import Agent
from utils import plot_learning_curve
import torch
import time
if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env.seed(0)
    torch.manual_seed(0)
    agent = Agent(input_dims=env.observation_space.shape,
                                n_actions=env.action_space.n)
    agent.load_models()
    #env = wrappers.Monitor(env, "tmp/lunar-lander",
    #                        video_callable=lambda episode_id: True, force=True)
    observation = env.reset()
    done = False
    while not done :
        env.render()
        action, prob, val = agent.choose_action(observation)
        observation, reward, done, info = env.step(action)
        time.sleep(1)
            