import gym
from agent import PolicyGradientAgent
import torch
import time
if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env.seed(0)
    torch.manual_seed(0)
    agent = PolicyGradientAgent(ALPHA=0.0003, input_dims=env.observation_space.shape, GAMMA=0.99,
                                n_actions=env.action_space.n, layer1_size=128, layer2_size=128)
    agent.load_model()
    #env = wrappers.Monitor(env, "tmp/lunar-lander",
    #                        video_callable=lambda episode_id: True, force=True)
    observation = env.reset()
    done = False
    while not done :
        env.render()
        action = agent.choose_action(observation)
        observation, reward, done, info = env.step(action)
        time.sleep(0.01)
        
