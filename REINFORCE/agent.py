import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as func
from net import PolicyNetwork
class PolicyGradientAgent(object):
    def __init__(self, ALPHA, input_dims, GAMMA=0.99, n_actions=4,
                 layer1_size=256, layer2_size=256):
        self.gamma = GAMMA
        self.reward_memory = []
        self.action_memory = []
        self.policy = PolicyNetwork(ALPHA, input_dims, layer1_size, layer2_size,
                                    n_actions)

    def choose_action(self, observation):
        probabilities = func.softmax(self.policy.forward(observation))
        action_probs = torch.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)

        return action.item()

    def store_rewards(self,reward):
        self.reward_memory.append(reward)

    def learn(self):
        self.policy.optimizer.zero_grad()
        # Assumes only a single episode for reward_memory
        G = np.zeros_like(self.reward_memory, dtype=np.float64)
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        G = (G - mean) / std

        G = torch.tensor(G, dtype=torch.float)

        loss = 0
        for g, logprob in zip(G, self.action_memory):
            loss += -g * logprob

        loss.backward()
        self.policy.optimizer.step()

        self.action_memory = []
        self.reward_memory = []
    def save_model(self):
        self.policy.save_model()
    def load_model(self):
        self.policy.load_model()