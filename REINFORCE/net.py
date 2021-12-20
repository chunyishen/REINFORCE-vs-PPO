import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as func

class PolicyNetwork(nn.Module):
    def __init__(self, ALPHA, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(PolicyNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)

       
    def forward(self, observation):
        state = torch.Tensor(observation)
        x = func.relu(self.fc1(state))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def save_model(self,path = 'model'):
        torch.save(self.state_dict(),path)
    def load_model(self,path = 'model'):
        self.load_state_dict(torch.load(path))