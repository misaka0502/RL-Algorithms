import numpy as np
import torch
from ..enviroment.grid_world import GridWorld

class PolicyEvaluator():
    """
        A class to evaluate state value for a given policy, using Bellman expectation equation.
    """
    def __init__(self, env: GridWorld, gamma: float = 0.99):
        self.env = env
        self.gamma = gamma
        self.state_values = np.zeros(env.get_state_space_size())

    def evaluate_policy(self, policy, theta: float = 1e-6, max_iterations: int = 1000) -> np.array:
        pass