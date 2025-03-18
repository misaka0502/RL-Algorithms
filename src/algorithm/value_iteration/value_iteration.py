import numpy as np
from enviroment.grid_world import GridWorld
from utils.visualization import Visualizer
import matplotlib.pyplot as plt

def value_iteration(env: GridWorld, gamma: float=0.9, theta: float=1e-10, max_iterations: int=1000):
    # initialize policy
    policy = {}
    policy = np.zeros((env.num_states, len(env.action_space)))
    # initialize state values
    state_values = np.zeros(env.num_states)
    for iter in range(max_iterations):
        delta = 0
        env.reset()
        for state in range(env.num_states):
            old_value = state_values[state]

            q_values = []
            state = env._state_to_pos(state)
            for action in env.action_space:
                # next_state, reward = env.get_next_state_reward(state, action)
                env.agent_state = state
                next_state, reward, done, _ = env.step(action)
                next_state = env._pos_to_state(next_state)
                q_values.append(reward + gamma * state_values[next_state])
            # Policy update
            state = env._pos_to_state(state)
            best_action = np.argmax(q_values)
            policy[state, best_action] = 1
            policy[state, np.arange(env.num_actions) != best_action] = 0
            # Value update
            state_values[state] = q_values[best_action]
            delta = max(delta, abs(old_value - state_values[state]))
        print(f"Iteration {iter}, delta: {delta}")
        if delta < theta:
            break
    
    return policy, state_values


if __name__ == "__main__":
    env = GridWorld()
    policy, state_values = value_iteration(env)
    if isinstance(env, GridWorld):
        env.render()
        env.add_policy(policy)
        env.add_state_values(state_values)
        env.render(animation_interval=5)