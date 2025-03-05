import numpy as np
class P2PEnergyMarketEnv:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def get_reward(self, rol, state):
        """Determine trading results and compute rewards"""
    
        if rol == 'generator':
            generator_state = state[0]
            consumer_state = state[1]
            # generation_costs = self.a*generator_state**2 + self.b*generator_state + self.c
            reward = -generator_state * 1/np.log(1+consumer_state)
        elif rol == 'consumer':
            generator_state = state[1]
            consumer_state = state[0]
            reward = generator_state * 1/np.log(1+consumer_state)
        return reward
    
    def step(self, agent, joint_action):
        next_state = (round(agent.state[0]+joint_action[0],2), round(agent.state[1]+joint_action[1],2))
        

        if abs(next_state[0]) > abs(agent.agent_states[0]) and abs(next_state[0]) < abs(agent.agent_states[-1]):
            if abs(next_state[1]) > abs(agent.opponent_states[0]) and abs(next_state[1]) < abs(agent.opponent_states[-1]):
                reward = self.get_reward(agent.rol, next_state)
                return next_state, reward
            else:
                reward = self.get_reward(agent.rol, agent.state)
                return agent.state, reward
        else:
            reward = self.get_reward(agent.rol, agent.state)
            return agent.state, reward
        