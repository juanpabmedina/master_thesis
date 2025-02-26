import numpy as np
class P2PEnergyMarket:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def get_reward(self, rol, generator_state, consumer_state):
        """Determine trading results and compute rewards"""

        if rol == 'generator':
            generation_costs = -self.a*generator_state**2 + self.b*generator_state + self.c
            reward = generator_state * 1/np.log(1+consumer_state) - generation_costs
        elif rol == 'consumer':
            reward = generator_state * 1/np.log(1+consumer_state) 
        return reward