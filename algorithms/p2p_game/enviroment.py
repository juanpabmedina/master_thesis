import numpy as np
class P2PEnergyMarket:
    def __init__(self):
        self.generation_costs = 100

    
    def get_reward(self, rol, generator_state, consumer_state):
        """Determine trading results and compute rewards"""

        if rol == 'generator':
            reward = generator_state * consumer_state - self.generation_costs
        elif rol == 'consumer':
            reward = generator_state * np.log(1/(consumer_state)+1)
        return reward