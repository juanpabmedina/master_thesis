import numpy as np

class EnergyMarketEnv:
    def __init__(self, a=0.1, b=2, c=0, 
                 init_gen_power=10, init_con_price=1,
                 min_power=0.1, max_power=0.5,
                 min_price=1.0, max_price=5, 
                 threshold=0.0):
        """
        Initialize the Energy Market Environment.
        
        Args:
            a, b, c: Coefficients for generator quadratic cost.
            init_gen_power: Initial generator power offer.
            init_con_price: Initial consumer price offer.
            power_step: Amount generator can adjust power by.
            price_step: Amount consumer can adjust price by.
            min_power, max_power: Generator power bounds.
            min_price, max_price: Consumer price bounds.
        """
        self.a = a
        self.b = b
        self.c = c

        self.init_gen_power = init_gen_power
        self.init_con_price = init_con_price
        
        self.min_power = min_power
        self.max_power = max_power
        self.min_price = min_price
        self.max_price = max_price

        self.threshold = threshold
        
        self.reset()

    def reset(self):
        """Resets the environment to the initial state."""
        self.gen_power = self.init_gen_power
        self.con_price = self.init_con_price
        return (self.gen_power, self.con_price)
    
    def is_trade_valid(self, threshold=0.0):
        Hg = self.a * self.gen_power**2 + self.b * self.gen_power + self.c
        return self.gen_power * self.con_price - Hg >= threshold


    def step(self, actions):
        # Apply actions
        gen_action = actions['generator']
        con_action = actions['consumer']

        self.gen_power += gen_action 
        self.con_price += con_action

        self.gen_power = np.clip(self.gen_power, self.min_power, self.max_power)
        self.con_price = np.clip(self.con_price, self.min_price, self.max_price)

        # Generator profit (Sell price - cost generation)
        Hg = self.a * self.gen_power**2 + self.b * self.gen_power + self.c
        # self.profit = self.con_price - Hg

        self.profit = - self.gen_power * 1/np.log(1+self.con_price) - Hg

        # Check for valid trade
        if self.profit >= self.threshold:
            generator_reward = 1
            consumer_reward = -1
            done = True
        else:
            generator_reward = 0.0
            consumer_reward = 0.0  # negotiation step, not accepted
            done = False

        rewards = {
            'generator': generator_reward,
            'consumer': consumer_reward
        }

        next_state = (self.gen_power, self.con_price)

        return next_state, rewards, done


    def get_state(self):
        """Returns the current state."""
        return (self.gen_power, self.con_price)
