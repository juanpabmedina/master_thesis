from scipy.optimize import linprog

class MinimaxQ:

    def __init__(self, agent, gamma=0.9, epsilon=0.2, alpha=1, decay=0.9999954):

        self.q_table = agent.q_table
        self.pi_table = agent.pi_table
        self.v_table = agent.v_table 

        self.agent_actions = agent.agent_actions
        self.opponent_actions = agent.opponent_actions

        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.decay = decay

    def update(self, state, action, opponent_action, reward, next_state):
        """Update Q, π, and V after observing reward and transitioning to next state."""
            
        # Update Q-value
        self.q_table[state][action][opponent_action] = (1 - self.alpha) *  self.q_table[state][action][opponent_action] + self.alpha * (reward + self.gamma*self.v_table[next_state])
        
        # Update π using Linear Programming 
        self.pi_table[state] = self.optimize_policy(state)

        worst_case_value = float('inf')
        
        # Update V based on min over opponent actions
        for o in self.opponent_actions:
            # Calculate the expected value for this opponent action o
            expected_value = 0
            for a in self.agent_actions:
                expected_value += self.pi_table[state][a]* self.q_table[state][a][o]
            
            # Update the worst-case (minimum over opponent actions)
            worst_case_value = min(worst_case_value, expected_value)

        self.v_table[state] = worst_case_value
        # Decay learning rate
        self.alpha *= self.decay
        return self.pi_table


    def optimize_policy(self, state):
        """
        Compute the optimal policy π[s, .] using linear programming.

        Args:
            s (int): The current state.

        Returns:
            dict: A dictionary π[s, a] representing the optimal policy for state s.
        """
        num_actions = len(self.agent_actions)
        
        # Decision variables: π[a] for each action + v (worst-case value)
        c = [-1] + [0] * num_actions  # Maximize v (converted to minimization by negation)

        # Constraints: Gx <= h
        G = []
        h = []
        
        # Worst-case expected value constraint
        for o in self.opponent_actions:
            row = [1]  # Coefficient for v
            for a in self.agent_actions:
                row.append(self.q_table[state][a][o])  # Coefficients for -π[a] * Q[s, a, o]
            G.append(row)
            h.append(0)

        # Probability distribution constraint: sum(π[a]) = 1
        equality_row = [0] + [1] * num_actions
        A_eq = [equality_row]
        b_eq = [1]

        # # π[a] >= 0 for all a
        # for i in range(num_actions):
        #     constraint = [0] + [-1 if j == i else 0 for j in range(num_actions)]
        #     G.append(constraint)
        #     h.append(0)

            # Define bounds: v is unbounded, π[a] ∈ [0,1]
        bounds = [(None, None)] + [(0, 1)] * num_actions

        result = linprog(c, A_ub=G, b_ub=h, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if result.success:
            v = result.x[0]
            policy = {a: prob for a, prob in zip(self.agent_actions, result.x[1:])}
            return policy
        else:
            raise ValueError(f"Linear programming failed at state {state}")