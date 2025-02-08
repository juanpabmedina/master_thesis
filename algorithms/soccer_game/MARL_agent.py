import numpy as np
import random
from scipy.optimize import linprog

class MARLAgent:
    def __init__(self, actions, opponent_actions, gamma=0.9, epsilon=0.2, alpha=1, decay=0.9999954):
        self.actions = actions
        self.opponent_actions = opponent_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.decay = decay

        # initialize possiible sattes
        self.possible_states = []

        # Grid dimensions
        self.grid_rows = 4
        self.grid_cols = 5

        # Initialize Q, V, and policy π
        self.q_table = {}
        self.v_table = {}
        self.pi_table = {}

        self.getPossibleStates()
        self.getInitLists()

    def getPossibleStates(self):
        # Generate all possible positions for both players
        for ax in range(self.grid_rows):
            for ay in range(self.grid_cols):
                for bx in range(self.grid_rows):
                    for by in range(self.grid_cols):
                        # Ensure players occupy distinct positions
                        if (ax, ay) != (bx, by):
                            # Ball possession can either be with Player A or Player B
                            for ball_possession in ['A', 'B']:
                                # Store the state as a tuple: ((Player A position), (Player B position), Ball possession)
                                state = ((ax, ay), (bx, by), ball_possession)
                                self.possible_states.append(state)

    def getInitLists(self):
        for state in self.possible_states:
            self.q_table[state] = {}
            self.pi_table[state] = {}
            self.v_table[state] = 1
            for action in self.actions:
                self.pi_table[state][action] = 1 / len(self.actions)
                self.q_table[state][action] = {}
                for opponent_action in self.opponent_actions:
                    self.q_table[state][action][opponent_action] = 1 

          
    def choose_action(self, state):
        """Choose an action based on epsilon-greedy strategy."""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            if any(self.pi_table[state]) < 0:
                print("jueputa 0")
            return self.select_action(self.pi_table[state])
    
    import numpy as np

    def select_action(self, policy):
        """
        Select an action based on the optimized policy π[s, .].
        
        Args:
            policy (dict): A dictionary where keys are actions and values are probabilities.
            
        Returns:
            int: The selected action.
        """
        actions = list(policy.keys())  # List of available actions
        probabilities = np.array(list(policy.values()))  # Convert to numpy array

        # Clip small negative values due to floating-point precision errors
        probabilities = np.clip(probabilities, 0, 1)

        # Normalize to ensure sum is exactly 1
        probabilities /= np.sum(probabilities)

        # Try to sample an action
        try:
            chosen_action = np.random.choice(actions, p=probabilities)
            return chosen_action
        except ValueError as e:
            print(f"Error selecting action: {e}")
            print(f"Probabilities: {probabilities}, Sum: {np.sum(probabilities)}")
            print(f"Actions: {actions}")
            raise  # Re-raise the error for debugging


    def update(self, state, action, opponent_action, joint_reward, next_state, agent = 'A'):
        """Update Q, π, and V after observing reward and transitioning to next state."""
        reward = joint_reward[agent]
        # if agent == 'B':
        #     a = action
        #     action = opponent_action
        #     opponent_action = a
            
        # Update Q-value
        self.q_table[state][action][opponent_action] = (1 - self.alpha) *  self.q_table[state][action][opponent_action] + self.alpha * (reward + self.gamma*self.v_table[next_state])
        
        # Update π using Linear Programming 
        self.pi_table[state] = self.optimize_policy(state)

        worst_case_value = float('inf')
        
        # Update V based on min over opponent actions
        for o in self.opponent_actions:
            # Calculate the expected value for this opponent action o
            expected_value = 0
            for a in self.actions:
                expected_value += self.pi_table[state][a]* self.q_table[state][a][o]
            
            # Update the worst-case (minimum over opponent actions)
            worst_case_value = min(worst_case_value, expected_value)

        self.v_table[state] = worst_case_value
        # Decay learning rate
        self.alpha *= self.decay


    def optimize_policy(self, state):
        """
        Compute the optimal policy π[s, .] using linear programming.

        Args:
            s (int): The current state.

        Returns:
            dict: A dictionary π[s, a] representing the optimal policy for state s.
        """
        num_actions = len(self.actions)
        
        # Decision variables: π[a] for each action + v (worst-case value)
        c = [-1] + [0] * num_actions  # Maximize v (converted to minimization by negation)

        # Constraints: Gx <= h
        G = []
        h = []
        
        # Worst-case expected value constraint
        for o in self.opponent_actions:
            row = [1]  # Coefficient for v
            for a in self.actions:
                row.append(-self.q_table[state][a][o])  # Coefficients for -π[a] * Q[s, a, o]
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
            policy = {a: prob for a, prob in zip(self.actions, result.x[1:])}
            return policy
        else:
            raise ValueError(f"Linear programming failed at state {state}")


class QLAgent:
    def __init__(self, actions, gamma=0.9, epsilon=0.2, alpha=1, decay=0.9999954):
        self.actions = actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.decay = decay

        # initialize possiible sattes
        self.possible_states = []

        # Grid dimensions
        self.grid_rows = 4
        self.grid_cols = 5

        # Initialize Q, V, and policy π
        self.q_table = {}

        self.getPossibleStates()
        self.getInitLists()

    def getPossibleStates(self):
        # Generate all possible positions for both players
        for ax in range(self.grid_rows):
            for ay in range(self.grid_cols):
                for bx in range(self.grid_rows):
                    for by in range(self.grid_cols):
                        # Ensure players occupy distinct positions
                        if (ax, ay) != (bx, by):
                            # Ball possession can either be with Player A or Player B
                            for ball_possession in ['A', 'B']:
                                # Store the state as a tuple: ((Player A position), (Player B position), Ball possession)
                                state = ((ax, ay), (bx, by), ball_possession)
                                self.possible_states.append(state)

    def getInitLists(self):
        for state in self.possible_states:
            self.q_table[state] = {}
            for action in self.actions:
                self.q_table[state][action] = 1

          
    def choose_action(self, state):
        """Choose an action based on epsilon-greedy strategy."""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            a = np.argmax(list(self.q_table[state].values()))
            return self.actions[a]

    def update(self, state, action, joint_reward, next_state, agent = 'A'):
        reward = joint_reward[agent]
        """
        Update the Q-value based on the reward and next state.

        Args:
            s (int): Current state.
            a (str): Action taken.
            s_prime (int): Next state.
            reward (float): Received reward.
        """
        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        best_next_q_value = self.q_table[next_state][best_next_action]

        # Standard Q-learning update
        self.q_table[state][action] = (1 - self.alpha) * self.q_table[state][action] + \
                       self.alpha * (reward + self.gamma * best_next_q_value)

        self.alpha *= self.decay
