import random
import numpy as np 
from scipy.optimize import linprog

class MiniMaxQAgent:
    def __init__(self, agent_parameters):

        # Agent information
        self.name = agent_parameters['name']
        self.agent_states = agent_parameters['states']
        self.agent_actions = agent_parameters['actions']
        self.opponent_actions = agent_parameters['opponent_actions']
        self.opponent_states = agent_parameters['opponent_states']
        self.rol = agent_parameters['rol']
        self.gamma=agent_parameters['gamma']
        self.epsilon=agent_parameters['epsilon']
        self.alpha=agent_parameters['alpha']
        self.decay=agent_parameters['decay']
        # self.agent_generation = agent_parameters['generation']
        # self.agent_consumption = agent_parameters['consumption']

        self.possible_states = []

        # self.gdr = self.agent_generation / self.agent_consumption

        # if self.rol == 'generator':
        #     self.agent_states = self.agent_generation
        # elif self.rol == 'consumer':
        #     self.agent_states = self.agent_consumption

        # Initialize Q, V, and policy π
        self.q_table = {}
        self.v_table = {}
        self.pi_table = {}

        self.state = (0,0)
        # self.get_rol(0)
        self.getPossibleStates()
        self.getInitLists()


    def getPossibleStates(self):
        # Generate all possible positions for both players
        for ag_s in self.agent_states:
            for op_s in self.opponent_states:
                state = (ag_s, op_s)
                self.possible_states.append(state)

    def getInitLists(self):
        for state in self.possible_states:
            self.q_table[state] = {}
            self.pi_table[state] = {}
            self.v_table[state] = 1
            for action in self.agent_actions:
                self.pi_table[state][action] = 1 / len(self.agent_actions)
                self.q_table[state][action] = {}
                for opponent_action in self.opponent_actions:
                    self.q_table[state][action][opponent_action] = 1 

    def resetAgent(self):
        self.getInitLists()
          
    def choose_action(self, state):
        """Choose an action based on epsilon-greedy strategy."""
        if random.random() < self.epsilon:
            return random.choice(self.agent_actions)
        else:
            if any(self.pi_table[state]) < 0:
                print("jueputa 0")
            return self.select_action(self.pi_table[state])
    

    def select_action(self, policy):
        """
        Select an action based on the optimized policy π[s, .].
        
        Args:
            policy (dict): A dictionary where keys are actions and values are probabilities.
            
        Returns:
            int: The selected action.
        """
        actions = list(policy.keys())  # List of available actions
        actions = [str(action) for action in actions]
        probabilities = np.array(list(policy.values()))  # Convert to numpy array

        # Clip small negative values due to floating-point precision errors
        probabilities = np.clip(probabilities, 0, 1)

        # Normalize to ensure sum is exactly 1
        probabilities /= np.sum(probabilities)

        # Try to sample an action
        try:
            chosen_action = np.random.choice(actions, p=probabilities)
            return eval(chosen_action)
        except ValueError as e:
            print(f"Error selecting action: {e}")
            print(f"Probabilities: {probabilities}, Sum: {np.sum(probabilities)}")
            print(f"Actions: {actions}")
            raise  # Re-raise the error for debugging

    def update(self, state, action, opponent_action, reward, next_state):
        """Update Q, π, and V after observing reward and transitioning to next state."""
        self.state = state
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
                row.append(-self.q_table[state][a][o])  # Coefficients for -π[a] * Q[s, a, o]
            G.append(row)
            h.append(0)

        # Probability distribution constraint: sum(π[a]) = 1
        equality_row = [0] + [1] * num_actions
        A_eq = [equality_row]
        b_eq = [1]


        # Define bounds: v is unbounded, π[a] ∈ [0,1]
        bounds = [(None, None)] + [(0, 1)] * num_actions

        result = linprog(c, A_ub=G, b_ub=h, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if result.success:
            v = result.x[0]
            policy = {a: prob for a, prob in zip(self.agent_actions, result.x[1:])}
            return policy
        else:
            raise ValueError(f"Linear programming failed at state {state}")
    

    