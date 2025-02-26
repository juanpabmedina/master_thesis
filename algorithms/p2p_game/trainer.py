import numpy as np
from tqdm import tqdm

class MinimaxTrainer:
    def __init__(self, agent, opponent, enviroment, algorithm):
        self.agent = agent
        self.opponent = opponent
        self.enviroment = enviroment
        self.algorithm = algorithm

        self.hist_reward = []
        
    
    def train(self, steps):
        """Ejecuta el entrenamiento minimax."""
        state = self.agent.get_state(20)
        opponent_state = self.opponent.get_state(2)

        with tqdm(total=steps, desc="Progreso", unit="step") as pbar:
            for i in range(steps):
                t = 10
                # Agent action and next state
                action = round(self.agent.choose_action(state),2)
                next_state = self.agent.get_next_state(state, action)

                # Opponent action and next opponent state
                opponent_action = self.opponent.choose_action(opponent_state)
                next_opponent_state = self.opponent.get_next_state(opponent_state, opponent_action)

                # Obtein the reward
                agent_rol = self.agent.get_rol(t)
                reward = self.enviroment.get_reward(rol=agent_rol, generator_state=state, consumer_state=opponent_state)
                self.hist_reward.append(reward)

                # Calculate the new policy
                policy = self.algorithm.update(state, action, opponent_action, reward, next_state)

                # Update states and policy
                state = next_state
                opponent_state = next_opponent_state
                self.agent.pi_table = policy

                pbar.update(1)

        return self.hist_reward