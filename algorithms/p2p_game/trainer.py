import numpy as np
from tqdm import tqdm

class MinimaxTrainer:
    def __init__(self, agent1, agent2, enviroment):
        self.agent1 = agent1
        self.agent2 = agent2
        self.market_env = enviroment
    
    def trainMR(self, init_generator_state, init_consumer_state, steps):
        """Ejecuta el entrenamiento minimax."""
        self.agent1.state = (init_generator_state, init_consumer_state)
        self.agent2.state = (init_consumer_state, init_generator_state)
        hist_reward = []
        self.agent1.resetAgent()
        self.agent2.resetAgent()
        with tqdm(total=steps, desc="Progreso", unit="step") as pbar:
            for i in range(steps):
                # print('state: ', agent1.state, agent2.state)
                # Agent action and next state
                action = round(self.agent1.choose_action(self.agent1.state),2)
                opponent_action = round(np.random.choice(self.agent2.agent_actions),2)

                # print('action: ', action, opponent_action)

                joint_action1 = (action, opponent_action)
                joint_action2 = (opponent_action, action)
                # print('Joint action: ',joint_action1, joint_action2)

                next_state1, reward1 = self.market_env.step(self.agent1, joint_action=joint_action1)
                next_state2, reward2 = self.market_env.step(self.agent2, joint_action=joint_action2)

                # print('step 1: ', next_state1, reward1)
                # print('step 2: ', next_state2, reward2)

                self.agent1.update(self.agent1.state, action, opponent_action, reward1, next_state1)

                self.agent1.state = next_state1
                self.agent2.state = next_state2

                hist_reward.append(reward1)
                pbar.update(1)

        return hist_reward, self.agent1.pi_table
    
    def trainMM(self, init_generator_state, init_consumer_state, steps):
        """Ejecuta el entrenamiento minimax."""
        self.agent1.state = (init_generator_state, init_consumer_state)
        self.agent2.state = (init_consumer_state, init_generator_state)
        hist_reward1 = []
        hist_reward2 = []
        self.agent1.resetAgent()
        self.agent2.resetAgent()
        with tqdm(total=steps, desc="Progreso", unit="step") as pbar:
            for i in range(steps):
                # print('state: ', agent1.state, agent2.state)
                # Agent action and next state
                action = round(self.agent1.choose_action(self.agent1.state),2)
                opponent_action = round(self.agent2.choose_action(self.agent2.state),2)

                # print('action: ', action, opponent_action)

                joint_action1 = (action, opponent_action)
                joint_action2 = (opponent_action, action)
                # print('Joint action: ',joint_action1, joint_action2)

                next_state1, reward1 = self.market_env.step(self.agent1, joint_action=joint_action1)
                next_state2, reward2 = self.market_env.step(self.agent2, joint_action=joint_action2)

                # print('step 1: ', next_state1, reward1)
                # print('step 2: ', next_state2, reward2)

                self.agent1.update(self.agent1.state, action, opponent_action, reward1, next_state1)
                self.agent2.update(self.agent2.state, opponent_action, action, reward2, next_state2)

                self.agent1.state = next_state1
                self.agent2.state = next_state2

                hist_reward1.append(reward1)
                hist_reward2.append(reward2)
                pbar.update(1)

        return hist_reward1, hist_reward2, self.agent1.pi_table, self.agent2.pi_table
    
    def evaluate(self, init_generator_state, init_consumer_state, pi_table1, pi_table2, steps):
        # self.agent1.resetAgent()
        # self.agent2.resetAgent()
        self.agent1.state = (init_generator_state, init_consumer_state)
        self.agent2.state = (init_consumer_state, init_generator_state)
        power =[]
        cost =[]
        with tqdm(total=steps, desc="Progreso", unit="step") as pbar:
            for i in range(steps):
                # print('state: ', agent1.state, agent2.state)
                # Agent action and next state
                action = round(self.agent1.select_action(pi_table1[self.agent1.state]),2)
                opponent_action = round(self.agent2.select_action(pi_table2[self.agent2.state]),2)

                # print('action: ', action, opponent_action)

                joint_action1 = (action, opponent_action)
                joint_action2 = (opponent_action, action)
                # print('Joint action: ',joint_action1, joint_action2)

                next_state1, _ = self.market_env.step(self.agent1, joint_action=joint_action1)
                next_state2, _ = self.market_env.step(self.agent2, joint_action=joint_action2)
                # print('step 1: ', next_state1, reward1)
                # print('step 2: ', next_state2, reward2)

                self.agent1.state = next_state1
                self.agent2.state = next_state2

                power.append(self.agent1.state[0])
                cost.append(self.agent2.state[0])
                pbar.update(1)
        
        return power, cost