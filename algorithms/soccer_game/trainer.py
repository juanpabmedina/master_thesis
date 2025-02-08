import random
import numpy as np
from tqdm import tqdm
from MARL_agent import MARLAgent, QLAgent
from soccer import SoccerGame
from time import sleep

class MinimaxTrainer:
    def __init__(self, steps, player='A', opponent='random', pi_table={}):
        self.steps = steps
        self.player = player
        self.opponent = opponent
        self.pi_table = pi_table
        self.actions = ['N', 'S', 'E', 'W', 'stand']
        
        # Agent creation
        self.agent = MARLAgent(self.actions, self.actions, epsilon=0.2)
        self.agent2 = MARLAgent(self.actions, self.actions, epsilon=0.2) if opponent == 'minimax' else None
        
        # Game creation
        self.game = SoccerGame()
        self.state = ((1, 3), (2, 1), self.game.ball_possession)
        
        # Reward tracking
        self.cum_reward = 0
        self.hist_reward = []
        self.goal_count = 0

    def choose_opponent_action(self):
        """Determina la acción del oponente según la estrategia elegida."""
        if self.opponent == 'random':
            return random.choice(self.actions)
        elif self.opponent == 'pi_table':
            return self.actions[np.argmax(list(self.pi_table[self.state].values()))]
        elif self.opponent == 'minimax':
            return self.agent2.choose_action(self.state)
    
    def update_agents(self, action, opponent_action, reward, new_state):
        """Actualiza los parámetros de los agentes."""
        self.agent.update(self.state, action, opponent_action, reward, new_state, agent=self.player)
        if self.agent2:
            self.agent2.update(self.state, opponent_action, action, reward, new_state, agent=self.player)
    
    def train(self):
        """Ejecuta el entrenamiento minimax."""
        self.game.print_state()  # Print initial state
        
        with tqdm(total=self.steps, desc="Progreso", unit="step") as pbar:
            for i in range(self.steps):
                action = self.agent.choose_action(self.state)
                opponent_action = self.choose_opponent_action()
                
                joint_action = {'A': action, 'B': opponent_action} if self.player == 'A' else {'A': opponent_action, 'B': action}
                
                new_state, reward, done = self.game.play_turn(joint_action)
                self.cum_reward += reward[self.player]
                
                if i % 1000 == 0:
                    self.hist_reward.append(self.cum_reward)
                    self.cum_reward = 0

                if done:
                    self.goal_count += 1

                self.update_agents(action, opponent_action, reward, new_state)
                self.state = new_state
                pbar.update(1)

        print(f'Played games: {self.goal_count}')
        print("...Running validation...")
        self.run_validation(self.agent.pi_table, player = self.player, render=False, opponent='random', pi_table_opponent={})
        return (self.hist_reward, self.agent.pi_table, self.agent2.pi_table) if self.agent2 else (self.hist_reward, self.agent.pi_table)

    def run_validation(self, pi_table, player='A', render=False, opponent='random', pi_table_opponent={}, steps = 100):
        """Valida el resultado del algoritmo de manera más eficiente."""

        self.game.reset_game

        self.state = ((1, 3), (2, 1), self.game.ball_possession)
        agent = MARLAgent(self.actions, self.actions, epsilon=0.2)
        count, count_win = 0, 0
        
        for _ in range(steps):
            action = agent.select_action(pi_table[self.state])
            opponent_action = random.choice(self.actions) if opponent == 'random' else self.actions[np.argmax(list(pi_table_opponent[self.state].values()))] if opponent == 'pi_table' else None
            
            joint_action = {'A': action, 'B': opponent_action} if player == 'A' else {'A': opponent_action, 'B': action}
            new_state, reward, done = self.game.play_turn(joint_action)
            
            if done:
                count += 1
                count_win += (reward[player] == 1)
            
            if random.random() < 0.1:
                self.game.reset_pos()
                self.state = ((1,3),(2,1), self.game.ball_possession)
            else:
                self.state = new_state
            
            if render:
                self.game.print_state()
                sleep(0.5)
                if self.game.scores[player] == 2:
                    break
        
        win_rate = (count_win / count * 100) if count > 0 else 0
        print(f'Se completaron {count} juegos y se ganó el {win_rate}% ')

class QLearningTrainer:
    def __init__(self, steps, player='A', opponent='random', pi_table={}, render = False):
        self.steps = steps
        self.render = render
        self.player = player
        self.opponent = opponent
        self.pi_table = pi_table
        self.actions = ['N', 'S', 'E', 'W', 'stand']
        
        # Agent creation
        self.agent = QLAgent(self.actions, epsilon=0.2)
        self.agent2 = QLAgent(self.actions, epsilon=0.2) if opponent == 'q-learning' else None
        
        # Game creation
        self.game = SoccerGame()
        self.state = ((1, 3), (2, 1), self.game.ball_possession)
        
        # Reward tracking
        self.cum_reward = 0
        self.hist_reward = []
        self.goal_count = 0
    
    def choose_opponent_action(self):
        """Determina la acción del oponente según la estrategia elegida."""
        if self.opponent == 'random':
            return random.choice(self.actions)
        elif self.opponent == 'pi_table':
            return self.actions[np.argmax(list(self.pi_table[self.state].values()))]
        elif self.opponent == 'q-learning':
            return self.agent2.choose_action(self.state)
    
    def update_agents(self, action, opponent_action, reward, new_state):
        """Actualiza los parámetros de los agentes."""
        self.agent.update(self.state, action, reward, new_state, agent=self.player)
        if self.agent2:
            self.agent2.update(self.state, opponent_action, reward, new_state, agent=self.player)
    
    def train(self):
        """Ejecuta el entrenamiento Q-learning."""
        self.game.print_state()
        
        with tqdm(total=self.steps, desc="Progreso", unit="step") as pbar:
            for i in range(self.steps):
                action = self.agent.choose_action(self.state)
                opponent_action = self.choose_opponent_action()
                
                joint_action = {'A': action, 'B': opponent_action} if self.player == 'A' else {'A': opponent_action, 'B': action}
                new_state, reward, done = self.game.play_turn(joint_action)
                
                self.cum_reward += reward[self.player]
                if i % 1000 == 0:
                    self.hist_reward.append(self.cum_reward)
                    self.cum_reward = 0
                
                if done:
                    self.goal_count += 1
                
                self.update_agents(action, opponent_action, reward, new_state)
                self.state = new_state
                pbar.update(1)

        print(f'Played games: {self.goal_count}')
        print("...Running validation...")
        self.run_validation(self.agent.q_table, player = self.player, render=self.render, opponent='random', Q_table_opponent={})
        return (self.hist_reward, self.agent.q_table, self.agent2.q_table) if self.agent2 else (self.hist_reward, self.agent.q_table)

    def run_validation(self, Q_table, player='A', render=False, opponent='random', Q_table_opponent={}, steps = 100):
        """Valida el resultado del algoritmo de manera más eficiente."""

        self.game.reset_game

        self.state = ((1, 3), (2, 1), self.game.ball_possession)
        count, count_win = 0, 0
        
        for _ in range(steps):
            a = np.argmax(list(Q_table[self.state].values()))
            action = self.actions[a]
            opponent_action = random.choice(self.actions) if opponent == 'random' else self.actions[np.argmax(list(Q_table_opponent[self.state].values()))] if opponent == 'pi_table' else None
            
            joint_action = {'A': action, 'B': opponent_action} if player == 'A' else {'A': opponent_action, 'B': action}
            new_state, reward, done = self.game.play_turn(joint_action)
            
            if done:
                count += 1
                count_win += (reward[player] == 1)
            
            if random.random() < 0.1:
                self.game.reset_pos()
                self.state = ((1,3),(2,1), self.game.ball_possession)
            else:
                self.state = new_state
            
            if render:
                self.game.print_state()
                sleep(0.5)
                if self.game.scores[player] == 5:
                    break
        
        win_rate = (count_win / count * 100) if count > 0 else 0
        print(f'Se completaron {count} juegos y se ganó el {win_rate}% ')