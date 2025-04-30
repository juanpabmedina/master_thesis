import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from soccer_env import SoccerGame, ACTIONS
from minimax_agent import MinimaxQAgent

def plot_wins(wins_history, window):
    plt.figure(figsize=(10, 5))
    plt.plot(range(window, window * len(wins_history) + 1, window), wins_history, label=f'Games Won per {window} Steps')
    plt.xlabel('Step')
    plt.ylabel(f'Games Won in last {window} Steps')
    plt.title('Minimax-Q Agent Wins over Training')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_rewards(rewards, title, window_size=10):
    episodes = np.arange(len(rewards))
    smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, label='Recompensa por episodio', alpha=0.5)
    plt.plot(episodes[:len(smoothed_rewards)], smoothed_rewards, label=f'Promedio móvil (window={window_size})', color='red', linewidth=2)
    plt.xlabel('Episodios')
    plt.ylabel('Recompensa')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def train_agent(mode='MR', total_steps=100000, eval_interval=1000):
    env = SoccerGame()
    actions = ACTIONS

    agent_A = MinimaxQAgent(player_id='A', actions=actions)

    if mode == 'MR':
        agent_B = None
    elif mode == 'MM':
        agent_B = MinimaxQAgent(player_id='B', actions=actions)
    elif mode == 'MC':
        with open('trained_challenger.pkl', 'rb') as f:
            agent_B = pickle.load(f)
    else:
        raise ValueError("Mode must be 'MR', 'MM', or 'MC'")

    step_count = 0
    game_count = 0
    cumulative_reward = 0
    reward_history = []

    state = env.reset()

    with tqdm(total=total_steps, desc=f"Training Mode: {mode}") as pbar:
        while step_count < total_steps:
            action_A = agent_A.select_action(state)
            action_B = random.choice(actions) if agent_B is None else agent_B.select_action(state)
            actions_dict = {'A': action_A, 'B': action_B}

            next_state, reward, done = env.step(actions_dict)

            agent_A.update(state, action_A, action_B, reward['A'], next_state)
            if mode == 'MM':
                agent_B.update(state, action_B, action_A, reward['B'], next_state)

            cumulative_reward += reward['A']
            step_count += 1
            pbar.update(1)
            state = next_state

            if done:
                game_count += 1
                state = env.reset()

            if step_count % eval_interval == 0:
                reward_history.append(cumulative_reward)
                cumulative_reward = 0
                pbar.set_postfix(games_played=game_count)

    return agent_A, agent_B, reward_history

def evaluate_policy(agent_A, agent_B=None, episodes=100):
    env = SoccerGame()
    actions = ACTIONS

    wins = 0
    losses = 0
    draws = 0

    for _ in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action_A = agent_A.select_action(state, evaluate=True)
            action_B = random.choice(actions) if agent_B is None else agent_B.select_action(state, evaluate=True)
            actions_dict = {'A': action_A, 'B': action_B}
            state, reward, done = env.step(actions_dict)

        if reward['A'] > 0:
            wins += 1
        elif reward['A'] < 0:
            losses += 1
        else:
            draws += 1

    print(f"Evaluation over {episodes} episodes:")
    print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")

def evaluate_against_challenger(agent_path='trained_agent.pkl', challenger_path='trained_challenger.pkl', episodes=100):
    env = SoccerGame()
    actions = ACTIONS

    with open(agent_path, 'rb') as f:
        agent_A = pickle.load(f)
    with open(challenger_path, 'rb') as f:
        agent_B = pickle.load(f)

    wins = 0
    losses = 0
    draws = 0

    for _ in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action_A = agent_A.select_action(state, evaluate=True)
            action_B = agent_B.select_action(state, evaluate=True)
            actions_dict = {'A': action_A, 'B': action_B}
            state, reward, done = env.step(actions_dict)

        if reward['A'] > 0:
            wins += 1
        elif reward['A'] < 0:
            losses += 1
        else:
            draws += 1

    print(f"Evaluation vs Challenger over {episodes} episodes:")
    print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")
