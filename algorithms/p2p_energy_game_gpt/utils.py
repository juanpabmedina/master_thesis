import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from energy_market_env import EnergyMarketEnv
from minimax_agent import MinimaxQAgent

# Define the action sets directly
# GENERATOR_ACTIONS = [-0.1, 0.0, 0.1]  # Power adjustments
# CONSUMER_ACTIONS = [-1, 0, 1]          # Price adjustments

def plot_rewards(rewards, title, window_size=10):
    episodes = np.arange(len(rewards))
    smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, label='Reward per Transaction', alpha=0.5)
    plt.plot(episodes[:len(smoothed_rewards)], smoothed_rewards, label=f'Moving Average (window={window_size})', color='red', linewidth=2)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def evaluate(agent1, agent2, kwargs, plot=True):
    total_steps = kwargs.get('total_steps', 10000)
    eval_interval = kwargs.get('eval_interval', 100)
    init_gen_power = kwargs.get('init_gen_power', 0.3)
    init_con_price = kwargs.get('init_con_price', 2.0)
    min_power = kwargs.get('min_power', 0.1)
    max_power = kwargs.get('max_power', 0.5)
    min_price = kwargs.get('min_price', 1.0)
    max_price = kwargs.get('max_price', 5.0)
    threshold = kwargs.get('threshold', 1)
    agent1_actions = kwargs.get('agent1_actions', [-0.1, 0.0, 0.1])
    agent2_actions = kwargs.get('agent2_actions', [-1, 0, 1])
    a = kwargs.get('a', 0.1)
    b = kwargs.get('b', 2)
    c = kwargs.get('c', 0)
    agent1_id = kwargs.get('agent1_id', 'G')
    agent2_id = kwargs.get('agent2_id', 'C')

    env = EnergyMarketEnv( a, b, c, 
                        init_gen_power, init_con_price,
                        min_power, max_power,
                        min_price, max_price, 
                        threshold,
                        agent1_id)

    gen_states = []
    con_states = []

    gen_state, con_state = env.reset()
    state = (gen_state, con_state)

    next_state = state

    for _ in range(total_steps):
        
        gen_states.append(next_state[0])
        con_states.append(next_state[1])

        a1 = agent1.select_action(state, evaluate=True)
        if agent2 is None:
            a2 = random.choice(agent2_actions)
        else:
            a2 = agent2.select_action(state, evaluate=True)

        actions_dict = {agent1_id: a1, agent2_id: a2}

        next_state, _, _ = env.step(actions_dict, agent1_id)

        state = next_state


    if plot == True:
        print('Generator 10 last mean', np.mean(gen_states[-10:]))
        print('Consummer 10 last mean', np.mean(con_states[-10:]))
        steps_range = range(total_steps)
        plt.figure(figsize=(12, 6))
        plt.plot(steps_range, gen_states, label='Generator Power Offer (kW)', color='blue', marker='o')
        plt.plot(steps_range, con_states, label='Consumer Price Offer ($/kWh)', color='green', marker='x')
        plt.title('Evolution of Transaction Offers Over Time')
        plt.xlabel('Transaction (Step)')
        plt.ylabel('Offer Values')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        return np.mean(gen_states[-10:]), np.mean(con_states[-10:])


def train(mode, kwargs):
    total_steps = kwargs.get('total_steps', 10000)
    eval_interval = kwargs.get('eval_interval', 100)
    init_gen_power = kwargs.get('init_gen_power', 0.3)
    init_con_price = kwargs.get('init_con_price', 2.0)
    min_power = kwargs.get('min_power', 0.1)
    max_power = kwargs.get('max_power', 0.5)
    min_price = kwargs.get('min_price', 1.0)
    max_price = kwargs.get('max_price', 5.0)
    gen_threshold = kwargs.get('gen_threshold', 1)
    con_threshold = kwargs.get('con_threshold', 1)
    agent1_actions = kwargs.get('agent1_actions', [-0.1, 0.0, 0.1])
    agent2_actions = kwargs.get('agent2_actions', [-1, 0, 1])
    a = kwargs.get('a', 0.1)
    b = kwargs.get('b', 2)
    c = kwargs.get('c', 0)
    agent1_id = kwargs.get('agent1_id', 'G')
    agent2_id = kwargs.get('agent2_id', 'C')

    env = EnergyMarketEnv( a, b, c, 
                        init_gen_power, init_con_price,
                        min_power, max_power,
                        min_price, max_price, 
                        gen_threshold, con_threshold,
                        agent1_id)

    agent1 = MinimaxQAgent(actions=agent1_actions, opponent_actions=agent2_actions)

    if mode == 'MR':
        agent2 = None  # Consumer acts randomly
    elif mode == 'MM':
        agent2 = MinimaxQAgent(actions=agent2_actions, opponent_actions=agent1_actions)
    elif mode == 'MC':
        with open('trained_challenger.pkl', 'rb') as f:
            agent2 = pickle.load(f)
    else:
        raise ValueError("Mode must be 'MR', 'MM', or 'MC'.")

    step_count = 0
    games_played = 0
    cumulative_reward = 0
    reward_history = []
    profit_history = []

    ag1_win = 0
    ag2_win = 0

    with tqdm(total=total_steps, desc=f"Training Mode: {mode}") as pbar:
        gen_state, con_state = env.reset()

        state = (gen_state, con_state)

        while step_count < total_steps:
            
            a1 = agent1.select_action(state)
            if agent2 is None:
                a2 = random.choice(agent2_actions)
            else:
                a2 = agent2.select_action(state)

            actions_dict = {agent1_id: a1, agent2_id: a2}

            next_state, reward, done = env.step(actions_dict, agent1_id) ##### BUUUUG ARREGLAR ESTO -> SI NO LE DOY EL ID NO ENTRENA BIEN

            agent1.update(state, a1, a2, reward[agent1_id], next_state)
            if mode == 'MM':
                agent2.update(state, a2, a1, reward[agent2_id], next_state)

            cumulative_reward += reward[agent1_id]
            step_count += 1
            pbar.update(1)

            if done:
                if reward[agent1_id] == 1:
                    ag1_win += 1
                else:
                    ag2_win += 1

                gen_state, con_state = env.reset()
                state = (gen_state, con_state)
                games_played += 1

            state = next_state

            if step_count % eval_interval == 0:
                reward_history.append(cumulative_reward)
                cumulative_reward = 0
                pbar.set_postfix(games_played=games_played)

    plot_rewards(reward_history, title=f'Rewards (MM)', window_size=10)


    return agent1, agent2, reward_history, profit_history, ag1_win, ag2_win