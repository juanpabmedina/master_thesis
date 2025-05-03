import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from energy_market_env import EnergyMarketEnv
from minimax_agent import MinimaxQAgent

# Define the action sets directly
GENERATOR_ACTIONS = [-0.1, 0.0, 0.1]  # Power adjustments
CONSUMER_ACTIONS = [-1, 0, 1]          # Price adjustments

def plot_rewards(rewards, title, window_size=10):
    episodes = np.arange(len(rewards))
    smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, label='Reward per Transaction', alpha=0.5)
    plt.plot(episodes[:len(smoothed_rewards)], smoothed_rewards, label=f'Moving Average (window={window_size})', color='red', linewidth=2)
    plt.xlabel('Transactions')
    plt.ylabel('Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def evaluate_and_plot_transactions(agent_G, agent_C, gen_state, con_state,episodes=100):
    env = EnergyMarketEnv(a=0.1, b=2, c=0, 
                        init_gen_power=gen_state, init_con_price=con_state,
                        min_power=0.1, max_power=0.5,
                        min_price=1.0, max_price=5, 
                        threshold=0)
    gen_states = []
    con_states = []

    for _ in range(episodes):
        

        action_G = agent_G.select_action((gen_state, con_state), evaluate=True)
        if agent_C is None:
            action_C = random.choice(CONSUMER_ACTIONS)
        else:
            action_C = agent_C.select_action((gen_state, con_state), evaluate=True)

        next_state, reward, _ = env.step({'generator': action_G, 'consumer': action_C})

        gen_states.append(next_state[0])
        con_states.append(next_state[1])

    steps_range = range(episodes)
    plt.figure(figsize=(12, 6))
    plt.plot(steps_range, gen_states, label='Generator Power Offer (kW)', color='blue', marker='o')
    plt.plot(steps_range, con_states, label='Consumer Price Offer ($/kWh)', color='green', marker='x')
    plt.title('Evolution of Transaction Offers Over Time')
    plt.xlabel('Transaction (Step)')
    plt.ylabel('Offer Values')
    plt.legend()
    plt.grid(True)
    plt.show()

def train_agent(mode='MR', total_steps=100000, eval_interval=1000):
    env = EnergyMarketEnv( a=0.1, b=2, c=0, 
                        init_gen_power=2, init_con_price=0.2,
                        min_power=0.1, max_power=0.5,
                        min_price=1.0, max_price=5, 
                        threshold=0)

    agent_G = MinimaxQAgent(player_id='G', actions=GENERATOR_ACTIONS)

    if mode == 'MR':
        agent_C = None  # Consumer acts randomly
    elif mode == 'MM':
        agent_C = MinimaxQAgent(player_id='C', actions=CONSUMER_ACTIONS)
    elif mode == 'MC':
        with open('trained_challenger.pkl', 'rb') as f:
            agent_C = pickle.load(f)
    else:
        raise ValueError("Mode must be 'MR', 'MM', or 'MC'.")

    step_count = 0
    cumulative_reward = 0
    reward_history = []

    with tqdm(total=total_steps, desc=f"Training Mode: {mode}") as pbar:
        gen_state, con_state = env.reset()

        while step_count < total_steps:
            
            action_G = agent_G.select_action((gen_state, con_state))
            if agent_C is None:
                action_C = random.choice(CONSUMER_ACTIONS)
            else:
                action_C = agent_C.select_action((gen_state, con_state))

            actions_dict = {'generator': action_G, 'consumer': action_C}

            next_state, reward, done = env.step(actions_dict)

            agent_G.update((gen_state, con_state), action_G, action_C, reward['generator'], next_state)
            if mode == 'MM':
                agent_C.update((gen_state, con_state), action_C, action_G, reward['consumer'], next_state)

            cumulative_reward += reward['generator']
            step_count += 1
            pbar.update(1)

            if done:
                gen_state, con_state = env.reset()

            if step_count % eval_interval == 0:
                reward_history.append(cumulative_reward)
                cumulative_reward = 0
                pbar.set_postfix(games_played=step_count)



    return agent_G, agent_C, reward_history
