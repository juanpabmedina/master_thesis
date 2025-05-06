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

def evaluate_and_plot_transactions(agent_G, agent_C, kwargs, evaluate_param):
    
    total_steps = kwargs.get('total_steps', 100)
    init_gen_power = kwargs.get('init_gen_power', 0.3)
    init_con_price = kwargs.get('init_con_price', 2.0)
    min_power = kwargs.get('min_power', 0.1)
    max_power = kwargs.get('max_power', 0.5)
    min_price = kwargs.get('min_price', 1.0)
    max_price = kwargs.get('max_price', 5.0)
    threshold = kwargs.get('threshold', 1)
    gen_actions = kwargs.get('gen_actions', [-0.1, 0.0, 0.1])
    con_actions = kwargs.get('con_actions', [-1, 0, 1])
    a = kwargs.get('a', 0.1)
    b = kwargs.get('b', 2)
    c = kwargs.get('c', 0)

    env = EnergyMarketEnv(a, b, c, 
                        init_gen_power, init_con_price,
                        min_power, max_power,
                        min_price, max_price, 
                        threshold)
    
    GENERATOR_ACTIONS = gen_actions  # Power adjustments
    CONSUMER_ACTIONS = con_actions         # Price adjustments

    gen_states = []
    con_states = []

    gen_state, con_state = env.reset()
    state = (gen_state, con_state)

    for _ in range(total_steps):
        

        action_G = agent_G.select_action(state, evaluate=evaluate_param)
        if agent_C is None:
            action_C = random.choice(CONSUMER_ACTIONS)
        else:
            action_C = agent_C.select_action(state, evaluate=evaluate_param)

        next_state, reward, _ = env.step({'generator': action_G, 'consumer': action_C})

        state = next_state

        gen_states.append(next_state[0])
        con_states.append(next_state[1])

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

def train_agent(mode, kwargs):
    total_steps = kwargs.get('total_steps', 10000)
    eval_interval = kwargs.get('eval_interval', 100)
    init_gen_power = kwargs.get('init_gen_power', 0.3)
    init_con_price = kwargs.get('init_con_price', 2.0)
    min_power = kwargs.get('min_power', 0.1)
    max_power = kwargs.get('max_power', 0.5)
    min_price = kwargs.get('min_price', 1.0)
    max_price = kwargs.get('max_price', 5.0)
    threshold = kwargs.get('threshold', 1)
    gen_actions = kwargs.get('gen_actions', [-0.1, 0.0, 0.1])
    con_actions = kwargs.get('con_actions', [-1, 0, 1])
    a = kwargs.get('a', 0.1)
    b = kwargs.get('b', 2)
    c = kwargs.get('c', 0)
    
    GENERATOR_ACTIONS = gen_actions  # Power adjustments
    CONSUMER_ACTIONS = con_actions         # Price adjustments

    env = EnergyMarketEnv( a, b, c, 
                        init_gen_power, init_con_price,
                        min_power, max_power,
                        min_price, max_price, 
                        threshold)

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
    profit_history = []

    with tqdm(total=total_steps, desc=f"Training Mode: {mode}") as pbar:
        gen_state, con_state = env.reset()

        state = (gen_state, con_state)

        while step_count < total_steps:
            
            action_G = agent_G.select_action(state)
            if agent_C is None:
                action_C = random.choice(CONSUMER_ACTIONS)
            else:
                action_C = agent_C.select_action(state)

            actions_dict = {'generator': action_G, 'consumer': action_C}

            next_state, reward, done = env.step(actions_dict)

            agent_G.update(state, action_G, action_C, reward['generator'], next_state)
            if mode == 'MM':
                agent_C.update(state, action_C, action_G, reward['consumer'], next_state)

            cumulative_reward += reward['generator']
            step_count += 1
            pbar.update(1)

            if done:
                profit_history.append([env.profit,step_count, state])
                gen_state, con_state = env.reset()
                state = (gen_state, con_state)

            state = next_state

            if step_count % eval_interval == 0:
                reward_history.append(cumulative_reward)
                cumulative_reward = 0
                pbar.set_postfix(games_played=step_count)

    plot_rewards(reward_history, title=f'Rewards (MM)', window_size=10)


    return agent_G, agent_C, reward_history, profit_history
