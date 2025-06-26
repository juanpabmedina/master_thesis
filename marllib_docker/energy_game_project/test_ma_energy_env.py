# from energy_market_env import env
from ma_energy_market_env import parallel_env
from pettingzoo.test import parallel_api_test
from pettingzoo.test import api_test
import time


# Create environment instances for testing
env_config = {
    "max_cycles": 100,             # Episode length
    "max_gen_power": 1,
    "min_gen_power": 0.1,
    "max_con_price": 1,
    "min_con_price": 0.1,
    "profit_threshold": 5,
    "max_steps": 100,
    "num_generators": 2,           # Number of generators
    "num_consumers": 2             # Number of consumers
}

max_cycles = env_config.get("max_cycles", 100)
max_gen_power = env_config.get("max_gen_power", 1)
min_gen_power = env_config.get("min_gen_power", 0.1)
max_con_price = env_config.get("max_con_price", 1)
min_con_price = env_config.get("min_con_price", 0.1)
profit_threshold = env_config.get("profit_threshold", 5)
max_steps = env_config.get("max_steps", 100)
num_generators = env_config.get("num_generators", 2)
num_consumers = env_config.get("num_consumers", 2)

parallel_energy_env = parallel_env(
    max_gen_power, min_gen_power, 
    max_con_price, min_con_price, 
    profit_threshold, max_steps,
    num_generators, num_consumers
)

print("\n=== Running Parallel API Test ===")
parallel_api_test(parallel_energy_env, num_cycles=10)

print("\n=== API Tests Completed Successfully ===")


# Action mappings for readable output
ACTIONS = ['DECREASE', 'STAY', 'INCREASE']

# Initialize environment for manual testing
env_instance = parallel_energy_env
observations = env_instance.reset()

print("\n=== Starting Multi-Agent Energy Market Game ===")
print(f"Number of Generators: {env_instance.num_generators}")
print(f"Number of Consumers: {env_instance.num_consumers}")
print(f"Agents: {env_instance.agents}")
print(f"Profit Threshold: {env_instance.profit_threshold}")
print(f"Max Steps: {env_instance.max_steps}")

# Initialize local dones
dones = {agent: False for agent in env_instance.agents}

for step_count in range(10):
    # Prepare actions for non-done agents
    actions = {
        agent: env_instance.action_spaces[agent].sample()
        for agent in env_instance.agents
    }

    print(f"\n--- Step {step_count + 1} ---")
    for agent, action in actions.items():
        print(f"Agent {agent} → Action {action} ({ACTIONS[action]})")

    # Step environment
    observations, rewards, dones, infos = env_instance.step(actions)

    # Print detailed information
    print("\nRewards:")
    for agent in rewards:
        print(f"  {agent}: {rewards[agent]:.3f}")
    
    print("\nCurrent State:")
    for agent in env_instance.agents:
        if agent in infos:
            info = infos[agent]
            if agent.startswith("generator"):
                gen_powers = info['generator_powers']
                gen_profits = info['generator_profits']
                print(f"  {agent}:")
                print(f"    Power: {gen_powers.get(agent, 0):.3f}")
                print(f"    Profit: {gen_profits.get(agent, 0):.3f}")
            else:  # consumer
                con_prices = info['consumer_prices']
                con_profits = info['consumer_profits']
                print(f"  {agent}:")
                print(f"    Price: {con_prices.get(agent, 0):.3f}")
                print(f"    Profit: {con_profits.get(agent, 0):.3f}")
    
    print("\nObservations (first 6 values):")
    for agent, obs in observations.items():
        print(f"  {agent}: {obs[:6]}")  # Show first 6 values to avoid clutter

    # Render the environment
    env_instance.render()

    # Check win conditions
    all_profits = []
    for agent in env_instance.agents:
        if agent in infos:
            if agent.startswith("generator"):
                profit = infos[agent]['generator_profits'].get(agent, 0)
                all_profits.append((agent, profit))
            else:
                profit = infos[agent]['consumer_profits'].get(agent, 0)
                all_profits.append((agent, profit))
    
    # Check if any agent reached profit threshold
    winners = [(agent, profit) for agent, profit in all_profits if profit > env_instance.profit_threshold]
    if winners:
        print(f"\n🏆 WINNERS:")
        for agent, profit in winners:
            print(f"  {agent}: {profit:.3f}")

    # If all agents are done, end the loop
    if all(dones.values()):
        print("\n=== Game Over ===")
        if step_count + 1 >= env_instance.max_steps:
            print("Reason: Maximum steps reached")
        break

    # Uncomment for slower visualization
    # time.sleep(0.5)

env_instance.close()

print("\n=== Energy Market Test Completed ===")


# Additional testing function for specific scenarios
def test_energy_market_scenarios():
    """Test specific scenarios in the energy market"""
    print("\n=== Testing Specific Multi-Agent Scenarios ===")
    
    # Test 1: High generator power, low consumer price
    print("\n--- Test 1: High Power, Low Price Scenario ---")

    env_config = {
        'profit_threshold': 5,
        'num_generators': 2,
        'num_consumers': 2    
    }

    profit_threshold = env_config.get("profit_threshold", 5)
    num_generators = env_config.get("num_generators", 2)
    num_consumers = env_config.get("num_consumers", 2)

    env_test = parallel_env(
        max_gen_power, min_gen_power, 
        max_con_price, min_con_price, 
        profit_threshold, max_steps,
        num_generators, num_consumers
    )
    
    env_test.reset()
    
    # Set specific values for testing
    for i in range(num_generators):
        agent_name = f"generator_{i}"
        env_test._generator_powers[agent_name] = 0.8 + (i * 0.1)  # Different values
    
    for i in range(num_consumers):
        agent_name = f"consumer_{i}"
        env_test._consumer_prices[agent_name] = 0.2 + (i * 0.1)  # Different values
    
    env_test._update_profits()
    
    print("Generator States:")
    for agent_name, power in env_test._generator_powers.items():
        profit = env_test.gen_profits.get(agent_name, 0)
        print(f"  {agent_name}: Power={power:.3f}, Profit={profit:.3f}")
    
    print("Consumer States:")
    for agent_name, price in env_test._consumer_prices.items():
        profit = env_test.con_profits.get(agent_name, 0)
        print(f"  {agent_name}: Price={price:.3f}, Profit={profit:.3f}")
    
    # Test 2: Boundary conditions
    print("\n--- Test 2: Boundary Conditions ---")
    
    # Set max values
    for i in range(num_generators):
        agent_name = f"generator_{i}"
        env_test._generator_powers[agent_name] = env_test.max_gen_power
    
    for i in range(num_consumers):
        agent_name = f"consumer_{i}"
        env_test._consumer_prices[agent_name] = env_test.max_con_price
    
    env_test._update_profits()
    
    print("Generator States (max values):")
    for agent_name, power in env_test._generator_powers.items():
        profit = env_test.gen_profits.get(agent_name, 0)
        print(f"  {agent_name}: Power={power:.3f}, Profit={profit:.3f}")
    
    print("Consumer States (max values):")
    for agent_name, price in env_test._consumer_prices.items():
        profit = env_test.con_profits.get(agent_name, 0)
        print(f"  {agent_name}: Price={price:.3f}, Profit={profit:.3f}")
    
    # Test 3: Different agent configurations
    print("\n--- Test 3: Different Agent Configurations ---")
    
    configs = [
        (1, 1, "Single Generator vs Single Consumer"),
        (3, 1, "Multiple Generators vs Single Consumer"),
        (1, 3, "Single Generator vs Multiple Consumers"),
        (3, 3, "Multiple Generators vs Multiple Consumers")
    ]
    
    for n_gen, n_con, description in configs:
        print(f"\n{description}:")
        test_env = parallel_env(
            max_gen_power, min_gen_power, 
            max_con_price, min_con_price, 
            profit_threshold, max_steps,
            n_gen, n_con
        )
        
        obs = test_env.reset()
        print(f"  Agents: {test_env.agents}")
        print(f"  Observation shapes: {[obs[agent].shape for agent in test_env.agents]}")
        
        test_env.close()
    
    env_test.close()


# Run additional tests
test_energy_market_scenarios()


# Performance test
def performance_test():
    """Test environment performance"""
    print("\n=== Performance Test ===")
    
    env_config = {
        'max_steps': 1000,
        'num_generators': 3,
        'num_consumers': 2
    }

    max_steps = env_config.get("max_steps", 1000)
    num_generators = env_config.get("num_generators", 3)
    num_consumers = env_config.get("num_consumers", 2)

    env_perf = parallel_env(
        max_gen_power, min_gen_power, 
        max_con_price, min_con_price, 
        profit_threshold, max_steps,
        num_generators, num_consumers
    )
    
    start_time = time.time()
    
    total_steps = 0
    num_episodes = 2
    
    for episode in range(num_episodes):
        obs = env_perf.reset()
        done = False
        episode_steps = 0
        
        while not done and episode_steps < 1000:
            actions = {
                agent: env_perf.action_spaces[agent].sample()
                for agent in env_perf.agents
            }
            
            obs, rewards, dones, infos = env_perf.step(actions)
            done = all(dones.values())
            episode_steps += 1
            total_steps += 1

            # Only render occasionally to avoid performance impact
            if episode_steps % 50 == 0:
                env_perf.render()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Completed {num_episodes} episodes in {duration:.2f} seconds")
    print(f"Total steps: {total_steps}")
    print(f"Steps per second: {total_steps/duration:.1f}")
    print(f"Average steps per episode: {total_steps/num_episodes:.1f}")
    print(f"Agents per episode: {len(env_perf.possible_agents)}")
    
    env_perf.close()


# Run performance test
performance_test()


# Observation structure test
def test_observation_structure():
    """Test the observation structure for different agent configurations"""
    print("\n=== Observation Structure Test ===")
    
    env_obs_test = parallel_env(
        max_gen_power, min_gen_power, 
        max_con_price, min_con_price, 
        profit_threshold, max_steps,
        2, 2  # 2 generators, 2 consumers
    )
    
    obs = env_obs_test.reset()
    
    print(f"Total agents: {len(env_obs_test.agents)}")
    print(f"Expected observation length: {len(env_obs_test.possible_agents) * 2}")
    
    for agent_name, observation in obs.items():
        print(f"\n{agent_name} observation:")
        print(f"  Shape: {observation.shape}")
        print(f"  Values: {observation}")
        
        # Decode observation structure
        print(f"  Decoded structure:")
        print(f"    Own state: [{observation[0]:.3f}, {observation[1]:.3f}] (value, profit)")
        
        idx = 2
        for other_agent in env_obs_test.possible_agents:
            if other_agent != agent_name:
                print(f"    {other_agent}: [{observation[idx]:.3f}, {observation[idx+1]:.3f}] (value, profit)")
                idx += 2
    
    env_obs_test.close()


# Run observation structure test
test_observation_structure()