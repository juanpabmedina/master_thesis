# from energy_market_env import env
from energy_market_env import parallel_env
from pettingzoo.test import parallel_api_test
from pettingzoo.test import api_test
import time


# Create environment instances for testing
env_config = {
        # Add your soccer environment specific parameters here
        "max_cycles": 100,             # Episode length
        # Add other parameters your soccer env needs
    }

max_cycles = env_config.get("max_cycles", 100)  # default to 100 steps
max_gen_power = env_config.get("max_gen_power", 1)  # default to 100 steps
min_gen_power = env_config.get("min_gen_power", 0.1)  # default to 100 steps
max_con_price = env_config.get("max_con_price", 1)  # default to 100 steps
min_con_price = env_config.get("min_con_price", 0.1)  # default to 100 steps
profit_threshold = env_config.get("profit_threshold", 500)  # default to 100 steps
max_steps = env_config.get("max_steps", 100)  # default to 100 steps

parallel_energy_env = parallel_env(
        max_gen_power, min_gen_power, 
        max_con_price, min_con_price, 
        profit_threshold, max_steps
        )


# aec_energy_env = env()

# # Run PettingZoo API tests
# print("=== Running AEC API Test ===")
# api_test(aec_energy_env, num_cycles=10, verbose_progress=True)

print("\n=== Running Parallel API Test ===")
parallel_api_test(parallel_energy_env, num_cycles=10)

print("\n=== API Tests Completed Successfully ===")


# Action mappings for readable output
ACTIONS = ['DECREASE', 'STAY', 'INCREASE']

# Initialize environment for manual testing
env_instance = parallel_energy_env
observations = env_instance.reset()

print("\n=== Starting Parallel Energy Market Game ===")
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
            print(f"  {agent}:")
            if agent == "generator":
                print(f"    Power: {info['generator_power']:.3f}")
                print(f"    Profit: {info['generator_profit']:.3f}")
            else:  # consumer
                print(f"    Price: {info['consumer_price']:.3f}")
                print(f"    Profit: {info['consumer_profit']:.3f}")
    
    print("\nObservations:")
    for agent, obs in observations.items():
        print(f"  {agent}: {obs}")

    # Render the environment
    env_instance.render()

    # Check win conditions
    gen_profit = infos.get("generator", {}).get("generator_profit", 0)
    con_profit = infos.get("consumer", {}).get("consumer_profit", 0)
    
    if gen_profit > env_instance.profit_threshold:
        print(f"\n🏆 GENERATOR WINS! Profit: {gen_profit:.3f}")
    elif con_profit > env_instance.profit_threshold:
        print(f"\n🏆 CONSUMER WINS! Profit: {con_profit:.3f}")

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
    print("\n=== Testing Specific Scenarios ===")
    
    # Test 1: High generator power, low consumer price
    print("\n--- Test 1: High Power, Low Price Scenario ---")

    env_config = {
        'profit_threshold': 5,    
    }

    profit_threshold = env_config.get("profit_threshold", 500) 

    env_test = parallel_env(
        max_gen_power, min_gen_power, 
        max_con_price, min_con_price, 
        profit_threshold, max_steps
        )
    
    env_test.reset()
    env_test._generator_power = 1
    env_test._consumer_price = 0.1
    env_test._update_profits()
    
    print(f"Generator Power: {env_test._generator_power}")
    print(f"Consumer Price: {env_test._consumer_price}")
    print(f"Generator Profit: {env_test.gen_profit:.3f}")
    print(f"Consumer Profit: {env_test.con_profit:.3f}")
    
    # Test 2: Low generator power, high consumer price  
    print("\n--- Test 2: Low Power, High Price Scenario ---")
    env_test._generator_power = 0.1
    env_test._consumer_price = 1
    env_test._update_profits()
    
    print(f"Generator Power: {env_test._generator_power}")
    print(f"Consumer Price: {env_test._consumer_price}")
    print(f"Generator Profit: {env_test.gen_profit:.3f}")
    print(f"Consumer Profit: {env_test.con_profit:.3f}")
    
    # Test 3: Boundary conditions
    print("\n--- Test 3: Boundary Conditions ---")
    env_test._generator_power = env_test.max_gen_power
    env_test._consumer_price = env_test.max_con_price
    env_test._update_profits()
    
    print(f"Generator Power (max): {env_test._generator_power}")
    print(f"Consumer Price (max): {env_test._consumer_price}")
    print(f"Generator Profit: {env_test.gen_profit:.3f}")
    print(f"Consumer Profit: {env_test.con_profit:.3f}")
    
    env_test.close()


# Run additional tests
test_energy_market_scenarios()


# Performance test
def performance_test():
    """Test environment performance"""
    print("\n=== Performance Test ===")
    
    env_config = {
        'max_steps': 1000,    
    }

    max_steps = env_config.get("max_steps", 500) 

    env_perf = parallel_env(
        max_gen_power, min_gen_power, 
        max_con_price, min_con_price, 
        profit_threshold, max_steps
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

            env_perf.render()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Completed {num_episodes} episodes in {duration:.2f} seconds")
    print(f"Total steps: {total_steps}")
    print(f"Steps per second: {total_steps/duration:.1f}")
    print(f"Average steps per episode: {total_steps/num_episodes:.1f}")
    
    env_perf.close()


# Run performance test
performance_test()