from soccer_environment import env
from soccer_environment import parallel_env
from pettingzoo.test import parallel_api_test
from pettingzoo.test import api_test
import time


aec_soccer_env = env()
parallel_soccer_env = parallel_env()

api_test(aec_soccer_env, num_cycles=10, verbose_progress=True)
parallel_api_test(parallel_soccer_env, num_cycles=10)



ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']

# Initialize environment
env = parallel_env()
observations = env.reset()

print("=== Starting Parallel Soccer Game ===")

# Initialize local dones
dones = {agent: False for agent in env.agents}

for step_count in range(100):
    # Prepare actions for non-done agents
    actions = {
        agent: env.action_spaces[agent].sample()
        for agent in env.agents
    }

    print(f"\n--- Step {step_count} ---")
    for agent, action in actions.items():
        print(f"Agent {agent} → Action {action} ({ACTIONS[action]})")

    # Step environment
    observations, rewards, dones, infos = env.step(actions)

    # Print observations and rewards
    for agent in rewards:
        print(f"  Reward for {agent}: {rewards[agent]}")
        print(f"  Observation: {observations[agent]}")

    # Render the environment
    env.render()

    # If all agents are done, end the loop
    if all(dones.values()):
        print("\n=== Game Over ===")
        break

    time.sleep(0.5)

env.close()

