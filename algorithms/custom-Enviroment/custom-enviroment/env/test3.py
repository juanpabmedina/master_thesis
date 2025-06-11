from soccer_environment2 import env
# Example usage and testing

soccer_env = env()

# Test the environment
soccer_env.reset()


# Constants
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']


for step_count in range(10):  # Run for 100 steps or until done
    if soccer_env.dones[soccer_env.agent_selection]:
        break
        
    # Random action for current agent
    action = soccer_env.action_spaces[soccer_env.agent_selection].sample()
    print(f"Agent {soccer_env.agent_selection} takes action {action} ({ACTIONS[action]})")
    soccer_env.step(action)

    
    # Render after both agents have acted (when we're back to the first agent)
    if step_count > 1 and soccer_env.agent_selection == soccer_env.agents[0]:
        obs = soccer_env.observe(soccer_env.agent_selection)
        print(f"Observation: {obs}")    
        soccer_env.render()
    
    # Check if game is done
    if all(soccer_env.dones.values()):
        print("Game finished!")
        print(f"Final scores: {soccer_env.scores}")
        break

soccer_env.close()