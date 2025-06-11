from custom_environment import CustomActionMaskedEnvironment

env = CustomActionMaskedEnvironment()
obs, infos = env.reset()
print("Initial observation:", obs)

for i in range(10):
    actions = {
        "prisoner": env.action_space("prisoner").sample(),
        "guard": env.action_space("guard").sample(),
    }
    print(f"\nStep {i+1}, actions: {actions}")
    obs, rewards, terminations, truncations = env.step(actions)
    env.render()
    print("Observations:", obs)
    print("Rewards:", rewards)
    print("Terminations:", terminations)
    print("Truncations:", truncations)
    if not env.agents:
        print("Episode ended.")
        break
