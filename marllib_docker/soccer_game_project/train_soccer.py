from marllib import marl
import soccer_marl_wrapper  # Make sure this registers your env in ENV_REGISTRY

# Step 1: Create your custom environment
env = marl.make_env(environment_name="soccer_marl", map_name="soccer")

# Step 2: Initialize the algorithm and load hyperparameters
mappo = marl.algos.mappo(hyperparam_source="mpe")
# customize model
model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-128"})
# start learning
mappo.fit(env, model, stop={'episode_reward_mean': 2000, 'timesteps_total': 200000}, local_mode=False, num_gpus=1,
          num_workers=1, share_policy='all', checkpoint_freq=500)
