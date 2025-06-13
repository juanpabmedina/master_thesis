from marllib import marl
import soccer_marl_wrapper  # Make sure this registers your env in ENV_REGISTRY

# Step 1: Create your custom environment
env = marl.make_env(environment_name="soccer_marl", map_name="soccer")

# Step 2: Initialize the algorithm and load hyperparameters
mappo = marl.algos.mappo(hyperparam_source="common")
# customize model
model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-128"})
# start learning

config = {
    "algo_args": {
        "use_gae": True,
        "lambda": 0.95,                # Slightly reduced for soccer stability
        "kl_coeff": 0.2,
        "batch_episode": 64,           # Reduced from 128 for soccer training
        "num_sgd_iter": 10,
        "vf_loss_coeff": 1.0,
        "lr": 0.0005,
        "entropy_coeff": 0.02,         # Increased for better exploration
        "clip_param": 0.3,
        "vf_clip_param": 20.0,
        "batch_mode": "complete_episodes"
    },
    
    "env_args": {
        # Add your soccer environment specific parameters here
        "max_cycles": 500,             # Episode length
        # Add other parameters your soccer env needs
    }
}

mappo.fit(env, 
          model, 
          stop={'episode_reward_mean': 2000, 'timesteps_total': 200000}, 
          local_mode=False, 
          num_gpus=1,
          num_workers=10, 
          share_policy='all', 
          checkpoint_freq=50,
            config=config)
