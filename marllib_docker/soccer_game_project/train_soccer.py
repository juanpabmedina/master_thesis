from marllib import marl
import soccer_marl_wrapper  # Make sure this registers your env in ENV_REGISTRY

# Step 1: Create your custom environment
env = marl.make_env(environment_name="soccer_marl", map_name="soccer")

#Step 2: Initialize the algorithm and load hyperparameters
maa2c = marl.algos.maa2c(hyperparam_source="common")

#customize model
model = marl.build_model(env, maa2c, {"core_arch": "mlp", "encode_layer": "128-128"})

#start learning

config = {
    # "algo_args": {
    #         "use_gae": True,
    #         "lambda": 1.0,
    #         "vf_loss_coeff": 1.0,
    #         "batch_episode":  10,
    #         "batch_mode": "truncate_episodes",
    #         "lr": 0.0005,
    #         "entropy_coeff": 0.01
    # },
    
    "env_args": {
        # Add your soccer environment specific parameters here
        "max_cycles": 500,             # Episode length
        # Add other parameters your soccer env needs
    }
}

# # Step 2: Initialize the algorithm and load hyperparameters
# iql = marl.algos.iql(hyperparam_source="common")

# # customize model
# model = marl.build_model(env, iql, {"core_arch": "mlp", "encode_layer": "128-128"})

# config = {
#     "algo_args": {
#         "batch_episode":  32,
#         "lr": 0.0005,
#         "rollout_fragment_length": 1,
#         "buffer_size": 5000,
#         "target_network_update_freq": 100,
#         "final_epsilon": 0.05,
#         "epsilon_timesteps": 50000,
#         "optimizer": "rmsprop", # "adam"
#         "reward_standardize": True
#     },
    
#     "env_args": {
#         # Add your soccer environment specific parameters here
#         "max_cycles": 500,             # Episode length
#         # Add other parameters your soccer env needs
#     }
# }

maa2c.fit(env, 
          model, 
          stop={'episode_reward_mean': 2000, 'timesteps_total': 200000}, 
          local_mode=False, 
          num_gpus=1,
          num_workers=10, 
          share_policy='all', 
          checkpoint_freq=50,
            config=config)



# iql.fit(env, 
#         model, 
#         stop={'episode_reward_mean': 2000, 'timesteps_total': 200000}, 
#         local_mode=False, 
#         num_gpus=1,
#         num_workers=10, 
#         share_policy='all', 
#         checkpoint_freq=50,
#         config=config)
