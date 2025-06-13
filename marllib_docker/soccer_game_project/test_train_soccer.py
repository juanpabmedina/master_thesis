from marllib import marl
import soccer_marl_wrapper  # Make sure this registers your env in ENV_REGISTRY

# Step 1: Create your custom environment
env = marl.make_env(environment_name="soccer_marl", map_name="soccer")

# Step 2: Initialize the algorithm and load hyperparameters
mappo = marl.algos.mappo(hyperparam_source="common")
# customize model
model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-128"})

# rendering
mappo.render(env, model,
             restore_path={'params_path': "/workspace/soccer_game_project/exp_results/mappo_mlp_soccer/MAPPOTrainer_soccer_marl_soccer_ad802_00000_0_2025-06-13_01-35-24/params.json",  # experiment configuration
                           'model_path': "/workspace/soccer_game_project/exp_results/mappo_mlp_soccer/MAPPOTrainer_soccer_marl_soccer_ad802_00000_0_2025-06-13_01-35-24/checkpoint_000200/checkpoint-200", # checkpoint path
                           'render': True},  # render
             local_mode=True,
             share_policy="all",
             checkpoint_end=False)