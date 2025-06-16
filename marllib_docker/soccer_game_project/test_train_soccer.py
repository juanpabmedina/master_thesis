from marllib import marl
import soccer_marl_wrapper  # Make sure this registers your env in ENV_REGISTRY

# Step 1: Create your custom environment
env = marl.make_env(environment_name="soccer_marl", map_name="soccer")

# Step 2: Initialize the algorithm and load hyperparameters
maa2c = marl.algos.maa2c(hyperparam_source="common")
# customize model
model = marl.build_model(env, maa2c, {"core_arch": "mlp", "encode_layer": "128-128"})

# rendering
maa2c.render(env, model,
             restore_path={'params_path': "/workspace/soccer_game_project/exp_results/maa2c_mlp_soccer/MAA2CTrainer_soccer_marl_soccer_ae459_00000_0_2025-06-13_17-13-10/params.json",  # experiment configuration
                           'model_path': "/workspace/soccer_game_project/exp_results/maa2c_mlp_soccer/MAA2CTrainer_soccer_marl_soccer_ae459_00000_0_2025-06-13_17-13-10/checkpoint_000008/checkpoint-8", # checkpoint path
                           'render': True},  # render
             local_mode=True,
             share_policy="all",
             checkpoint_end=False)