from marllib import marl
from utils import clean_folder, rename_and_move_result, copy_config_file
import energy_marl_wrapper2  # Make sure this registers your env in ENV_REGISTRY

EXP_FOLDER_NAME = "maa2c_mlp_energy_market"
CHECK_NAME = "energy_market_case4_ma_2025-06-25_19-44-17"
CHECK_FOL = "checkpoint_000010"
CHECK_NUM = "checkpoint-10"

FOLDER_TO_CLEAN = 'exp_results'
TRAINING_OUTPUT_DIR = 'exp_results/maa2c_mlp_energy_market'  # where results are generated
DESTINATION_ROOT = 'evaluated_policies/maa2c_mlp_energy_market'  # Where experiments are stored
EXP_NAME = 'energy_market_case3'  # Change this to your experiment name
CONFIG_FILE = 'energy.yaml'
CONFIG_DIR = 'config/env_config'

clean_folder(FOLDER_TO_CLEAN)
copy_config_file(CONFIG_FILE, config_dir=CONFIG_DIR)

# Step 1: Create your custom environment
env = marl.make_env(environment_name="energy", map_name="energy_market")

# Step 2: Initialize the algorithm and load hyperparameters
algorithm = marl.algos.maa2c(hyperparam_source="common")
# customize model
model = marl.build_model(env, algorithm, {"core_arch": "mlp", "encode_layer": "128-128"})

# rendering
algorithm.render(env, model,
            restore_path={'params_path': f"/workspace/energy_game_project/trained_policies/{EXP_FOLDER_NAME}/{CHECK_NAME}/params.json",  # experiment configuration
                           'model_path': F"/workspace/energy_game_project/trained_policies/{EXP_FOLDER_NAME}/{CHECK_NAME}/{CHECK_FOL}/{CHECK_NUM}", # checkpoint path
                           'render': True},  # render
            stop={'episode_reward_mean': 2000, 'timesteps_total': 200}, 
            local_mode=True,
            share_policy="all",
            checkpoint_end=False)

rename_and_move_result(TRAINING_OUTPUT_DIR, DESTINATION_ROOT, EXP_NAME)