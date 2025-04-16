import optuna
import yaml
import sys
import os
import logging
import copy
import time
import numpy as np

# Import hàm chạy thử nghiệm từ main.py
try:
    from main import run_experiment  # run_experiment(config, trial) phải trả về metric
except ImportError as e:
    print(f"CRITICAL: Could not import 'run_experiment' from 'main.py'. {e}")
    sys.exit(1)

# Cấu hình logging cho optimize.py
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OptunaOptimizer")

# Đường dẫn đến file config cơ sở
BASE_CONFIG_PATH = "config_optimize.yaml"


# -------------------------
# Hàm merge đệ quy để kết hợp cấu hình
def merge_dicts(base, update):
    for k, v in update.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            base[k] = merge_dicts(base[k], v)
        else:
            base[k] = v
    return base

# -------------------------
# Định nghĩa không gian tìm kiếm tham số chung
def define_common_search_space(trial: optuna.Trial, base_config: dict) -> dict:
    params = {}
    # Gợi ý loại tokenizer
    params["tokenizer"] = trial.suggest_categorical(
        "tokenizer", ["ipa", "bpe", "wordpiece", "unigram", "char", "byte"]
    )
    # Với các tokenizer cần train (bpe, unigram, wordpiece), cố định vocab_size; nếu không dùng giá trị gốc.
    if params["tokenizer"] in ["bpe", "unigram", "wordpiece"]:
         params["vocab_size"] = 30000  # Giữ cố định ban đầu, có thể mở rộng sau
         logger.debug(f"Trial {trial.number}: Using fixed vocab_size={params['vocab_size']} for {params['tokenizer']}")
    else:
         params["vocab_size"] = base_config.get('vocab_size')
    
    params["lr"] = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    params["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 64])
    params["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 0.1, log=True)
    params["max_len"] = trial.suggest_categorical("max_len", [64, 96, 128, 192, 256])
    params["patience"] = trial.suggest_int("patience", 3, 10, step=1)
    params["scheduler_patience"] = trial.suggest_int("scheduler_patience", 1, 5, step=1)
    params["scheduler_factor"] = trial.suggest_categorical("scheduler_factor", [0.1, 0.2, 0.3, 0.5])
    return params

# -------------------------
# Định nghĩa không gian tham số cho các model, với ràng buộc dựa vào tokenizer
def define_model_search_space(trial: optuna.Trial, base_config: dict, common_params: dict) -> dict:
    model_params = {}
    tokenizer_choice = common_params["tokenizer"]
    # Nếu tokenizer không phải "ipa", loại bỏ các model chuyên dụng IPA (_ipa)
    if tokenizer_choice == "ipa":
        all_models = [
            "bilstm", "bigru", "transformer", "textcnn", "ebilstm",
            "bilstm_ipa", "bigru_ipa", "transformer_ipa", "textcnn_ipa"
        ]
    else:
        all_models = [
            "bilstm", "bigru", "transformer", "textcnn", "ebilstm"
        ]
    model_choice = trial.suggest_categorical("model", all_models)
    common_params["model"] = model_choice

    # Tham số dropout chung
    model_params["dropout"] = trial.suggest_float("dropout", 0.05, 0.5)

    if "bilstm" in model_choice or "bigru" in model_choice:
        model_params["embedding_dim"] = trial.suggest_categorical("embedding_dim_rnn", [128, 256, 300, 512])
        model_params["hidden_dim"] = trial.suggest_categorical("hidden_dim_rnn", [128, 256, 512])
        model_params["num_layers"] = trial.suggest_int("num_layers_rnn", 1, 4)
        if model_choice.endswith("_ipa"):
            model_params["combine_mode"] = trial.suggest_categorical("combine_mode_rnn", ["sum", "concat"])
            model_params["custom_pos_vocab_size"] = 8
    elif "transformer" in model_choice:
        nhead = trial.suggest_categorical("nhead", [4, 8])
        d_model_options = [h for h in [128, 256, 512] if h % nhead == 0]
        if not d_model_options:
            d_model_options = [nhead * 32]
        chosen_d_model = trial.suggest_categorical("d_model", d_model_options)
        model_params["d_model"] = chosen_d_model
        model_params["nhead"] = nhead
        model_params["num_layers"] = trial.suggest_int("num_layers_tf", 2, 6)
        # Sử dụng bảng ánh xạ cố định cho dim_feedforward dựa trên d_model
        ff_options_map = {
            128: [256, 512],
            256: [512, 1024],
            512: [1024, 2048],
        }
        possible_ff = ff_options_map.get(chosen_d_model, [chosen_d_model * 2, chosen_d_model * 4])
        model_params["dim_feedforward"] = trial.suggest_categorical("dim_feedforward", possible_ff)
        if model_choice.endswith("_ipa"):
            model_params["combine_mode"] = trial.suggest_categorical("combine_mode_tf", ["sum", "concat"])
            model_params["custom_pos_vocab_size"] = 8
    elif "textcnn" in model_choice:
        model_params["embedding_dim"] = trial.suggest_categorical("embedding_dim_cnn", [128, 256, 300])
        model_params["filter_sizes"] = [3, 4, 5]  # cố định
        model_params["num_filters"] = trial.suggest_categorical("num_filters", [50, 100, 150, 200])
        if model_choice.endswith("_ipa"):
            model_params["combine_mode"] = trial.suggest_categorical("combine_mode_cnn", ["sum", "concat"])
            model_params["custom_pos_vocab_size"] = 8
    elif "ebilstm" in model_choice:
        model_params["embedding_dim"] = trial.suggest_categorical("embedding_dim_ebilstm", [128, 256, 300, 512])
        model_params["hidden_dim"] = trial.suggest_categorical("hidden_dim_ebilstm", [128, 256, 512])
        model_params["num_layers"] = trial.suggest_int("num_layers_ebilstm", 1, 4)
    else:
        logger.warning(f"Model choice {model_choice} not specifically handled; using common parameters only.")

    common_params["model_params"] = model_params
    return common_params

# -------------------------
# Hàm tổng hợp không gian tìm kiếm hyperparameter
def define_search_space(trial: optuna.Trial, base_config: dict) -> dict:
    params = define_common_search_space(trial, base_config)
    params = define_model_search_space(trial, base_config, params)
    return params

# -------------------------
# Hàm merge cấu hình (recursive update)
def recursive_update(base, updates):
    for k, v in updates.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            base[k] = recursive_update(base[k], v)
        else:
            base[k] = v
    return base

# -------------------------
# Hàm mục tiêu của HPO
def objective(trial: optuna.Trial):
    trial_number = trial.number
    logger.info(f"\n===== Starting Optuna Trial {trial_number} =====")
    try:
        with open(BASE_CONFIG_PATH, 'r', encoding='utf-8') as f:
            base_config = yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Trial {trial_number}: Error loading base config: {e}", exc_info=True)
        return -float('inf')
    
    try:
        params_to_try = define_search_space(trial, base_config)
        logger.info(f"Trial {trial_number} Parameters: {params_to_try}")
    except optuna.TrialPruned as e:
        logger.warning(f"Trial {trial_number} pruned during suggestion: {e}")
        raise
    except Exception as e:
        logger.error(f"Trial {trial_number}: Suggestion error: {e}", exc_info=True)
        return -float('inf')
    
    try:
        trial_config = copy.deepcopy(base_config)
        trial_config = recursive_update(trial_config, params_to_try)
        trial_config['run_mode'] = 'train'
        trial_config['seed'] = trial_number  # Sử dụng số trial làm seed
        trial_config['epochs'] = base_config.get('epochs', 40)
        trial_config['resume'] = False

        # Đặt đường dẫn checkpoint và log dựa trên trial
        trial_base_name = f"trial_{trial_number}_{trial_config['task']}_{trial_config['tokenizer']}_{trial_config['model']}"
        trial_config['checkpoint_path'] = os.path.join("checkpoints", f"{trial_base_name}.pth")
        trial_config['log_file'] = os.path.join("logs", f"results_{trial_base_name}.txt")
        logger.info(f"Trial {trial_number}: Running with generated config.")
        logger.debug(f"Trial {trial_number} Config: {trial_config}")
    except Exception as e:
        logger.error(f"Trial {trial_number}: Config creation error: {e}", exc_info=True)
        return -float('inf')
    
    metric_direction = 'maximize' if trial_config['task'] != 'stsb' else 'minimize'
    worst_metric = -float('inf') if metric_direction == 'maximize' else float('inf')
    metric = worst_metric

    try:
        metric = run_experiment(config=trial_config, trial=trial)
        logger.info(f"Trial {trial_number} finished. Reported Metric: {metric:.4f}")
        if metric is None or not isinstance(metric, (int, float)) or np.isnan(metric) or np.isinf(metric):
            logger.warning(f"Trial {trial_number}: Invalid metric ({metric}). Reporting worst metric.")
            metric = worst_metric
    except optuna.TrialPruned:
        logger.info(f"Trial {trial_number} pruned.")
        intermediate = trial.intermediate_values
        metric = intermediate[max(intermediate.keys())] if intermediate else worst_metric
    except Exception as e:
        logger.error(f"Trial {trial_number}: Error during run_experiment: {e}", exc_info=True)
        metric = worst_metric
    
    return float(metric)

# -------------------------
# Chạy Study Optuna
if __name__ == "__main__":
    try:
        with open(BASE_CONFIG_PATH, 'r', encoding='utf-8') as f:
            base_config = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Could not load base config {BASE_CONFIG_PATH}: {e}")
        base_config = {}
    
    current_task = base_config.get('task', 'unknown_task')
    study_name = base_config.get("optuna_study_name", f"HPO_{current_task}")
    n_trials = base_config.get("optuna_n_trials", 100)
    storage_name = base_config.get("optuna_storage", f"sqlite:///HPO_{current_task}_study.db")
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1)

    logger.info(f"--- Starting Optuna Study ---")
    logger.info(f"Study Name: {study_name}")
    logger.info(f"Task: {current_task}")
    logger.info(f"Number of Trials: {n_trials}")
    logger.info(f"Storage: {storage_name}")
    logger.info(f"Pruner: {type(pruner).__name__}")
    logger.info(f"Objective: {'Maximize' if current_task != 'stsb' else 'Minimize'} Validation Metric")

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize" if current_task != 'stsb' else "minimize",
        storage=storage_name,
        load_if_exists=True,
        pruner=pruner
    )
    study.set_user_attr("base_config_file", BASE_CONFIG_PATH)
    study.set_user_attr("base_task", current_task)
    study.set_user_attr("base_epochs", base_config.get('epochs'))

    optimize_start_time = time.time()
    try:
        study.optimize(objective, n_trials=n_trials, timeout=base_config.get("optuna_timeout", None))
    except KeyboardInterrupt:
        logger.warning("Optimization interrupted by user.")
    except Exception as e:
        logger.error(f"An error occurred during optimization: {e}", exc_info=True)
    optimize_duration = time.time() - optimize_start_time
    logger.info(f"Total Optimization Time: {optimize_duration:.2f}s")

    logger.info("\n===== Optimization Finished =====")
    logger.info(f"Number of finished trials: {len(study.trials)}")
    try:
        best_trial = study.best_trial
        logger.info(f"Best trial number: {best_trial.number}")
        logger.info(f"Best trial value (Validation Metric): {best_trial.value:.4f}")
        logger.info("Best trial parameters found:")
        for key, value in best_trial.params.items():
            logger.info(f"  {key}: {value}")

        # Tách model_params từ các tham số phẳng
        best_params = best_trial.params.copy()
        best_model_params = {}
        keys_to_remove = ['tokenizer', 'model', 'lr', 'batch_size', 'weight_decay', 'vocab_size', 'max_len', 'patience', 'scheduler_patience', 'scheduler_factor']
        for k in list(best_params.keys()):
            if k not in keys_to_remove:
                clean_k = k.replace('_rnn', '').replace('_tf', '').replace('_cnn', '')
                best_model_params[clean_k] = best_params.pop(k)
        final_config = copy.deepcopy(base_config)
        final_config.update(best_params)
        final_config['model_params'] = final_config.get('model_params', {})
        final_config['model_params'].update(best_model_params)
        final_config['task'] = current_task
        final_config['run_mode'] = 'train'
        final_config['epochs'] = base_config.get('epochs')
        final_config.setdefault('dataset_path', base_config.get('dataset_path'))
        final_config.setdefault('vocab_file', base_config.get('vocab_file'))

        best_config_file = f"best_config_{study_name}_trial{best_trial.number}.yaml"
        try:
            os.makedirs(os.path.dirname(best_config_file) or ".", exist_ok=True)
            with open(best_config_file, 'w', encoding='utf-8') as f:
                yaml.dump(final_config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            logger.info(f"Best configuration saved to: {best_config_file}")
        except Exception as e:
            logger.error(f"Could not save best configuration: {e}")
    except Exception as e:
        logger.error(f"An error occurred while reporting best trial: {e}", exc_info=True)
