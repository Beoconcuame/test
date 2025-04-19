import copy
import logging
import os
import time
from typing import Dict, Any

import optuna
import yaml
import numpy as np

from main import run_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("OptunaOptimizer")

BASE_CONFIG_PATH = "config_optimize.yaml"
FIXED_VOCAB_SIZE = 30_000
TOKENIZER_CHOICES = ["ipa", "bpe", "wordpiece", "unigram", "char", "byte"]
STANDARD_MODELS = ["bilstm", "bigru", "transformer", "textcnn", "ebilstm"]
IPA_MODELS = [f"{m}_ipa" for m in ["bilstm", "bigru", "transformer", "textcnn"]]
BATCH_SIZE_CHOICES = [16, 32, 64]
MAX_LEN_CHOICES = [64, 96, 128, 192, 256]
SCHEDULER_FACTORS = [0.1, 0.2, 0.3, 0.5]


def _recursive_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _recursive_update(base[k], v)
        else:
            base[k] = v
    return base


def _common_search_space(trial: optuna.Trial, base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    tok = trial.suggest_categorical("tokenizer", TOKENIZER_CHOICES)
    return {
        "tokenizer": tok,
        "vocab_size": FIXED_VOCAB_SIZE if tok in {"bpe", "unigram", "wordpiece"} else base_cfg.get("vocab_size"),
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", BATCH_SIZE_CHOICES),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
        "max_len": trial.suggest_categorical("max_len", MAX_LEN_CHOICES),
        "patience": trial.suggest_int("patience", 3, 10),
        "scheduler_patience": trial.suggest_int("scheduler_patience", 1, 5),
        "scheduler_factor": trial.suggest_categorical("scheduler_factor", SCHEDULER_FACTORS),
    }


def _model_search_space(trial: optuna.Trial, common: Dict[str, Any]) -> Dict[str, Any]:
    tok = common["tokenizer"]
    models = IPA_MODELS if tok == "ipa" else STANDARD_MODELS
    model_name = trial.suggest_categorical("model", models)
    p: Dict[str, Any] = {"dropout": trial.suggest_float("dropout", 0.05, 0.5)}

    if model_name.endswith("_ipa"):
        p["combine_mode"] = trial.suggest_categorical("combine_mode", ["sum", "concat"])
        p["custom_pos_vocab_size"] = 8

    if model_name in {"bilstm", "bilstm_ipa", "bigru", "bigru_ipa"}:
        p.update(
            embedding_dim=trial.suggest_categorical("embedding_dim_rnn", [128, 256, 300, 512]),
            hidden_dim=trial.suggest_categorical("hidden_dim_rnn", [128, 256, 512]),
            num_layers=trial.suggest_int("num_layers_rnn", 1, 4),
        )
    elif "transformer" in model_name:
        nhead = trial.suggest_categorical("nhead", [4, 8])
        d_model = trial.suggest_categorical("d_model", [h for h in [128, 256, 512] if h % nhead == 0])
        ff_map = {128: [256, 512], 256: [512, 1024], 512: [1024, 2048]}
        dim_ff = trial.suggest_categorical(f"dim_feedforward_{d_model}", ff_map[d_model])
        layers = trial.suggest_int("num_layers_tf", 2, 6)
        p.update(
            {
                "d_model": d_model,
                "nhead": nhead,
                "num_layers": layers,
                "dim_feedforward": dim_ff,
            }
        )
    elif "textcnn" in model_name:
        p.update(
            embedding_dim=trial.suggest_categorical("embedding_dim_cnn", [128, 256, 300]),
            filter_sizes=[3, 4, 5],
            num_filters=trial.suggest_categorical("num_filters", [50, 100, 150, 200]),
        )
    elif model_name == "ebilstm":
        p.update(
            embedding_dim=trial.suggest_categorical("embedding_dim_ebilstm", [128, 256, 300, 512]),
            hidden_dim=trial.suggest_categorical("hidden_dim_ebilstm", [128, 256, 512]),
            num_layers=trial.suggest_int("num_layers_ebilstm", 1, 4),
        )
    else:
        logger.warning("Model %s is not supported", model_name)

    common.update({"model": model_name, "model_params": p})
    return common


def build_search_space(trial: optuna.Trial, base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    params = _common_search_space(trial, base_cfg)
    return _model_search_space(trial, params)


def _objective(trial: optuna.Trial) -> float:
    idx = trial.number
    logger.info("Trial %d start", idx)
    base_cfg = yaml.safe_load(open(BASE_CONFIG_PATH, encoding="utf-8")) or {}
    params = build_search_space(trial, base_cfg)
    cfg = _recursive_update(copy.deepcopy(base_cfg), params)
    cfg.update({"run_mode": "train", "seed": idx, "epochs": base_cfg.get("epochs", 40), "resume": False})
    base_name = f"trial_{idx}_{cfg.get('task','task')}_{cfg['tokenizer']}_{cfg['model']}"
    cfg["checkpoint_path"] = os.path.join("checkpoints", f"{base_name}.pth")
    cfg["log_file"] = os.path.join("logs", f"results_{base_name}.txt")

    metric = run_experiment(config=cfg, trial=trial) or 0.0
if metric == 0.0:
    logger.warning(f"Metric is zero for trial {idx}, trying additional hyperparameter sets")
    fallback_configs = [
        {"lr": 1e-4, "batch_size": 32, "dropout": 0.1},
        {"lr": 5e-5, "batch_size": 64, "dropout": 0.2},
        {"lr": 1e-3, "batch_size": 16, "dropout": 0.3},
    ]
    for fb in fallback_configs:
        fb_params = params.copy()
        fb_params.update(fb)
        cfg_fb = _recursive_update(copy.deepcopy(base_cfg), fb_params)
        cfg_fb.update({"run_mode": "train", "seed": idx, "epochs": base_cfg.get("epochs", 40), "resume": False})
        cfg_fb["checkpoint_path"] = cfg["checkpoint_path"]
        cfg_fb["log_file"] = cfg["log_file"].replace('.txt', f'_fb_{fb["lr"]}.txt')
        metric = run_experiment(config=cfg_fb, trial=trial) or 0.0
        if metric > 0.0:
            logger.info(f"Fallback succeeded for trial {idx} with config {fb}")
            break
return float(metric)


if __name__ == "__main__":
    base_cfg = yaml.safe_load(open(BASE_CONFIG_PATH, encoding="utf-8")) or {}
    study = optuna.create_study(direction="maximize")
    study.optimize(_objective, n_trials=50)
    logger.info("Best trial %d: %s", study.best_trial.number, study.best_trial.params)

    best_cfg = copy.deepcopy(base_cfg)
    best_cfg = _recursive_update(best_cfg, study.best_trial.params)
    out_yaml = "best_config.yaml"
    os.makedirs(os.path.dirname(out_yaml) or ".", exist_ok=True)
    with open(out_yaml, "w", encoding="utf-8") as fh:
        yaml.dump(best_cfg, fh, allow_unicode=True, sort_keys=False)
    logger.info("Best config saved to %s", out_yaml)
