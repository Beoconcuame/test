import copy
import logging
import os
import sys
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
BATCH_SIZE_CHOICES = [16, 32, 64]
MAX_LEN_CHOICES = [64, 96, 128, 192, 256]
SCHEDULER_FACTORS = [0.1, 0.2, 0.3, 0.5]

STANDARD_MODELS = ["bilstm", "bigru", "transformer", "textcnn", "ebilstm"]
IPA_MODELS = [f"{m}_ipa" for m in ["bilstm", "bigru", "transformer", "textcnn"]]



def _recursive_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _recursive_update(base[k], v)  
        else:
            base[k] = v
    return base

merge_dicts = _recursive_update  


def _common_search_space(trial: optuna.Trial, base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    tok = trial.suggest_categorical("tokenizer", TOKENIZER_CHOICES)
    params = {
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
    return params


def _model_search_space(trial: optuna.Trial, common: Dict[str, Any]) -> Dict[str, Any]:
    tok = common["tokenizer"]
    model_choices = STANDARD_MODELS + (IPA_MODELS if tok == "ipa" else [])
    model_name = trial.suggest_categorical("model", model_choices)
    common["model"] = model_name

    p = {"dropout": trial.suggest_float("dropout", 0.05, 0.5)}

    def _combine_if_ipa():
        if model_name.endswith("_ipa"):
            p["combine_mode"] = trial.suggest_categorical("combine_mode", ["sum", "concat"])
            p["custom_pos_vocab_size"] = 8

    if any(m in model_name for m in ("bilstm", "bigru")):
        p.update(
            embedding_dim=trial.suggest_categorical("embedding_dim_rnn", [128, 256, 300, 512]),
            hidden_dim=trial.suggest_categorical("hidden_dim_rnn", [128, 256, 512]),
            num_layers=trial.suggest_int("num_layers_rnn", 1, 4),
        )
        _combine_if_ipa()
    elif "transformer" in model_name:
        nhead = trial.suggest_categorical("nhead", [4, 8])
        d_model = trial.suggest_categorical("d_model", [h for h in [128, 256, 512] if h % nhead == 0])
        ff_map = {128: [256, 512], 256: [512, 1024], 512: [1024, 2048]}
        p.update(
            d_model=d_model,
            nhead=nhead,
            num_layers=trial.suggest_int("num_layers_tf", 2, 6),
            dim_feedforward=trial.suggest_categorical("dim_feedforward", ff_map[d_model]),
        )
        _combine_if_ipa()
    elif "textcnn" in model_name:
        p.update(
            embedding_dim=trial.suggest_categorical("embedding_dim_cnn", [128, 256, 300]),
            filter_sizes=[3, 4, 5],
            num_filters=trial.suggest_categorical("num_filters", [50, 100, 150, 200]),
        )
        _combine_if_ipa()
    elif "ebilstm" in model_name:
        p.update(
            embedding_dim=trial.suggest_categorical("embedding_dim_ebilstm", [128, 256, 300, 512]),
            hidden_dim=trial.suggest_categorical("hidden_dim_ebilstm", [128, 256, 512]),
            num_layers=trial.suggest_int("num_layers_ebilstm", 1, 4),
        )
    else:
        logger.warning("Model choice %s not explicitly handled", model_name)

    common["model_params"] = p
    return common


def build_search_space(trial: optuna.Trial, base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    params = _common_search_space(trial, base_cfg)
    return _model_search_space(trial, params)



def _objective(trial: optuna.Trial) -> float:  # noqa: C901
    idx = trial.number
    logger.info("===== Optuna Trial %d =====", idx)

    try:
        with open(BASE_CONFIG_PATH, "r", encoding="utf-8") as fh:
            base_cfg = yaml.safe_load(fh) or {}
    except Exception as e:
        logger.error("Trial %d: Cannot load base config: %s", idx, e)
        return -float("inf")
    try:
        params = build_search_space(trial, base_cfg)
        logger.info("Trial %d params: %s", idx, params)
    except optuna.TrialPruned:
        raise
    except Exception as e:
        logger.error("Trial %d: Param suggestion failed: %s", idx, e, exc_info=True)
        return -float("inf")

    try:
        cfg = _recursive_update(copy.deepcopy(base_cfg), params)
        cfg.update(
            run_mode="train",
            seed=idx,
            epochs=base_cfg.get("epochs", 40),
            resume=False,
        )
        base_name = f"trial_{idx}_{cfg['task']}_{cfg['tokenizer']}_{cfg['model']}"
        cfg["checkpoint_path"] = os.path.join("checkpoints", f"{base_name}.pth")
        cfg["log_file"] = os.path.join("logs", f"results_{base_name}.txt")
    except Exception as e:
        logger.error("Trial %d: Config creation error: %s", idx, e, exc_info=True)
        return -float("inf")

    maximize = cfg["task"] != "stsb"
    worst = -float("inf") if maximize else float("inf")

    try:
        metric = run_experiment(config=cfg, trial=trial)
        logger.info("Trial %d metric: %.4f", idx, metric)
        if metric is None or not np.isfinite(metric):
            metric = worst
    except optuna.TrialPruned:
        logger.info("Trial %d pruned.", idx)
        hist = trial.intermediate_values
        metric = hist[max(hist)] if hist else worst
    except Exception as e:
        logger.error("Trial %d: run_experiment error: %s", idx, e, exc_info=True)
        metric = worst

    return float(metric)


if __name__ == "__main__":
    try:
        with open(BASE_CONFIG_PATH, "r", encoding="utf-8") as fh:
            base_cfg = yaml.safe_load(fh) or {}
    except Exception as e:
        logger.warning("Cannot load %s: %s", BASE_CONFIG_PATH, e)
        base_cfg = {}

    task = base_cfg.get("task", "unknown_task")
    study_name = base_cfg.get("optuna_study_name", f"HPO_{task}")
    n_trials = base_cfg.get("optuna_n_trials", 100)
    storage = base_cfg.get("optuna_storage", f"sqlite:///HPO_{task}.db")
    timeout = base_cfg.get("optuna_timeout")

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1)
    logger.info("Starting Optuna study '%s' for task '%s' (%d trials)", study_name, task, n_trials)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize" if task != "stsb" else "minimize",
        storage=storage,
        load_if_exists=True,
        pruner=pruner,
    )
    study.set_user_attr("base_config_file", BASE_CONFIG_PATH)

    start = time.time()
    try:
        study.optimize(_objective, n_trials=n_trials, timeout=timeout)
    except KeyboardInterrupt:
        logger.warning("Optimization interrupted by user.")
    except Exception as e:
        logger.error("Optimization error: %s", e, exc_info=True)
    logger.info("Optimization finished in %.2fs", time.time() - start)

    if study.best_trial:
        bt = study.best_trial
        logger.info("Best trial #%d metric=%.4f", bt.number, bt.value)
        logger.info("Best params: %s", bt.params)

        best_cfg = _recursive_update(copy.deepcopy(base_cfg), bt.params)
        model_params = {}
        flat = {}
        to_skip = {"tokenizer", "model", "lr", "batch_size", "weight_decay", "vocab_size", "max_len", "patience", "scheduler_patience", "scheduler_factor"}
        for k, v in bt.params.items():
            if k in to_skip:
                flat[k] = v
            else:
                model_params[k.replace("_rnn", "").replace("_tf", "").replace("_cnn", "")] = v
        best_cfg.update(flat)
        best_cfg.setdefault("model_params", {}).update(model_params)

        out_yaml = f"best_config_{study_name}_trial{bt.number}.yaml"
        try:
            with open(out_yaml, "w", encoding="utf-8") as fh:
                yaml.dump(best_cfg, fh, allow_unicode=True, sort_keys=False)
            logger.info("Best configuration saved to %s", out_yaml)
        except Exception as e:
            logger.error("Cannot save best config: %s", e)