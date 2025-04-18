import yaml 
import os
import sys 
import logging
import datetime

logger = logging.getLogger(__name__)
DEFAULT_CONFIG_FILE = "config.yaml"

def load_config(args):

    cfg_file = args.config_file or DEFAULT_CONFIG_FILE
    if os.path.exists(cfg_file):
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            logger.info("Loaded configuration from %s", cfg_file)
        except yaml.YAMLError as e:
            logger.error("YAML error in %s: %s", cfg_file, e)
            sys.exit(1)
    else:
        logger.warning("Config file %s not found – using CLI/defaults.", cfg_file)
        cfg = {}


    cli = vars(args)
    final = cfg.copy()
    for k, v in cli.items():
        if k == "config_file":
            continue
        if v is not None or k not in final:
            final[k] = v


    defaults = {
        "run_mode": "train",
        "task": "cola",
        "tokenizer": "ipa",
        "model": "bilstm",
        "max_len": 50,
        "batch_size": 32,
        "epochs": 5,
        "lr": 1e-3,
        "patience": 3,
        "resume": False,
        "weight_decay": 0.01,
        "scheduler_patience": 2,
        "scheduler_factor": 0.1,
        "model_params": {},
        "seed": 42,              
    }
    for k, v in defaults.items():
        final.setdefault(k, v)

    os.makedirs("logs", exist_ok=True)

    if not final.get("checkpoint_path"):
        final["checkpoint_path"] = os.path.join(
            "checkpoint",
            f"best_model_{final['task']}_{final['tokenizer']}_{final['model']}.pth"
        )
        logger.info("Using default checkpoint path: %s", final["checkpoint_path"])

    if not final.get("log_file"):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        final["log_file"] = os.path.join(
            "logs",
            f"results_{final['task']}_{final['tokenizer']}_{final['model']}_{ts}.txt"
        )
        logger.info("Using default log file: %s", final["log_file"])

    required = [
        "task", "tokenizer", "model", "max_len", "batch_size",
        "epochs", "lr", "dataset_path", "checkpoint_path"
    ]
    if final["tokenizer"] == "ipa":
        required.append("vocab_file")
    if final["model"].endswith("_ipa"):
        if final.get("model_params", {}).get("custom_pos_vocab_size") is None:
            logger.error("Model '%s' yêu cầu 'custom_pos_vocab_size' trong 'model_params'.", final["model"])
            sys.exit(1)

    missing = [k for k in required if final.get(k) is None]
    if missing:
        logger.error("Thiếu khóa cấu hình bắt buộc: %s", missing)
        sys.exit(1)

    logger.info("--- Final Configuration ---")
    for k, v in sorted(final.items()):
        logger.info("  %s: %s", k, v)
    logger.info("---------------------------")

    return final
