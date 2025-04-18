import argparse
import datetime
import sys
import os
import time
import logging
import random

import numpy as np
import torch
import optuna

from core.config import load_config
from core.tokenizer_factory import create_tokenizer
from core.model_factory import get_model
from core.runner import run_training, run_evaluation


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def set_seed(seed_value: int) -> None:
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    logger.info(f"Random seed set to {seed_value}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GLUE task runner")
    parser.add_argument("--config_file", type=str, help="Path to YAML configuration file")
    parser.add_argument("--run_mode", choices=["train", "validation", "test"], help="Run mode override")
    parser.add_argument("--task", type=str, help="GLUE task override")
    parser.add_argument("--model", type=str, help="Model name override")
    parser.add_argument("--tokenizer", type=str, help="Tokenizer name override")
    parser.add_argument("--vocab_file", type=str, help="Vocab file path override")
    parser.add_argument("--max_len", type=int, help="Max sequence length override")
    parser.add_argument("--batch_size", type=int, help="Batch size override")
    parser.add_argument("--epochs", type=int, help="Number of epochs override")
    parser.add_argument("--lr", type=float, help="Learning rate override")
    parser.add_argument("--patience", type=int, help="Early stopping patience override")
    parser.add_argument("--dataset_path", type=str, help="Dataset path override")
    parser.add_argument("--checkpoint_path", type=str, help="Checkpoint path override")
    parser.add_argument("--resume", action="store_true", help="Resume training flag")
    parser.add_argument("--seed", type=int, help="Random seed override")
    return parser.parse_args()


def run_experiment(config: dict, trial: optuna.Trial = None) -> float:
    """Execute training or evaluation based on run_mode."""
    seed_value = config.get("seed", 42)
    set_seed(seed_value)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device'] = device

    run_mode = config['run_mode']
    task = config['task']
    direction = 'maximize' if task != 'stsb' else 'minimize'
    worst_value = -float('inf') if direction == 'maximize' else float('inf')

    logger.info("Starting experiment: mode=%s, task=%s, seed=%d", run_mode, task, seed_value)
    logger.info("Device: %s", device)

    try:
        tokenizer, vocab_size = create_tokenizer(config)
        num_classes = {
            'cola':2,'sst2':2,'mrpc':2,'qqp':2,'mnli':3,'qnli':2,'rte':2,'stsb':1
        }[task]
        model = get_model(config, vocab_size, num_classes, device)
    except Exception as e:
        logger.error("Initialization error: %s", e, exc_info=True)
        return worst_value

    try:
        if run_mode == 'train':
            return run_training(config, model, tokenizer, device, trial)
        elif run_mode in ('validation', 'test'):
            return run_evaluation(config, model, tokenizer, device)
        else:
            logger.error("Invalid run_mode: %s", run_mode)
            return worst_value
    except optuna.TrialPruned as e:
        logger.warning("Optuna trial pruned: %s", e)
        raise
    except Exception as e:
        logger.error("Execution error: %s", e, exc_info=True)
        return worst_value



def main() -> None:
    args = get_args()
    config = load_config(args)
    value = run_experiment(config, trial=None)
    logger.info("Experiment completed with metric: %.4f", value)

if __name__ == '__main__':
    logger.info("Script started")
    start_time = time.time()
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(130)
    except SystemExit as e:
        logger.warning("System exit with code %s", e.code)
        sys.exit(e.code)
    except Exception as e:
        logger.critical("Unhandled error: %s", e, exc_info=True)
        sys.exit(1)
    finally:
        elapsed = time.time() - start_time
        logger.info("Script finished in %.2fs", elapsed)
