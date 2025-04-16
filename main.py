import argparse
import datetime
import sys
import torch
import logging
import os
import time
import random
import numpy as np
import optuna 

try:
    from config import load_config
    from tokenizer_factory import create_tokenizer
    from model_factory import get_model
    from runner import run_training, run_evaluation
except ImportError as e:
    print(f"CRITICAL: Failed to import necessary top-level modules: {e}. Check file paths.")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed_value=42):
    """
    Sets the seed for random, numpy, and torch to ensure reproducibility.

    Args:
        seed_value (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        logger.debug("CUDA seed set.")
    logger.info(f"Set random seed to {seed_value}")

def get_args():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="GLUE tasks runner (Refactored v2)")
    parser.add_argument("--config_file", type=str, default="config.yaml", help="Path to YAML configuration file")
    parser.add_argument("--run_mode", type=str, choices=["train", "validation", "test"], default=None, help="Override run mode")
    parser.add_argument("--task", type=str, default=None, help="Override GLUE task")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--tokenizer", type=str, default=None, help="Override tokenizer name")
    parser.add_argument("--vocab_file", type=str, default=None, help="Override path to vocab file")
    parser.add_argument("--max_len", type=int, default=None, help="Override max sequence length")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--patience", type=int, default=None, help="Override early stopping patience")
    parser.add_argument("--dataset_path", type=str, default=None, help="Override path to dataset")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Override path for checkpoints")
    parser.add_argument("--resume", action="store_true", default=None, help="Resume training from checkpoint")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    return parser.parse_args()

def run_experiment(config: dict, trial: optuna.Trial = None):
    """
    Runs a full training or evaluation experiment based on the provided configuration.
    Designed to be callable from both main() and optimize.py (for HPO).

    Args:
        config (dict): The merged configuration dictionary for this run.
        trial (optuna.Trial, optional): Optuna trial object if running HPO,
                                        used for reporting and pruning checks. Defaults to None.

    Returns:
        float: The final metric result (best validation metric or evaluation metric).
               Returns a worst possible value (-inf or +inf) if errors occur.
    """
    run_mode = config['run_mode']
    task = config['task']
    metric_direction = 'maximize' if task != 'stsb' else 'minimize'
    worst_metric = -float('inf') if metric_direction == 'maximize' else float('inf')
    seed_value = config.get("seed", 42)
    set_seed(seed_value)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device'] = device
    logger.info(f"--- Starting Experiment ---")
    logger.info(f"Run Mode: {run_mode}, Task: {task}, Seed: {seed_value}")
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        try: 
            logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        except Exception as e: 
            logger.warning(f"Could not get CUDA device details: {e}")
    logger.info(f"Model: {config.get('model')}, Tokenizer: {config.get('tokenizer')}, LR: {config.get('lr')}, Batch Size: {config.get('batch_size')}")
    try:
        logger.info("Creating tokenizer...")
        tokenizer, vocab_size = create_tokenizer(config)
        num_classes_map = {"cola": 2, "sst2": 2, "mrpc": 2, "qqp": 2, "mnli": 3, "qnli": 2, "rte": 2, "stsb": 1}
        if task not in num_classes_map: 
            raise ValueError(f"Invalid task '{task}'.")
        num_classes = num_classes_map[task]
        logger.info("Creating model...")
        model = get_model(config, vocab_size, num_classes, device)
    except Exception as e:
        logger.error(f"Failed during tokenizer or model creation: {e}", exc_info=True)
        return worst_metric
    final_metric = None
    try:
        if run_mode == 'train':
            logger.info("Starting training process via runner...")
            final_metric = run_training(config, model, tokenizer, device, trial)
        elif run_mode in ['validation', 'test']:
            logger.info("Starting evaluation process via runner...")
            final_metric = run_evaluation(config, model, tokenizer, device)
        else:
            logger.error(f"Invalid run_mode specified: {run_mode}. Choose 'train', 'validation', or 'test'.")
            final_metric = worst_metric
    except optuna.TrialPruned as e:
         logger.warning(f"Trial pruned by Optuna: {e}")
         raise
    except Exception as e:
        logger.error(f"An error occurred during run execution (mode: {run_mode}): {e}", exc_info=True)
        final_metric = worst_metric
    if final_metric is None or not isinstance(final_metric, (int, float)) or np.isnan(final_metric) or np.isinf(final_metric):
         logger.warning(f"Run finished but final metric is invalid ({final_metric}). Returning worst possible value.")
         final_metric = worst_metric
    logger.info(f"--- Experiment Finished ---")
    return float(final_metric)

def main():
    """
    Main execution function called when script is run directly.
    Parses arguments, loads configuration, and runs a single experiment.
    """
    args = get_args()
    try:
        config = load_config(args)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}", exc_info=True)
        sys.exit(1)
    run_experiment(config, trial=None)

if __name__ == "__main__":
    logger.info("Script execution started.")
    script_start_time = time.time()
    try:
        main()
    except SystemExit as se:
         logger.warning(f"Script exited with code: {se.code}")
    except KeyboardInterrupt:
         logger.warning("Script execution interrupted by user (KeyboardInterrupt).")
         sys.exit(130)
    except Exception as e:
        logger.critical(f"An unhandled critical error occurred in main execution: {e}", exc_info=True)
        sys.exit(1)
    finally:
         script_end_time = time.time()
         total_script_time = script_end_time - script_start_time
         logger.info(f"Script execution finished. Total Time: {total_script_time:.2f}s")
