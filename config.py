import yaml
import os
import sys
import logging
import datetime

logger = logging.getLogger(__name__)
DEFAULT_CONFIG_FILE = "config.yaml"

def load_config(args):
    """
    Loads configuration from a YAML file and merges it with command-line arguments.
    Command-line arguments override values specified in the YAML file.

    Args:
        args: The parsed command-line arguments object from argparse.

    Returns:
        dict: The final configuration dictionary after merging and applying defaults.
    """
    config_file = args.config_file if args.config_file else DEFAULT_CONFIG_FILE

    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                if config is None:
                    config = {}
                logger.info(f"Loaded configuration from: {config_file}")
        except yaml.YAMLError as e:
            logger.error(f"Error loading YAML file {config_file}: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"An unexpected error occurred while reading {config_file}: {e}", exc_info=True)
            sys.exit(1)
    else:
        logger.warning(f"Configuration file {config_file} not found. Using command-line args and defaults.")
        config = {}

    cli_args = vars(args)
    final_config = config.copy()

    for key, value in cli_args.items():
        if key == 'config_file':
            continue
        if value is not None or key not in final_config:
            if isinstance(value, bool) and value:
                final_config[key] = True
            elif isinstance(value, bool) and not value and args.resume is False:
                final_config[key] = False
            elif value is not None:
                final_config[key] = value

    final_config.setdefault("run_mode", "train")
    final_config.setdefault("task", "cola")
    final_config.setdefault("tokenizer", "ipa")
    final_config.setdefault("model", "bilstm")
    final_config.setdefault("max_len", 50)
    final_config.setdefault("batch_size", 32)
    final_config.setdefault("epochs", 5)
    final_config.setdefault("lr", 1e-3)
    final_config.setdefault("patience", 3)
    final_config.setdefault("resume", False)
    final_config.setdefault("model_params", {})
    final_config.setdefault("weight_decay", 0.01)
    final_config.setdefault("scheduler_patience", 2)
    final_config.setdefault("scheduler_factor", 0.1)

    os.makedirs("logs", exist_ok=True)

    if 'checkpoint_path' not in final_config or final_config['checkpoint_path'] is None:
         final_config['checkpoint_path'] = os.path.join(
             "checkpoint",
             f"best_model_{final_config['task']}_{final_config['tokenizer']}_{final_config['model']}.pth"
         )
         logger.info(f"Checkpoint path not specified, using default folder 'checkpoint': {final_config['checkpoint_path']}")

    if 'log_file' not in final_config or final_config['log_file'] is None:
         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
         final_config['log_file'] = os.path.join(
             "logs",
             f"results_{final_config['task']}_{final_config['tokenizer']}_{final_config['model']}_{timestamp}.txt"
          )
         logger.info(f"Log file path not specified, using default: {final_config['log_file']}")

    logger.info("--- Final Configuration ---")
    for key, value in sorted(final_config.items()):
         logger.info(f"  {key}: {value}")
    logger.info("--------------------------")

    required_keys = ['task', 'tokenizer', 'model', 'max_len', 'batch_size', 'epochs', 'lr', 'dataset_path', 'checkpoint_path']
    if final_config['tokenizer'] == 'ipa' and 'vocab_file' not in final_config:
         required_keys.append('vocab_file')
    if final_config['model'].endswith('_ipa') and ('model_params' not in final_config or 'custom_pos_vocab_size' not in final_config['model_params']):
         logger.error(f"CRITICAL: Model '{final_config['model']}' requires 'custom_pos_vocab_size' in 'model_params'.")
         sys.exit(1)

    missing_keys = [key for key in required_keys if key not in final_config or final_config[key] is None]
    if missing_keys:
        logger.error(f"CRITICAL: Missing required configuration keys: {missing_keys}")
        sys.exit(1)

    if 'vocab_file' in required_keys and not os.path.exists(final_config['vocab_file']):
         logger.error(f"CRITICAL: Required vocab_file '{final_config['vocab_file']}' not found.")
         sys.exit(1)

    return final_config
