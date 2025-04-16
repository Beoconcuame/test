import torch
import os
import sys
import logging
import time
import datetime
import optuna
import numpy as np

try:
    from dataset_utils import prepare_dataloaders
    from training_utils import setup_training, save_checkpoint, load_checkpoint
    from trainer import train_model_fn
    from evaluate import evaluate_model_fn
except ImportError as e:
    print(f"CRITICAL: Failed to import components into runner.py: {e}. Check file paths and definitions.")
    sys.exit(1)

logger = logging.getLogger(__name__)

def run_training(config, model, tokenizer, device, trial: optuna.Trial = None):
    """
    Coordinates the complete training and evaluation process, optionally integrating with Optuna.

    Loads training and validation dataloaders, sets up training components (optimizer,
    criterion, scaler, scheduler), handles checkpoint resuming, writes training logs,
    runs the training loop with early stopping, saves the best model checkpoint,
    reports intermediate results to Optuna (if a trial object is provided),
    and handles Optuna pruning requests.

    Args:
        config (dict): The configuration dictionary for this run.
        model (torch.nn.Module): The model instance to train.
        tokenizer: The initialized tokenizer instance.
        device (torch.device): The device ('cuda' or 'cpu') to run training on.
        trial (optuna.Trial, optional): An Optuna trial object if running as part of
                                        an HPO study. Used for reporting intermediate
                                        metrics and checking for pruning. Defaults to None.

    Returns:
        float or str: The best validation metric achieved during training. Returns 'N/A'
                      or a worst possible value if training fails or no valid metric is obtained.
    """
    task = config['task']
    epochs = config['epochs']
    checkpoint_path = config['checkpoint_path']
    patience = config['patience']
    resume = config.get('resume', False)
    model_name_cfg = config.get('model', 'unknown_model')
    tokenizer_name_cfg = config.get('tokenizer', 'unknown_tokenizer')
    scheduler_mode = 'max' if task != 'stsb' else 'min'
    worst_metric = -float('inf') if scheduler_mode == 'max' else float('inf')

    try:
        dataloaders, num_classes = prepare_dataloaders(config, tokenizer)
        train_loader = dataloaders.get('train')
        val_loader = dataloaders.get('val')
        if not train_loader or not val_loader:
            raise ValueError("Could not get train or validation dataloaders.")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to prepare dataloaders: {e}", exc_info=True)
        if trial:
            return worst_metric
        else:
            sys.exit(1)

    try:
        optimizer, criterion, scaler, scheduler = setup_training(model, config)
        if scheduler: scheduler_mode = scheduler.mode
    except Exception as e:
        logger.error(f"CRITICAL: Failed to setup training components: {e}", exc_info=True)
        if trial:
            return worst_metric
        else:
            sys.exit(1)

    start_epoch = 0
    best_metric = worst_metric
    patience_counter = 0

    if resume:
        checkpoint_info = load_checkpoint(checkpoint_path, model, optimizer, scaler, scheduler, device)
        start_epoch = checkpoint_info.get('start_epoch', 0)
        loaded_best_metric = checkpoint_info.get('best_metric')
        if loaded_best_metric is not None:
            if (scheduler_mode == 'max' and isinstance(loaded_best_metric, (int, float)) and loaded_best_metric > -float('inf')) or \
               (scheduler_mode == 'min' and isinstance(loaded_best_metric, (int, float)) and loaded_best_metric < float('inf')):
                best_metric = loaded_best_metric
            else:
                logger.warning(f"Loaded best_metric ({loaded_best_metric}) incompatible with mode ('{scheduler_mode}'). Using initial {best_metric}")
        patience_counter = checkpoint_info.get('patience_counter', 0)
    elif os.path.isfile(checkpoint_path) and not trial:
        logger.warning(f"Checkpoint {checkpoint_path} exists but resume=False. OVERWRITING.")

    log_dir = os.path.dirname(config.get("log_file", ""))
    log_filename = config.get("log_file") or os.path.join("logs", f"results_{task}_{tokenizer_name_cfg}_{model_name_cfg}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    try:
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)
        with open(log_filename, "w", encoding="utf-8") as f:
            f.write("--- Training Configuration ---\n")
            for key, value in sorted(config.items()): f.write(f"{key}: {value}\n")
            f.write("\n--- Epoch Results ---\n")
        logger.info(f"Logging results to: {log_filename}")
    except Exception as e:
        logger.error(f"Could not open log file {log_filename} for writing: {e}")

    logger.info(f"Starting training loop: {epochs} epochs, Patience: {patience}, Start Epoch: {start_epoch + 1}")
    training_start_time = time.time()
    completed_epochs = start_epoch - 1
    final_best_metric_epoch = -1 if start_epoch == 0 else start_epoch
    last_val_metric = None

    try:
        for epoch in range(start_epoch, epochs):
            completed_epochs = epoch
            logger.info(f"\n===== Epoch {epoch+1}/{epochs} =====")
            epoch_start_time = time.time()

            train_loss = train_model_fn(model, train_loader, optimizer, criterion, device, scaler, task)
            val_metric = evaluate_model_fn(model, val_loader, device, task)
            epoch_end_time = time.time()

            if val_metric is None or not isinstance(val_metric, (int, float)) or np.isnan(val_metric) or np.isinf(val_metric):
                 logger.error(f"Epoch {epoch+1}: Invalid validation metric received ({val_metric}). Stopping training.")
                 if trial: return worst_metric
                 else: break

            last_val_metric = val_metric
            current_lr = optimizer.param_groups[0]['lr']
            epoch_summary = f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss:.4f}, Val Metric = {val_metric:.4f}, Time = {epoch_end_time - epoch_start_time:.2f}s, LR = {current_lr:.2e}"
            logger.info(epoch_summary)
            try:
                with open(log_filename, "a", encoding="utf-8") as f: f.write(epoch_summary + "\n")
            except Exception as e: logger.warning(f"Could not write epoch results to log file: {e}")

            if scheduler: scheduler.step(val_metric)

            improved = (scheduler_mode == 'max' and val_metric > best_metric) or \
                       (scheduler_mode == 'min' and val_metric < best_metric)

            if improved:
                improvement_delta = abs(val_metric - best_metric)
                logger.info(f"Validation metric improved ({best_metric:.4f} --> {val_metric:.4f}, Delta: {improvement_delta:.4f}). Saving model...")
                best_metric = val_metric
                final_best_metric_epoch = epoch + 1
                patience_counter = 0
                checkpoint_state = {
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'best_metric': best_metric,
                    'task': task, 'model_name': model_name_cfg, 'tokenizer_name': tokenizer_name_cfg,
                    'patience_counter': patience_counter,
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'scaler_state_dict': scaler.state_dict() if scaler and scaler.is_enabled() else None,
                    'config': config
                }
                save_checkpoint(checkpoint_state, checkpoint_path)
            else:
                patience_counter += 1
                logger.info(f"Validation metric did not improve from {best_metric:.4f}. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    logger.info(f"EARLY STOPPING triggered after {patience} epochs without improvement.")
                    break

            if trial:
                trial.report(val_metric, epoch)
                if trial.should_prune():
                    logger.info(f"Optuna Trial {trial.number} pruned at epoch {epoch+1}.")
                    raise optuna.TrialPruned()

    except optuna.TrialPruned:
         logger.warning(f"Trial {trial.number if trial else 'N/A'} was pruned.")
         metric_to_return = best_metric if abs(best_metric)!=float('inf') else last_val_metric if last_val_metric is not None else worst_metric
         return metric_to_return if isinstance(metric_to_return, (int, float)) else worst_metric
    except Exception as e:
        logger.error(f"An error occurred during the training loop (epoch {completed_epochs+1}): {e}", exc_info=True)
        try:
            with open(log_filename, "a", encoding="utf-8") as f:
                f.write(f"\n--- TRAINING LOOP ERROR (Epoch {completed_epochs+1}) ---\n{e}\n"); import traceback; traceback.print_exc(file=f)
        except Exception as log_e: logger.error(f"Could not write training error to log: {log_e}")
        if trial: return worst_metric
        else: raise e

    finally:
        training_end_time = time.time(); total_training_time = training_end_time - training_start_time
        logger.info(f"Training loop finished. Total Training Time: {total_training_time:.2f}s")

        final_best_metric_to_log = best_metric if isinstance(best_metric, (int, float)) and abs(best_metric) != float('inf') else "N/A"
        log_summary = f"\n--- Training Summary ---\n"
        if os.path.exists(checkpoint_path):
            try:
                 checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                 final_best_metric_to_log = checkpoint.get('best_metric', best_metric)
                 final_best_metric_epoch = checkpoint.get('epoch', completed_epochs) + 1
                 log_summary += f"Best Validation Metric ({task}): {final_best_metric_to_log:.4f} (Epoch {final_best_metric_epoch})\n"
                 log_summary += f"Best model checkpoint saved to: {checkpoint_path}\n"
            except Exception as e:
                 logger.error(f"Could not reload final metric from checkpoint: {e}")
                 log_summary += f"Best Validation Metric during training ({task}): {final_best_metric_to_log:.4f}\n"
                 log_summary += f"Attempted to save best model to: {checkpoint_path}\n"
        else:
             logger.warning("No best model checkpoint was found after training.")
             log_summary += "No best model checkpoint was saved.\n"
        log_summary += f"Total Training Time: {total_training_time:.2f}s\n"
        try:
            with open(log_filename, "a", encoding="utf-8") as f: f.write(log_summary)
            logger.info(f"Training summary appended to {log_filename}")
        except Exception as log_e: logger.warning(f"Could not write final summary to log file: {log_e}")

        metric_print_val = f"{final_best_metric_to_log:.4f}" if isinstance(final_best_metric_to_log, (int, float)) else "N/A"
        print(f"FINAL_BEST_VAL_METRIC: {metric_print_val}")

        return final_best_metric_to_log if isinstance(final_best_metric_to_log, (int, float)) else worst_metric

def run_evaluation(config, model, tokenizer, device):
    """
    Coordinates the evaluation process on the validation or test dataset.

    Loads the evaluation DataLoader, restores the model from checkpoint, runs evaluation,
    and prints the final metric along with key configuration details.

    Args:
        config (dict): The evaluation configuration.
        model (torch.nn.Module): The model to evaluate.
        tokenizer: The tokenizer used for data processing.
        device (torch.device): The device to run the evaluation on.

    Returns:
        float: The calculated evaluation metric. Returns 0.0 if evaluation fails.
    """
    task = config['task']
    run_mode = config['run_mode']
    checkpoint_path = config['checkpoint_path']
    model_name_cfg = config.get('model', 'unknown_model')
    tokenizer_name_cfg = config.get('tokenizer', 'unknown_tokenizer')
    metric = 0.0

    try:
        dataloaders, num_classes = prepare_dataloaders(config, tokenizer)
        eval_loader = dataloaders.get('eval')
        eval_split_name = getattr(eval_loader.dataset, 'split', run_mode) if eval_loader and hasattr(eval_loader, 'dataset') else run_mode
        if not eval_loader:
            raise ValueError(f"Could not get evaluation dataloader for specified run_mode '{run_mode}'.")
    except Exception as e:
        logger.error(f"Failed to prepare evaluation dataloader: {e}", exc_info=True)
        sys.exit(1)

    if not os.path.isfile(checkpoint_path):
        logger.error(f"CRITICAL: Checkpoint file '{checkpoint_path}' not found. Cannot run '{run_mode}' without a trained model.")
        sys.exit(1)
    try:
        logger.info(f"Loading model from checkpoint for evaluation: {checkpoint_path}")
        load_info = load_checkpoint(checkpoint_path, model, device=device)
        if not load_info or 'start_epoch' not in load_info :
            raise RuntimeError("Failed to load model state from checkpoint (load_checkpoint indication).")
        logger.info("Model loaded successfully for evaluation.")
    except Exception as e:
        logger.error(f"Error loading model state dict from checkpoint '{checkpoint_path}': {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Evaluating model on '{eval_split_name}' split...")
    metric = evaluate_model_fn(model, eval_loader, device, task)
    logger.info(f"Evaluation complete on '{eval_split_name}'.")

    print("\n" + "="*40)
    print(f"   Evaluation Result ({run_mode} mode)")
    print("="*40)
    print(f" Task:          {task}")
    print(f" Split:         {eval_split_name}")
    print(f" Model:         {model_name_cfg}")
    print(f" Tokenizer:     {tokenizer_name_cfg}")
    print(f" Checkpoint:    {checkpoint_path}")
    print(f" Final Metric = {metric:.4f}")
    print("="*40 + "\n")

    print(f"FINAL_EVAL_METRIC: {metric:.4f}")

    return metric if isinstance(metric, (int, float)) else 0.0