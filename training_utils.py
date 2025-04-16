import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
import os
import logging

logger = logging.getLogger(__name__)

def setup_training(model, config):
    """
    Sets up the optimizer, loss criterion, gradient scaler (for AMP),
    and learning rate scheduler for training.

    Reads relevant hyperparameters (lr, weight_decay, scheduler_patience,
    scheduler_factor) from the configuration dictionary.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        config (dict): The configuration dictionary containing training parameters.

    Returns:
        tuple: A tuple containing (optimizer, criterion, scaler, scheduler).
    """
    task = config['task']
    lr = config['lr']
    device_type = config.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu")).type

    logger.info("Setting up optimizer, criterion, scaler, and scheduler...")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=config.get('weight_decay', 0.01))
    logger.info(f"Using optimizer: AdamW (lr={lr}, weight_decay={config.get('weight_decay', 0.01)})")

    scaler = GradScaler(enabled=(device_type == 'cuda'))
    scaler = GradScaler(enabled=False) 
    if scaler.is_enabled():
        logger.info("Using Automatic Mixed Precision (AMP) via GradScaler.")

    criterion = nn.MSELoss() if task == "stsb" else nn.CrossEntropyLoss()
    logger.info(f"Using loss function: {type(criterion).__name__}")

    scheduler_mode = 'max' if task != 'stsb' else 'min'
    scheduler_patience_val = config.get('scheduler_patience', 2)
    scheduler_factor_val = config.get('scheduler_factor', 0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_mode,
        patience=scheduler_patience_val,
        factor=scheduler_factor_val,
        verbose=True
    )
    logger.info(f"Using ReduceLROnPlateau scheduler with mode='{scheduler_mode}', patience={scheduler.patience}, factor={scheduler.factor}")

    return optimizer, criterion, scaler, scheduler

def save_checkpoint(state, filepath):
    """
    Saves the checkpoint state dictionary to the specified file path.
    Creates the directory if it doesn't exist. Uses atomic save via temporary file.

    Args:
        state (dict): A dictionary containing the model state, optimizer state,
                      epoch, metric, and other relevant training information.
        filepath (str): The full path to the checkpoint file to be saved.
    """
    try:
        checkpoint_dir = os.path.dirname(filepath)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        tmp_filepath = filepath + ".tmp"
        torch.save(state, tmp_filepath)
        os.replace(tmp_filepath, filepath)

        logger.info(f"Checkpoint saved successfully to {filepath}")
    except Exception as e:
        logger.error(f"Error saving checkpoint to {filepath}: {e}", exc_info=True)
        if 'tmp_filepath' in locals() and os.path.exists(tmp_filepath):
             try:
                 os.remove(tmp_filepath)
                 logger.info(f"Removed temporary checkpoint file: {tmp_filepath}")
             except Exception as rm_e:
                 logger.error(f"Error removing temporary checkpoint file {tmp_filepath}: {rm_e}")

def load_checkpoint(filepath, model, optimizer=None, scaler=None, scheduler=None, device='cpu'):
    """
    Loads training state (model, optimizer, scaler, scheduler, metadata)
    from a checkpoint file. Uses weights_only=False to load all metadata.

    Args:
        filepath (str): The path to the checkpoint file.
        model (torch.nn.Module): The model instance to load state_dict into.
        optimizer (Optimizer, optional): The optimizer instance to load state_dict into. Defaults to None.
        scaler (GradScaler, optional): The AMP scaler instance to load state_dict into. Defaults to None.
        scheduler (Scheduler, optional): The learning rate scheduler instance to load state_dict into. Defaults to None.
        device (str or torch.device): The device ('cuda' or 'cpu') to map the loaded checkpoint to.

    Returns:
        dict: A dictionary containing metadata loaded from the checkpoint, such as:
              {'start_epoch': int, 'best_metric': float, 'patience_counter': int}.
              Returns a dictionary with default values if the checkpoint file is not found
              or if loading fails critically.
    """
    start_epoch = 0
    best_metric = None
    patience_counter = 0
    loaded_info = {'start_epoch': start_epoch, 'best_metric': best_metric, 'patience_counter': patience_counter}

    if not os.path.isfile(filepath):
        logger.warning(f"Checkpoint file not found at {filepath}. Cannot load checkpoint.")
        return loaded_info

    logger.info(f"Attempting to load checkpoint: {filepath}")
    try:
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)

        if 'model_state_dict' in checkpoint:
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Model state loaded successfully.")
            except Exception as model_load_err:
                 logger.error(f"Error loading model_state_dict: {model_load_err}. Model weights might be incompatible.")
        else:
            logger.warning("Checkpoint missing 'model_state_dict' key. Attempting to load entire checkpoint as state_dict.")
            try:
                model.load_state_dict(checkpoint)
                logger.info("Loaded model state dict from entire checkpoint (potentially older format).")
            except Exception as load_err:
                logger.error(f"Failed to load model state from entire checkpoint: {load_err}. Model weights not loaded.")
                raise ValueError("Checkpoint does not contain a valid model state dict.") from load_err

        if optimizer and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("Optimizer state loaded successfully.")
            except Exception as e:
                logger.warning(f"Could not load optimizer state: {e}. Optimizer state not restored.")
        elif optimizer:
            logger.warning("Checkpoint missing 'optimizer_state_dict'. Optimizer state not restored.")

        if scaler and scaler.is_enabled() and 'scaler_state_dict' in checkpoint:
            try:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                logger.info("GradScaler state loaded successfully.")
            except Exception as e:
                logger.warning(f"Could not load GradScaler state: {e}. GradScaler state not restored.")
        elif scaler and scaler.is_enabled():
            logger.warning("Checkpoint missing 'scaler_state_dict'. GradScaler state not restored.")

        if scheduler and 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info("Scheduler state loaded successfully.")
            except Exception as e:
                 logger.warning(f"Could not load scheduler state: {e}. Scheduler state not restored.")
        elif scheduler:
            logger.warning("Checkpoint missing 'scheduler_state_dict'. Scheduler state not restored.")

        start_epoch = checkpoint.get('epoch', -1) + 1
        best_metric = checkpoint.get('best_metric')
        patience_counter = checkpoint.get('patience_counter', 0)

        loaded_info['start_epoch'] = start_epoch
        loaded_info['best_metric'] = best_metric
        loaded_info['patience_counter'] = patience_counter
        loaded_info['task'] = checkpoint.get('task')
        loaded_info['model_name'] = checkpoint.get('model_name')

        logger.info(f"Checkpoint loaded. Resuming from epoch {start_epoch}. Best metric so far: {best_metric}. Patience: {patience_counter}")

    except FileNotFoundError:
         logger.error(f"Checkpoint file {filepath} not found during load attempt.")
    except Exception as e:
         logger.error(f"Error loading checkpoint {filepath}: {e}. Training will likely start from scratch.", exc_info=True)
         loaded_info = {'start_epoch': 0, 'best_metric': None, 'patience_counter': 0}

    return loaded_info