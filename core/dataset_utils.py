import os
import sys
import logging
from torch.utils.data import DataLoader
from core.dataset import GLUEDataset

logger = logging.getLogger(__name__)

def prepare_dataloaders(config, tokenizer):

    task = config['task']
    run_mode = config['run_mode']
    dataset_path = config['dataset_path']
    max_len = config['max_len']
    batch_size = config['batch_size']
    device_type = "cuda" if config.get("device", "cpu") == "cuda" else "cpu"

    num_classes_map = {"cola": 2, "sst2": 2, "mrpc": 2, "qqp": 2, "mnli": 3, "qnli": 2, "rte": 2, "stsb": 1}
    if task not in num_classes_map:
        logger.error(f"Invalid task '{task}'. Supported tasks are: {list(num_classes_map.keys())}")
        sys.exit(1)
    num_classes = num_classes_map[task]
    logger.info(f"Task: {task}, Number of classes: {num_classes}")

    num_workers = 0
    pin_memory = (device_type == 'cuda')
    logger.info(f"DataLoader settings: num_workers={num_workers}, pin_memory={pin_memory}")

    dataloaders = {}

    try:
        if run_mode == 'train':
            logger.info("Loading training dataset...")
            dataset_train = GLUEDataset(split="train", tokenizer=tokenizer, max_len=max_len, task=task, dataset_path=dataset_path)

            val_split_name = 'validation_matched' if task == 'mnli' and config.get('mnli_eval_split', 'matched') == 'matched' else \
                             'validation_mismatched' if task == 'mnli' and config.get('mnli_eval_split') == 'mismatched' else \
                             'validation'
            logger.info(f"Loading validation dataset (split: {val_split_name})...")
            dataset_val = GLUEDataset(split=val_split_name, tokenizer=tokenizer, max_len=max_len, task=task, dataset_path=dataset_path)

            if len(dataset_train) == 0 or len(dataset_val) == 0:
                logger.error(f"Training ({len(dataset_train)}) or validation ({len(dataset_val)}) dataset is empty. Check dataset path ('{dataset_path}') and task '{task}'.")
                sys.exit(1)

            dataloaders['train'] = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
            dataloaders['val'] = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
            logger.info(f"Train samples: {len(dataset_train)}, Validation samples ({val_split_name}): {len(dataset_val)}")
            logger.info(f"Train batches: {len(dataloaders['train'])}, Validation batches: {len(dataloaders['val'])}")

        else:
            eval_split = run_mode
            if task == 'mnli':
                split_suffix = config.get('mnli_eval_split', 'matched')
                eval_split_name = f"{eval_split}_{split_suffix}"
            else:
                eval_split_name = eval_split
            logger.info(f"Loading evaluation split: '{eval_split_name}'")

            dataset_eval = GLUEDataset(split=eval_split_name, tokenizer=tokenizer, max_len=max_len, task=task, dataset_path=dataset_path)
            if len(dataset_eval) == 0:
                logger.error(f"Evaluation dataset '{eval_split_name}' is empty. Check dataset path ('{dataset_path}') and task '{task}'.")
                sys.exit(1)

            dataloaders['eval'] = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
            logger.info(f"Evaluation samples ({eval_split_name}): {len(dataset_eval)}")
            logger.info(f"Evaluation batches: {len(dataloaders['eval'])}")

    except FileNotFoundError:
       logger.error(f"Dataset not found at path: {dataset_path}. Check 'dataset_path' config.", exc_info=True)
       sys.exit(1)
    except ValueError as ve:
       split_name_for_error = eval_split_name if 'eval_split_name' in locals() else run_mode
       logger.error(f"Invalid split name '{split_name_for_error}' for task '{task}'? Error: {ve}", exc_info=True)
       sys.exit(1)
    except Exception as e:
       split_name_for_error = eval_split_name if 'eval_split_name' in locals() else val_split_name if 'val_split_name' in locals() else run_mode
       logger.error(f"Error creating dataset/dataloader for split '{split_name_for_error}': {e}", exc_info=True)
       sys.exit(1)

    return dataloaders, num_classes
