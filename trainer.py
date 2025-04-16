import torch
import torch.nn as nn
import inspect
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)

def train_model_fn(model, train_loader, optimizer, criterion, device, scaler, task):
    """
    Trains the model for one epoch. Includes NaN/Inf checks, gradient clipping, and logit clipping.
    """
    model.train()
    total_loss = 0
    processed_batches = 0
    epoch_start_time = time.time()

    try:
        sig = inspect.signature(model.forward)
        param_names = list(sig.parameters.keys())
        if not param_names: raise ValueError("Model forward has no parameters!")
        input_arg_name = param_names[0]
    except Exception as e:
        logger.warning(f"[Train] Sig inspection error: {e}. Defaulting to 'input_ids'.")
        input_arg_name = 'input_ids'

    logger.info(f"Starting training epoch for task: {task}...")
    batch_count = len(train_loader)
    if batch_count == 0: logger.warning("Train DataLoader empty."); return 0.0

    for i, batch in enumerate(train_loader):
        batch_start_time = time.time()
        try:
            inputs, aux_input, labels = None, None, None
            if isinstance(batch, (list, tuple)):
                if len(batch)==3: inputs, aux_input, labels = batch
                elif len(batch)==2: inputs, labels = batch
                else: raise ValueError(f"Invalid batch len {len(batch)}")
            elif isinstance(batch, dict):
                keys_to_try = [input_arg_name, 'input_ids', 'x', 'tokens']
                inp_found = False
                for k in keys_to_try:
                     if k in batch: inputs=batch[k]; inp_found=True; break
                if not inp_found: raise ValueError(f"Input key not found in {list(batch.keys())}")
                labels = batch.get('labels')
                aux_input = batch.get('attention_mask', batch.get('custom_positions'))
                if labels is None: raise ValueError("Labels missing")
            else: raise ValueError(f"Invalid batch type {type(batch)}")

            if not isinstance(inputs, torch.Tensor) or not isinstance(labels, torch.Tensor): raise TypeError("Inputs/Labels not Tensors")
            inputs, labels = inputs.to(device), labels.to(device)
            attention_mask, custom_positions = None, None
            if aux_input is not None:
                if isinstance(aux_input, torch.Tensor):
                    aux_input = aux_input.to(device)
                    if "attention_mask" in sig.parameters: attention_mask = aux_input
                    elif "custom_positions" in sig.parameters: custom_positions = aux_input
        except Exception as e:
            logger.error(f"Batch {i} processing error: {e}", exc_info=True); raise e

        try:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16 if device.type=='cuda' else torch.bfloat16, enabled=(scaler is not None)):
                forward_args = {input_arg_name: inputs}
                if attention_mask is not None and "attention_mask" in sig.parameters: forward_args["attention_mask"] = attention_mask
                if custom_positions is not None and "custom_positions" in sig.parameters: forward_args["custom_positions"] = custom_positions
                raw_outputs = model(**forward_args)

                if torch.isnan(raw_outputs).any() or torch.isinf(raw_outputs).any():
                     logger.error(f"NaN or Inf detected in raw model output at batch {i}.")
                     logger.error(f"Input shape: {inputs.shape}, Input sample: {inputs[0, :10]}")
                     raise ValueError("NaN or Inf in model output before loss calculation")

                if task == "stsb":
                    if raw_outputs.shape[-1] == 1: outputs = raw_outputs.squeeze(-1)
                    elif len(raw_outputs.shape) == 2: outputs = raw_outputs[:, 0]
                    else: raise ValueError(f"Output shape {raw_outputs.shape} unusable for STS-B.")
                    loss = criterion(outputs, labels.float())
                else:
                    if len(raw_outputs.shape) == 3:
                         if raw_outputs.shape[1] == 1: outputs = raw_outputs.squeeze(1)
                         else: outputs = raw_outputs[:, 0, :]
                    elif len(raw_outputs.shape) == 2: outputs = raw_outputs
                    else: raise ValueError(f"Unexpected output shape {raw_outputs.shape}.")
                    if outputs.shape[0]!=labels.shape[0] or len(outputs.shape)!=2: raise ValueError(f"Processed shape {outputs.shape} incompatible with labels {labels.shape}.")

                    logit_clip_value = 15.0
                    outputs = torch.clamp(outputs, min=-logit_clip_value, max=logit_clip_value)

                    if i < 3:
                        logger.debug(f"Batch {i} - Logits clamped to [{-logit_clip_value}, {logit_clip_value}]. New min: {torch.min(outputs).item():.4f}, max: {torch.max(outputs).item():.4f}")
                        logger.debug(f"Labels unique in batch: {torch.unique(labels)}")

                    loss = criterion(outputs, labels.long())

                current_loss = loss.item()
                if np.isnan(current_loss) or np.isinf(current_loss):
                     logger.error(f"\nLoss became NaN or Inf AFTER criterion at batch {i}.")
                     logger.error(f"Clamped Outputs (logits) min: {torch.min(outputs).item():.4f}, max: {torch.max(outputs).item():.4f}")
                     logger.error(f"Labels unique: {torch.unique(labels)}")
                     raise ValueError("NaN or Inf loss detected after criterion")

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            processed_batches += 1
            total_loss += current_loss

        except Exception as e:
             logger.error(f"\nError during train step (Batch {i}): {e}", exc_info=True)
             raise e

        batch_end_time = time.time()
        log_interval = max(1, batch_count // 10)
        if (processed_batches % log_interval == 0 or processed_batches == batch_count):
             logger.info(f"  Batch {processed_batches}/{batch_count}, Loss: {current_loss:.4f}, Batch Time: {batch_end_time - batch_start_time:.2f}s")

    avg_epoch_loss = total_loss / processed_batches if processed_batches > 0 else 0
    epoch_duration = time.time() - epoch_start_time
    logger.info(f"Finished training epoch. Average Loss: {avg_epoch_loss:.4f}, Epoch Duration: {epoch_duration:.2f}s")
    return avg_epoch_loss