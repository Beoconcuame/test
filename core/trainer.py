import time
import inspect
import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)



def _parse_batch(batch, input_key: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return (inputs, aux, labels) from common DataLoader batch structures."""
    if isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            return batch  # type: ignore[return-value]
        if len(batch) == 2:
            return batch[0], None, batch[1]  # type: ignore[return-value]
        raise ValueError(f"Unexpected tuple length {len(batch)}")

    if isinstance(batch, dict):
        inputs = next((batch[k] for k in (input_key, "input_ids", "x", "tokens") if k in batch), None)
        if inputs is None or "labels" not in batch:
            raise ValueError("Missing required inputs or labels in batch dict")
        aux = batch.get("attention_mask") or batch.get("custom_positions")
        return inputs, aux, batch["labels"] 

    raise TypeError(f"Unsupported batch type {type(batch)}")


def train_model_fn(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    task: str,
) -> float:
    """Run one training epoch and return average loss."""
    model.train()
    total_loss, processed = 0.0, 0
    start = time.time()


    try:
        first_param = next(iter(inspect.signature(model.forward).parameters))
    except Exception as e:
        logger.warning("Signature inspection failed: %s", e)
        first_param = "input_ids"

    if not len(train_loader):
        logger.warning("Training DataLoader is empty")
        return 0.0

    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    for idx, batch in enumerate(train_loader, 1):
        batch_t0 = time.time()
        inputs, aux, labels = _parse_batch(batch, first_param)
        if not isinstance(inputs, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise TypeError("Inputs and labels must be tensors")

        inputs, labels = inputs.to(device), labels.to(device)
        kwargs = {first_param: inputs}
        if aux is not None:
            aux = aux.to(device)
            if "attention_mask" in model.forward.__code__.co_varnames:
                kwargs["attention_mask"] = aux
            elif "custom_positions" in model.forward.__code__.co_varnames:
                kwargs["custom_positions"] = aux

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=scaler is not None):
            logits = model(**kwargs)

            if task == "stsb":
                logits = logits.squeeze(-1) if logits.dim() == 2 else logits.squeeze()
                loss = criterion(logits, labels.float())
            else:
                if logits.dim() == 3:
                    logits = logits.squeeze(1) if logits.size(1) == 1 else logits[:, 0, :]
                logits = torch.clamp(logits, -15.0, 15.0)
                loss = criterion(logits, labels.long())


        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        processed += 1
        batch_loss = loss.item()
        total_loss += batch_loss

        if processed % max(1, len(train_loader) // 10) == 0 or processed == len(train_loader):
            logger.info("Batch %d/%d, Loss %.4f, Time %.2fs", processed, len(train_loader), batch_loss, time.time() - batch_t0)

        if not np.isfinite(batch_loss):
            raise ValueError(f"Non‑finite loss detected at batch {idx}: {batch_loss}")

    epoch_loss = total_loss / processed
    logger.info("Epoch finished: avg loss %.4f, duration %.2fs", epoch_loss, time.time() - start)
    return epoch_loss