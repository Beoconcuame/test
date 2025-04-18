import time
import inspect
import logging

import torch
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import matthews_corrcoef, accuracy_score

logger = logging.getLogger(__name__)

def evaluate_model_fn(model: torch.nn.Module,
                      data_loader: torch.utils.data.DataLoader,
                      device: torch.device,
                      task: str) -> float:

    model.eval()
    all_preds, all_trues = [], []
    processed = 0
    start = time.time()

    # Determine main input argument name
    try:
        params = inspect.signature(model.forward).parameters
        input_key = next(iter(params))
    except Exception:
        logger.warning("Cannot inspect forward signature; defaulting to 'input_ids'")
        input_key = 'input_ids'

    logger.info(f"Evaluating task={task} on {len(data_loader)} batches...")

    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            try:
                inputs, aux, labels = _parse_batch(batch, input_key)
                inputs, labels = inputs.to(device), labels.to(device)
                kwargs = {input_key: inputs}
                if aux is not None:
                    aux = aux.to(device)
                    if 'attention_mask' in params:
                        kwargs['attention_mask'] = aux
                    elif 'custom_positions' in params:
                        kwargs['custom_positions'] = aux

                outputs = model(**kwargs)
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    raise ValueError("NaN/Inf in model outputs")

                preds = _process_outputs(outputs, task)
                all_trues.extend(labels.cpu().numpy())
                all_preds.extend(preds)
                processed += len(preds)
            except Exception as e:
                logger.warning(f"Batch {idx} skipped: {e}")
                continue

    duration = time.time() - start
    logger.info(f"Processed {processed} samples in {duration:.2f}s")

    return _compute_metric(np.array(all_trues), np.array(all_preds), task)


def _parse_batch(batch, input_key):
    if isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            return batch
        if len(batch) == 2:
            return batch[0], None, batch[1]
        raise ValueError(f"Batch tuple length {len(batch)}")
    if isinstance(batch, dict):
        inputs = batch.get(input_key) or batch.get('input_ids') or batch.get('x') or batch.get('tokens')
        if inputs is None or 'labels' not in batch:
            raise ValueError("Missing inputs or labels in batch dict")
        aux = batch.get('attention_mask') or batch.get('custom_positions')
        return inputs, aux, batch['labels']
    raise ValueError(f"Unsupported batch type {type(batch)}")


def _process_outputs(raw, task):
    if task == 'stsb':
        if raw.dim() == 2:
            raw = raw[:, 0]
        preds = raw.squeeze(-1).cpu().numpy()
    else:
        if raw.dim() == 3 and raw.shape[1] != 1:
            raw = raw[:, 0, :]
        logits = raw.squeeze(1) if raw.dim() == 3 else raw
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
    return preds


def _compute_metric(trues: np.ndarray, preds: np.ndarray, task: str) -> float:
    if trues.size == 0:
        logger.warning("No samples for metric computation")
        return 0.0
    if task == 'stsb':
        trues_f, preds_f = trues.astype(float), preds.astype(float)
        if np.std(trues_f) < 1e-6 or np.std(preds_f) < 1e-6:
            return 0.0
        corr, _ = pearsonr(preds_f, trues_f)
        return float(corr) if not np.isnan(corr) else 0.0
    if task == 'cola':
        if len(np.unique(trues)) < 2 or len(np.unique(preds)) < 2:
            return 0.0
        return matthews_corrcoef(trues, preds)
    return accuracy_score(trues, preds)
