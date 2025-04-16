import torch
import inspect
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import matthews_corrcoef, accuracy_score
import time
import logging

logger = logging.getLogger(__name__)

def evaluate_model_fn(model, data_loader, device, task):
    """
    Evaluate the model on a validation or test dataset.

    This function performs the following steps:
      1. Sets the model to evaluation mode and disables gradient computation.
      2. Automatically determines the main input parameter name for the model’s forward method.
      3. Iterates over the data loader, processing each batch:
         - Extracts inputs, auxiliary inputs, and labels from the batch in a flexible manner.
         - Moves tensors to the specified device.
         - Performs a forward pass through the model with the proper input arguments.
         - Checks for NaN/Inf values in the raw outputs and skips problematic batches.
         - Processes the raw outputs to obtain predictions (handles both regression and classification tasks).
         - Collects all predictions and true labels from valid batches.
      4. Once all batches are processed, computes an appropriate metric:
         - For "stsb" the Pearson correlation is computed.
         - For "cola" the Matthews correlation coefficient is computed.
         - For all other tasks accuracy is used.
      5. Returns the computed metric score.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (iterable): DataLoader that yields batches of data.
        device (torch.device): The device on which to perform evaluation (e.g. 'cpu', 'cuda').
        task (str): The task name (e.g. "stsb", "cola", etc.) which influences prediction processing and metric.

    Returns:
        float: The computed evaluation metric score (or 0.0 if evaluation fails or there are no valid samples).
    """
    model.eval()
    all_preds = []
    all_trues = []
    total_processed_samples = 0
    eval_start_time = time.time()

    try:
        sig = inspect.signature(model.forward)
        param_names = list(sig.parameters.keys())
        if not param_names:
            raise ValueError("Model's forward method has no parameters!")
        input_arg_name = param_names[0]
    except Exception as e:
        logger.warning(f"[Eval] Error inspecting model signature: {e}. Defaulting to 'input_ids'.")
        input_arg_name = 'input_ids'

    logger.info(f"Starting evaluation for task: {task}...")
    batch_count = len(data_loader)
    if batch_count == 0:
        logger.warning("Evaluation DataLoader is empty.")
        return 0.0

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            try:
                inputs, aux_input, labels = None, None, None
                if isinstance(batch, (list, tuple)):
                    if len(batch) == 3:
                        inputs, aux_input, labels = batch
                    elif len(batch) == 2:
                        inputs, labels = batch
                    else:
                        raise ValueError(f"Expected tuple/list len 2/3, got {len(batch)}")
                elif isinstance(batch, dict):
                    input_keys_to_try = [input_arg_name, 'input_ids', 'x', 'tokens']
                    input_found = False
                    for key in input_keys_to_try:
                        if key in batch:
                            inputs = batch[key]
                            input_found = True
                            break
                    if not input_found:
                        raise ValueError(f"Could not find main input in batch dict: {list(batch.keys())}")
                    labels = batch.get('labels')
                    aux_input = batch.get('attention_mask', batch.get('custom_positions'))
                    if labels is None:
                        raise ValueError(f"Missing 'labels' key in batch dict: {list(batch.keys())}")
                else:
                    raise ValueError(f"Unexpected batch format: {type(batch)}")

                if not isinstance(inputs, torch.Tensor) or not isinstance(labels, torch.Tensor):
                    raise TypeError("Inputs and Labels must be torch.Tensors.")

                inputs = inputs.to(device)
                labels = labels.to(device)
                attention_mask = None
                custom_positions = None
                if aux_input is not None:
                    if isinstance(aux_input, torch.Tensor):
                        aux_input = aux_input.to(device)
                        if "attention_mask" in sig.parameters:
                            attention_mask = aux_input
                        elif "custom_positions" in sig.parameters:
                            custom_positions = aux_input
            except Exception as e:
                logger.warning(f"\nSkipping eval batch {i} due to processing error: {e}")
                continue

            try:
                forward_args = {input_arg_name: inputs}
                if attention_mask is not None and "attention_mask" in sig.parameters:
                    forward_args["attention_mask"] = attention_mask
                if custom_positions is not None and "custom_positions" in sig.parameters:
                    forward_args["custom_positions"] = custom_positions
                raw_outputs = model(**forward_args)
                if torch.isnan(raw_outputs).any() or torch.isinf(raw_outputs).any():
                    logger.warning(f"Eval Batch {i}: NaN or Inf detected in raw model output. Skipping batch.")
                    continue
            except Exception as e:
                logger.warning(f"\nSkipping eval batch {i} due to model forward pass error: {e}")
                continue

            try:
                true_labels_batch = labels.cpu().numpy()
                if task == "stsb":
                    if raw_outputs.shape[-1] == 1:
                        outputs = raw_outputs.squeeze(-1)
                    elif len(raw_outputs.shape) == 2:
                        outputs = raw_outputs[:, 0]
                    else:
                        logger.warning(f"Eval Batch {i}: Unexpected output shape {raw_outputs.shape} for STS-B. Skipping.")
                        continue
                    predictions_batch = outputs.cpu().numpy()
                else:
                    if len(raw_outputs.shape) == 3:
                        if raw_outputs.shape[1] == 1:
                            outputs = raw_outputs.squeeze(1)
                        else:
                            outputs = raw_outputs[:, 0, :]
                    elif len(raw_outputs.shape) == 2:
                        outputs = raw_outputs
                    else:
                        logger.warning(f"Eval Batch {i}: Unexpected output shape {raw_outputs.shape} from model for classification. Skipping.")
                        continue
                    if len(outputs.shape) != 2 or outputs.shape[0] == 0:
                        logger.warning(f"Eval Batch {i}: Unexpected processed output shape {outputs.shape} for argmax. Skipping.")
                        continue
                    predictions = torch.argmax(outputs, dim=-1)
                    predictions_batch = predictions.cpu().numpy()
                all_trues.extend(true_labels_batch)
                all_preds.extend(predictions_batch)
                total_processed_samples += len(true_labels_batch)
            except Exception as e:
                logger.warning(f"\nSkipping eval batch {i} due to prediction processing error: {e}", exc_info=True)
                continue

    eval_duration = time.time() - eval_start_time
    logger.info(f"Evaluation finished ({total_processed_samples} samples processed). Calculating metrics... Eval Duration: {eval_duration:.2f}s")

    if total_processed_samples == 0 or not all_trues or not all_preds:
        logger.warning("No valid predictions or true labels were collected during evaluation. Cannot calculate metrics.")
        return 0.0

    all_trues = np.array(all_trues)
    all_preds = np.array(all_preds)
    if len(all_trues) != len(all_preds):
        logger.error(f"CRITICAL: Length mismatch after collecting results ({len(all_trues)} trues, {len(all_preds)} preds). This indicates a bug in batch skipping logic.")
        return 0.0

    logger.info(f"Calculating metrics for {len(all_trues)} pairs.")
    try:
        unique_trues, counts_trues = np.unique(all_trues, return_counts=True)
        unique_preds, counts_preds = np.unique(all_preds, return_counts=True)
        logger.info(f"Final True Labels Unique (counts): {dict(zip(unique_trues, counts_trues))}")
        logger.info(f"Final Predictions Unique (counts): {dict(zip(unique_preds, counts_preds))}")
    except Exception as e:
        logger.warning(f"Could not log unique labels/predictions: {e}")

    try:
        score = 0.0
        metric_name = "N/A"
        if task == "stsb":
            all_trues = all_trues.astype(float)
            all_preds = all_preds.astype(float)
            nan_preds = np.isnan(all_preds).sum()
            inf_preds = np.isinf(all_preds).sum()
            if nan_preds > 0 or inf_preds > 0:
                logger.warning(f"Found {nan_preds} NaNs and {inf_preds} Infs in predictions. Replacing with 0.")
                all_preds = np.nan_to_num(all_preds, nan=0.0, posinf=0.0, neginf=0.0)
            if np.std(all_trues) < 1e-6 or np.std(all_preds) < 1e-6:
                logger.warning("Std deviation near zero. Pearson correlation undefined.")
                score = 0.0
            else:
                pearson_r, _ = pearsonr(all_preds, all_trues)
                if np.isnan(pearson_r):
                    score = 0.0
                    logger.warning("Pearson correlation result is NaN.")
                else:
                    score = pearson_r
            metric_name = "Pearson Correlation"
        elif task == "cola":
            unique_true_labels = np.unique(all_trues)
            unique_pred_labels = np.unique(all_preds)
            if len(unique_true_labels) <= 1 or len(unique_pred_labels) <= 1:
                logger.warning(f"Cannot calculate Matthews correlation: Predictions are constant (Unique: {unique_pred_labels}) or True labels are constant (Unique: {unique_true_labels}). Returning 0.")
                score = 0.0
            else:
                score = matthews_corrcoef(all_trues, all_preds)
            metric_name = "Matthews Correlation"
        else:
            score = accuracy_score(all_trues, all_preds)
            metric_name = "Accuracy"

        logger.info(f"Evaluation Metric ({task} - {metric_name}): {score:.4f}")
        return score

    except Exception as e:
        logger.error(f"\nError calculating metrics for task {task}: {e}", exc_info=True)
        logger.error(f"Number of true labels array: {len(all_trues)}")
        logger.error(f"Number of predictions array: {len(all_preds)}")
        return 0.0