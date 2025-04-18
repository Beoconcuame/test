import os
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler

logger = logging.getLogger(__name__)


def setup_training(model: nn.Module, cfg: Dict[str, Any]) -> Tuple[optim.Optimizer, nn.Module, GradScaler, optim.lr_scheduler.ReduceLROnPlateau]:
    """Return optimizer, criterion, scaler, scheduler based on *cfg*."""
    lr = cfg["lr"]
    device_type = (
        cfg.get("device") if isinstance(cfg.get("device"), torch.device) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ).type

    logger.info("Initialising training utilities (device=%s, lr=%.2e)", device_type, lr)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.get("weight_decay", 0.01))
    scaler = GradScaler(device_type, enabled=(device_type == "cuda"))

    task = cfg["task"]
    criterion = nn.MSELoss() if task == "stsb" else nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max" if task != "stsb" else "min",
        factor=cfg.get("scheduler_factor", 0.1),
        patience=cfg.get("scheduler_patience", 2),
    )

    logger.info(
        "Optimizer=AdamW, Criterion=%s, AMP=%s, Scheduler=ReduceLROnPlateau(patience=%d, factor=%.2f)",
        type(criterion).__name__,
        scaler.is_enabled(),
        scheduler.patience,
        scheduler.factor,
    )

    return optimizer, criterion, scaler, scheduler


def _atomic_save(obj: Dict[str, Any], path: str) -> None:
    tmp_path = f"{path}.tmp"
    torch.save(obj, tmp_path)
    Path(tmp_path).replace(path)


def save_checkpoint(state: Dict[str, Any], filepath: str) -> None:
    """Atomically save *state* dict to *filepath*."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    try:
        _atomic_save(state, filepath)
        logger.info("Checkpoint saved → %s", filepath)
    except Exception as e:
        logger.error("Failed to save checkpoint: %s", e, exc_info=True)
        # Clean tmp file if present
        tmp = f"{filepath}.tmp"
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
    scaler: GradScaler | None = None,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau | None = None,
    device: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """Load checkpoint and restore states; returns meta‑info dict."""
    info = {"start_epoch": 0, "best_metric": None, "patience_counter": 0}

    if not Path(filepath).is_file():
        logger.warning("Checkpoint not found: %s", filepath)
        return info

    ckpt = torch.load(filepath, map_location=device)

    # ---- model ----
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except KeyError:
        model.load_state_dict(ckpt)  # fallback older format
    except Exception as e:
        logger.error("Model state load error: %s", e)

    # ---- optimizer / scaler / scheduler (if provided) ----
    if optimizer and "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as e:
            logger.warning("Optimizer state load failed: %s", e)
    if scaler and scaler.is_enabled() and "scaler_state_dict" in ckpt:
        try:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        except Exception as e:
            logger.warning("Scaler state load failed: %s", e)
    if scheduler and "scheduler_state_dict" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception as e:
            logger.warning("Scheduler state load failed: %s", e)

    info.update(
        start_epoch=ckpt.get("epoch", -1) + 1,
        best_metric=ckpt.get("best_metric"),
        patience_counter=ckpt.get("patience_counter", 0),
        task=ckpt.get("task"),
        model_name=ckpt.get("model_name"),
    )
    logger.info("Checkpoint loaded, resume at epoch %d (best_metric=%s)", info["start_epoch"], info["best_metric"])
    return info
