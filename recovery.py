from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from masks import (
    apply_masks,
    freeze_all,
    get_weight_key_and_tensor,
    register_zero_freeze_hooks,
)
from models import ModelKind
from representations import (
    LazyDenseTargets,
    build_dense_targets,
    cosine_alignment_loss,
    extract_representations,
)


@dataclass
class RecoveryConfig:
    epochs: int
    lr: float
    min_lr: float
    lr_schedule: str
    weight_decay: float
    grad_clip: float
    store_dense_float16: bool
    no_dense_cache: bool = False


def starfish_recovery(
    *,
    pruned_model: nn.Module,
    dense_model: nn.Module,
    kind: ModelKind,
    block_names: list[str],
    masks: dict[str, torch.Tensor],
    calib_batches: list[torch.Tensor],
    device: str,
    is_prunable_weight,
    config: RecoveryConfig,
) -> list[dict[str, float]]:
    if config.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if not calib_batches:
        raise ValueError("No calibration batches were provided")

    if config.no_dense_cache:
        print("[Dense] No-cache mode: recomputing dense representations each step.")
        dense_targets = LazyDenseTargets(
            dense_model,
            kind=kind,
            block_names=block_names,
            calib_batches=calib_batches,
            device=device,
        )
    else:
        print("[Dense] Caching dense model representations.")
        dense_targets = build_dense_targets(
            dense_model,
            kind=kind,
            block_names=block_names,
            calib_batches=calib_batches,
            device=device,
            store_float16=config.store_dense_float16,
        )
        print(f"[Dense] Cached {len(dense_targets)} dense model batches.")

    apply_masks(pruned_model, masks)
    optimizer = _build_optimizer(
        pruned_model,
        lr=config.lr,
        weight_decay=config.weight_decay,
        is_prunable_weight=is_prunable_weight,
    )

    total_steps = config.epochs * len(calib_batches)
    scheduler = _build_scheduler(
        optimizer,
        schedule=config.lr_schedule,
        total_steps=total_steps,
        min_lr=config.min_lr,
    )

    handles = register_zero_freeze_hooks(pruned_model, masks)
    history: list[dict[str, float]] = []

    pruned_model.train()
    dense_model.eval()

    try:
        for epoch in range(config.epochs):
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(calib_batches):
                images = batch.to(device, non_blocking=True)
                dense_reps = dense_targets[batch_idx]

                optimizer.zero_grad(set_to_none=True)
                _, pruned_reps = extract_representations(
                    pruned_model,
                    kind=kind,
                    block_names=block_names,
                    images=images,
                )
                loss = cosine_alignment_loss(pruned_reps, dense_reps, block_names)
                loss.backward()

                grad_norm = _clip_or_measure_grad_norm(optimizer, config.grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                apply_masks(pruned_model, masks)

                loss_value = float(loss.detach().item())
                epoch_loss += loss_value

                if batch_idx == 0 or batch_idx == len(calib_batches) - 1:
                    print(
                        f"[Recover] epoch {epoch + 1}/{config.epochs} "
                        f"batch {batch_idx + 1}/{len(calib_batches)} "
                        f"loss={loss_value:.6f} grad_norm={float(grad_norm):.4f} "
                        f"lr={optimizer.param_groups[0]['lr']:.2e}"
                    )

            mean_loss = epoch_loss / len(calib_batches)
            history.append({"epoch": float(epoch + 1), "loss": mean_loss})
            print(f"[Recover] epoch {epoch + 1}/{config.epochs} mean_loss={mean_loss:.6f}")

    finally:
        for handle in handles:
            handle.remove()

    pruned_model.eval()
    return history


def _build_optimizer(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
    is_prunable_weight,
) -> torch.optim.Optimizer:
    freeze_all(model)
    params: list[nn.Parameter] = []

    for name, module in model.named_modules():
        if is_prunable_weight(name, module):
            _weight_key, weight = get_weight_key_and_tensor(name, module)
            if weight is not None:
                weight.requires_grad = True
                params.append(weight)

    if not params:
        raise ValueError("No trainable parameters were selected.")

    if weight_decay > 0:
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, foreach=False)
    return torch.optim.Adam(params, lr=lr, foreach=False)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    schedule: str,
    total_steps: int,
    min_lr: float,
):
    if schedule == "constant":
        return None
    if schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps),
            eta_min=min_lr,
        )
    raise ValueError(f"Unknown LR schedule: {schedule}")


def _clip_or_measure_grad_norm(optimizer: torch.optim.Optimizer, grad_clip: float):
    params = [
        param
        for group in optimizer.param_groups
        for param in group["params"]
        if param.grad is not None
    ]
    if not params:
        return torch.tensor(0.0)
    max_norm = grad_clip if grad_clip > 0 else float("inf")
    return torch.nn.utils.clip_grad_norm_(params, max_norm)
