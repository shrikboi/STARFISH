from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from checkpoints import torch_load


ModulePredicate = Callable[[str, nn.Module], bool]


def get_weight_key_and_tensor(name: str, module: nn.Module):
    if isinstance(module, nn.MultiheadAttention):
        return f"{name}.in_proj_weight", module.in_proj_weight
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        return f"{name}.weight", module.weight
    return None, None


def freeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def load_masks(
    path: str | Path,
    model: nn.Module,
    is_trainable_weight: ModulePredicate,
) -> dict[str, torch.Tensor]:
    raw_masks = torch_load(path, map_location="cpu")
    if not _looks_like_mask_mapping(raw_masks):
        raise ValueError(
            "Mask checkpoint must be a plain dict of {weight_name: mask_tensor}."
        )

    expected = _expected_mask_shapes(model, is_trainable_weight)

    masks: dict[str, torch.Tensor] = {}
    unexpected_keys: list[str] = []
    for raw_key, raw_mask in raw_masks.items():
        if raw_key not in expected:
            unexpected_keys.append(raw_key)
            continue
        expected_shape = expected[raw_key]
        if tuple(raw_mask.shape) != expected_shape:
            raise ValueError(
                f"Mask shape mismatch for '{raw_key}': expected {expected_shape}, "
                f"got {tuple(raw_mask.shape)}."
            )
        masks[raw_key] = _normalize_mask_tensor(raw_mask, raw_key)

    if not masks:
        expected_sample = ", ".join(list(expected)[:5])
        unexpected_sample = ", ".join(unexpected_keys[:5])
        raise ValueError(
            "No mask tensors matched the selected prune scope. "
            f"Expected keys like: {expected_sample}. "
            f"Unmatched mask keys included: {unexpected_sample}."
        )

    missing = sorted(set(expected) - set(masks))
    if missing:
        sample = ", ".join(missing[:5])
        raise ValueError(
            "Mask checkpoint is missing masks for prunable tensors in the selected "
            f"scope. Missing examples: {sample}."
        )
    return masks


def apply_masks(model: nn.Module, masks: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, module in model.named_modules():
            weight_key, weight = get_weight_key_and_tensor(name, module)
            if weight_key in masks:
                weight.mul_(masks[weight_key].to(device=weight.device, dtype=weight.dtype))


def register_zero_freeze_hooks(
    model: nn.Module,
    masks: dict[str, torch.Tensor],
) -> list[torch.utils.hooks.RemovableHandle]:
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for name, module in model.named_modules():
        weight_key, weight = get_weight_key_and_tensor(name, module)
        if weight_key not in masks:
            continue
        mask = masks[weight_key].to(device=weight.device, dtype=weight.dtype)

        def make_hook(local_mask):
            def hook(grad):
                return grad * local_mask
            return hook

        handles.append(weight.register_hook(make_hook(mask)))
    return handles


def mask_density(masks: dict[str, torch.Tensor]) -> float:
    total = sum(mask.numel() for mask in masks.values())
    alive = sum(int(mask.sum().item()) for mask in masks.values())
    return alive / max(1, total)


def model_nonzero_fraction(model: nn.Module) -> float:
    total = 0
    nonzero = 0
    for param in model.parameters():
        total += param.numel()
        nonzero += int(torch.count_nonzero(param.detach()).item())
    return nonzero / max(1, total)


def _looks_like_mask_mapping(value) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(isinstance(key, str) for key in value.keys())
        and all(torch.is_tensor(tensor) for tensor in value.values())
    )


def _expected_mask_shapes(
    model: nn.Module,
    is_trainable_weight: ModulePredicate,
) -> dict[str, tuple[int, ...]]:
    expected: dict[str, tuple[int, ...]] = {}
    for name, module in model.named_modules():
        if not is_trainable_weight(name, module):
            continue
        weight_key, weight = get_weight_key_and_tensor(name, module)
        if weight_key is not None and weight is not None:
            expected[weight_key] = tuple(weight.shape)
    if not expected:
        raise ValueError("No prunable weight tensors were found for the selected scope.")
    return expected


def _normalize_mask_tensor(mask: torch.Tensor, key: str) -> torch.Tensor:
    mask = mask.detach().to(device="cpu")
    if mask.dtype == torch.bool:
        return mask.to(dtype=torch.float32)

    finite = True
    if mask.is_floating_point() or mask.is_complex():
        finite = bool(torch.isfinite(mask).all().item())
    binary = bool(((mask == 0) | (mask == 1)).all().item())
    if not finite or not binary:
        raise ValueError(
            f"Mask '{key}' must contain only 0/1 or boolean values; "
            "1 keeps weights trainable and 0 zeros/freezes them."
        )
    return mask.to(dtype=torch.float32)
