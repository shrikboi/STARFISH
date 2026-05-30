from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def evaluate_top1(model: nn.Module, loader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        valid = labels >= 0
        if not valid.any():
            continue
        images = images[valid].to(device, non_blocking=True)
        labels = labels[valid].to(device, non_blocking=True)
        logits = model(images)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.numel()

    if total == 0:
        return float("nan")
    return 100.0 * correct / total
