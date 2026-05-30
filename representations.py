from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import ModelKind


def extract_representations(
    model: nn.Module,
    *,
    kind: ModelKind,
    block_names: list[str],
    images: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if kind == "vit":
        return forward_vit_with_tokens(model, block_names, images)
    return forward_cnn_with_activations(model, block_names, images)


@torch.no_grad()
def build_dense_targets(
    dense_model: nn.Module,
    *,
    kind: ModelKind,
    block_names: list[str],
    calib_batches: list[torch.Tensor],
    device: str,
    store_float16: bool = True,
) -> list[dict[str, torch.Tensor]]:
    dense_model.eval()
    targets: list[dict[str, torch.Tensor]] = []
    dtype = torch.float16 if store_float16 else torch.float32

    for batch in calib_batches:
        images = batch.to(device, non_blocking=True)
        _, reps = extract_representations(
            dense_model,
            kind=kind,
            block_names=block_names,
            images=images,
        )
        targets.append({
            name: tensor.detach().to(dtype=dtype).cpu()
            for name, tensor in reps.items()
        })
    return targets


class LazyDenseTargets:
    """Recompute dense block representations on each access instead of caching."""
    def __init__(
        self,
        dense_model: nn.Module,
        *,
        kind: ModelKind,
        block_names: list[str],
        calib_batches: list[torch.Tensor],
        device: str,
    ) -> None:
        self.dense_model = dense_model
        self.kind = kind
        self.block_names = block_names
        self.calib_batches = calib_batches
        self.device = device

    def __len__(self) -> int:
        return len(self.calib_batches)

    @torch.no_grad()
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        self.dense_model.eval()
        images = self.calib_batches[idx].to(self.device, non_blocking=True)
        _, reps = extract_representations(
            self.dense_model,
            kind=self.kind,
            block_names=self.block_names,
            images=images,
        )
        return {name: tensor.detach() for name, tensor in reps.items()}


def cosine_alignment_loss(
    pruned_reps: dict[str, torch.Tensor],
    dense_reps: dict[str, torch.Tensor],
    block_names: list[str],
) -> torch.Tensor:
    device = next(iter(pruned_reps.values())).device
    total = torch.zeros((), device=device)

    for name in block_names:
        pruned = flatten_for_cosine(pruned_reps[name])
        dense = flatten_for_cosine(dense_reps[name].to(device=device, dtype=pruned.dtype))
        pruned = F.normalize(pruned, p=2, dim=-1)
        dense = F.normalize(dense, p=2, dim=-1)
        layer_loss = (1.0 - (pruned * dense).sum(dim=-1)).mean()
        total = total + layer_loss

    return total / len(block_names)


def forward_vit_with_tokens(
    model: nn.Module,
    block_names: list[str],
    images: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    outputs: dict[str, torch.Tensor] = {}
    handles = []
    module_map = dict(model.named_modules())
    batch_size = images.size(0)

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            outputs[name] = normalize_vit_block_output(output, batch_size=batch_size)
        return hook

    for name in block_names:
        if name not in module_map:
            raise ValueError(f"Block '{name}' was not found in model.named_modules().")
        handles.append(module_map[name].register_forward_hook(make_hook(name)))

    try:
        logits = model(images)
    finally:
        for handle in handles:
            handle.remove()

    missing = [name for name in block_names if name not in outputs]
    if missing:
        raise RuntimeError(f"Failed to collect ViT block outputs: {missing}")
    return logits, outputs


def forward_cnn_with_activations(
    model: nn.Module,
    block_names: list[str],
    images: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    outputs: dict[str, torch.Tensor] = {}
    handles = []
    module_map = dict(model.named_modules())

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            outputs[name] = flatten_activation(output)
        return hook

    for name in block_names:
        if name not in module_map:
            raise ValueError(f"Block '{name}' was not found in model.named_modules().")
        handles.append(module_map[name].register_forward_hook(make_hook(name)))

    try:
        logits = model(images)
    finally:
        for handle in handles:
            handle.remove()

    missing = [name for name in block_names if name not in outputs]
    if missing:
        raise RuntimeError(f"Failed to collect CNN block activations: {missing}")
    return logits, outputs


def normalize_vit_block_output(output, *, batch_size: int) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    if not torch.is_tensor(output):
        raise TypeError(f"Expected tensor block output, got {type(output)}")
    if output.ndim != 3:
        raise ValueError(f"Expected block output with 3 dims, got {tuple(output.shape)}")
    if output.shape[0] != batch_size and output.shape[1] == batch_size:
        output = output.transpose(0, 1)
    if output.shape[0] != batch_size:
        raise ValueError(f"Could not identify batch dimension in {tuple(output.shape)}")
    return output


def flatten_activation(activation: torch.Tensor) -> torch.Tensor:
    if activation.ndim != 4:
        raise ValueError(f"Expected 4D activation map, got {tuple(activation.shape)}")
    return activation.flatten(1)


def flatten_for_cosine(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor
    if tensor.ndim == 3:
        return tensor.reshape(-1, tensor.shape[-1])
    if tensor.ndim == 4:
        return tensor.flatten(1)
    raise ValueError(f"Unsupported representation shape: {tuple(tensor.shape)}")
