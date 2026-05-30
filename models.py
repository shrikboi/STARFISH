from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

from checkpoints import extract_state_dict, load_checkpoint_into_model, torch_load
from constants import (
    MOBILENET_MODEL,
    MOBILENET_PRUNE_SCOPES,
    SUPPORTED_VIT_MODELS,
    VIT_PRUNE_SCOPES,
)


ModelKind = Literal["vit", "mobilenet"]

def infer_model_kind(model_name: str) -> ModelKind:
    if model_name in SUPPORTED_VIT_MODELS:
        return "vit"
    if model_name == MOBILENET_MODEL:
        return "mobilenet"
    raise ValueError(
        f"Unsupported model '{model_name}'. Supported ViTs: "
        f"{', '.join(SUPPORTED_VIT_MODELS)}. MobileNetV1: {MOBILENET_MODEL}."
    )


def build_empty_model(model_name: str) -> nn.Module:
    kind = infer_model_kind(model_name)
    if kind == "mobilenet":
        try:
            import timm
        except ImportError as exc:
            raise ImportError("MobileNetV1 support requires timm: pip install timm") from exc
        return timm.create_model(MOBILENET_MODEL, pretrained=False, num_classes=1000)

    if model_name == "vit_b_16":
        from torchvision.models import vit_b_16
        return vit_b_16(weights=None)
    if model_name == "vit_l_16":
        from torchvision.models import vit_l_16
        return vit_l_16(weights=None)

    try:
        import timm
    except ImportError as exc:
        raise ImportError("DeiT/timm ViT support requires timm: pip install timm") from exc
    return timm.create_model(model_name, pretrained=False, num_classes=1000)


def load_model(model_name: str, checkpoint_path: str | Path, device: str) -> nn.Module:
    model = build_empty_model(model_name)
    try:
        loaded = load_checkpoint_into_model(model, checkpoint_path, strict=True)
    except RuntimeError:
        if infer_model_kind(model_name) != "mobilenet":
            raise
        if not _try_load_mobilenet_str_checkpoint(model, checkpoint_path):
            raise
        loaded = model

    loaded.to(device)
    loaded.eval()
    return loaded


def force_mobilenet_str_runtime(model: nn.Module) -> None:
    """Match the STR MobileNetV1 runtime: PyTorch ReLU activations."""
    for module in model.modules():
        if hasattr(module, "act") and isinstance(module.act, nn.ReLU6):
            module.act = nn.ReLU(inplace=True)


def enable_grad_checkpointing(model: nn.Module) -> bool:
    """Turn on activation (gradient) checkpointing to cut backprop memory.
    timm models (DeiT/ViT and the MobileNetV1 backbone) expose
    ``set_grad_checkpointing``; torchvision ViTs (vit_b_16 / vit_l_16) do not, so
    this is a no-op there. Returns True if checkpointing was enabled.
    """

    setter = getattr(model, "set_grad_checkpointing", None)
    if callable(setter):
        setter(True)
        return True
    return False


def all_block_names(model_name: str, model: nn.Module) -> list[str]:
    kind = infer_model_kind(model_name)
    if kind == "mobilenet":
        names = _mobilenet_block_names(model)
    else:
        names = _vit_block_names(model_name, model)
    if not names:
        raise ValueError(f"Could not find representation blocks for {model_name}.")
    return names


def is_prunable_weight(
    model_name: str,
    prune_scope: str,
):
    kind = infer_model_kind(model_name)
    if kind == "vit":
        if prune_scope not in VIT_PRUNE_SCOPES:
            raise ValueError(f"Invalid ViT prune scope: {prune_scope}")
        return lambda name, module: is_prunable_vit(name, module, prune_scope)
    if prune_scope not in MOBILENET_PRUNE_SCOPES:
        raise ValueError(f"Invalid MobileNetV1 prune scope: {prune_scope}")
    return lambda name, module: is_prunable_mobilenet(name, module, prune_scope)


def default_prune_scope(model_name: str) -> str:
    return "qkv_out_mlp" if infer_model_kind(model_name) == "vit" else "conv_fc"


def is_prunable_vit(name: str, module: nn.Module, prune_scope: str) -> bool:
    is_mha = isinstance(module, nn.MultiheadAttention)
    is_linear = isinstance(module, nn.Linear)

    is_qkv_tv = is_mha and "self_attention" in name
    is_out_tv = is_linear and "self_attention.out_proj" in name
    is_mlp_tv = is_linear and "mlp" in name

    is_qkv_timm = is_linear and ".attn.qkv" in name
    is_out_timm = is_linear and ".attn.proj" in name
    is_mlp_timm = is_linear and (".mlp.fc1" in name or ".mlp.fc2" in name)

    is_qkv = is_qkv_tv or is_qkv_timm
    is_out = is_out_tv or is_out_timm
    is_mlp = is_mlp_tv or is_mlp_timm

    if prune_scope == "qkv":
        return is_qkv
    if prune_scope == "qkv_out":
        return is_qkv or is_out
    if prune_scope == "qkv_out_mlp":
        return is_qkv or is_out or is_mlp
    if prune_scope == "mlp":
        return is_mlp
    if prune_scope == "qk":
        return is_qkv
    if prune_scope == "qk_mlp":
        return is_qkv or is_mlp
    raise ValueError(f"Unknown ViT prune scope: {prune_scope}")


def is_prunable_mobilenet(name: str, module: nn.Module, prune_scope: str) -> bool:
    if prune_scope == "conv_only":
        return isinstance(module, nn.Conv2d) and not _is_depthwise(module)
    if prune_scope == "fc_only":
        return isinstance(module, nn.Linear)
    if prune_scope == "conv_fc":
        return (
            isinstance(module, nn.Conv2d) and not _is_depthwise(module)
        ) or isinstance(module, nn.Linear)
    if prune_scope == "conv_dw":
        return isinstance(module, nn.Conv2d)
    if prune_scope == "all":
        return isinstance(module, (nn.Conv2d, nn.Linear))
    raise ValueError(f"Unknown MobileNetV1 prune scope: {prune_scope}")


def _is_depthwise(module: nn.Conv2d) -> bool:
    return module.groups == module.in_channels


def _vit_block_names(model_name: str, model: nn.Module) -> list[str]:
    modules = dict(model.named_modules())
    names: list[str] = []
    count = _vit_num_blocks(model_name)
    for idx in range(count):
        torchvision_name = f"encoder.layers.encoder_layer_{idx}"
        timm_name = f"blocks.{idx}"
        if torchvision_name in modules:
            names.append(torchvision_name)
        elif timm_name in modules:
            names.append(timm_name)
        else:
            raise ValueError(
                f"Could not find ViT block {idx}. Tried {torchvision_name} and {timm_name}."
            )
    return names


def _vit_num_blocks(model_name: str) -> int:
    if model_name in (
        "vit_b_16",
        "deit_tiny_patch16_224",
        "deit_small_patch16_224",
        "deit_base_patch16_224",
        "deit3_base_patch16_224",
    ):
        return 12
    if model_name in ("vit_l_16", "deit3_large_patch16_224"):
        return 24
    if model_name == "deit3_huge_patch14_224":
        return 32
    raise ValueError(f"Unknown ViT model: {model_name}")


def _mobilenet_block_names(model: nn.Module) -> list[str]:
    modules = dict(model.named_modules())
    expected_layout = [1, 2, 2, 6, 2]
    names: list[str] = []
    missing: list[str] = []
    for stage, count in enumerate(expected_layout):
        for idx in range(count):
            name = f"blocks.{stage}.{idx}"
            if name in modules:
                names.append(name)
            else:
                missing.append(name)
    if missing:
        raise ValueError(f"Could not find MobileNetV1 blocks: {missing}")
    return names


def _try_load_mobilenet_str_checkpoint(model: nn.Module, path: str | Path) -> bool:
    checkpoint = torch_load(path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)
    if state_dict is None:
        return False
    keys = list(state_dict.keys())
    if not any(key.startswith("model.0.") for key in keys):
        return False

    remapped = _remap_str_mobilenet_state_dict(state_dict, model)
    model.load_state_dict(remapped, strict=True)
    force_mobilenet_str_runtime(model)
    return True


def _remap_str_mobilenet_state_dict(
    state_dict: OrderedDict[str, torch.Tensor],
    model: nn.Module,
) -> OrderedDict[str, torch.Tensor]:
    layout = [1, 2, 2, 6, 2]
    flat_to_timm: dict[int, str] = {}
    flat_idx = 1
    for stage, count in enumerate(layout):
        for idx in range(count):
            flat_to_timm[flat_idx] = f"blocks.{stage}.{idx}"
            flat_idx += 1

    def remap_key(key: str) -> str:
        key = key.replace("model.0.0.", "conv_stem.")
        key = key.replace("model.0.1.", "bn1.")
        key = key.replace("fc.weight", "classifier.weight")
        for flat, timm_name in flat_to_timm.items():
            key = key.replace(f"model.{flat}.0.", f"{timm_name}.conv_dw.")
            key = key.replace(f"model.{flat}.1.", f"{timm_name}.bn1.")
            key = key.replace(f"model.{flat}.3.", f"{timm_name}.conv_pw.")
            key = key.replace(f"model.{flat}.4.", f"{timm_name}.bn2.")
        return key

    remapped: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        target_key = remap_key(key)
        if target_key == "classifier.weight" and value.ndim == 4:
            value = value.reshape(value.shape[0], value.shape[1])
        remapped[target_key] = value

    model_state = model.state_dict()
    if "classifier.bias" in model_state and "classifier.bias" not in remapped:
        remapped["classifier.bias"] = torch.zeros_like(model_state["classifier.bias"])

    extra = [key for key in remapped if key not in model_state]
    if extra:
        raise RuntimeError(f"Unexpected STR MobileNet keys after remap: {extra[:5]}")
    return remapped
