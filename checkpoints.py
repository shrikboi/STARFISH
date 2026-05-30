from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn


STATE_DICT_KEY = "state_dict"


def torch_load(path: str | Path, map_location: str = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_state_dict(checkpoint) -> OrderedDict[str, torch.Tensor] | None:
    if isinstance(checkpoint, nn.Module):
        return None

    if _looks_like_state_dict(checkpoint):
        return OrderedDict(checkpoint)

    if isinstance(checkpoint, Mapping):
        value = checkpoint.get(STATE_DICT_KEY)
        if _looks_like_state_dict(value):
            return OrderedDict(value)

    raise ValueError(
        "Could not find a model state_dict in checkpoint. Expected a plain "
        f"state_dict or a '{STATE_DICT_KEY}' key."
    )


def load_checkpoint_into_model(
    model: nn.Module,
    path: str | Path,
    *,
    strict: bool = True,
) -> nn.Module:
    checkpoint = torch_load(path, map_location="cpu")
    if isinstance(checkpoint, nn.Module):
        return checkpoint

    state_dict = extract_state_dict(checkpoint)
    assert state_dict is not None

    model.load_state_dict(state_dict, strict=strict)
    return model


def _looks_like_state_dict(obj) -> bool:
    if not isinstance(obj, Mapping) or not obj:
        return False
    return all(isinstance(k, str) for k in obj.keys()) and all(
        torch.is_tensor(v) for v in obj.values()
    )
