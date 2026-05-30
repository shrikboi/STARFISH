from __future__ import annotations

from typing import Iterable, Literal

import torch
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms


CalibSource = Literal["train", "val", "test"]


class HFImageStream(IterableDataset):
    def __init__(self, stream: Iterable, transform):
        self.stream = stream
        self.transform = transform

    def __iter__(self):
        for item in self.stream:
            image = _image_from_item(item)
            if image.mode != "RGB":
                image = image.convert("RGB")
            label = item.get("label", -100)
            if label is None:
                label = -100
            yield self.transform(image), int(label)


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def make_hf_loader(
    *,
    dataset_name: str,
    split: str,
    batch_size: int,
    transform,
    shuffle_buffer: int = 0,
    seed: int = 0,
    take: int | None = None,
    skip: int = 0,
) -> DataLoader:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install HuggingFace datasets: pip install datasets") from exc

    stream = load_dataset(dataset_name, streaming=True, split=split)
    if shuffle_buffer > 0:
        stream = stream.shuffle(buffer_size=shuffle_buffer, seed=seed)
    if skip > 0:
        stream = stream.skip(skip)
    if take is not None:
        stream = stream.take(take)

    dataset = HFImageStream(stream, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def make_calibration_loader(
    *,
    dataset_name: str,
    source: CalibSource,
    batch_size: int,
    transform,
    calib_samples: int,
    seed: int,
    shuffle_buffer: int,
) -> DataLoader:
    if source == "train":
        return make_hf_loader(
            dataset_name=dataset_name,
            split="train",
            batch_size=batch_size,
            transform=transform,
            shuffle_buffer=shuffle_buffer,
            seed=seed,
            take=calib_samples,
        )
    if source == "val":
        # No shuffle: eval will use .skip(calib_samples) on the same unshuffled stream
        # for a guaranteed positional split with zero overlap.
        return make_hf_loader(
            dataset_name=dataset_name,
            split="validation",
            batch_size=batch_size,
            transform=transform,
            take=calib_samples,
        )
    if source == "test":
        return make_hf_loader(
            dataset_name=dataset_name,
            split="test",
            batch_size=batch_size,
            transform=transform,
            shuffle_buffer=shuffle_buffer,
            seed=seed,
            take=calib_samples,
        )
    raise ValueError(f"Unknown calibration source: {source}")


def make_eval_loader(
    *,
    dataset_name: str,
    batch_size: int,
    transform,
    eval_samples: int,
    skip: int = 0,
) -> DataLoader:
    return make_hf_loader(
        dataset_name=dataset_name,
        split="validation",
        batch_size=batch_size,
        transform=transform,
        take=eval_samples,
        skip=skip,
    )


def cache_calibration_data(loader: DataLoader, num_samples: int) -> list[torch.Tensor]:
    if num_samples <= 0:
        raise ValueError("--calib-samples must be positive")

    batches: list[torch.Tensor] = []
    seen = 0

    for images, _labels in loader:
        if seen >= num_samples:
            break
        remaining = num_samples - seen
        if images.size(0) > remaining:
            images = images[:remaining]
        batches.append(images.cpu())
        seen += images.size(0)

    if seen == 0:
        raise RuntimeError("No calibration images were read from the HuggingFace stream.")

    print(f"[Data] Cached {seen} calibration images in {len(batches)} batches.")
    return batches


def _image_from_item(item: dict):
    from PIL import Image
    import io

    for key in ("image", "jpg", "jpeg", "png", "webp"):
        value = item.get(key)
        if value is None:
            continue
        if hasattr(value, "mode"):
            return value
        if isinstance(value, bytes):
            return Image.open(io.BytesIO(value))
        if isinstance(value, dict) and value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"]))
    raise ValueError(f"Could not find an image field. Available keys: {list(item.keys())}")
