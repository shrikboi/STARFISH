# STARFISH
This repository contains the official code for the paper:

**STARFISH: faST Accuracy Recovery in pruned networks From Internal State Healing.**

STARFISH recovers the accuracy of a pruned model *without labels and
without full retraining*, offering an efficient way to recover most of the accuracy lost when pruning,
while keeping the benefits of the smaller models. Its key idea is aligning the 
**internal representations** of the pruned model with the dense model's. The objective
is a per-block cosine-alignment loss (`mean over blocks of 1 − cos`) on a small
set of unlabeled calibration images — so the network "heals" its internal
state while staying sparse. 

Crucially, it never touches the original training data — a few thousand images (by default ImageNet's held-out test 
split) are all it needs. This makes recovery cheap and usable even when the training set is unavailable.



## Install

```powershell
pip install -r requirements.txt
```

Calibration/eval images are streamed from Hugging Face
(`ILSVRC/imagenet-1k` by default), which is gated — run `huggingface-cli login`
and accept the dataset terms once, or point `--hf-dataset` at another image set.

## Example run

Recover a pruned DeiT-B, calibrating on 5000 images and
reporting top-1 before/after:

```powershell
python starfish.py `
  --model-name deit_base_patch16_224 `
  --dense-path path\dense.pt `
  --mask-path  path\mask.pt `
  --calib-source test `
  --calib-samples 5000 `
  --batch-size 32 `
  --epochs 10 `
  --lr 6e-4 --min-lr 1e-6 `
  --eval-samples 50000 `
  --output-path outputs\recovered.pt
```

## Masks

The mask checkpoint must be a plain dictionary of `{weight_name: mask_tensor}`,
keyed to match the pruned weights (e.g. `encoder.layers.encoder_layer_0.mlp.0.weight`).
Mask values must be boolean or 0/1 tensors with the same shape as the weight:
`1` keeps a weight trainable, `0` zeros it and freezes its gradient.

## Useful options

| Flag | Default | Notes                                                                                                                                                           |
| --- | --- |-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--prune-scope` | `auto` | Which layers are sparse/trainable. ViT: `qkv`, `qkv_out`, `qkv_out_mlp`, `mlp`, `qk`, `qk_mlp`. MobileNet: `conv_only`, `fc_only`, `conv_fc`, `conv_dw`, `all`. |
| `--calib-source` | `test` | `train` / `val` / `test` split to draw calibration images from. `val` uses non-overlapping eval slice.                                                          |
| `--lr-schedule` | `cosine` | `cosine` (to `--min-lr`) or `constant`.                                                                                                                         |
| `--eval-samples` | `0` | `>0` runs a top-1 check before and after recovery.                                                                                                              |
| `--grad-checkpointing` | off | Trades ~20–30% speed for lower memory (timm models only).                                                                                                       |
| `--no-dense-cache` | off | Recompute teacher features each step instead of caching them (saves RAM).                                                                                       |
| `--device` / `--seed` | auto / `0` | Standard overrides.                                                                                                                                             |

## Supported models

torchvision **ViT-B/16**, **ViT-L/16**; timm **DeiT / DeiT3** (`deit_tiny/small/base_patch16_224`,
`deit3_base/large/huge`); and timm **MobileNetV1** (`mobilenetv1_100`).

> **MobileNetV1 note.** MobileNetV1 checkpoints are assumed to follow the STR
> training recipe: at load time any `ReLU6` activations are converted to `ReLU`
> to match that runtime. If your checkpoint was trained with `ReLU6`, this
> conversion will change its behavior and degrade accuracy — convert your model
> or remove this step (`force_mobilenet_str_runtime` in `models.py`).

