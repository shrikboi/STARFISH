from __future__ import annotations
import argparse
import json
import os
import random
import torch
from pathlib import Path

from constants import MOBILENET_PRUNE_SCOPES, VIT_PRUNE_SCOPES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recover a pruned model by cosine-aligning internal representations."
    )

    parser.add_argument("--dense-path", required=True, help="Path to dense model checkpoint.")
    parser.add_argument(
        "--mask-path",
        required=True,
        help="Path to mask checkpoint. 1/True keeps weights trainable; 0/False zeros and freezes them.",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="One of the supported ViT or MobileNetV1 model names.",
    )
    parser.add_argument("--output-path", default="outputs/recovered.pt")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--calib-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="cosine")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument(
        "--grad-checkpointing",
        action="store_true",
        help=(
            "Enable activation/gradient checkpointing on the pruned model to cut "
            "GPU memory during backprop. Works on timm models (DeiT/ViT, "
            "MobileNetV1); no-op on torchvision ViTs. Default off."
        ),
    )

    parser.add_argument("--hf-dataset", default="ILSVRC/imagenet-1k")
    parser.add_argument("--calib-source", choices=["train", "val", "test"], default="test")
    parser.add_argument("--shuffle-buffer", type=int, default=5000)
    parser.add_argument("--eval-samples", type=int, default=0, help="0 disables validation eval.")

    parser.add_argument(
        "--prune-scope",
        default="auto",
        help=(
            "auto, a ViT scope "
            f"({', '.join(VIT_PRUNE_SCOPES)}), or a MobileNetV1 scope "
            f"({', '.join(MOBILENET_PRUNE_SCOPES)})."
        ),
    )
    parser.add_argument("--store-dense-float32", action="store_true")
    parser.add_argument(
        "--no-dense-cache",
        action="store_true",
        help=(
            "Do not cache dense teacher representations. Each calibration step "
            "re-runs the dense model forward instead of indexing a cached list, "
            "trading extra compute for the RAM the cache would use. The dense "
            "model stays on-device."
        ),
    )

    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    from data import (
        build_transform,
        cache_calibration_data,
        make_calibration_loader,
        make_eval_loader,
    )
    from eval import evaluate_top1
    from masks import (
        apply_masks,
        load_masks,
        mask_density,
        model_nonzero_fraction,
    )
    from models import (
        all_block_names,
        default_prune_scope,
        enable_grad_checkpointing,
        force_mobilenet_str_runtime,
        infer_model_kind,
        is_prunable_weight,
        load_model,
    )
    from recovery import RecoveryConfig, starfish_recovery

    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    kind = infer_model_kind(args.model_name)
    prune_scope = default_prune_scope(args.model_name) if args.prune_scope == "auto" else args.prune_scope
    is_prunable = is_prunable_weight(args.model_name, prune_scope)

    transform = build_transform()

    print("=" * 88)
    print("[STARFISH] recovery")
    print(f"[STARFISH] model={args.model_name} kind={kind} device={device}")
    print(f"[STARFISH] calib_source={args.calib_source} epochs={args.epochs} samples={args.calib_samples}")
    print("=" * 88)

    print("[Load] Dense model.")
    dense_model = load_model(args.model_name, args.dense_path, device)
    print("[Load] Pruned model.")
    pruned_model = load_model(args.model_name, args.dense_path, device)
    if kind == "mobilenet":
        force_mobilenet_str_runtime(dense_model)
        force_mobilenet_str_runtime(pruned_model)
        print("[Model] MobileNetV1 STR mode: standard normalization, ReLU activations.")

    if args.grad_checkpointing:
        if enable_grad_checkpointing(pruned_model):
            print("[Model] Gradient checkpointing enabled on the pruned model.")
        else:
            print(
                f"[Model] WARNING: --grad-checkpointing requested but {args.model_name} "
                "has no set_grad_checkpointing(); continuing without it."
            )

    blocks = all_block_names(args.model_name, pruned_model)
    print(f"[Blocks] Aligning {len(blocks)} blocks.")

    masks = load_masks(args.mask_path, pruned_model, is_prunable)
    apply_masks(pruned_model, masks)
    print(
        f"[Masks] {len(masks)} tensors, mask density={mask_density(masks):.4f}, "
        f"model nonzero fraction={model_nonzero_fraction(pruned_model):.4f}"
    )

    calib_loader = make_calibration_loader(
        dataset_name=args.hf_dataset,
        source=args.calib_source,
        batch_size=args.batch_size,
        transform=transform,
        calib_samples=args.calib_samples,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
    )
    calib_batches = cache_calibration_data(calib_loader, args.calib_samples)

    if args.eval_samples > 0:
        eval_loader = make_eval_loader(
            dataset_name=args.hf_dataset,
            batch_size=args.batch_size,
            transform=transform,
            eval_samples=args.eval_samples,
            skip=args.calib_samples if args.calib_source == "val" else 0,
        )
        before = evaluate_top1(pruned_model, eval_loader, device)
        print(f"[Eval] Before recovery top1={before:.2f}")
    else:
        before = float("nan")

    config = RecoveryConfig(
        epochs=args.epochs,
        lr=args.lr,
        min_lr=args.min_lr,
        lr_schedule=args.lr_schedule,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        store_dense_float16=not args.store_dense_float32,
        no_dense_cache=args.no_dense_cache,
    )

    history = starfish_recovery(
        pruned_model=pruned_model,
        dense_model=dense_model,
        kind=kind,
        block_names=blocks,
        masks=masks,
        calib_batches=calib_batches,
        device=device,
        is_prunable_weight=is_prunable,
        config=config,
    )

    if args.eval_samples > 0:
        eval_loader = make_eval_loader(
            dataset_name=args.hf_dataset,
            batch_size=args.batch_size,
            transform=transform,
            eval_samples=args.eval_samples,
            skip=args.calib_samples if args.calib_source == "val" else 0,
        )
        after = evaluate_top1(pruned_model, eval_loader, device)
        print(f"[Eval] After recovery top1={after:.2f} delta={after - before:+.2f}")
    else:
        after = float("nan")

    save_recovered_checkpoint(
        path=args.output_path,
        model=pruned_model,
        args=args,
        blocks=blocks,
        history=history,
        before=before,
        after=after,
        mask_density_value=mask_density(masks),
    )


def save_recovered_checkpoint(
    *,
    path: str,
    model,
    args,
    blocks: list[str],
    history: list[dict[str, float]],
    before: float,
    after: float,
    mask_density_value: float,
) -> None:

    output_path = Path(path)
    if output_path.parent:
        os.makedirs(output_path.parent, exist_ok=True)

    payload = {
        "state_dict": model.state_dict(),
        "metadata": {
            "model_name": args.model_name,
            "dense_path": args.dense_path,
            "mask_path": args.mask_path,
            "calib_source": args.calib_source,
            "calib_samples": args.calib_samples,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "grad_checkpointing": args.grad_checkpointing,
            "blocks": blocks,
            "history": history,
            "top1_before": before,
            "top1_after": after,
            "mask_density": mask_density_value,
        },
    }
    torch.save(payload, output_path)

    meta_path = output_path.with_suffix(output_path.suffix + ".json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(payload["metadata"], handle, indent=2)
    print(f"[Save] Recovered checkpoint: {output_path}")
    print(f"[Save] Metadata: {meta_path}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
