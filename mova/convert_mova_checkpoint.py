#!/usr/bin/env python3
"""
convert_mova_checkpoint.py
==========================
Merge MOVA sharded video_dit / video_dit_2 checkpoints into single safetensors files
that can be used with WanVideoWrapper's standard model loaders.

Background
----------
The MOVA HuggingFace repos (OpenMOSS-Team/MOVA-360p and OpenMOSS-Team/MOVA-720p) store
video_dit and video_dit_2 as three ~10 GB shards:

  video_dit/
      diffusion_pytorch_model.safetensors.index.json   (key → shard mapping)
      diffusion_pytorch_model-00001-of-00003.safetensors  (9.97 GB)
      diffusion_pytorch_model-00002-of-00003.safetensors  (9.89 GB)
      diffusion_pytorch_model-00003-of-00003.safetensors  (8.72 GB)
      config.json

The merged single file can then be quantised and placed in ComfyUI's
models/diffusion_models/ folder for loading with WanVideoModelLoader.

Usage
-----
# Merge both video_dit and video_dit_2 from the 360p checkpoint:
python convert_mova_checkpoint.py \\
    --ckpt_path /path/to/MOVA-360p \\
    --output_dir /path/to/ComfyUI/models/diffusion_models

# Merge only video_dit_2 with FP8 quantisation (saves ~14 GB on disk):
python convert_mova_checkpoint.py \\
    --ckpt_path /path/to/MOVA-360p \\
    --output_dir /path/to/output \\
    --model video_dit_2 \\
    --quantize fp8_e4m3fn

Options
-------
--ckpt_path   Path to the MOVA checkpoint directory (contains audio_dit/, video_dit/ …)
--output_dir  Directory where the merged .safetensors files will be written
--model       Which model to merge: video_dit | video_dit_2 | both  (default: both)
--quantize    Optional quantisation for the output: fp8_e4m3fn | fp8_e5m2
              (keeps norm / bias / embedding layers in float32)
--suffix      Optional name suffix, e.g. "360p" → mova_360p_video_dit.safetensors
"""

import argparse
import json
import os
import sys


QUANT_KEEP_PATTERNS = (
    "norm",
    "bias",
    "pos_emb",
    "embedding",
    "time_embedding",
    "text_embedding",
    "modulation",
    "head",
    "img_emb",
    "freqs",
)


def _maybe_quantize(tensor, quant_dtype, key: str):
    """Cast tensor to quant_dtype unless the key matches a sensitive pattern."""
    if quant_dtype is None:
        return tensor
    if any(p in key for p in QUANT_KEEP_PATTERNS):
        return tensor
    if not tensor.is_floating_point():
        return tensor
    return tensor.to(quant_dtype)


def merge_shards(
    model_dir: str,
    output_path: str,
    quant_dtype=None,
) -> None:
    """Merge a sharded HuggingFace safetensors checkpoint into a single file."""
    import torch
    from safetensors.torch import load_file, save_file

    index_path = os.path.join(model_dir, "diffusion_pytorch_model.safetensors.index.json")
    single_path = os.path.join(model_dir, "diffusion_pytorch_model.safetensors")

    if os.path.isfile(single_path):
        print(f"  Single-file checkpoint detected: {single_path}")
        if quant_dtype is None:
            # Nothing to do – just copy?  We still run through save_file to normalise metadata.
            print("  No quantisation requested – output will be a clean copy.")
        state_dict = load_file(single_path)
    elif os.path.isfile(index_path):
        print(f"  Sharded checkpoint detected: {index_path}")
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        shard_files = sorted(set(weight_map.values()))
        print(f"  Merging {len(shard_files)} shard(s) …")
        state_dict = {}
        for shard_name in shard_files:
            shard_path = os.path.join(model_dir, shard_name)
            if not os.path.isfile(shard_path):
                print(f"  ERROR: shard file not found: {shard_path}", file=sys.stderr)
                sys.exit(1)
            print(f"    Loading {shard_name} …")
            state_dict.update(load_file(shard_path))
    else:
        print(
            f"  ERROR: No weights found in {model_dir}\n"
            "  Expected diffusion_pytorch_model.safetensors or "
            "diffusion_pytorch_model.safetensors.index.json (+shards)",
            file=sys.stderr,
        )
        sys.exit(1)

    if quant_dtype is not None:
        print(f"  Quantising to {quant_dtype} (keeping norms/biases/embeddings in fp32) …")
        state_dict = {k: _maybe_quantize(v, quant_dtype, k) for k, v in state_dict.items()}

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"  Saving → {output_path}")
    save_file(state_dict, output_path)
    size_gb = os.path.getsize(output_path) / 1e9
    print(f"  Done! ({size_gb:.2f} GB)")


def main():
    parser = argparse.ArgumentParser(
        description="Merge MOVA sharded checkpoints into single safetensors files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ckpt_path",
        required=True,
        help="Root directory of the MOVA checkpoint (e.g. /data/MOVA-360p)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where merged safetensors files will be written.",
    )
    parser.add_argument(
        "--model",
        choices=["video_dit", "video_dit_2", "both"],
        default="both",
        help="Which model(s) to merge (default: both).",
    )
    parser.add_argument(
        "--quantize",
        choices=["fp8_e4m3fn", "fp8_e5m2"],
        default=None,
        help=(
            "Quantise the merged weights. Recommended for video_dit_2 (~28 GB) "
            "to save disk and VRAM. Linear weights are cast to the chosen FP8 dtype; "
            "norms/biases/embeddings stay in float32."
        ),
    )
    parser.add_argument(
        "--suffix",
        default="",
        help=(
            "Optional string to include in the output filename, e.g. '360p' → "
            "mova_360p_video_dit.safetensors  (default: no suffix)."
        ),
    )
    args = parser.parse_args()

    try:
        import torch
        import safetensors  # noqa: F401
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}\nInstall with: pip install torch safetensors", file=sys.stderr)
        sys.exit(1)

    quant_dtype = None
    if args.quantize:
        dtype_map = {
            "fp8_e4m3fn": torch.float8_e4m3fn,
            "fp8_e5m2": torch.float8_e5m2,
        }
        quant_dtype = dtype_map[args.quantize]
        print(f"Quantisation: {args.quantize}")

    name_infix = f"_{args.suffix}" if args.suffix else ""

    os.makedirs(args.output_dir, exist_ok=True)

    if args.model in ("video_dit", "both"):
        src = os.path.join(args.ckpt_path, "video_dit")
        out = os.path.join(args.output_dir, f"mova{name_infix}_video_dit.safetensors")
        print(f"\n[video_dit] {src} → {out}")
        merge_shards(src, out, quant_dtype)

    if args.model in ("video_dit_2", "both"):
        src = os.path.join(args.ckpt_path, "video_dit_2")
        out = os.path.join(args.output_dir, f"mova{name_infix}_video_dit_2.safetensors")
        print(f"\n[video_dit_2] {src} → {out}")
        merge_shards(src, out, quant_dtype)

    print("\nAll done.")


if __name__ == "__main__":
    main()
