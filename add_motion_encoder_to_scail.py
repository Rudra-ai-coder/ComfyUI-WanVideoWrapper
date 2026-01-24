#!/usr/bin/env python3
"""
Script to add motion_encoder weights from WAN Animate model to a SCAIL model.

This script merges the motion_encoder component from a WAN Animate model into
a SCAIL model that already has face_encoder and fuser_block weights.

Usage:
    python add_motion_encoder_to_scail.py \
        --scail /path/to/scail_with_face_adapter.safetensors \
        --wananimate /path/to/wananimate_model.safetensors \
        --output /path/to/merged_output.safetensors

Components merged from WAN Animate:
    - motion_encoder.* (InsightFace-based motion extractor)

Optionally also merges (if missing):
    - face_encoder.* (face embedding processor)
    - face_adapter.fuser_blocks.* (injection blocks)
"""

import argparse
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm


def get_model_info(sd, name="Model"):
    """Print summary info about a model's state dict."""
    print(f"\n{'='*60}")
    print(f"{name} Info:")
    print(f"{'='*60}")
    print(f"Total keys: {len(sd)}")
    
    # Check for WAN Animate components
    motion_keys = [k for k in sd if k.startswith("motion_encoder.")]
    face_keys = [k for k in sd if k.startswith("face_encoder.")]
    fuser_orig = [k for k in sd if "face_adapter.fuser_blocks" in k]
    fuser_renamed = [k for k in sd if ".fuser_block." in k]
    pose_patch = [k for k in sd if k.startswith("pose_patch_embedding.")]
    patch_pose = [k for k in sd if k.startswith("patch_embedding_pose.")]
    
    print(f"\nWAN Animate components:")
    print(f"  motion_encoder.*: {len(motion_keys)} keys")
    print(f"  face_encoder.*: {len(face_keys)} keys")
    print(f"  face_adapter.fuser_blocks.*: {len(fuser_orig)} keys")
    print(f"  blocks.*.fuser_block.*: {len(fuser_renamed)} keys")
    
    print(f"\nPose embedding:")
    print(f"  pose_patch_embedding.* (WAN Animate): {len(pose_patch)} keys")
    print(f"  patch_embedding_pose.* (SCAIL): {len(patch_pose)} keys")
    
    return {
        "motion_encoder": motion_keys,
        "face_encoder": face_keys,
        "fuser_orig": fuser_orig,
        "fuser_renamed": fuser_renamed,
        "pose_patch": pose_patch,
        "patch_pose": patch_pose,
    }


def merge_motion_encoder(scail_path, wananimate_path, output_path, 
                         also_merge_face_encoder=True,  # Changed to True - face_encoder is required!
                         also_merge_fuser_blocks=True):  # Changed to True - fuser_blocks are required!
    """
    Merge WAN Animate face adapter components from WAN Animate into SCAIL model.
    
    Args:
        scail_path: Path to SCAIL model (base model)
        wananimate_path: Path to WAN Animate model (source of face adapter)
        output_path: Path to save merged model
        also_merge_face_encoder: Also merge face_encoder if missing (default: True)
        also_merge_fuser_blocks: Also merge fuser blocks if missing (default: True)
    """
    print("Loading SCAIL model...")
    scail_sd = load_file(scail_path)
    scail_info = get_model_info(scail_sd, "SCAIL Model")
    
    print("\nLoading WAN Animate model...")
    wananimate_sd = load_file(wananimate_path)
    wananimate_info = get_model_info(wananimate_sd, "WAN Animate Model")
    
    # Start with SCAIL model
    merged_sd = dict(scail_sd)
    merged_count = 0
    
    # Always merge motion_encoder
    print("\n" + "="*60)
    print("Merging components...")
    print("="*60)
    
    if wananimate_info["motion_encoder"]:
        print(f"\n✓ Adding motion_encoder ({len(wananimate_info['motion_encoder'])} keys)...")
        for key in tqdm(wananimate_info["motion_encoder"], desc="motion_encoder"):
            if key in merged_sd:
                print(f"  Warning: Overwriting existing key: {key}")
            merged_sd[key] = wananimate_sd[key]
            merged_count += 1
    else:
        print("\n⚠️  WAN Animate model has no motion_encoder weights!")
        return False
    
    # Optionally merge face_encoder
    if also_merge_face_encoder:
        if not scail_info["face_encoder"] and wananimate_info["face_encoder"]:
            print(f"\n✓ Adding face_encoder ({len(wananimate_info['face_encoder'])} keys)...")
            for key in tqdm(wananimate_info["face_encoder"], desc="face_encoder"):
                merged_sd[key] = wananimate_sd[key]
                merged_count += 1
        elif scail_info["face_encoder"]:
            print("\n• Skipping face_encoder (already exists in SCAIL model)")
    
    # Optionally merge fuser blocks
    if also_merge_fuser_blocks:
        # Check if SCAIL already has fuser blocks
        has_fuser = scail_info["fuser_orig"] or scail_info["fuser_renamed"]
        
        if not has_fuser and wananimate_info["fuser_orig"]:
            print(f"\n✓ Adding fuser_blocks ({len(wananimate_info['fuser_orig'])} keys)...")
            for key in tqdm(wananimate_info["fuser_orig"], desc="fuser_blocks"):
                merged_sd[key] = wananimate_sd[key]
                merged_count += 1
        elif has_fuser:
            print("\n• Skipping fuser_blocks (already exists in SCAIL model)")
    
    print(f"\n{'='*60}")
    print(f"Merged {merged_count} keys from WAN Animate into SCAIL model")
    print(f"Total keys in merged model: {len(merged_sd)}")
    print(f"{'='*60}")
    
    # Save merged model
    print(f"\nSaving merged model to: {output_path}")
    
    # Get metadata from SCAIL model if available
    try:
        import json
        from safetensors import safe_open
        
        with safe_open(scail_path, framework="pt") as f:
            metadata = f.metadata()
        
        if metadata:
            print(f"Preserving {len(metadata)} metadata entries from SCAIL model")
        else:
            metadata = {}
    except:
        metadata = {}
    
    # Add merge info to metadata
    metadata["merged_from"] = f"SCAIL: {scail_path}, WanAnimate: {wananimate_path}"
    metadata["merged_components"] = "motion_encoder"
    if also_merge_face_encoder:
        metadata["merged_components"] += ", face_encoder"
    if also_merge_fuser_blocks:
        metadata["merged_components"] += ", fuser_blocks"
    
    save_file(merged_sd, output_path, metadata=metadata)
    print("✅ Done!")
    
    # Verify the output
    print("\nVerifying merged model...")
    get_model_info(load_file(output_path), "Merged Model")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Add motion_encoder from WAN Animate to SCAIL model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage - just add motion_encoder:
    python add_motion_encoder_to_scail.py \\
        --scail my_scail_model.safetensors \\
        --wananimate wanvideo_animate.safetensors \\
        --output my_scail_with_motion_encoder.safetensors
    
    # Also merge face_encoder if missing:
    python add_motion_encoder_to_scail.py \\
        --scail my_scail_model.safetensors \\
        --wananimate wanvideo_animate.safetensors \\
        --output merged.safetensors \\
        --also-face-encoder
    
    # Merge all WAN Animate face adapter components:
    python add_motion_encoder_to_scail.py \\
        --scail my_scail_model.safetensors \\
        --wananimate wanvideo_animate.safetensors \\
        --output merged.safetensors \\
        --also-face-encoder \\
        --also-fuser-blocks
        """
    )
    
    parser.add_argument("--scail", required=True, 
                        help="Path to SCAIL model (target)")
    parser.add_argument("--wananimate", required=True,
                        help="Path to WAN Animate model (source of motion_encoder)")
    parser.add_argument("--output", required=True,
                        help="Path to save merged model")
    parser.add_argument("--also-face-encoder", action="store_true",
                        help="Also merge face_encoder if missing in SCAIL model")
    parser.add_argument("--also-fuser-blocks", action="store_true",
                        help="Also merge fuser_blocks if missing in SCAIL model")
    
    args = parser.parse_args()
    
    success = merge_motion_encoder(
        scail_path=args.scail,
        wananimate_path=args.wananimate,
        output_path=args.output,
        also_merge_face_encoder=args.also_face_encoder,
        also_merge_fuser_blocks=args.also_fuser_blocks
    )
    
    if not success:
        print("\n❌ Merge failed!")
        exit(1)


if __name__ == "__main__":
    main()
