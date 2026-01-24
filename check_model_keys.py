#!/usr/bin/env python3
"""
Quick diagnostic script to check what face adapter related weights exist in a safetensors model.
Usage: python check_model_keys.py /path/to/model.safetensors
"""

import sys
from safetensors import safe_open

def check_model(model_path):
    print(f"Checking model: {model_path}\n")
    
    with safe_open(model_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
    
    # Check for WAN Animate components
    print("=" * 60)
    print("WAN Animate Components Check:")
    print("=" * 60)
    
    # Motion encoder
    motion_keys = [k for k in keys if k.startswith("motion_encoder.")]
    print(f"\n✓ motion_encoder.* keys: {len(motion_keys)}")
    if motion_keys:
        for k in motion_keys[:5]:
            print(f"   - {k}")
        if len(motion_keys) > 5:
            print(f"   ... and {len(motion_keys) - 5} more")
    else:
        print("   ⚠️  NO MOTION ENCODER WEIGHTS FOUND!")
    
    # Face encoder
    face_keys = [k for k in keys if k.startswith("face_encoder.")]
    print(f"\n✓ face_encoder.* keys: {len(face_keys)}")
    if face_keys:
        for k in face_keys[:5]:
            print(f"   - {k}")
        if len(face_keys) > 5:
            print(f"   ... and {len(face_keys) - 5} more")
    else:
        print("   ⚠️  NO FACE ENCODER WEIGHTS FOUND!")
    
    # Fuser blocks (original WAN Animate format)
    fuser_orig_keys = [k for k in keys if "face_adapter.fuser_blocks" in k]
    print(f"\n✓ face_adapter.fuser_blocks.* keys: {len(fuser_orig_keys)}")
    if fuser_orig_keys:
        for k in fuser_orig_keys[:5]:
            print(f"   - {k}")
    
    # Fuser blocks (renamed format)
    fuser_renamed_keys = [k for k in keys if ".fuser_block." in k]
    print(f"\n✓ blocks.*.fuser_block.* keys: {len(fuser_renamed_keys)}")
    if fuser_renamed_keys:
        for k in fuser_renamed_keys[:5]:
            print(f"   - {k}")
    
    # Pose embeddings
    print("\n" + "=" * 60)
    print("Pose Embedding Check:")
    print("=" * 60)
    
    pose_patch = [k for k in keys if k.startswith("pose_patch_embedding.")]
    print(f"\n✓ pose_patch_embedding.* (WAN Animate): {len(pose_patch)}")
    
    patch_pose = [k for k in keys if k.startswith("patch_embedding_pose.")]
    print(f"✓ patch_embedding_pose.* (SCAIL): {len(patch_pose)}")
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    if motion_keys and face_keys and (fuser_orig_keys or fuser_renamed_keys):
        print("✅ Full WAN Animate face adapter is present")
    elif face_keys and (fuser_orig_keys or fuser_renamed_keys) and not motion_keys:
        print("⚠️  Face adapter present but MISSING motion_encoder!")
        print("   You need to add motion_encoder weights to your merged model.")
    elif not (motion_keys or face_keys or fuser_orig_keys or fuser_renamed_keys):
        print("❌ No WAN Animate face adapter components found")
    else:
        print("⚠️  Partial WAN Animate face adapter - some components missing")
    
    if patch_pose:
        print("✅ SCAIL pose embedding present")
    if pose_patch:
        print("✅ WAN Animate pose embedding present")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_model_keys.py /path/to/model.safetensors")
        sys.exit(1)
    
    check_model(sys.argv[1])
