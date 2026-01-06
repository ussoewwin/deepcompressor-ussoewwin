#!/usr/bin/env python
"""
[CRITICAL] FLUX Quantization Pre-flight and Post-flight Verification

This script checks EVERY stage of the quantization pipeline to ensure
lora_down/lora_up (Low-Rank Branch) are correctly generated and exported.

RUN THIS BEFORE EXPORT to verify intermediate files.
RUN THIS AFTER EXPORT to verify the final output.

Usage:
    python verify_lora_complete.py --branch-pt <path_to_branch.pt>
    python verify_lora_complete.py --output <path_to_output.safetensors>
    python verify_lora_complete.py --both --branch-pt <path> --output <path>
"""

import argparse
import sys
from pathlib import Path


def check_branch_pt(branch_pt_path: str) -> bool:
    """Check that branch.pt contains the expected Low-Rank branches."""
    import torch
    
    print(f"\n{'='*60}")
    print(f"[STEP 1] Checking branch.pt: {branch_pt_path}")
    print(f"{'='*60}")
    
    if not Path(branch_pt_path).exists():
        print(f"[FAIL] branch.pt not found at: {branch_pt_path}")
        return False
    
    branch_dict = torch.load(branch_pt_path, map_location="cpu")
    
    if not branch_dict:
        print("[FAIL] branch.pt is EMPTY!")
        return False
    
    print(f"[INFO] branch.pt contains {len(branch_dict)} entries")
    
    # Expected patterns for FLUX
    expected_patterns = [
        # Double transformer blocks (19 blocks)
        ("transformer_blocks.0.attn.to_q", "qkv_proj (block 0)"),
        ("transformer_blocks.0.attn.add_q_proj", "qkv_proj_context (block 0)"),
        # Single transformer blocks (38 blocks)
        ("single_transformer_blocks.0.attn.to_q", "single_qkv_proj (block 0)"),
    ]
    
    all_ok = True
    for pattern, desc in expected_patterns:
        if pattern in branch_dict:
            entry = branch_dict[pattern]
            if "a.weight" in entry and "b.weight" in entry:
                a_shape = entry["a.weight"].shape
                b_shape = entry["b.weight"].shape
                print(f"[OK] {desc}: a.weight={list(a_shape)}, b.weight={list(b_shape)}")
            else:
                print(f"[FAIL] {desc}: Missing a.weight or b.weight! Keys: {list(entry.keys())}")
                all_ok = False
        else:
            print(f"[FAIL] {desc}: Key '{pattern}' NOT FOUND in branch.pt!")
            all_ok = False
    
    # Show sample of all keys
    print(f"\n[INFO] Sample keys in branch.pt (first 10):")
    for i, key in enumerate(sorted(branch_dict.keys())[:10]):
        print(f"  - {key}")
    
    # Count by category
    qkv_count = sum(1 for k in branch_dict if ".attn.to_q" in k)
    add_qkv_count = sum(1 for k in branch_dict if ".attn.add_q_proj" in k)
    print(f"\n[INFO] qkv_proj entries (attn.to_q): {qkv_count}")
    print(f"[INFO] qkv_proj_context entries (attn.add_q_proj): {add_qkv_count}")
    
    if qkv_count == 0:
        print("[FAIL] No qkv_proj branches found!")
        all_ok = False
    if add_qkv_count == 0:
        print("[FAIL] No qkv_proj_context branches found!")
        all_ok = False
    
    return all_ok


def check_output_safetensors(output_path: str) -> bool:
    """Check that the output safetensors contains lora_down/lora_up."""
    from safetensors import safe_open
    
    print(f"\n{'='*60}")
    print(f"[STEP 2] Checking output safetensors: {output_path}")
    print(f"{'='*60}")
    
    if not Path(output_path).exists():
        print(f"[FAIL] Output file not found at: {output_path}")
        return False
    
    with safe_open(output_path, framework="pt") as f:
        keys = list(f.keys())
    
    print(f"[INFO] Output contains {len(keys)} tensors")
    
    # Check for lora_down/lora_up
    lora_down_keys = [k for k in keys if ".lora_down" in k]
    lora_up_keys = [k for k in keys if ".lora_up" in k]
    
    print(f"[INFO] lora_down tensors: {len(lora_down_keys)}")
    print(f"[INFO] lora_up tensors: {len(lora_up_keys)}")
    
    if len(lora_down_keys) == 0:
        print("[FAIL] NO lora_down tensors found in output!")
        return False
    if len(lora_up_keys) == 0:
        print("[FAIL] NO lora_up tensors found in output!")
        return False
    
    # Check specific patterns
    all_ok = True
    patterns_to_check = [
        "transformer_blocks.0.qkv_proj.lora_down",
        "transformer_blocks.0.qkv_proj.lora_up",
        "transformer_blocks.0.qkv_proj_context.lora_down",
        "transformer_blocks.0.qkv_proj_context.lora_up",
        "single_transformer_blocks.0.qkv_proj.lora_down",
        "single_transformer_blocks.0.qkv_proj.lora_up",
    ]
    
    for pattern in patterns_to_check:
        if pattern in keys:
            print(f"[OK] {pattern}")
        else:
            print(f"[FAIL] {pattern} NOT FOUND!")
            all_ok = False
    
    # Also check qweight and other essential keys
    qweight_keys = [k for k in keys if ".qweight" in k]
    print(f"\n[INFO] qweight tensors: {len(qweight_keys)}")
    
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Verify FLUX quantization lora_down/lora_up")
    parser.add_argument("--branch-pt", type=str, help="Path to branch.pt")
    parser.add_argument("--output", type=str, help="Path to output safetensors")
    parser.add_argument("--both", action="store_true", help="Check both branch.pt and output")
    args = parser.parse_args()
    
    if not args.branch_pt and not args.output:
        parser.print_help()
        sys.exit(1)
    
    all_passed = True
    
    if args.branch_pt:
        if not check_branch_pt(args.branch_pt):
            all_passed = False
    
    if args.output:
        if not check_output_safetensors(args.output):
            all_passed = False
    
    print(f"\n{'='*60}")
    if all_passed:
        print("[SUCCESS] All checks passed!")
        sys.exit(0)
    else:
        print("[FAILURE] Some checks failed! DO NOT proceed with deployment.")
        sys.exit(1)


if __name__ == "__main__":
    main()
