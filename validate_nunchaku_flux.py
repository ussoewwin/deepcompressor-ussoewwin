"""
Nunchaku FLUX Model Validation Script
Run after quantization to verify the output before loading in ComfyUI.
"""
import sys
from safetensors import safe_open

def validate_flux_nunchaku_model(model_path: str, verbose: bool = True):
    """
    Validate a Nunchaku FLUX model for correctness.
    
    Checks:
    1. All transformer_blocks have qkv_proj_context.qweight (not .weight)
    2. All expected quantization keys exist (qweight, wscales, smooth, lora_down, lora_up)
    3. Metadata exists (config, comfy_config, quantization_config, model_class)
    """
    print(f"Validating: {model_path}")
    print("=" * 60)
    
    errors = []
    warnings = []
    
    with safe_open(model_path, framework='pt') as f:
        keys = set(f.keys())
        metadata = f.metadata() or {}
        
        # Check metadata
        required_metadata = ['config', 'comfy_config', 'quantization_config', 'model_class']
        for m in required_metadata:
            if m not in metadata:
                errors.append(f"Missing metadata: {m}")
            elif verbose:
                print(f"[OK] Metadata '{m}' exists")
        
        # Find all transformer_blocks
        tb_indices = set()
        stb_indices = set()
        for k in keys:
            if k.startswith('transformer_blocks.'):
                idx = k.split('.')[1]
                if idx.isdigit():
                    tb_indices.add(int(idx))
            if k.startswith('single_transformer_blocks.'):
                idx = k.split('.')[1]
                if idx.isdigit():
                    stb_indices.add(int(idx))
        
        print(f"\nFound {len(tb_indices)} transformer_blocks, {len(stb_indices)} single_transformer_blocks")
        
        # Check transformer_blocks for qkv_proj_context
        for idx in sorted(tb_indices):
            prefix = f'transformer_blocks.{idx}.qkv_proj_context'
            has_qweight = f'{prefix}.qweight' in keys
            has_weight = f'{prefix}.weight' in keys
            
            if has_weight and not has_qweight:
                errors.append(f"{prefix}: Has .weight but missing .qweight (NOT QUANTIZED!)")
            elif has_qweight:
                # Check for other required quantization keys
                required_keys = ['wscales', 'smooth', 'lora_down', 'lora_up']
                missing = [k for k in required_keys if f'{prefix}.{k}' not in keys]
                if missing:
                    warnings.append(f"{prefix}: Missing keys: {missing}")
                elif verbose:
                    print(f"[OK] {prefix} is properly quantized")
            else:
                warnings.append(f"{prefix}: Neither .qweight nor .weight found")
        
        # Check qkv_proj (regular, not context)
        for idx in sorted(tb_indices):
            prefix = f'transformer_blocks.{idx}.qkv_proj'
            has_qweight = f'{prefix}.qweight' in keys
            has_weight = f'{prefix}.weight' in keys
            
            if has_weight and not has_qweight:
                errors.append(f"{prefix}: Has .weight but missing .qweight (NOT QUANTIZED!)")
            elif has_qweight and verbose:
                print(f"[OK] {prefix} is properly quantized")
        
        # Check single_transformer_blocks qkv_proj
        for idx in sorted(stb_indices):
            prefix = f'single_transformer_blocks.{idx}.qkv_proj'
            has_qweight = f'{prefix}.qweight' in keys
            has_weight = f'{prefix}.weight' in keys
            
            if has_weight and not has_qweight:
                errors.append(f"{prefix}: Has .weight but missing .qweight (NOT QUANTIZED!)")
    
    # Summary
    print("\n" + "=" * 60)
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  ❌ {e}")
    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  ⚠️ {w}")
    
    if not errors and not warnings:
        print("✅ All checks passed! Model is ready for ComfyUI.")
        return True
    elif not errors:
        print("⚠️ Model has warnings but should work.")
        return True
    else:
        print("❌ Model has errors and will likely fail in ComfyUI.")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_nunchaku_flux.py <model.safetensors>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    success = validate_flux_nunchaku_model(model_path)
    sys.exit(0 if success else 1)
