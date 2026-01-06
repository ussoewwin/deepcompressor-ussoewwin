import safetensors
import json

output_file = r'd:\nu\deepcompressor\qkv_result.txt'
results = []

f = safetensors.safe_open(r'D:\nu\flux1-dev-r128-svdq-fp4.safetensors', framework='pt')
keys = sorted(f.keys())

# Check qkv_proj_context in user model
user_qkv = [k for k in keys if 'qkv_proj_context' in k]

# Load official model
f_official = safetensors.safe_open(r'D:\USERFILES\ComfyUI\ComfyUI\models\unet\svdq-fp4_r32-flux.1-dev.safetensors', framework='pt')
official_keys = sorted(f_official.keys())
official_qkv = [k for k in official_keys if 'qkv_proj_context' in k and 'transformer_blocks.0' in k]

results.append("=== User Model: transformer_blocks.0.qkv_proj_context ===")
user_t0 = [k for k in user_qkv if 'transformer_blocks.0' in k]
for k in sorted(user_t0):
    results.append(f"  {k}")

results.append("\n=== Official Model: transformer_blocks.0.qkv_proj_context ===")
for k in sorted(official_qkv):
    results.append(f"  {k}")

results.append("\n=== Key differences ===")
user_suffixes = set(k.replace('transformer_blocks.0.qkv_proj_context.', '') for k in user_t0)
official_suffixes = set(k.replace('transformer_blocks.0.qkv_proj_context.', '') for k in official_qkv)

missing_in_user = official_suffixes - user_suffixes
extra_in_user = user_suffixes - official_suffixes

results.append(f"Missing in user (need these for Nunchaku): {sorted(missing_in_user)}")
results.append(f"Extra in user (not expected): {sorted(extra_in_user)}")

# Write to file
with open(output_file, 'w', encoding='utf-8') as f_out:
    f_out.write('\n'.join(results))

print(f"Analysis saved to {output_file}")
