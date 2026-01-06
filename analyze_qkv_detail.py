import safetensors
import json

output_file = r'd:\nu\deepcompressor\deepcompressor\qkv_detailed_analysis.txt'
results = []

f = safetensors.safe_open(r'D:\nu\flux1-dev-r128-svdq-fp4.safetensors', framework='pt')
keys = sorted(f.keys())

# Check for keys related to transformer_blocks.0 context attention
t0_add_keys = [k for k in keys if k.startswith('transformer_blocks.0.') and 'add' in k.lower()]
results.append('=== transformer_blocks.0 add_* keys ===')
for k in sorted(t0_add_keys):
    results.append(k)
results.append(f'\nTotal add_* keys: {len(t0_add_keys)}')

# Check qkv_proj keys to compare
qkv_keys = [k for k in keys if 'transformer_blocks.0.' in k and 'qkv_proj' in k]
results.append('\n=== transformer_blocks.0 qkv_proj* keys ===')
for k in sorted(qkv_keys):
    results.append(k)

# Check if regular qkv_proj has qweight (quantized)
has_qkv_qweight = any('qkv_proj.qweight' in k for k in keys)
has_ctx_qweight = any('qkv_proj_context.qweight' in k for k in keys)
has_ctx_weight = any('qkv_proj_context.weight' in k for k in keys)

results.append(f'\n=== Summary ===')
results.append(f'qkv_proj.qweight exists: {has_qkv_qweight}')
results.append(f'qkv_proj_context.qweight exists: {has_ctx_qweight}')
results.append(f'qkv_proj_context.weight exists: {has_ctx_weight}')

# Check total transformer_blocks count
tb_count = len(set(k.split('.')[1] for k in keys if k.startswith('transformer_blocks.')))
results.append(f'Total transformer_blocks: {tb_count}')

# Write to file
with open(output_file, 'w', encoding='utf-8') as f_out:
    f_out.write('\n'.join(results))

print(f"Analysis saved to {output_file}")
