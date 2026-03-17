"""Fix KV cache layer name registration AND allocation count for PP workers.

Two issues:
1. kv_cache_tensor.shared_by has wrong layer names (from wrong PP stage)
2. KV cache count may not match local layer count (wrong config distributed)

Fix: After KV cache allocation, check local attention layers. If there are
more local layers than KV caches, allocate additional caches. Then re-register
all layer names from local attention modules.
"""

import os

PATH = "/workspace/tpu_inference/tpu_inference/runner/kv_cache_manager.py"

with open(PATH) as f:
    code = f.read()

old = '''        logger.info(
            f"Init kv-cache | "
            f"num_layers={len(kv_caches)} | "'''

new = '''        # Fix for PP: re-register KV cache with LOCAL attention layer names
        # and allocate extra caches if needed for mismatched PP stage sizes.
        try:
            from vllm.model_executor.layers.attention import Attention
            from vllm.config import get_layers_from_vllm_config
            local_layers = {}
            try:
                from tpu_inference.layers.vllm.mla_attention import MLAAttention
                for cls in [Attention, MLAAttention]:
                    local_layers.update(get_layers_from_vllm_config(
                        self.runner.vllm_config, cls))
            except Exception:
                local_layers.update(get_layers_from_vllm_config(
                    self.runner.vllm_config, Attention))
            # Remove shared KV cache layers
            for name in self.shared_kv_cache_layers:
                local_layers.pop(name, None)
            local_layer_names = sorted(local_layers.keys(),
                key=lambda n: int(''.join(c for c in n.split('layers.')[1].split('.')[0] if c.isdigit())) if 'layers.' in n else 0)

            if local_layer_names:
                # Allocate extra KV caches if local layers > existing caches
                while len(kv_caches) < len(local_layer_names):
                    # Clone the last KV cache's spec for the extra layer
                    last_spec = layer_name_to_spec.get(list(layer_name_to_spec.keys())[-1] if layer_name_to_spec else None)
                    if last_spec is None:
                        break
                    extra_kv = create_kv_caches(
                        num_blocks=num_blocks_list[-1],
                        block_size=last_spec.block_size,
                        num_kv_heads=last_spec.num_kv_heads,
                        head_size=head_size,
                        mesh=self.runner.mesh,
                        layer_names=[f'kv_cache_tensor.extra.{len(kv_caches)}'],
                        cache_dtype=t2j_dtype(last_spec.dtype),
                        use_mla=self.use_mla,
                    )[0]
                    kv_caches.append(extra_kv)
                    num_blocks_list.append(num_blocks_list[-1])
                    logger.info(f"PP KV cache fix: allocated extra cache #{len(kv_caches)-1}")

                # Re-register with local names
                n = min(len(local_layer_names), len(kv_caches))
                self.runner.layer_name_to_kvcache_index.clear()
                for i in range(n):
                    self.runner.layer_name_to_kvcache_index[local_layer_names[i]] = i
                logger.info(
                    f"PP KV cache fix: registered {n} local names "
                    f"(local={len(local_layer_names)}, caches={len(kv_caches)}): "
                    f"{local_layer_names[:2]}...{local_layer_names[-2:]}")
        except Exception as e:
            logger.warning(f"PP KV cache fix failed: {e}")

        logger.info(
            f"Init kv-cache | "
            f"num_layers={len(kv_caches)} | "'''

if old in code:
    code = code.replace(old, new)
    with open(PATH, "w") as f:
        f.write(code)
    print("PATCHED kv_cache_manager.py: re-register KV cache with local names + allocate extras")
else:
    print("SKIP: target not found")
