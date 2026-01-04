import gc
import json
import os
import pprint
import traceback

import safetensors
import safetensors.torch
import torch
from diffusers import DiffusionPipeline

import inspect

from deepcompressor.app.llm.nn.patch import patch_attention, patch_gemma_rms_norm
from deepcompressor.app.llm.ptq import ptq as llm_ptq
from deepcompressor.backend.nunchaku.convert import (
    convert_to_nunchaku_flux_state_dicts,
    convert_to_nunchaku_w4x4y16_linear_state_dict,
)
from deepcompressor.calib.config import SkipBasedSmoothCalibConfig, SmoothTransfomerConfig
from deepcompressor.utils import tools

from .config import DiffusionPtqCacheConfig, DiffusionPtqRunConfig, DiffusionQuantCacheConfig, DiffusionQuantConfig
from .nn.struct import DiffusionModelStruct
from .quant import (
    load_diffusion_weights_state_dict,
    quantize_diffusion_activations,
    quantize_diffusion_weights,
    rotate_diffusion,
    smooth_diffusion,
)

__all__ = ["ptq"]


def _build_flux_comfy_config(*, transformer_cfg: dict[str, object]) -> str:
    """
    Build ComfyUI-style `comfy_config` JSON for FLUX models from the (Diffusers) transformer config.

    This is used only for `--export-nunchaku-flux` to better match "official" Nunchaku metadata
    and mitigate noisy outputs caused by missing/incorrect runtime config.
    """

    def _get_int(k: str, default: int | None = None) -> int | None:
        v = transformer_cfg.get(k, default)
        try:
            return int(v) if v is not None else None
        except Exception:
            return default

    def _get_bool(k: str, default: bool | None = None) -> bool | None:
        v = transformer_cfg.get(k, default)
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            if v.lower() in ("true", "1", "yes", "y"):
                return True
            if v.lower() in ("false", "0", "no", "n"):
                return False
        return default

    head_dim = _get_int("attention_head_dim", 128) or 128
    num_heads = _get_int("num_attention_heads", 24) or 24
    hidden_size = head_dim * num_heads

    # Diffusers FluxTransformer2DModel uses in_channels=64, patch_size=1,
    # while ComfyUI expects in/out_channels=16, patch_size=2.
    in_channels = _get_int("in_channels", 64) or 64
    out_channels = in_channels
    if in_channels % 4 == 0:
        in_channels //= 4
        out_channels //= 4
    patch_size = _get_int("patch_size", 1) or 1
    patch_size *= 2

    # Prefer explicit values if present; otherwise match the known Flux.1-dev defaults.
    axes_dim = transformer_cfg.get("axes_dim") or transformer_cfg.get("axes_dims") or [16, 56, 56]
    if not (isinstance(axes_dim, (list, tuple)) and len(axes_dim) == 3):
        axes_dim = [16, 56, 56]
    axes_dim = [int(x) for x in axes_dim]

    model_cfg: dict[str, object] = {
        "axes_dim": axes_dim,
        "context_in_dim": _get_int("joint_attention_dim", 4096) or 4096,
        "depth": _get_int("num_layers", 19) or 19,
        "depth_single_blocks": _get_int("num_single_layers", 38) or 38,
        "disable_unet_model_creation": True,
        "guidance_embed": bool(_get_bool("guidance_embeds", True)),
        "hidden_size": hidden_size,
        "image_model": "flux",
        "in_channels": in_channels,
        "mlp_ratio": float(transformer_cfg.get("mlp_ratio", 4.0) or 4.0),
        "num_heads": num_heads,
        "out_channels": out_channels,
        "patch_size": patch_size,
        "qkv_bias": bool(transformer_cfg.get("qkv_bias", True)),
        "theta": int(transformer_cfg.get("theta", 10000) or 10000),
        "vec_in_dim": _get_int("pooled_projection_dim", 768) or 768,
    }
    comfy_obj = {"model_class": "Flux", "model_config": model_cfg}
    return json.dumps(comfy_obj, indent=2, ensure_ascii=False)


def _load_safetensors_state_dict(path: str) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    with safetensors.safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
    return state_dict


def _looks_like_comfyui_or_sd_ckpt(state_dict: dict[str, torch.Tensor]) -> bool:
    # Typical original SD/ComfyUI checkpoints include keys like:
    #   model.diffusion_model.*
    # while Diffusers UNet state dict uses down_blocks/mid_block/up_blocks etc.
    for k in state_dict.keys():
        if k.startswith("model.diffusion_model."):
            return True
    return False


def _convert_single_file_to_diffusers_unet_state_dict(
    *, ckpt_path: str, torch_dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """
    Best-effort conversion for "single-file" SDXL checkpoints into a Diffusers UNet state_dict.

    This relies on Diffusers' `from_single_file` conversion logic when available.
    """
    # NOTE: We intentionally avoid importing StableDiffusionXLPipeline at module import time,
    # because diffusers versions vary across environments.
    try:
        from diffusers import StableDiffusionXLPipeline  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Your diffusers does not provide StableDiffusionXLPipeline; cannot convert a ComfyUI/original SDXL checkpoint "
            "to a Diffusers UNet state_dict. Please update diffusers or provide a Diffusers-format UNet safetensors."
        ) from e

    if not hasattr(StableDiffusionXLPipeline, "from_single_file"):
        raise RuntimeError(
            "Your diffusers version does not support StableDiffusionXLPipeline.from_single_file; "
            "cannot convert a ComfyUI/original SDXL checkpoint. Please update diffusers."
        )

    # Many diffusers versions accept slightly different kwargs; build kwargs dynamically.
    kwargs: dict[str, object] = {"torch_dtype": torch_dtype}
    try:
        sig = inspect.signature(StableDiffusionXLPipeline.from_single_file)
        if "use_safetensors" in sig.parameters:
            kwargs["use_safetensors"] = True
        if "safety_checker" in sig.parameters:
            kwargs["safety_checker"] = None
        if "feature_extractor" in sig.parameters:
            kwargs["feature_extractor"] = None
        if "requires_safety_checker" in sig.parameters:
            kwargs["requires_safety_checker"] = False
    except Exception:
        # If signature inspection fails, just try minimal kwargs.
        pass

    pipe = StableDiffusionXLPipeline.from_single_file(ckpt_path, **kwargs)  # type: ignore[arg-type]
    try:
        # Ensure CPU tensors for downstream concatenation / save_file.
        return {k: v.detach().cpu() for k, v in pipe.unet.state_dict().items()}
    finally:
        try:
            del pipe
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()


def _sdxl_export_build_metadata(*, unet, rank: int, precision: str) -> dict[str, str]:
    # NunchakuModelLoaderMixin expects metadata["config"] to be json.
    unet_cfg = getattr(unet, "config", None)
    if hasattr(unet_cfg, "to_dict"):
        unet_cfg = unet_cfg.to_dict()
    if not isinstance(unet_cfg, dict):
        raise RuntimeError("Failed to read UNet config for Nunchaku export (unet.config is not dict-like).")
    return {
        "config": json.dumps(unet_cfg),
        "quantization_config": json.dumps({"rank": int(rank), "precision": str(precision)}),
    }


def _sdxl_export_to_nunchaku_single_safetensors(
    *,
    output_path: str,
    orig_unet_path: str,
    dequant_unet_state: dict[str, torch.Tensor],
    scale_state: dict[str, torch.Tensor],
    branch_state: dict[str, dict[str, torch.Tensor]] | None,
    rank: int,
    precision: str,
    torch_dtype: torch.dtype,
    unet,
) -> None:  # noqa: C901
    """
    Export a single safetensors file that can be loaded by:
      nunchaku.models.unets.unet_sdxl.NunchakuSDXLUNet2DConditionModel.from_pretrained(...)
    """
    logger = tools.logging.getLogger(__name__)
    assert orig_unet_path, "orig_unet_path is required for SDXL Nunchaku export."
    assert os.path.exists(orig_unet_path), f"orig_unet_path does not exist: {orig_unet_path}"

    # Load float UNet weights:
    # - If user passed a ComfyUI/original SDXL checkpoint, convert it to Diffusers UNet state_dict.
    # - Otherwise, treat it as a Diffusers-format UNet safetensors and filter to UNet keys.
    orig_state_raw = _load_safetensors_state_dict(orig_unet_path)
    if _looks_like_comfyui_or_sd_ckpt(orig_state_raw):
        orig_state = _convert_single_file_to_diffusers_unet_state_dict(ckpt_path=orig_unet_path, torch_dtype=torch_dtype)
    else:
        # Keep only UNet keys (user might pass a file that contains extra tensors).
        expected = set(unet.state_dict().keys()) if hasattr(unet, "state_dict") else None
        if expected:
            orig_state = {k: v for k, v in orig_state_raw.items() if k in expected}
        else:
            orig_state = orig_state_raw
    out_state: dict[str, torch.Tensor] = dict(orig_state)  # start from float UNet weights

    # Helpers
    def _del(k: str) -> None:
        if k in out_state:
            del out_state[k]

    def _get_scale(module_name: str) -> tuple[torch.Tensor, torch.Tensor | None]:
        s0 = scale_state.get(f"{module_name}.weight.scale.0", None)
        if s0 is None:
            raise KeyError(f"Missing scale for {module_name}: {module_name}.weight.scale.0")
        if not isinstance(s0, torch.Tensor):
            # scale_state_dict can store Python floats for some quantizers.
            s0 = torch.tensor([float(s0)], dtype=torch_dtype, device="cpu")
        s1 = scale_state.get(f"{module_name}.weight.scale.1", None)
        if s1 is not None and not isinstance(s1, torch.Tensor):
            s1 = torch.tensor([float(s1)], dtype=torch_dtype, device="cpu")
        return s0, s1

    def _get_branch(module_name: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not branch_state:
            return None
        b = branch_state.get(module_name, None)
        if not b:
            return None
        if "a.weight" not in b or "b.weight" not in b:
            return None
        return b["a.weight"], b["b.weight"]

    def _ensure_wcscales(prefix: str, converted: dict[str, torch.Tensor], out_features: int) -> None:
        # SVDQW4A4Linear(nvfp4) always has .wcscales parameter; SDXL loader does not patch missing keys.
        if precision == "nvfp4" and "wcscales" not in converted:
            converted["wcscales"] = torch.ones(out_features, dtype=torch_dtype, device="cpu")

    float_point = precision == "nvfp4"

    # Collect transformer block prefixes (Diffusers SDXL UNet naming).
    block_prefixes = sorted(
        {k.split(".attn1.to_q.weight")[0] for k in orig_state.keys() if k.endswith(".attn1.to_q.weight")}
    )
    logger.info(f"* Exporting Nunchaku SDXL UNet safetensors: found {len(block_prefixes)} transformer blocks")

    for block in block_prefixes:
        # --- attn1: self-attention, Nunchaku fuses qkv into to_qkv ---
        q = f"{block}.attn1.to_q"
        k = f"{block}.attn1.to_k"
        v = f"{block}.attn1.to_v"
        q_w = orig_state[f"{q}.weight"]
        k_w = orig_state[f"{k}.weight"]
        v_w = orig_state[f"{v}.weight"]
        q_b = orig_state.get(f"{q}.bias", None)
        k_b = orig_state.get(f"{k}.bias", None)
        v_b = orig_state.get(f"{v}.bias", None)
        fused_w = torch.cat([q_w, k_w, v_w], dim=0)
        fused_b = None
        if q_b is not None and k_b is not None and v_b is not None:
            fused_b = torch.cat([q_b, k_b, v_b], dim=0)

        # Dequant (float) weights from DeepCompressor run, for residual SVD (to build fused low-rank branch)
        dq_q_w = dequant_unet_state[f"{q}.weight"]
        dq_k_w = dequant_unet_state[f"{k}.weight"]
        dq_v_w = dequant_unet_state[f"{v}.weight"]
        dq_fused_w = torch.cat([dq_q_w, dq_k_w, dq_v_w], dim=0)
        residual = (fused_w.to(dtype=torch.float32) - dq_fused_w.to(dtype=torch.float32)).to(dtype=torch.float16)
        # SVD rank-128 (R128) low-rank branch: residual ≈ (U*S) @ V^T
        # LowRankBranch stores a.weight=V^T[:r] (rank, in_features), b.weight=U[:,:r]*S[:r] (out, rank)
        u, s, vh = torch.linalg.svd(residual.double())
        b_w = (u[:, :rank] * s[:rank]).to(dtype=torch_dtype, device="cpu")
        a_w = vh[:rank].to(dtype=torch_dtype, device="cpu")

        q_s0, q_s1 = _get_scale(q)
        k_s0, k_s1 = _get_scale(k)
        v_s0, v_s1 = _get_scale(v)
        # When q/k/v are stored with per-tensor scale (numel()==1), fusing requires
        # switching to per-channel scale, same as deepcompressor.backend.nunchaku.convert.py
        if q_s0.numel() == 1:
            if not (k_s0.numel() == 1 and v_s0.numel() == 1):
                raise AssertionError("Inconsistent per-tensor scales across q/k/v (scale.0).")
            fused_s0 = torch.cat(
                [
                    q_s0.view(-1).expand(q_w.shape[0]).reshape(q_w.shape[0], 1, 1, 1),
                    k_s0.view(-1).expand(k_w.shape[0]).reshape(k_w.shape[0], 1, 1, 1),
                    v_s0.view(-1).expand(v_w.shape[0]).reshape(v_w.shape[0], 1, 1, 1),
                ],
                dim=0,
            )
        else:
            fused_s0 = torch.cat([q_s0, k_s0, v_s0], dim=0)
        fused_s1 = None
        if q_s1 is not None or k_s1 is not None or v_s1 is not None:
            if q_s1 is None or k_s1 is None or v_s1 is None:
                raise KeyError(f"Missing subscale for fused qkv in {block} (scale.1 inconsistent).")
            if q_s1.numel() == 1:
                if not (k_s1.numel() == 1 and v_s1.numel() == 1):
                    raise AssertionError("Inconsistent per-tensor scales across q/k/v (scale.1).")
                fused_s1 = torch.cat(
                    [
                        q_s1.view(-1).expand(q_w.shape[0]).reshape(q_w.shape[0], 1, 1, 1),
                        k_s1.view(-1).expand(k_w.shape[0]).reshape(k_w.shape[0], 1, 1, 1),
                        v_s1.view(-1).expand(v_w.shape[0]).reshape(v_w.shape[0], 1, 1, 1),
                    ],
                    dim=0,
                )
            else:
                fused_s1 = torch.cat([q_s1, k_s1, v_s1], dim=0)

        converted = convert_to_nunchaku_w4x4y16_linear_state_dict(
            weight=fused_w.to(dtype=torch_dtype, device="cpu"),
            scale=fused_s0.to(device="cpu"),
            bias=fused_b.to(dtype=torch_dtype, device="cpu") if fused_b is not None else None,
            smooth=None,
            lora=(a_w, b_w),
            float_point=float_point,
            subscale=fused_s1.to(device="cpu") if fused_s1 is not None else None,
        )
        _ensure_wcscales(f"{block}.attn1.to_qkv", converted, fused_w.shape[0])

        # Remove original to_q/to_k/to_v params (will not exist in Nunchaku patched model)
        for base in (q, k, v):
            _del(f"{base}.weight")
            _del(f"{base}.bias")

        # Write fused to_qkv params
        fused_prefix = f"{block}.attn1.to_qkv"
        for kk, vv in converted.items():
            out_state[f"{fused_prefix}.{kk}"] = vv

        # --- attn1: to_out.0 quantized ---
        out0 = f"{block}.attn1.to_out.0"
        out0_w = orig_state[f"{out0}.weight"]
        out0_b = orig_state.get(f"{out0}.bias", None)
        out0_s0, out0_s1 = _get_scale(out0)
        out0_branch = _get_branch(out0)
        converted = convert_to_nunchaku_w4x4y16_linear_state_dict(
            weight=out0_w.to(dtype=torch_dtype, device="cpu"),
            scale=out0_s0.to(device="cpu"),
            bias=out0_b.to(dtype=torch_dtype, device="cpu") if out0_b is not None else None,
            smooth=None,
            lora=out0_branch,
            float_point=float_point,
            subscale=out0_s1.to(device="cpu") if out0_s1 is not None else None,
        )
        _ensure_wcscales(out0, converted, out0_w.shape[0])
        _del(f"{out0}.weight")
        _del(f"{out0}.bias")
        for kk, vv in converted.items():
            out_state[f"{out0}.{kk}"] = vv

        # --- attn2: cross-attention: Nunchaku quantizes to_q only; to_k/to_v stay float ---
        ca_q = f"{block}.attn2.to_q"
        ca_q_w = orig_state[f"{ca_q}.weight"]
        ca_q_b = orig_state.get(f"{ca_q}.bias", None)
        ca_q_s0, ca_q_s1 = _get_scale(ca_q)
        ca_q_branch = _get_branch(ca_q)
        converted = convert_to_nunchaku_w4x4y16_linear_state_dict(
            weight=ca_q_w.to(dtype=torch_dtype, device="cpu"),
            scale=ca_q_s0.to(device="cpu"),
            bias=ca_q_b.to(dtype=torch_dtype, device="cpu") if ca_q_b is not None else None,
            smooth=None,
            lora=ca_q_branch,
            float_point=float_point,
            subscale=ca_q_s1.to(device="cpu") if ca_q_s1 is not None else None,
        )
        _ensure_wcscales(ca_q, converted, ca_q_w.shape[0])
        _del(f"{ca_q}.weight")
        _del(f"{ca_q}.bias")
        for kk, vv in converted.items():
            out_state[f"{ca_q}.{kk}"] = vv

        ca_out0 = f"{block}.attn2.to_out.0"
        ca_out0_w = orig_state[f"{ca_out0}.weight"]
        ca_out0_b = orig_state.get(f"{ca_out0}.bias", None)
        ca_out0_s0, ca_out0_s1 = _get_scale(ca_out0)
        ca_out0_branch = _get_branch(ca_out0)
        converted = convert_to_nunchaku_w4x4y16_linear_state_dict(
            weight=ca_out0_w.to(dtype=torch_dtype, device="cpu"),
            scale=ca_out0_s0.to(device="cpu"),
            bias=ca_out0_b.to(dtype=torch_dtype, device="cpu") if ca_out0_b is not None else None,
            smooth=None,
            lora=ca_out0_branch,
            float_point=float_point,
            subscale=ca_out0_s1.to(device="cpu") if ca_out0_s1 is not None else None,
        )
        _ensure_wcscales(ca_out0, converted, ca_out0_w.shape[0])
        _del(f"{ca_out0}.weight")
        _del(f"{ca_out0}.bias")
        for kk, vv in converted.items():
            out_state[f"{ca_out0}.{kk}"] = vv

        # --- ff: quantize typical SDXL FeedForward linears ---
        ff_fc1 = f"{block}.ff.net.0.proj"
        if f"{ff_fc1}.weight" in orig_state:
            ff_fc1_w = orig_state[f"{ff_fc1}.weight"]
            ff_fc1_b = orig_state.get(f"{ff_fc1}.bias", None)
            ff_fc1_s0, ff_fc1_s1 = _get_scale(ff_fc1)
            ff_fc1_branch = _get_branch(ff_fc1)
            converted = convert_to_nunchaku_w4x4y16_linear_state_dict(
                weight=ff_fc1_w.to(dtype=torch_dtype, device="cpu"),
                scale=ff_fc1_s0.to(device="cpu"),
                bias=ff_fc1_b.to(dtype=torch_dtype, device="cpu") if ff_fc1_b is not None else None,
                smooth=None,
                lora=ff_fc1_branch,
                float_point=float_point,
                subscale=ff_fc1_s1.to(device="cpu") if ff_fc1_s1 is not None else None,
            )
            _ensure_wcscales(ff_fc1, converted, ff_fc1_w.shape[0])
            _del(f"{ff_fc1}.weight")
            _del(f"{ff_fc1}.bias")
            for kk, vv in converted.items():
                out_state[f"{ff_fc1}.{kk}"] = vv

        ff_fc2 = f"{block}.ff.net.2"
        if f"{ff_fc2}.weight" in orig_state:
            ff_fc2_w = orig_state[f"{ff_fc2}.weight"]
            ff_fc2_b = orig_state.get(f"{ff_fc2}.bias", None)
            ff_fc2_s0, ff_fc2_s1 = _get_scale(ff_fc2)
            ff_fc2_branch = _get_branch(ff_fc2)
            converted = convert_to_nunchaku_w4x4y16_linear_state_dict(
                weight=ff_fc2_w.to(dtype=torch_dtype, device="cpu"),
                scale=ff_fc2_s0.to(device="cpu"),
                bias=ff_fc2_b.to(dtype=torch_dtype, device="cpu") if ff_fc2_b is not None else None,
                smooth=None,
                lora=ff_fc2_branch,
                float_point=float_point,
                subscale=ff_fc2_s1.to(device="cpu") if ff_fc2_s1 is not None else None,
            )
            _ensure_wcscales(ff_fc2, converted, ff_fc2_w.shape[0])
            _del(f"{ff_fc2}.weight")
            _del(f"{ff_fc2}.bias")
            for kk, vv in converted.items():
                out_state[f"{ff_fc2}.{kk}"] = vv

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    metadata = _sdxl_export_build_metadata(unet=unet, rank=rank, precision=precision)
    logger.info(f"* Saving Nunchaku SDXL UNet safetensors to {output_path}")
    safetensors.torch.save_file(out_state, output_path, metadata=metadata)


def _flux_export_to_nunchaku_single_safetensors(
    *,
    output_path: str,
    dequant_state: dict[str, torch.Tensor],
    scale_state: dict[str, torch.Tensor],
    smooth_state: dict[str, torch.Tensor] | None,
    branch_state: dict[str, dict[str, torch.Tensor]] | None,
    transformer,
    float_point: bool = False,
    rank: int | None = None,
    quantization_config: dict[str, object] | None = None,
    comfy_config: str | None = None,
) -> None:
    """
    Export a single safetensors file that can be loaded by:
      nunchaku.models.transformers.transformer_flux.NunchakuFluxTransformer2dModel.from_pretrained(...)
    """
    logger = tools.logging.getLogger(__name__)
    logger.info("* Exporting Nunchaku FLUX.1-dev transformer safetensors")

    # Convert to Nunchaku format using the existing conversion function
    converted_state_dict, other_state_dict = convert_to_nunchaku_flux_state_dicts(
        state_dict=dequant_state,
        scale_dict=scale_state,
        smooth_dict=smooth_state or {},
        branch_dict=branch_state or {},
        float_point=float_point,
    )

    # Merge converted and other state dicts
    out_state: dict[str, torch.Tensor] = {}
    out_state.update(converted_state_dict)
    out_state.update(other_state_dict)

    # Validate output for ComfyUI-nunchaku compatibility BEFORE writing:
    # - For each quantized linear (qweight), ensure low-rank branch shapes are consistent
    #   with Nunchaku runtime expectations.
    #   out_dim = qweight.shape[0]
    #   in_dim  = qweight.shape[1] * 2   (packed into int8)
    for k, qw in out_state.items():
        if not isinstance(qw, torch.Tensor) or qw.ndim != 2:
            continue
        if not k.endswith(".qweight"):
            continue
        base = k[: -len(".qweight")]
        out_dim = int(qw.shape[0])
        in_dim = int(qw.shape[1]) * 2
        dk = base + ".lora_down"
        uk = base + ".lora_up"
        if dk in out_state:
            d = out_state[dk]
            if not (isinstance(d, torch.Tensor) and d.ndim == 2 and int(d.shape[0]) == in_dim):
                raise RuntimeError(f"Invalid low-rank shape: {dk}={getattr(d,'shape',None)} expected [in_dim={in_dim}, rank]")
        if uk in out_state:
            u = out_state[uk]
            if not (isinstance(u, torch.Tensor) and u.ndim == 2 and int(u.shape[0]) == out_dim):
                raise RuntimeError(f"Invalid low-rank shape: {uk}={getattr(u,'shape',None)} expected [out_dim={out_dim}, rank]")

    # Build metadata
    transformer_cfg = getattr(transformer, "config", None)
    if hasattr(transformer_cfg, "to_dict"):
        transformer_cfg = transformer_cfg.to_dict()
    if not isinstance(transformer_cfg, dict):
        raise RuntimeError("Failed to read transformer config for Nunchaku export (transformer.config is not dict-like).")
    # Build metadata.
    #
    # Align with "official" FLUX safetensors metadata layout:
    # - model_class: loader hint for ComfyUI/Nunchaku
    # - comfy_config: ComfyUI-side model config JSON (optional)
    # - config: diffusers model config JSON
    # - quantization_config: include both weight and activation sections when available
    qcfg = quantization_config
    if qcfg is None:
        # Fallback (backward compatible) if caller didn't provide full quantization_config.
        qcfg = {
            "method": "svdquant",
            "weight": {
                "dtype": "fp4_e2m1_all" if float_point else "int4",
                "group_size": 16 if float_point else 64,
            },
        }
        if rank is not None:
            qcfg["rank"] = int(rank)
    metadata: dict[str, str] = {
        "model_class": "NunchakuFluxTransformer2dModel",
        "config": json.dumps(transformer_cfg),
        "quantization_config": json.dumps(qcfg),
    }
    # Optional comfy_config passthrough (only when explicitly provided/available).
    if comfy_config:
        metadata["comfy_config"] = comfy_config

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    logger.info(f"* Saving Nunchaku FLUX.1-dev transformer safetensors to {output_path}")
    safetensors.torch.save_file(out_state, output_path, metadata=metadata)


def ptq(  # noqa: C901
    model: DiffusionModelStruct,
    config: DiffusionQuantConfig,
    cache: DiffusionPtqCacheConfig | None = None,
    load_dirpath: str = "",
    save_dirpath: str = "",
    copy_on_save: bool = False,
    save_model: bool = False,
    export_nunchaku_sdxl: dict | None = None,
    export_nunchaku_flux: dict | None = None,
) -> DiffusionModelStruct:
    """Post-training quantization of a diffusion model.

    Args:
        model (`DiffusionModelStruct`):
            The diffusion model.
        config (`DiffusionQuantConfig`):
            The diffusion model post-training quantization configuration.
        cache (`DiffusionPtqCacheConfig`, *optional*, defaults to `None`):
            The diffusion model quantization cache path configuration.
        load_dirpath (`str`, *optional*, defaults to `""`):
            The directory path to load the quantization checkpoint.
        save_dirpath (`str`, *optional*, defaults to `""`):
            The directory path to save the quantization checkpoint.
        copy_on_save (`bool`, *optional*, defaults to `False`):
            Whether to copy the cache to the save directory.
        save_model (`bool`, *optional*, defaults to `False`):
            Whether to save the quantized model checkpoint.

    Returns:
        `DiffusionModelStruct`:
            The quantized diffusion model.
    """
    logger = tools.logging.getLogger(__name__)
    if not isinstance(model, DiffusionModelStruct):
        model = DiffusionModelStruct.construct(model)
    assert isinstance(model, DiffusionModelStruct)

    quant_wgts = config.enabled_wgts
    quant_ipts = config.enabled_ipts
    quant_opts = config.enabled_opts
    quant_acts = quant_ipts or quant_opts
    quant = quant_wgts or quant_acts

    load_model_path, load_path, save_path = "", None, None
    if load_dirpath:
        load_path = DiffusionQuantCacheConfig(
            smooth=os.path.join(load_dirpath, "smooth.pt"),
            branch=os.path.join(load_dirpath, "branch.pt"),
            wgts=os.path.join(load_dirpath, "wgts.pt"),
            acts=os.path.join(load_dirpath, "acts.pt"),
        )
        load_model_path = os.path.join(load_dirpath, "model.pt")
        if os.path.exists(load_model_path):
            if config.enabled_wgts and config.wgts.enabled_low_rank:
                if os.path.exists(load_path.branch):
                    load_model = True
                else:
                    logger.warning(f"Model low-rank branch checkpoint {load_path.branch} does not exist")
                    load_model = False
            else:
                load_model = True
            if load_model:
                logger.info(f"* Loading model from {load_model_path}")
                save_dirpath = ""  # do not save the model if loading
        else:
            logger.warning(f"Model checkpoint {load_model_path} does not exist")
            load_model = False
    else:
        load_model = False
    if save_dirpath:
        os.makedirs(save_dirpath, exist_ok=True)
        save_path = DiffusionQuantCacheConfig(
            smooth=os.path.join(save_dirpath, "smooth.pt"),
            branch=os.path.join(save_dirpath, "branch.pt"),
            wgts=os.path.join(save_dirpath, "wgts.pt"),
            acts=os.path.join(save_dirpath, "acts.pt"),
        )
    else:
        save_model = False

    if quant and config.enabled_rotation:
        logger.info("* Rotating model for quantization")
        tools.logging.Formatter.indent_inc()
        rotate_diffusion(model, config=config)
        tools.logging.Formatter.indent_dec()
        gc.collect()
        torch.cuda.empty_cache()

    # region smooth quantization
    # Keep smooth state in-memory for export runs (so we can include `smooth`/`smooth_orig`
    # in the final safetensors without writing intermediate *.pt files).
    smooth_cache_for_export: dict[str, torch.Tensor] | None = None
    if quant and config.enabled_smooth:
        logger.info("* Smoothing model for quantization")
        tools.logging.Formatter.indent_inc()
        # FLUX smoothing is memory-hungry. For FLUX Nunchaku export runs, reduce calib batch_size
        # during smooth generation to avoid CUDA OOM, then restore.
        _orig_calib_bs: int | None = None
        if export_nunchaku_flux and hasattr(config, "calib") and hasattr(config.calib, "batch_size"):
            try:
                _orig_calib_bs = int(config.calib.batch_size)
            except Exception:
                _orig_calib_bs = None
            if _orig_calib_bs and _orig_calib_bs > 1:
                config.calib.batch_size = 1
                logger.info(
                    "- FLUX export: forcing calib.batch_size=%s -> %s for smoothing (OOM mitigation)",
                    _orig_calib_bs,
                    config.calib.batch_size,
                )
        load_from = ""
        if load_path and os.path.exists(load_path.smooth):
            load_from = load_path.smooth
        elif cache and cache.path.smooth and os.path.exists(cache.path.smooth):
            load_from = cache.path.smooth
        if load_from:
            logger.info(f"- Loading smooth scales from {load_from}")
            smooth_cache = torch.load(load_from)
            smooth_diffusion(model, config, smooth_cache=smooth_cache)
        else:
            logger.info("- Generating smooth scales")
            smooth_cache = smooth_diffusion(model, config)
            if cache and cache.path.smooth:
                logger.info(f"- Saving smooth scales to {cache.path.smooth}")
                os.makedirs(cache.dirpath.smooth, exist_ok=True)
                torch.save(smooth_cache, cache.path.smooth)
                load_from = cache.path.smooth
        # Preserve a CPU copy for exporters (FLUX/SDXL) regardless of cache settings.
        try:
            smooth_cache_for_export = {k: v.detach().cpu() for k, v in smooth_cache.items()}
        except Exception:
            smooth_cache_for_export = None
        if save_path:
            if not copy_on_save and load_from:
                logger.info(f"- Linking smooth scales to {save_path.smooth}")
                os.symlink(os.path.relpath(load_from, save_dirpath), save_path.smooth)
            else:
                logger.info(f"- Saving smooth scales to {save_path.smooth}")
                torch.save(smooth_cache, save_path.smooth)
        del smooth_cache
        # Restore calib batch size after smoothing
        if _orig_calib_bs is not None and hasattr(config, "calib") and hasattr(config.calib, "batch_size"):
            try:
                config.calib.batch_size = _orig_calib_bs
            except Exception:
                pass
        tools.logging.Formatter.indent_dec()
        gc.collect()
        torch.cuda.empty_cache()
    # endregion
    # region collect original state dict
    if config.needs_acts_quantizer_cache:
        if load_path and os.path.exists(load_path.acts):
            orig_state_dict = None
        elif cache and cache.path.acts and os.path.exists(cache.path.acts):
            orig_state_dict = None
        else:
            orig_state_dict: dict[str, torch.Tensor] = {
                name: param.detach().clone() for name, param in model.module.named_parameters() if param.ndim > 1
            }
    else:
        orig_state_dict = None
    # endregion
    if load_model:
        logger.info(f"* Loading model checkpoint from {load_model_path}")
        load_diffusion_weights_state_dict(
            model,
            config,
            state_dict=torch.load(load_model_path),
            branch_state_dict=torch.load(load_path.branch) if os.path.exists(load_path.branch) else None,
        )
        gc.collect()
        torch.cuda.empty_cache()
    elif quant_wgts:
        logger.info("* Quantizing weights")
        tools.logging.Formatter.indent_inc()
        quantizer_state_dict, quantizer_load_from = None, ""
        if load_path and os.path.exists(load_path.wgts):
            quantizer_load_from = load_path.wgts
        elif cache and cache.path.wgts and os.path.exists(cache.path.wgts):
            quantizer_load_from = cache.path.wgts
        if quantizer_load_from:
            logger.info(f"- Loading weight settings from {quantizer_load_from}")
            quantizer_state_dict = torch.load(quantizer_load_from)
        branch_state_dict, branch_load_from = None, ""
        if load_path and os.path.exists(load_path.branch):
            branch_load_from = load_path.branch
        elif cache and cache.path.branch and os.path.exists(cache.path.branch):
            branch_load_from = cache.path.branch
        if branch_load_from:
            logger.info(f"- Loading branch settings from {branch_load_from}")
            branch_state_dict = torch.load(branch_load_from)
        if not quantizer_load_from:
            logger.info("- Generating weight settings")
        if not branch_load_from:
            logger.info("- Generating branch settings")
        quantizer_state_dict, branch_state_dict, scale_state_dict = quantize_diffusion_weights(
            model,
            config,
            quantizer_state_dict=quantizer_state_dict,
            branch_state_dict=branch_state_dict,
            return_with_scale_state_dict=bool(save_dirpath) or bool(save_model) or bool(export_nunchaku_sdxl) or bool(export_nunchaku_flux),
        )
        if not quantizer_load_from and cache and cache.dirpath.wgts:
            logger.info(f"- Saving weight settings to {cache.path.wgts}")
            os.makedirs(cache.dirpath.wgts, exist_ok=True)
            torch.save(quantizer_state_dict, cache.path.wgts)
            quantizer_load_from = cache.path.wgts
        if not branch_load_from and cache and cache.dirpath.branch:
            logger.info(f"- Saving branch settings to {cache.path.branch}")
            os.makedirs(cache.dirpath.branch, exist_ok=True)
            torch.save(branch_state_dict, cache.path.branch)
            branch_load_from = cache.path.branch
        if save_path:
            if not copy_on_save and quantizer_load_from:
                logger.info(f"- Linking weight settings to {save_path.wgts}")
                os.symlink(os.path.relpath(quantizer_load_from, save_dirpath), save_path.wgts)
            else:
                logger.info(f"- Saving weight settings to {save_path.wgts}")
                torch.save(quantizer_state_dict, save_path.wgts)
            if not copy_on_save and branch_load_from:
                logger.info(f"- Linking branch settings to {save_path.branch}")
                os.symlink(os.path.relpath(branch_load_from, save_dirpath), save_path.branch)
            else:
                logger.info(f"- Saving branch settings to {save_path.branch}")
                torch.save(branch_state_dict, save_path.branch)
        if save_model:
            logger.info(f"- Saving model to {save_dirpath}")
            torch.save(scale_state_dict, os.path.join(save_dirpath, "scale.pt"))
            torch.save(model.module.state_dict(), os.path.join(save_dirpath, "model.pt"))
        if export_nunchaku_sdxl:
            # Export final Nunchaku SDXL UNet checkpoint (single-file safetensors).
            # This is the intended deliverable for ComfyUI/Nunchaku usage.
            export_path = export_nunchaku_sdxl["output_path"]
            orig_unet_path = export_nunchaku_sdxl["orig_unet_path"]
            unet = export_nunchaku_sdxl["unet"]
            rank = int(export_nunchaku_sdxl["rank"])
            precision = export_nunchaku_sdxl["precision"]
            torch_dtype = export_nunchaku_sdxl["torch_dtype"]
            # dequantized weights are stored in the current in-memory model
            dequant_state = {k: v.detach().cpu() for k, v in model.module.state_dict().items()}
            _sdxl_export_to_nunchaku_single_safetensors(
                output_path=export_path,
                orig_unet_path=orig_unet_path,
                dequant_unet_state=dequant_state,
                scale_state=scale_state_dict,
                branch_state=branch_state_dict,
                rank=rank,
                precision=precision,
                torch_dtype=torch_dtype,
                unet=unet,
            )
            if export_nunchaku_sdxl.get("cleanup_run_cache", False) and save_dirpath:
                try:
                    import shutil

                    shutil.rmtree(save_dirpath, ignore_errors=True)
                except Exception:
                    pass
        if export_nunchaku_flux:
            # Export final Nunchaku FLUX.1-dev transformer checkpoint (single-file safetensors).
            # This is the intended deliverable for ComfyUI/Nunchaku usage.
            export_path = export_nunchaku_flux["output_path"]
            transformer = export_nunchaku_flux["transformer"]
            float_point = export_nunchaku_flux.get("float_point", False)
            rank = int(export_nunchaku_flux.get("rank", 0))
            # dequantized weights are stored in the current in-memory model
            dequant_state = {k: v.detach().cpu() for k, v in model.module.state_dict().items()}
            # Load smooth state if available
            smooth_state = smooth_cache_for_export
            if smooth_state is None and config.enabled_smooth and cache and cache.path.smooth and os.path.exists(cache.path.smooth):
                smooth_state = torch.load(cache.path.smooth, map_location="cpu")
            _flux_export_to_nunchaku_single_safetensors(
                output_path=export_path,
                dequant_state=dequant_state,
                scale_state=scale_state_dict,
                smooth_state=smooth_state,
                branch_state=branch_state_dict,
                transformer=transformer,
                float_point=float_point,
                rank=rank,
                quantization_config=export_nunchaku_flux.get("quantization_config"),
                comfy_config=export_nunchaku_flux.get("comfy_config"),
            )
            if export_nunchaku_flux.get("cleanup_run_cache", False) and save_dirpath:
                try:
                    import shutil

                    shutil.rmtree(save_dirpath, ignore_errors=True)
                except Exception:
                    pass
        del quantizer_state_dict, branch_state_dict, scale_state_dict
        tools.logging.Formatter.indent_dec()
        gc.collect()
        torch.cuda.empty_cache()
    if quant_acts:
        logger.info("  * Quantizing activations")
        tools.logging.Formatter.indent_inc()
        if config.needs_acts_quantizer_cache:
            load_from = ""
            if load_path and os.path.exists(load_path.acts):
                load_from = load_path.acts
            elif cache and cache.path.acts and os.path.exists(cache.path.acts):
                load_from = cache.path.acts
            if load_from:
                logger.info(f"- Loading activation settings from {load_from}")
                quantizer_state_dict = torch.load(load_from)
                quantize_diffusion_activations(
                    model, config, quantizer_state_dict=quantizer_state_dict, orig_state_dict=orig_state_dict
                )
            else:
                logger.info("- Generating activation settings")
                quantizer_state_dict = quantize_diffusion_activations(model, config, orig_state_dict=orig_state_dict)
                if cache and cache.dirpath.acts and quantizer_state_dict is not None:
                    logger.info(f"- Saving activation settings to {cache.path.acts}")
                    os.makedirs(cache.dirpath.acts, exist_ok=True)
                    torch.save(quantizer_state_dict, cache.path.acts)
                load_from = cache.path.acts
            if save_dirpath:
                if not copy_on_save and load_from:
                    logger.info(f"- Linking activation quantizer settings to {save_path.acts}")
                    os.symlink(os.path.relpath(load_from, save_dirpath), save_path.acts)
                else:
                    logger.info(f"- Saving activation quantizer settings to {save_path.acts}")
                    torch.save(quantizer_state_dict, save_path.acts)
            del quantizer_state_dict
        else:
            logger.info("- No need to generate/load activation quantizer settings")
            quantize_diffusion_activations(model, config, orig_state_dict=orig_state_dict)
        tools.logging.Formatter.indent_dec()
        del orig_state_dict
        gc.collect()
        torch.cuda.empty_cache()
    return model


def main(config: DiffusionPtqRunConfig, logging_level: int = tools.logging.DEBUG) -> DiffusionPipeline:  # noqa: C901
    """Post-training quantization of a diffusion model.

    Args:
        config (`DiffusionPtqRunConfig`):
            The diffusion model post-training quantization configuration.
        logging_level (`int`, *optional*, defaults to `logging.DEBUG`):
            The logging level.

    Returns:
        `DiffusionPipeline`:
            The diffusion pipeline with quantized model.
    """
    config.output.lock()
    # Compatibility + FLUX export safety:
    # - Keep legacy behavior (SDXL etc.): low-rank `exclusive=True` may be auto-forced by config.
    # - But for FLUX Nunchaku fused-QKV export, we must ensure a shared low-rank basis,
    #   i.e. `exclusive=False`, regardless of the legacy auto-forcing.
    _forced_flux_lowrank_exclusive: bool | None = None
    if config.export_nunchaku_flux and config.quant and config.quant.wgts and config.quant.wgts.low_rank:
        _forced_flux_lowrank_exclusive = bool(config.quant.wgts.low_rank.exclusive)
        config.quant.wgts.low_rank.exclusive = False
    # Mitigate "noisy" outputs for FLUX exports:
    # Official Nunchaku FLUX safetensors include `smooth`/`smooth_orig` for many linears.
    # Enable projection smoothing by default ONLY for FLUX export runs (SDXL unaffected).
    if bool(config.export_nunchaku_flux) and config.quant and config.quant.smooth is None:
        # SmoothCalibConfig in manual mode requires exactly one span combination.
        # Match the common default used in configs: spans = [(AbsMax, AbsMax)].
        config.quant.smooth = SmoothTransfomerConfig(
            proj=SkipBasedSmoothCalibConfig(spans=[("AbsMax", "AbsMax")])
        )
    config.dump(path=config.output.get_running_job_path("config.yaml"))
    tools.logging.setup(path=config.output.get_running_job_path("run.log"), level=logging_level)
    logger = tools.logging.getLogger(__name__)

    # Explicit English logs for SDXL/FLUX runs: show effective low-rank settings.
    # Also print an explicit run target to avoid ambiguity for users parsing logs.
    run_target = "UNKNOWN"
    if bool(config.export_nunchaku_flux):
        run_target = "FLUX"
    elif bool(config.export_nunchaku_sdxl):
        run_target = "SDXL"
    else:
        # Best-effort inference from pipeline name (still keep UNKNOWN if not obvious).
        pname = str(getattr(config.pipeline, "name", "") or "").lower()
        if "sdxl" in pname:
            run_target = "SDXL"
        elif "flux" in pname:
            run_target = "FLUX"

    logger.info(
        "* RUN TARGET (explicit): %s (pipeline.name=%s, pipeline.path=%s, pipeline.unet_path=%s)",
        run_target,
        getattr(config.pipeline, "name", None),
        getattr(config.pipeline, "path", None),
        getattr(config.pipeline, "unet_path", None),
    )
    lr = None
    try:
        lr = config.quant.wgts.low_rank if (config.quant and config.quant.wgts) else None
    except Exception:
        lr = None
    if lr is not None:
        logger.info(
            "* Low-rank config (effective) [%s]: rank=%s exclusive=%s compensate=%s num_iters=%s",
            run_target,
            getattr(lr, "rank", None),
            getattr(lr, "exclusive", None),
            getattr(lr, "compensate", None),
            getattr(lr, "num_iters", None),
        )
    logger.info(
        "* Export mode [%s]: nunchaku_sdxl=%s nunchaku_flux=%s",
        run_target,
        bool(config.export_nunchaku_sdxl),
        bool(config.export_nunchaku_flux),
    )
    if _forced_flux_lowrank_exclusive is not None:
        logger.info(
            "* Override applied for FLUX Nunchaku export [%s]: low_rank.exclusive %s -> %s",
            run_target,
            _forced_flux_lowrank_exclusive,
            False,
        )
    # Prevent intermediate *.pt cache artifacts for FLUX export runs (user requirement).
    # Note: we still write run.log/config.yaml (text) for traceability.
    ptq_cache = config.cache
    if bool(config.export_nunchaku_flux):
        ptq_cache = None
        logger.info("* Cache writes: DISABLED for FLUX Nunchaku export (no *.pt cache files will be written)")
        logger.info("* Smooth: ENABLED (proj) for FLUX Nunchaku export to reduce noisy outputs")

    logger.info("=== Configurations ===")
    tools.logging.info(config.formatted_str(), logger=logger)
    logger.info("=== Dumped Configurations ===")
    tools.logging.info(pprint.pformat(config.dump(), indent=2, width=120), logger=logger)
    logger.info("=== Output Directory ===")
    logger.info(config.output.job_dirpath)

    logger.info("=== Start Evaluating ===")
    logger.info("* Building diffusion model pipeline")
    tools.logging.Formatter.indent_inc()
    pipeline = config.pipeline.build()
    if config.pipeline.unet_path and hasattr(pipeline, "unet"):
        logger.info(f"* Loading UNet weights from {config.pipeline.unet_path}")
        unet_state_dict_raw = {}
        with safetensors.safe_open(config.pipeline.unet_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                unet_state_dict_raw[k] = f.get_tensor(k)
        if _looks_like_comfyui_or_sd_ckpt(unet_state_dict_raw):
            # Convert full checkpoint into Diffusers UNet weights.
            unet_state_dict = _convert_single_file_to_diffusers_unet_state_dict(
                ckpt_path=config.pipeline.unet_path, torch_dtype=config.pipeline.dtype
            )
        else:
            # Assume Diffusers-format UNet weights (possibly with extras); filter to known keys for safety.
            expected = set(pipeline.unet.state_dict().keys())
            unet_state_dict = {k: v for k, v in unet_state_dict_raw.items() if k in expected}
        incompatible = pipeline.unet.load_state_dict(unet_state_dict, strict=False)
        logger.info(
            f"* UNet load: matched={len(unet_state_dict)} missing={len(incompatible.missing_keys)} "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )
        del unet_state_dict_raw, unet_state_dict, incompatible
        gc.collect()
        torch.cuda.empty_cache()
    if "nf4" not in config.pipeline.name and "gguf" not in config.pipeline.name:
        model = DiffusionModelStruct.construct(pipeline)
        tools.logging.Formatter.indent_dec()
        # Default: save per-run cache under the job directory.
        save_dirpath = os.path.join(config.output.running_job_dirpath, "cache")
        # If user requests Nunchaku SDXL/FLUX export, they likely don't want intermediate *.pt artifacts in the run folder.
        # We'll keep everything in-memory and only write the final .safetensors.
        if config.export_nunchaku_sdxl or config.export_nunchaku_flux:
            save_dirpath = ""
            save_model = False
        if config.save_model and not config.export_nunchaku_sdxl and not config.export_nunchaku_flux:
            if config.save_model.lower() in ("false", "none", "null", "nil"):
                save_model = False
            elif config.save_model.lower() in ("true", "default"):
                save_dirpath, save_model = os.path.join(config.output.running_job_dirpath, "model"), True
            else:
                save_dirpath, save_model = config.save_model, True
        else:
            save_model = False
        export_sdxl_ctx = None
        if config.export_nunchaku_sdxl:
            if not config.pipeline.unet_path:
                raise ValueError("export_nunchaku_sdxl requires pipeline.unet_path (original UNet safetensors).")
            if not config.quant.enabled_wgts:
                raise ValueError("export_nunchaku_sdxl requires weight quantization enabled (quant.wgts).")
            if not config.quant.wgts.enabled_low_rank or not config.quant.wgts.low_rank:
                raise ValueError("export_nunchaku_sdxl requires SVDQuant low-rank enabled (quant.wgts.low_rank).")
            export_sdxl_ctx = {
                "output_path": config.export_nunchaku_sdxl,
                "orig_unet_path": config.pipeline.unet_path,
                "unet": pipeline.unet,
                "rank": int(config.quant.wgts.low_rank.rank),
                "precision": "nvfp4",
                "torch_dtype": config.pipeline.dtype,
                "cleanup_run_cache": bool(config.cleanup_run_cache_after_export),
            }

        export_flux_ctx = None
        if config.export_nunchaku_flux:
            if not config.quant.enabled_wgts:
                raise ValueError("export_nunchaku_flux requires weight quantization enabled (quant.wgts).")
            transformer = pipeline.transformer if hasattr(pipeline, "transformer") else pipeline.unet
            # Determine float_point from precision (sfp4 = float_point=True, int4 = float_point=False)
            float_point = config.quant.wgts.dtype and "fp" in str(config.quant.wgts.dtype).lower()
            # Build a Nunchaku-style quantization_config metadata block (match official layouts).
            def _norm_dtype_name(x: object | None) -> str | None:
                if x is None:
                    return None
                s = str(x)
                s = s.replace("torch.", "")
                s = s.replace("QuantDataType.", "")
                s = s.lower()
                if s.startswith("sfp"):
                    s = "fp" + s[3:]
                if s.startswith("sint"):
                    s = "int" + s[4:]
                return s

            wgts = config.quant.wgts
            ipts = config.quant.ipts
            weight_scale_dtype = [_norm_dtype_name(d) for d in (getattr(wgts, "scale_dtypes", None) or (None,))]
            activation_scale_dtype = _norm_dtype_name((getattr(ipts, "scale_dtypes", None) or (None,))[0])
            qcfg: dict[str, object] = {
                "method": "svdquant",
                "weight": {
                    "dtype": _norm_dtype_name(getattr(wgts, "dtype", None)),
                    "scale_dtype": weight_scale_dtype,
                    "group_size": 16 if float_point else 64,
                },
                "activation": {
                    "dtype": _norm_dtype_name(getattr(ipts, "dtype", None)),
                    "scale_dtype": activation_scale_dtype,
                    "group_size": 16,
                },
            }
            if config.quant.wgts.low_rank is not None:
                qcfg["rank"] = int(config.quant.wgts.low_rank.rank)

            # Build ComfyUI-style comfy_config from the transformer config (no hardcoded blob).
            comfy_cfg = None
            try:
                tcfg = getattr(transformer, "config", None)
                if hasattr(tcfg, "to_dict"):
                    tcfg = tcfg.to_dict()
                if isinstance(tcfg, dict):
                    comfy_cfg = _build_flux_comfy_config(transformer_cfg=tcfg)
            except Exception:
                comfy_cfg = None
            export_flux_ctx = {
                "output_path": config.export_nunchaku_flux,
                "transformer": transformer,
                "float_point": float_point,
                "rank": int(config.quant.wgts.low_rank.rank) if config.quant.wgts.low_rank else 0,
                "cleanup_run_cache": bool(config.cleanup_run_cache_after_export),
                "quantization_config": qcfg,
                "comfy_config": comfy_cfg,
            }

        model = ptq(
            model,
            config.quant,
            cache=ptq_cache,
            load_dirpath=config.load_from,
            save_dirpath=save_dirpath,
            copy_on_save=config.copy_on_save,
            save_model=save_model,
            export_nunchaku_sdxl=export_sdxl_ctx,
            export_nunchaku_flux=export_flux_ctx,
        )
    if config.pipeline.lora is not None:
        load_from = ""
        if config.quant.enabled_smooth:
            if config.load_from and os.path.exists(os.path.join(config.load_from, "smooth.pt")):
                load_from = os.path.join(config.load_from, "smooth.pt")
            elif config.cache.path and os.path.exists(config.cache.path.smooth):
                load_from = config.cache.path.smooth
            elif os.path.exists(os.path.join(save_dirpath, "smooth.pt")):
                load_from = os.path.join(save_dirpath, "smooth.pt")
            logger.info(f"* Loading smooth scales from {load_from}")
        config.pipeline.load_lora(pipeline, smooth_cache=torch.load(load_from) if load_from else None)
    if config.text is not None and config.text.is_enabled():
        for encoder_name, encoder, tokenizer in config.pipeline.extract_text_encoders(pipeline):
            logger.info(f"* Post-training quantizing the text encoder {encoder_name}")
            patch_attention(encoder)
            patch_gemma_rms_norm(encoder)
            save_dirpath = os.path.join(save_dirpath, "encoder")
            setattr(
                pipeline,
                encoder_name,
                llm_ptq(
                    encoder,
                    tokenizer,
                    config.text,
                    cache=config.text_cache,
                    load_dirpath=os.path.join(config.load_from, "encoder") if config.load_from else "",
                    save_dirpath=save_dirpath,
                    copy_on_save=config.copy_on_save,
                    save_model=save_model,
                ),
            )
    config.eval.gen_root = config.eval.gen_root.format(
        output=config.output.running_dirpath, job=config.output.running_job_dirname
    )
    if config.skip_eval:
        if not config.skip_gen:
            logger.info("* Generating image")
            tools.logging.Formatter.indent_inc()
            config.eval.generate(pipeline, task=config.pipeline.task)
            tools.logging.Formatter.indent_dec()
    else:
        logger.info(f"* Evaluating model {'(skipping generation)' if config.skip_gen else ''}")
        tools.logging.Formatter.indent_inc()
        results = config.eval.evaluate(pipeline, skip_gen=config.skip_gen, task=config.pipeline.task)
        tools.logging.Formatter.indent_dec()
        if results is not None:
            logger.info(f"* Saving results to {config.output.job_dirpath}")
            with open(config.output.get_running_job_path("results.json"), "w") as f:
                json.dump(results, f, indent=2, sort_keys=True)
    # Close file handlers (e.g. run.log) before renaming output directories on Windows.
    # unlock() must NEVER crash the whole run (quantization may already be completed).
    tools.logging.shutdown()
    try:
        config.output.unlock()
    except Exception:
        pass


if __name__ == "__main__":
    config, _, unused_cfgs, unused_args, unknown_args = DiffusionPtqRunConfig.get_parser().parse_known_args()
    assert isinstance(config, DiffusionPtqRunConfig)
    if len(unused_cfgs) > 0:
        tools.logging.warning(f"Unused configurations: {unused_cfgs}")
    if unused_args is not None:
        tools.logging.warning(f"Unused arguments: {unused_args}")
    assert len(unknown_args) == 0, f"Unknown arguments: {unknown_args}"
    try:
        main(config, logging_level=tools.logging.DEBUG)
    except Exception as e:
        tools.logging.Formatter.indent_reset()
        tools.logging.error("=== Error ===")
        tools.logging.error(traceback.format_exc())
        tools.logging.shutdown()
        traceback.print_exc()
        config.output.unlock(error=True)
        raise e
