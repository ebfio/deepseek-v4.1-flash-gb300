# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import fp8_einsum


_DG_PROBE_DONE = False


def _dg_probe():
    """Return a logger that fires exactly once per process, else None."""
    global _DG_PROBE_DONE
    if _DG_PROBE_DONE:
        return None
    _DG_PROBE_DONE = True
    import logging
    return logging.getLogger("dsv41.layout")


def compute_fp8_einsum_recipe(
    block_size: int = 128,
) -> tuple[tuple[int, int, int], bool]:
    """fp8_einsum recipe + scale layout for the current GPU arch.

    SM90 keeps block-row FP32 scales. SM100 uses packed per-row E8M0 scales.

    Returns ``(einsum_recipe, tma_aligned_scales)`` for ``deep_gemm_fp8_o_proj``.
    """
    cap = current_platform.get_device_capability()
    assert cap is not None, "DeepseekV4 attention requires a CUDA device"
    einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, block_size)
    tma_aligned_scales = cap.major >= 10
    return einsum_recipe, tma_aligned_scales


def deep_gemm_fp8_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: nn.Module,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    einsum_recipe: tuple[int, int, int],
    tma_aligned_scales: bool,
) -> torch.Tensor:
    """O projection: inverse RoPE + grouped wo_a + wo_b.

    Shared by the FlashMLA and FlashInfer CUDA backends. The attention
    layer selects the recipe at initialization.
    """
    use_fp8 = wo_a.weight.dtype == torch.float8_e4m3fn
    o_proj_input, o_scale = fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        quant_group_size=einsum_recipe[2],
        tma_aligned_scales=tma_aligned_scales,
        quantize=use_fp8,
    )
    z = torch.empty(
        (o.shape[0], n_groups, o_lora_rank),
        device=o.device,
        dtype=torch.bfloat16,
    )
    if use_fp8:
        weight_scale = (
            wo_a.weight_scale
            if hasattr(wo_a, "weight_scale")
            else wo_a.weight_scale_inv
        )
        # One-shot diagnostic. DeepGEMM validates these layouts in
        # csrc/utils/layout.hpp::check_sf_layout; a mismatch is an assert, not a
        # warning, so print the shapes once to make the next regression obvious.
        _lp = _dg_probe()
        if _lp is not None:
            _lp.error(
                "o_proj fp8_einsum layout: sfa=%s%s sfb=%s%s recipe=%s",
                tuple(o_scale.shape), tuple(o_scale.stride()),
                tuple(weight_scale.shape), tuple(weight_scale.stride()),
                einsum_recipe,
            )
        fp8_einsum(
            "bhr,hdr->bhd",
            (o_proj_input, o_scale),
            (wo_a.weight, weight_scale),
            z,
            recipe=einsum_recipe,
        )
    else:
        grouped_weight = wo_a.weight.view(n_groups, o_lora_rank, -1)
        torch.bmm(
            o_proj_input.transpose(0, 1),
            grouped_weight.transpose(1, 2),
            out=z.transpose(0, 1),
        )
    return wo_b(z.flatten(1))
