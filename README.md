# DeepSeek-V4.1-Flash on DGX Station GB300 (SM103) — patches for the official vLLM image

Run **DeepSeek-V4.1-Flash** (522B MoE / 8-16B active) on an **NVIDIA DGX Station GB300**
with the official `vllm/vllm-openai:deepseekv41-flash-*` image.

Companion to [ebfio/glm53-flash-dflash2-gb300](https://github.com/ebfio/glm53-flash-dflash2-gb300)
— same class of machine, same "make the upstream image work on this GPU" problem, different
failure modes.

> **SM103, not SM100.** nvidia-smi reports `compute_cap 10.3` on this box. The upstream
> image builds `TORCH_CUDA_ARCH_LIST="8.7 8.9 9.0 10.0+PTX 12.0"`, and inspection of the
> shipped `_C_stable_libtorch` shows what that actually produced:
>
> ```
> CUBINs: sm_80, sm_87, sm_89, sm_90, sm_90a, sm_100, sm_100a, sm_120, sm_120a
> PTX:    sm_80, sm_89, sm_90        ← no sm_100/sm_103 PTX, nothing JITs forward
> ```
>
> So there **is** SM100 code (`sm_100` and the arch-specific `sm_100a`), and **no**
> `sm_100`/`compute_100` PTX — which means nothing JIT-compiles forward onto 10.3. The
> `sm_100a` cubins are the sharp edge: an `a`-suffixed cubin runs on *exactly* its target
> arch and never on a later one, so SM103 silently has no image for those kernels.
>
> That explains **bug 2 below only**. Bugs 1 and 3 are host-side Python layout problems —
> no kernel is involved, and they would fire on any arch whose code path reaches them.
> They are not SM103-specific in principle; they are SM103-specific *in practice* here
> because of which arch gates select those paths, and I have not located the exact gate
> that makes them fire on 10.3 but not on 10.0. See the caveat at the end.

```
patches/
  deep_gemm.py             # attention o_proj weight-scale layout (DeepGEMM einsum)
  sparse_attn_indexer.py   # cooperative_topk has no SM103 cubin
  o_proj.py                # one-shot layout diagnostic (drop-safe)
examples/
  docker-compose.yml       # working config, all values measured
  verify.sh                # math / long-context / tool-calling checks
```

## Results

DGX Station GB300, single GPU, `--tensor-parallel-size 1`, thinking off:

| metric | value |
|---|---|
| decode (technical prose) | **91.7 tok/s** median of 4 |
| decode (essay) | 57.8 tok/s |
| prefill (9K-token prompt) | **~4,090 tok/s** (2.2 s wall) |
| KV cache | 4,458,466 tokens (9.36 GiB) |
| concurrency @ 1M ctx | 4.25× |
| correctness | `17*19 → 323`; needle-in-9K retrieved exactly |
| tool calling | clean `tool_calls`, `finish_reason=tool_calls` |

For reference, the same checkpoint on a GH200 (SM90, MARLIN, `--cpu-offload-gb 210`)
runs at **~14 tok/s**. Same model, **~6.5×** faster on the Station.

Reproduce the numbers with `examples/verify.sh` (tests correctness, not speed — a
wrong-but-legal scale layout produces garbage text, never an assert).

## The three bugs

### 1. `deep_gemm.py` — o_proj weight-scale layout

`vllm/models/deepseek_v4/nvidia/ops/o_proj.py` calls `fp8_einsum` for the attention
output projection. DeepGEMM validates the grouped scale tensors in
`csrc/utils/layout.hpp::check_sf_layout`:

```cpp
DG_HOST_ASSERT(sf.stride(-2) == 1);
DG_HOST_ASSERT(sf.stride(-1) == get_tma_aligned_size(mn, sf.element_size()));
if (num_groups.has_value())
    DG_HOST_ASSERT(sf.stride(-3) == sf.stride(-1) * sf.size(-1));
```

The **activation** scales (`sfa`, from `fused_inv_rope_fp8_quant(tma_aligned_scales=True)`)
already satisfy this. The **weight** scales (`sfb`, from the load-time
`deepgemm_post_process_weight_scale_block`) are emitted row-major and do not:

```
RuntimeError: Assertion error (csrc/utils/layout.hpp:113):
  sf.stride(-3) == sf.stride(-1) * sf.size(-1)
```

Fix: relayout **SFB only** (`transpose(-1,-2).contiguous().transpose(-1,-2)`), applied
inside the `fp8_einsum` wrapper. Verified against the real kernel:

```
SFA (8192, 8, 32) stride (1, 262144, 8192)   # vLLM's output — left untouched
SFB (8, 1024, 32) stride (32768, 1, 1024)    # patched
rel err vs float reference: 2.0e-3
```

> **Do not transpose SFA.** It is already correct. Transposing it moves the failure to
> the `size(-3) == num_groups` assert and sends you down a blind alley.

The relayout is a reinterpretation (no dtype change, no value change), so numerics are
identical to a correctly packed tensor.

**This is a per-call copy.** `transpose.contiguous.transpose` runs inside `fp8_einsum` on
every layer, every step — about 1 MB × 40 layers per forward, so cheap, but it is a
runtime fix. The cleaner version is a one-shot repack of `wo_a.weight_scale` right after
load (E8M0 → FP32 → DeepGEMM's own `get_mn_major_tma_aligned_packed_ue8m0_tensor`), which
also drops any assumption about the byte order of whatever produced the K-major tensor.
The 2.0e-3 rel-err below and the `17*19 → 323` check validate that assumption for this
build, so the switch should be safe — it just has not been done here.

### 2. `sparse_attn_indexer.py` — `cooperative_topk` has no SM103 cubin

```python
use_cooperative_topk = (
    current_platform.is_cuda()
    and topk_tokens in (512, 1024, 2048)
    and num_rows <= 64
    and logits.stride(0) % 4 == 0
    and current_platform.has_device_capability(90)     # ← admits SM103
    and not current_platform.is_device_capability_family(120)
)
```

The gate passes on SM103, but the prebuilt kernel does not exist for it, so **CUDA graph
capture** dies:

```
RuntimeError: launch_cooperative_cluster, cooperative_topk.cu:48,
  cooperative_topk launch failed: no kernel image is available for execution on the device
```

Fix: narrow the gate to the SM90 family. The existing `persistent_topk` fallback below it
covers the same conditions and works on SM103.

### 3. `o_proj.py` — diagnostic only

Prints the (shape, stride) of both scale tensors **once per process**. Safe to delete.

## Launch config notes

Two flags differ from a naive SM90 port, and both are load-bearing:

| flag | value | why |
|---|---|---|
| `--block-size` | **128** | `models/deepseek_v4_1/sparse_mla.py:90` hardcodes `64 if family(90) else 128`. Passing 64 gives `ValueError: No common block size for 64`. |
| `--gpu-memory-utilization` | **0.92** | 0.95 OOMs during load / graph capture. |
| `--cpu-offload-gb` | **75** | see the sweep below. |
| `--moe-backend` / `--linear-backend` | **marlin** | **required on this image** — see below. |

**Backends: both pinned to Marlin.** `--moe-backend marlin --linear-backend marlin` is what
the config above runs. Neither pin is a device-support limitation and neither is
permanent — they are two separate open items:

- **`DEEPGEMM_MXFP4`** (the auto-selected MoE backend) asserts inside
  `m_grouped_fp8_fp4_gemm_nt_contiguous` on the 3-D expert scale tensor. In the
  contiguous-grouped layout SFA is 2-D, so the `stride(-3)` check can only be firing on
  SFB — the same check bug 1 trips, on a tensor produced at load by
  `oracle/mxfp4.py::_pack_deepgemm_mxfp4_scales`. By analogy with bug 1 the fix should be
  the same K-major → grouped MN-major relayout of those packed scales, but **I did not
  probe or fix it** — treat that as expected, not established.
  What it costs: prefill throughput. At ≤4 streams decode is weight-bandwidth-bound and
  Marlin's dequant path is close to the native kernel; the number the native path would
  move is the ~4.1K tok/s prefill figure, not the decode figure.
- **FlashInfer CUTLASS MXFP8** (the auto-selected linear backend) fails
  `Mismatched mB.strides[1]` — a stride rejection, not a device-capability one.
  `MarlinMxfp8LinearKernel` accepts the same tensors, and these are the small dense
  projections, so there is nothing to gain at this concurrency.

To try the native MoE path: put the bug-1 diagnostic on SFB inside `_grouped_fp4_impl`,
apply the same relayout, drop `--moe-backend marlin`, re-run `verify.sh`.

### CPU offload: the floor is structural

Non-Engram weights total **287.7 GiB**; the HBM budget at `gmu 0.92` is **230.6 GiB**.
At least **57 GiB must be offloaded** no matter how you configure it. Measured sweep:

| `--cpu-offload-gb` | KV cache | concurrency @1M |
|---|---|---|
| 90 | 11.0M tok | 10.50× |
| **75** | **4.46M tok** | **4.25×** |
| 70 | 1.26M tok | 1.20× |

**75 is the choice for KV headroom**: it keeps 4.25× concurrency at 1M context while
still pulling 15 GiB of experts back into HBM versus 90. Below 75 the pool collapses
(1.20× at 70), which is a capacity problem for a shared endpoint, not a speed one.

> A note on the decode column this table used to carry: single-stream decode does not
depend on KV pool size, so the 91.7 vs 79.9 tok/s I first reported here was measuring
> DSpark acceptance on two different prompts, not the effect of the offload change.
> Acceptance varies a lot with prompt predictability, which makes single-sample tok/s a
> poor instrument for config decisions. The offload choice above rests on the KV numbers;
> if you want the speed column to mean something, fix one prompt, temperature 0, take ≥5
> reps, and record mean accepted length alongside tok/s.

## Usage

```bash
# 1. copy the patches somewhere on the host
install -D -m 0644 patches/*.py /opt/ai-services/deepseek-v4.1-flash/patches/

# 2. start the service (see examples/docker-compose.yml for the full config)
cd /opt/ai-services/deepseek-v4.1-flash
docker compose up -d
```

The three mounts to add to an existing compose file:

```yaml
volumes:
  - "…/patches/o_proj.py:/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/ops/o_proj.py:ro"
  - "…/patches/deep_gemm.py:/usr/local/lib/python3.12/dist-packages/vllm/utils/deep_gemm.py:ro"
  - "…/patches/sparse_attn_indexer.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer.py:ro"
```

## Verifying a deploy

```bash
curl -s localhost:8001/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-v4.1-flash",
  "messages": [{"role": "user", "content": "What is 17*19? Return only the integer."}],
  "max_tokens": 32, "temperature": 0,
  "chat_template_kwargs": {"thinking": false}}'
# expect content "323"
```

A wrong-but-legal scale layout produces **garbage text, not an assert** — always run a
correctness check, never just a health check.

## Tested environment

**NVIDIA DGX Station GB300** — single GB300 GPU (SM103 / compute capability 10.3),
256 GB HBM3e, 494 GB unified LPDDR5x via NVLink-C2C, 72-core Neoverse-V2 (aarch64),
Ubuntu 24.04, 64K-page kernel (`6.17.0-nvidia-64k`), driver 610.43.02.

- Image: `vllm/vllm-openai:deepseekv41-flash-0909-cu129-arm64`
  (vLLM `0.1.dev20904+g179dd0fa9`, torch 2.13.0+cu129), 2026 vendored DeepGEMM of
  unknown tag. The assert text and the INT-at-`gran_k=32` branch rules out the v2.1.1
  tag specifically (v2.1.1 gates that branch to `gran_k == 128`); the exact revision is
  not pinned anywhere in the image.
- Model: `deepseek-ai/DeepSeek-V4.1-Flash`
- GPU selected by **UUID** (`device_ids: ['GPU-…']`), not index — the box also exposes an
  RTX PRO 4000 that must not be picked up accidentally

## Credits

Standing on the shoulders of:

- [tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark)
  — the 4×DGX-Spark reference deployment for this exact model; its boot log (`docs/boot*.md`)
  is the best available field guide to running V4.1 on Blackwell. Their launcher's
  `--block-size 128`, `--engram-config` handling and DSpark settings informed this config.
- [vLLM](https://github.com/vllm-project/vllm) — the DeepSeek-V4.1 model code, the
  `deepseek_v41` tokenizer/parsers, and the layout asserts we work around.
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) — `csrc/utils/layout.hpp` is what
  defines the required grouped-scale layout.
- [DeepSeek-AI](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) — the model.

## Known dead ends

- **NVFP4 checkpoints** (`AtomicChat/DeepSeek-V4.1-Flash-NVFP4-nvidia`): the
  `FLASHINFER_TRTLLM` NVFP4 MoE backend reports
  `does not support the deployment configuration since kernel does not support current
  device cuda` on SM103 in this build.
- **Native `DEEPGEMM_MXFP4` experts**: blocked on the same layout bug class in
  `_pack_deepgemm_mxfp4_scales`. Worth fixing for long-prompt prefill; irrelevant for
  low-concurrency decode, where Marlin is fine.

## Open questions

Things I have not established, stated plainly so nobody builds on them:

- **Why bugs 1 and 3 are SM103-only.** They are host-side layout asserts, so an exact
  10.0 device should be able to reach the same code. vLLM's own recipe runs this image on
  GB200 NVL4 (SM100) at TP4, which suggests something selects these paths only on 10.3 —
  but `has_device_capability(100)` is a *minimum* check and returns `True` on SM103, and a
  grep for exact-100 gates in the V4/V4.1 model, quantization and MoE-oracle trees found
  nothing. The gate is either elsewhere or the difference is in the packed-weight layout
  rather than the gate. Not located.
- **Whether the MoE fix is really the same fix.** See the backend section — reasoned by
  analogy, not tested. A reproducing test needs real `expert_ids` from
  `deepgemm_moe_permute`; hand-built ones die earlier at `gemm.hpp:293 m == m__`.
- **Whether SM103 has any path to the native kernels at all** without an image rebuild.
  The `sm_100a` cubins are unusable here and no forward-compatible PTX is shipped, so the
  only routes are a rebuild with `10.3a`/`compute_103` in the arch list, or staying on
  Marlin.

## License / usage notes

- Everything in this repository is **Apache-2.0**; see `LICENSE`.
- `patches/*.py` are **derivative works of vLLM** (Apache-2.0). They contain
  upstream code, modified — not clean-room rewrites.
- The model is subject to **DeepSeek's own license**; this repo ships no weights.
- Validated on **one DGX Station GB300 at TP1**. Other SM103 configs (multi-GPU /
  P/D-disaggregated) are untested, and a different GB300 SKU could report a different
  compute capability — check `nvidia-smi --query-gpu=compute_cap` before assuming these
  patches apply.
