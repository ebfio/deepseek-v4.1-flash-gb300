# DeepSeek-V4.1-Flash on GB300 (SM103)

Run **DeepSeek-V4.1-Flash** on an NVIDIA **GB300 / B300 (SM103)** with the official
`vllm/vllm-openai:deepseekv41-flash-*` image.

The upstream image builds `TORCH_CUDA_ARCH_LIST="8.7 8.9 9.0 10.0+PTX 12.0"` — **there is
no SM103 cubin** — while vLLM's arch gates use `has_device_capability(90)` and
`is_device_capability_family(100)`, both of which admit SM103 and route into code paths
whose prebuilt kernels do not cover it. Three bugs fall out of that gap. All three are
fixed here by **bind-mounting patched Python over site-packages** — no rebuild, no fork,
no toolchain.

```
patches/
  deep_gemm.py             # attention o_proj weight-scale layout (DeepGEMM einsum)
  sparse_attn_indexer.py   # cooperative_topk has no SM103 cubin
  o_proj.py                # one-shot layout diagnostic (drop-safe)
examples/
  docker-compose.yml       # working config, all values measured
```

## Results

Single GB300, `--tensor-parallel-size 1`, thinking off:

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
runs at **~14 tok/s**. Same model, **~6.5×** faster here.

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

**Marlin is mandatory** even though the GB300 supports the native path. In auto mode
vLLM picks `DEEPGEMM_MXFP4`, which hits the same layout bug family on the *expert* scales
in `oracle/mxfp4.py::_pack_deepgemm_mxfp4_scales` (unfixed here — it is a prefill-time
win at most, and decode is fine on Marlin). FlashInfer's CUTLASS MXFP8 linear kernel also
rejects the offloaded weight strides (`Mismatched mB.strides[1]`). Pinning both to Marlin
sidesteps both.

### CPU offload: the floor is structural

Non-Engram weights total **287.7 GiB**; the HBM budget at `gmu 0.92` is **230.6 GiB**.
At least **57 GiB must be offloaded** no matter how you configure it. Measured sweep:

| `--cpu-offload-gb` | KV cache | concurrency @1M | median decode |
|---|---|---|---|
| 90 | 11.0M tok | 10.50× | — (wastes RAM) |
| **75** | **4.46M tok** | **4.25×** | **91.7 tok/s** |
| 70 | 1.26M tok | 1.20× | 79.9 tok/s |

Below 75 it gets **slower**: KV starvation costs more than the extra resident experts
gain. 75 is the sweet spot.

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

- GPU: NVIDIA GB300 (SM103), 256 GB HBM3e, arm64 Grace host
- Image: `vllm/vllm-openai:deepseekv41-flash-0909-cu129-arm64`
  (vLLM `0.1.dev20904+g179dd0fa9`), vendored DeepGEMM v2.1.x
- Model: `deepseek-ai/DeepSeek-V4.1-Flash`

## Known dead ends

- **NVFP4 checkpoints** (`AtomicChat/DeepSeek-V4.1-Flash-NVFP4-nvidia`): the
  `FLASHINFER_TRTLLM` NVFP4 MoE backend reports
  `does not support the deployment configuration since kernel does not support current
  device cuda` on SM103 in this build.
- **Native `DEEPGEMM_MXFP4` experts**: blocked on the same layout bug class in
  `_pack_deepgemm_mxfp4_scales`. Worth fixing for long-prompt prefill; irrelevant for
  low-concurrency decode, where Marlin is fine.

## License

Patch files are derivative works of vLLM (Apache-2.0). Everything in this repository is
provided under Apache-2.0; see `LICENSE`.
