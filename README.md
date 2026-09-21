# DeepSeek-V4.1-Flash on DGX Station GB300 (SM103) — one patch for the official vLLM nightly

Run **DeepSeek-V4.1-Flash** (522B MoE / 8-16B active) on an **NVIDIA DGX Station GB300**
with the official `vllm/vllm-openai:nightly` image — plus exactly **one** bind-mounted
patch.

> **Nightly, not the launch tag.** Any nightly from 2026-09-10 on serves this model
> natively ([vllm-project/vllm#56228](https://github.com/vllm-project/vllm/pull/56228));
> the official recipe has deprecated the launch-day `deepseekv41-flash-0909` tag. On the
> 0909 image this repo used to ship **six** patches. Five of them are now dead overlays
> that would *revert* newer upstream code — only the DeepGEMM scale-layout fix survives,
> because that bug is still live upstream.

Companion to [ebfio/glm53-flash-dflash2-gb300](https://github.com/ebfio/glm53-flash-dflash2-gb300)
— same class of machine, same "make the upstream image work on this GPU" problem, different
failure modes.

```
patches/
  deep_gemm.py             # grouped UE8M0 scale layout (the one live bug)
examples/
  docker-compose.yml       # working config, all values measured
  verify.sh                # math / long-context / tool-calling / vision checks
```

## The bug

### grouped UE8M0 scale layout — `deep_gemm.py`

DeepGEMM validates grouped scale tensors in `csrc/utils/layout.hpp::check_sf_layout`:

```cpp
DG_HOST_ASSERT(sf.size(-2) == ceil_div(mn, gran_mn));                     // shape
DG_HOST_ASSERT(sf.stride(-2) == 1);                                       // MN-major
DG_HOST_ASSERT(sf.stride(-1) == get_tma_aligned_size(mn, elem_size));     // TMA-aligned
if (num_groups.has_value())
    DG_HOST_ASSERT(sf.stride(-3) == sf.stride(-1) * sf.size(-1));         // no group padding
```

vLLM's load-time packer emits the packed int32 UE8M0 tensor **row-major**, so
`stride(-2) != 1` and the first grouped kernel to touch it dies. This defect predates
nightly (it is what two of the old 0909 patches worked around) — and nightly reaches it
through a **new** path: the `flash_mla_mega_attn` attention backend's `o_proj` calls
`fp8_einsum` with the load-time scale tensor during `profile_run`, so a clean boot dies
before the server ever comes up:

```
RuntimeError: Assertion error (/workspace/.deps/deepgemm-src/.../utils/layout.hpp:119):
  sf.stride(-3) == sf.stride(-1) * sf.size(-1)
  (in vllm/utils/deep_gemm.py fp8_einsum, called from
   vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py _o_proj)
```

**Fix** (`patches/deep_gemm.py`, ported onto nightly's own file — not a copy of the old
patch): a `_dg_fix_grouped_sf()` helper applied in two places —

1. inside the `fp8_einsum` wrapper, for grouped `(scale, weight)` operand pairs (covers
   the mega-attention `o_proj` and any other direct caller);
2. on the result of `transform_sf_into_required_layout`, the choke point every load-time
   scale producer routes through (covers the MoE expert scales).

The relayout is a reinterpretation — `sf.transpose(-1,-2).contiguous().transpose(-1,-2)`,
no dtype or value change — verified value-preserving and idempotent on nightly's own
torch, and boot-proven on the real checkpoint.

> **Do not transpose SFA.** The activation scales from
> `fused_inv_rope_fp8_quant(tma_aligned_scales=True)` are already in the required layout.
> Transposing them moves the failure to the `size(-3) == num_groups` assert and sends you
> down a blind alley. The patch checks the stride predicate and passes valid tensors
> through untouched.

A wrong-but-legal scale layout produces **garbage text, not an assert** — if you change
anything in this area, rerun the correctness checks below, never just a health check.

## The thinking-flag trap

The V4.1 chat template defaults to **thinking ON at effort 50** when no keys are sent.
Two consequences:

- `--default-chat-template-kwargs.thinking=True` on the serve line is redundant **and
  harmful**: clients that don't override it pay the reasoning tax on every request —
  image assessments that should take seconds take minutes, and small-`max_tokens`
  requests burn their whole budget on the trace and return empty `content` with
  `finish_reason: length`, which reads as a broken model and is not.
- The fix is per-call control: `chat_template_kwargs: {"thinking": false}` when you
  want a plain answer, or `reasoning_effort` (`low|high|xhigh|max`, or an integer
  1–100) when you want reasoning.

The compose file in this repo does **not** set the server-side flag.

## Launch config notes

Several flags differ from a naive SM90 port, and all are load-bearing:

| flag | value | why |
|---|---|---|
| `--block-size` | **128** | `models/deepseek_v41/sparse_mla.py:90` hardcodes `64 if family(90) else 128`. Passing 64 gives `ValueError: No common block size for 64`. |
| `--gpu-memory-utilization` | **0.92** | 0.95 OOMs during load / graph capture. |
| `--cpu-offload-gb` | **75** | see the sweep below. |
| `--max-model-len` | **1048576** | required for 1M context. |
| `--linear-backend` | **marlin** | FlashInfer CUTLASS rejects the offloaded weight strides; the dense projections are not the bottleneck. Attention runs the native `flash_mla_mega_attn` path regardless of this flag — the marlin pin only governs the dense projections, and it does **not** protect against the DeepGEMM assert (the crash is in the attention einsum). |
| `--speculative-config` | `dspark`, 5 tokens, **probabilistic** | `draft_sample_method: probabilistic`. |
| `--tokenizer-mode` / parsers | `deepseek_v41` | V4.1 DSML dialect; the `deepseek_v4` parser leaks raw markup into content. |

### CPU offload: the floor is structural

Non-Engram weights total **~288 GiB**; the HBM budget at `gmu 0.92` is ~230 GiB. At least
~57 GiB must be offloaded no matter how you configure it. Measured sweep (on the 0909
image; the floor is architectural and carries over):

| `--cpu-offload-gb` | KV cache | concurrency @1M |
|---|---|---|
| 90 | 11.0M tok | 10.50× |
| **75** | **4.46M tok** | **4.25×** |
| 70 | 1.26M tok | 1.20× |

**75 keeps 4.25× concurrency at 1M context** while pulling ~15 GiB of experts back into
HBM versus 90. Below 75 the pool collapses — a capacity problem for a shared endpoint,
not a speed one. On nightly the same 75 yields an even larger pool (below) thanks to the
`nvfp4_ds_mla` KV format.

> Single-stream decode does not depend on KV pool size, so decode numbers are a poor
> instrument for offload decisions — acceptance varies with prompt predictability. The
> offload choice above rests on the KV numbers. If you want decode numbers to mean
> anything: fix one prompt, temperature 0, ≥5 reps, record mean accepted length alongside
> tok/s.

## Results

DGX Station GB300, single GPU, `--tensor-parallel-size 1`, thinking off:

| metric | 0909 image (six patches) | **nightly (this repo)** |
|---|---|---|
| KV cache | 4,458,466 tokens (9.36 GiB) | **6,845,219 tokens (6.53× @ 1M ctx)** |
| attention path | pre-mega | **native `flash_mla_mega_attn`** |
| decode (code, warm) | 91.7 tok/s | ~91.5 tok/s |
| correctness | `17*19 → 323` | `17*19 → 323` |
| vision | untested | 1×1 image → correct short description |
| tool calling | via recovery patches | upstream rewritten parser; smoke-tested clean |

Boot: weights 48/48 shards in ~90 s, DSpark draft load, CUDA graph capture 3 s,
health 200 at ~14 min (first nightly boot JIT-compiles kernels).

Reproduce correctness with `examples/verify.sh` (tests math, long-context needle
retrieval, tool calling, and vision — a wrong-but-legal scale layout produces garbage
text, never an assert).

For reference, the same checkpoint on a GH200 (SM90, MARLIN, `--cpu-offload-gb 210`)
runs at **~14 tok/s**. Same model, **~6.5×** faster on the Station.

## Usage

```bash
# 1. clone the repo somewhere stable on the host (paths in the compose are absolute)
git clone https://github.com/ebfio/deepseek-v4.1-flash-gb300 /opt/deepseek-v4.1-flash-gb300

# 2. edit examples/docker-compose.yml: set your GB300 UUID and cache paths, then
docker compose -f examples/docker-compose.yml up -d
```

The single mount (adjust the left side to your checkout):

```yaml
volumes:
  - "/opt/deepseek-v4.1-flash-gb300/patches/deep_gemm.py:/usr/local/lib/python3.12/dist-packages/vllm/utils/deep_gemm.py:ro"
```

A changed patch needs a **recreate**, not a restart: bind mounts resolve at container
start, so a restarted container keeps running the old file. Use
`docker compose up -d --force-recreate`.

## Verifying a deploy

```bash
bash examples/verify.sh http://127.0.0.1:8001
```

Covers math (`323`), long-context needle retrieval, tool calling, and vision. Tool
calling deserves its own note: a malformed emission does **not** return an error — HTTP
200, `finish_reason: stop`, zero `tool_calls`, raw markup in `content`. Check for the
call, not for a 200. A clean refusal (zero calls, zero markup) is not a failure.

## Tool-call markup recovery — history, and a caveat

On the 0909 image, this repo shipped a rolling-buffer DSML repair
(`deepseek_v41.py` + `deepseek_v4.py`, seven malformed-emission shapes recovered,
15-case engine suite at step=1). Nightly **rewrote** the V4.1 parser wholesale (the
module is now a thin ~54-line file), so those patches are dead code against it and have
been removed.

What that means practically: on nightly, tool-call parsing rides upstream's rewritten
parser. Basic tool-calling and malformed-markup smoke tests pass — but the full
15-shape production suite has **not** been re-run against the new parser, so if you run
an agent-driven workload and a `tool_calls` entry silently turns into prose with markup
in `content`, the parser is the first suspect. The shapes to look for are documented in
this repo's git history (see the `2b6a366` and `8d6c6c0` commits).

## What died with the 0909 patches (and why)

For anyone running the deprecated launch tag, the old README documented five further
patches; they are **not** needed on nightly and mounting them there would overwrite
newer upstream code:

| old patch | why it existed (0909) | status on nightly |
|---|---|---|
| `sparse_attn_indexer.py` | `cooperative_topk` had no SM103 cubin; gate narrowed to SM90 family | `use_cooperative_topk` deleted upstream — patch is dead code |
| `o_proj.py` | layout diagnostic logging | path no longer exists (`models/deepseek_v41/` was restructured) |
| `fp8_utils.py` | load-time SFB relayout inside `deepgemm_post_process_weight_scale_block` | superseded by the `transform_sf_into_required_layout` fix in the one surviving patch |
| `deepseek_v4.py` / `deepseek_v41.py` | DSML tool-call recovery | parser rewritten upstream; see the caveat above |

## Tested environment

**NVIDIA DGX Station GB300** — single GB300 GPU (SM103 / compute capability 10.3),
256 GB HBM3e, 494 GB unified LPDDR5x via NVLink-C2C, 72-core Neoverse-V2 (aarch64),
Ubuntu 24.04, 64K-page kernel, driver 610.43.02.

- Image: `vllm/vllm-openai:nightly` (vLLM `0.29.1rc1.dev452+g3df4ae153`, pulled
  2026-09-21). Nightly moves — if a future nightly fixes the layout assert upstream,
  this patch becomes a no-op (the stride predicate passes valid tensors through) and can
  be dropped.
- Model: `deepseek-ai/DeepSeek-V4.1-Flash`
- GPU selected by **UUID** (`device_ids: ['GPU-…']`), not index — the box also exposes an
  RTX PRO 4000 that must not be picked up accidentally.

## Credits

Standing on the shoulders of:

- [tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark)
  — the 4×DGX-Spark reference deployment for this exact model; its `--block-size 128`
  and DSpark settings informed this config.
- [vLLM](https://github.com/vllm-project/vllm) — the DeepSeek-V4.1 model code, the
  `deepseek_v41` tokenizer/parsers, and the layout assert we work around.
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) — `csrc/utils/layout.hpp` is what
  defines the required grouped-scale layout.
- [DeepSeek-AI](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) — the model.

## License / usage notes

- Everything in this repository is **Apache-2.0**; see `LICENSE`.
- `patches/deep_gemm.py` is a **derivative work of vLLM** (Apache-2.0): nightly's own
  file, minimally modified — not a clean-room rewrite.
- The model is subject to **DeepSeek's own license**; this repo ships no weights.
- Validated on **one DGX Station GB300 at TP1**. Other SM103 configs (multi-GPU /
  P/D-disaggregated) are untested, and a different GB300 SKU could report a different
  compute capability — check `nvidia-smi --query-gpu=compute_cap` before assuming this
  applies.
