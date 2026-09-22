# DeepSeek-V4.1-Flash on DGX Station GB300 (SM103) — three small patches, official vLLM nightly, 100 tok/s

Run **DeepSeek-V4.1-Flash** (522B MoE / 8-16B active) on an **NVIDIA DGX Station GB300**
with the official `vllm/vllm-openai:nightly` image — plus three small bind-mounted
patches (one required, two optional-but-recommended) and a measured launch config.

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
  deep_gemm.py                 # grouped UE8M0 scale layout (the one live vLLM bug)
  kernel_warmup.py             # autotune floor-mapping fix, one line (vLLM side)
  flashinfer_mxfp8_rowmajor.py # row-major MXFP8 weight storage (FlashInfer stride bug)
examples/
  docker-compose.yml           # working config, all values measured
  verify.sh                    # math / long-context / tool-calling / vision checks
```

Two of the three are optional-but-recommended: `deep_gemm.py` is required for a clean
boot on any nightly right now. The other two unlock `--linear-backend auto` (pure
cute-dsl MXFP8 GEMMs — faster decode AND prefill than the default, plus ~2.4x KV
capacity via freed HBM); without them, set `--linear-backend marlin` instead and skip
those two mounts. All three are bind-mounts — upstream images stay untouched.

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

### MXFP8 linear weight storage — `flashinfer_mxfp8_rowmajor.py`

FlashInfer's cute-dsl `mm_mxfp8` (the `--linear-backend auto` path on SM100/103) stores
linear weights column-major `[K, N]` and passes them straight to the FFI. Any later
copy that flattens the tensor (e.g. vLLM's unpinned UVA re-offload of CPU-offloaded
weights) silently turns it row-major, and the first forward dies with
`ValueError: Mismatched mB.strides[1]` in `compiled_gemm`. The fix is to store the
fixed point of every copy path — row-major `[N, K]` — and pass `weight.t()` at apply
time (the same contract CUTLASS-class kernels use). Validated against the reference
GEMM at all dense-layer shapes; boot- and serving-proven at M 1…16384.

With this patch, pure cute-dsl beats the default marlin path on **both** sides of the
step: decode (8–10 µs vs 14–16 µs per fused_wqa_wkv under CUDA-graph replay) and
prefill (~204 µs vs ~914 µs at M=16384). It also deletes marlin's ~3.7 GiB HBM
quantization stash — on our 962 GB card that alone moved KV capacity 2.13M → 5.12M
tokens.

### FlashInfer warmup floor-mapping — `kernel_warmup.py`

Inside `autotune(tuning_buckets=...)` the FlashInfer autotuner *replaces* the op's own
round-up bucket mapper with a round-down one. vLLM's warmup runs the model inside that
context, so a dummy-run forward at a non-bucket M (the DSpark draft at 8 reqs × 3 =
24 tokens) looks up the bucket-16 tactic — whose split-K MMA tile (128,16) is then
rejected by the apply-time validator, because `mma_tiler_mn_for_m(24)` is (128,32):

```
ValueError: Invalid MXFP8 split-K tactic: ((128, 16), (1, 1), True, False, 4)
```

Outside the tune context (serving + CUDA-graph capture) the op's own round-up mapper
is used, so **only the warmup pass ever crashes** — the engine crash-loops at boot
while serving would have been fine. Fix is one line at `kernel_warmup.py:363`:
`autotune(tuning_buckets=tuning_buckets, round_up=True)`. Verified by a standalone
repro on the GB300: floor-mapped M=24/12/40 all fail with byte-identical tuples to the
engine crash; `round_up=True` clears all three in the same process, same cache.

> Broader hazard, not fixed here: the *base* (non-split-K) tactic candidates include
> small swap-AB tiles that get **no** apply-time check — under floor mapping a
> (128,16) winner tuned at bucket 16 can be applied at M=24, silently outside its
> envelope. The split-K validator is just the one place that notices. `mm_fp4`
> shares the same tactic generator, so NVFP4 models with speculative decoding are
> exposed to the same family of bugs.

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
| `--gpu-memory-utilization` | **0.9230** | vLLM logs its own safe max on boot; 0.95 OOMs during graph capture. |
| `--cpu-offload-gb` | **75** + `--cpu-offload-params experts` | see the sweep below; `experts` also pins the dense projections back to HBM (they're read every decode step). |
| `--max-model-len` | **1048576** | required for 1M context. |
| `--linear-backend` | **auto** | Pure cute-dsl MXFP8 with the two FlashInfer patches. Beats marlin on both decode (8–10 µs vs 14–16 µs under replay) and prefill (~204 µs vs ~914 µs at M=16384), and frees marlin's ~3.7 GiB stash (KV 2.13M → 5.12M tokens). Use `marlin` if you skip the patches — attention runs the native `flash_mla_mega_attn` path regardless of this flag. |
| `--speculative-config` | `dspark`, **3 tokens, greedy** | k=3 greedy beats k=5 probabilistic on this model: 63% acceptance efficiency (1.90 accepted/step vs 1.51) — greedy matches the draft's argmax training; the extra k=5 slots were rejected anyway. |
| `--tokenizer-mode` / parsers | `deepseek_v41` | V4.1 DSML dialect; the `deepseek_v4` parser leaks raw markup into content. |
| `cudagraph_capture_sizes` | **[4,8,12,16,20,24,28,32]** | under dspark, sizes must be multiples of `decode_query_len=4` AND ≤ `max_num_seqs×4` — vLLM silently DROPS anything else and those batches run **eager**. [1,6,12,18,24] kept only 5/5 and left 7-8-seq decode eager. |
| `--max-num-batched-tokens` | **16384** | ~5× faster long-prompt prefill vs 8192; costs ~1.7M KV tokens (see Results note). |

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

DGX Station GB300, single GPU, `--tensor-parallel-size 1`, thinking off.
Tuned config = the full recipe below (k=3 DSpark + pure cute-dsl + capture sizes
`[4,8,12,16,20,24,28,32]`):

| metric | 0909 image (six patches) | first nightly | **tuned (this repo)** |
|---|---|---|---|
| KV cache | 4,458,466 tokens | 6,845,219 (6.53× @1M) | **5,123,184 (4.89× @1M)** |
| decode (prose, two-length slope) | — | ~83 tok/s | **100.1 tok/s** |
| decode (counting; DSpark accepts 100%) | — | — | 198.5 tok/s |
| prefill (109k-token prompt) | — | ~4,000 tok/s | **22,975 tok/s** |
| CUDA graphs (FULL) | partial | 5/5 but top batches eager | **8/8 + DSpark 6/6** |
| attention path | pre-mega | native `flash_mla_mega_attn` | same |
| correctness | `17*19 → 323` | `17*19 → 323` | `17*19 → 323` |
| vision | untested | 1×1 image → correct short description | verified again |
| tool calling | via recovery patches | upstream rewritten parser; smoke-tested | smoke-tested clean |

> The tuned KV number is LOWER than the first nightly's 6.85M on purpose: 6.85M came
> from `--max-num-batched-tokens 8192` (prefill chunks capped at 8k). The tuned recipe
> raises it to 16384 for ~5× faster long-prompt prefill and pays ~1.7M tokens of KV —
> and still holds **~5 concurrent 1M-token sessions**. If you want max KV instead of
> prefill speed, set `--max-num-batched-tokens 8192`.

Boot: weights 48/48 shards in ~90 s, DSpark draft load, autotune warmup ~3 min
(cache-hit ~5 s), CUDA graph capture 3 s, health 200 at ~14 min (first boot
JIT-compiles kernels).

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

The three mounts (adjust the left sides to your checkout):

```yaml
volumes:
  - "/opt/deepseek-v4.1-flash-gb300/patches/deep_gemm.py:/usr/local/lib/python3.12/dist-packages/vllm/utils/deep_gemm.py:ro"
  # optional pair: unlocks --linear-backend auto (pure cute-dsl). Skip BOTH
  # and use --linear-backend marlin instead if you want a zero-extra-patch boot.
  - "/opt/deepseek-v4.1-flash-gb300/patches/flashinfer_mxfp8_rowmajor.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mxfp8/flashinfer.py:ro"
  - "/opt/deepseek-v4.1-flash-gb300/patches/kernel_warmup.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/warmup/kernel_warmup.py:ro"
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
