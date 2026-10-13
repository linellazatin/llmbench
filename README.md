# llmbench

Non-interactive benchmark harness for locally served models. Drives a model
through a deterministic, offline task suite and records **quality** (pass/fail
verdicts), **latency** (TTFT), **throughput** (tok/s), **token consumption**,
and **resource usage** (accelerator HBM + core %, power, CPU, RAM, page cache,
swap) — with both machine-readable (`run.json`) and human-readable
(`report.md`) output.

Single file, Python 3.10+ stdlib only, zero dependencies.

## Quick start

```bash
# run the smoke suite against the local vLLM endpoint (OpenAI-compatible API)
python3 llmbench.py run --backend api --endpoint http://127.0.0.1:8000/v1 --model qwen3.8

# browse tasks, re-render a report, or monitor resources standalone
python3 llmbench.py list
python3 llmbench.py report results/<run-id>
python3 llmbench.py sample --seconds 30
```

Each `run` writes to `results/<timestamp>_<model>_<backend>/`:

| file | what |
|---|---|
| `run.json` | machine-readable results (schema v1): config, host, per-task metrics, verdicts, resource aggregates, `metrics_legend` |
| `report.md` | human-readable report (scorecard, latency/throughput, per-chip resources, per-task table) |
| `tasks/<id>.md` | per-task prompt, reasoning, response, verdict |
| `tasks/<id>.samples.jsonl` | raw per-interval resource samples |

## Backends

| backend | status | how |
|---|---|---|
| `api` | ✅ | direct streaming `/chat/completions` against an OpenAI-compatible endpoint — cleanest TTFT/tok/s/token numbers (ground truth); also the only backend for the `tools` suite (OpenAI `tools` parameter) |
| `dsh` | ⏳ phase 2 | `dsh --profile headless` (already wired to the local vLLM via `~/.dsh/settings.yaml`) |
| `codex` | ⏳ phase 2 | `codex exec --json` with `-c` provider overrides pointing at the local endpoint |
| `claude` | ⏳ phase 2+ | `claude -p --output-format stream-json`; needs an Anthropic-protocol endpoint (built-in OpenAI→Anthropic bridge planned) |

## Metrics (see `metrics_legend` in run.json)

- `ttft_s` — request sent → first **content** token
- `ttft_first_reasoning_s` — request sent → first **reasoning** token (reasoning models stream thinking before content)
- `gen_tps` — completion tokens / (end − first token of any kind) → server decode speed incl. reasoning
- `decode_tps` — content chunks / (end − first content token) → content-only rate (suppressed for windows < 250 ms)
- `overall_tps` — completion tokens / total latency (incl. TTFT)
- tokens — from server `usage`; `completion_tokens` **includes reasoning tokens**

CLI backends (phase 2) report TTFT as time-to-first-stdout-byte, which
includes harness overhead; the `api` backend is the reference for pure
server metrics.

## Resource sampling

A background thread samples at `--sample-interval` (default 2 s) during each
task:

- **Accelerator** (auto-detected, in order): `nvidia-smi` → `rocm-smi` →
  `docker exec <container> npu-smi info` (this host: container `qwen3.8`;
  override with `--npu-container` or `LLMBENCH_NPU_CONTAINER`). Slow on NPU
  (~1.7 s/call), so NPU readings run in their own loop and the latest reading
  is attached to each fast /proc sample.
- **System**: CPU %, RAM used/total, page cache (Cached+SReclaimable), swap.
- **Process tree** (phase 2, for CLI backends): CPU % and RSS of the CLI
  process and descendants.

Per-chip aggregates (avg/min/peak/p50/p95) land in each task's `resources`
block and in the report's Resources section.

## Suites

All tasks are original and deterministic (no dataset downloads, no network
needed). Select with `--suites` (comma-separated; default = all).

| suite | tasks | what's scored | verifier |
|---|---|---|---|
| `smoke` | 3 | pipeline sanity | exact / math / answer |
| `reasoning` | 8 | word problems, knights-&-knaves, syllogisms, sequences, logic fallacies | `ANSWER:` contract + numeric/string equivalence |
| `spatial` | 8 | seeded CLEVR-style 3D table scenes (count/compare/relational), grid shortest path (length + move string), map directions, rotation, cube counting | math / answer / move-string simulation on the grid |
| `ifeval` | 8 | programmatically checkable constraints (exact sentence/word counts, keyword include/exclude, JSON-only shape, line prefixes, casing, punctuation bans, first line) | constraint checkers (all must pass) |
| `coding` | 8 | original unit-tested problems (anagram, roman, intervals, moving avg, parens, flatten, caesar, top-k) | solution extracted from first ` ```python ` block, unittest suite run in a temp dir (60 s timeout) |
| `tools` | 8 | OpenAI-style tool calls: correct function, argument subsets, arithmetic expressions safe-eval'd, multi-call, and a "don't over-call" task | tool-call matcher (api backend only) |

`python3 selftest.py` runs fast offline verifier self-tests.

## Notes / host facts (2026-10-13)

- Local model: `qwen3.8` = Qwen3.8-27B-w8a8 (W8A8, 262k ctx, MTP spec-decode)
  served by vLLM 0.23.0 on `0.0.0.0:8000` inside host-networked container
  `qwen3.8` (8× Ascend 910B4 NPUs, 64 GB HBM each).
- Host `npu-smi` is broken; NPU state is read via
  `docker exec qwen3.8 npu-smi info`.
- `dsh` headless couldn't be smoke-tested inside the DSH sandbox (it must
  create `~/.dsh/profiles/headless`); first live check happens on a normal
  shell.
