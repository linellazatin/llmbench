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

# run the same suite through an agentic CLI backend instead
python3 llmbench.py run --backend codex --suites agentic
python3 llmbench.py run --backend dsh --suites reasoning,spatial

# browse tasks, re-render a report, compare runs, or monitor resources standalone
python3 llmbench.py list
python3 llmbench.py report results/<run-id>
python3 llmbench.py compare results/<run-a> results/<run-b>
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
| `api` | ✅ | direct streaming `/chat/completions` against an OpenAI-compatible endpoint — cleanest TTFT/tok/s/token numbers (ground truth); only backend for the `tools` suite (OpenAI `tools` parameter) |
| `dsh` | ✅ | `dsh --profile headless --patch <overlay>`; the per-run overlay defines a **self-contained provider** (`baseURL` from `--endpoint`, key from `--dsh-api-key-env`, model) plus default-model routing — so `--endpoint` is honored and `~/.dsh/settings.yaml` is never touched. Prints the final assistant message, no usage → tokens estimated (~4 chars/token, `token_source=estimate`) |
| `codex` | ✅ | `codex exec --json` with `-c` provider overrides pointing at the local endpoint; JSONL events, per-turn `usage` (server-side tokens) |
| `claude` | ✅ | `claude -p --output-format stream-json --verbose` → `ANTHROPIC_BASE_URL` = endpoint minus `/v1`. **No bridge needed: vLLM ≥ 0.23 serves the Anthropic `/v1/messages` protocol natively** (verified non-streaming + streaming SSE incl. `thinking` blocks). Server-side tokens from the `result` event |

CLI backends run the model as a **full agent**: they get a working directory
(the per-task scratch dir) plus shell/file tools, which is what the `agentic`
suite exercises. All four backends are validated live against `qwen3.8` (see
*Host facts* below).

Claude-specific plumbing (non-interactive agentic as root):

- `--dangerously-skip-permissions` is refused for root, so instead each task
  gets a project-level allowlist (`<scratch>/.claude/settings.json` with
  `permissions.allow: Bash/Read/Write/Edit`) **and** a pre-accepted trust
  entry for the scratch path in `~/.claude.json`
  (`projects["<scratch>"].hasTrustDialogAccepted = true`) — the documented
  workaround for the untrusted-workspace gate. The trust file is
  read-modify-written atomically per task, and the entry is removed again when
  the task finishes. Caveat: the allowlist grants the benchmarked model full
  `Bash`/`Write` access in the scratch dir — fine on an isolated bench host,
  not on a machine with anything you care about.

## Metrics (see `metrics_legend` in run.json)

- `ttft_s` — `api`: request sent → first **content** token. `cli`: spawn → first stdout byte (includes CLI startup)
- `ttft_model_s` — request/spawn → first **model-originated** output (codex: first item event; claude: first assistant event; dsh: final message, it buffers). Use this for cross-backend *model* latency — `ttft_s` on a CLI is mostly process startup
- `ttft_first_reasoning_s` — `api` only: request sent → first **reasoning** token (reasoning models stream thinking before content)
- `gen_tps` — completion tokens / (end − `ttft_model_s`). For `api` this is the server decode rate (incl. reasoning); on CLI backends it also spans tool round-trips; suppressed when the window < 1 s (buffered CLIs)
- `decode_tps` — content chunks / (end − ttft) → content-only rate, `api` only (suppressed for windows < 250 ms)
- `overall_tps` — completion tokens / total latency (incl. TTFT)
- tokens — from server `usage` when the CLI reports it (`token_source=server`, cumulative over the task's internal API calls **including the harness system prompt**); else ~4 chars/token of the task prompt only (`token_source=estimate`, e.g. `dsh`). `completion_tokens` **includes reasoning tokens**

The `api` backend is the reference for pure server metrics; CLI numbers include harness overhead. **Run backends sequentially when they share one endpoint** (e.g. this host's single vLLM) — parallel runs contend on the NPU and corrupt each other's latency/throughput.

## Resource sampling

A background thread samples at `--sample-interval` (default 2 s) during each
task:

- **Accelerator** (auto-detected, in order): `nvidia-smi` → `rocm-smi` →
  `docker exec <container> npu-smi info` (this host: container `qwen3.8`;
  override with `--npu-container` or `LLMBENCH_NPU_CONTAINER`). Slow on NPU
  (~1.7 s/call), so NPU readings run in their own loop and the latest reading
  is attached to each fast /proc sample.
- **System**: CPU %, RAM used/total, page cache (Cached+SReclaimable), swap.
- **Process tree** (CLI backends): CPU % and RSS of the CLI process and all
  its descendants — attached via `set_root(pid)` right after spawn, so the
  sample window covers the whole agent run.

Per-chip aggregates (avg/min/peak/p50/p95) land in each task's `resources`
block and in the report's Resources section.

## Suites

All tasks are original and deterministic (no dataset downloads, no network
needed). Select with `--suites` (comma-separated; default = all **except the
opt-in `longctx` suite**, which you must name explicitly).

| suite | tasks | what's scored | verifier |
|---|---|---|---|
| `smoke` | 3 | pipeline sanity | exact / math / answer |
| `reasoning` | 8 | word problems, knights-&-knaves, syllogisms, sequences, logic fallacies | `ANSWER:` contract + numeric/string equivalence |
| `spatial` | 8 | seeded CLEVR-style 3D table scenes (count/compare/relational), grid shortest path (length + move string), map directions, rotation, cube counting | math / answer / move-string simulation on the grid |
| `ifeval` | 8 | programmatically checkable constraints (exact sentence/word counts, keyword include/exclude, JSON-only shape, line prefixes, casing, punctuation bans, first line) | constraint checkers (all must pass) |
| `coding` | 8 | original unit-tested problems (anagram, roman, intervals, moving avg, parens, flatten, caesar, top-k) | solution read from `scratch/solution.py` if the agent wrote one, else extracted from the first ` ```python ` block; unittest suite run (60 s timeout) |
| `tools` | 8 | OpenAI-style tool calls: correct function, argument subsets, arithmetic expressions safe-eval'd, multi-call, and a "don't over-call" task | tool-call matcher (api backend only) |
| `agentic` | 5 | CLI-only: create a file with exact content, sum a data file, shell-compute, script-and-run word count, fix a failing unit test in a provided repo | file-content / `ANSWER:` math / post-task command re-run (dsh, codex, claude) |
| `mullang` | 8 | prompts in zh / ja / ko / ar / ru / de / es / hi — word problems, a grid path (ru), a 2-sentence constraint (de), a knights-&-knaves (es), exact word count (hi) | existing verifiers (math / gridpath / constraints / answer) — language-neutral |
| `longctx` | 3 | opt-in: needle retrieval in a deterministic filler log — 8k, 32k (3 needles, order), 128k tokens | `ANSWER:` exact/ordered (api backend only; a 128k-token prompt exceeds the CLI per-argument length limit) |

`python3 selftest.py` runs fast offline verifier self-tests.

## Run modes

- `--retries N` (default 1) — re-attempt a task that **errors** (transport/
  crash), not one that merely fails its verifier (a fail is a real verdict).
  Each task records its `attempts`; the scratch dir is wiped between attempts so
  a crashed run's half-written `solution.py` can't leak into a retry.
- `--concurrency N` (default 1) — run N tasks in flight (load mode). Each task
  keeps its own resource sampler, so per-task NPU/CPU samples stay attributed;
  the shared endpoint is the thing under load.
- `--repeat N` — repeat every task N times. Aggregates then carry a 95% CI
  (`*_ci95` fields in `run.json`; `±` in the console summary) so single-run
  wiggles are distinguishable from signal.

## Notes / host facts (2026-10-13)

- Local model: `qwen3.8` = Qwen3.8-27B-w8a8 (W8A8, 262k ctx, MTP spec-decode)
  served by vLLM 0.23.0 on `0.0.0.0:8000` inside host-networked container
  `qwen3.8` (8× Ascend 910B4 NPUs, 64 GB HBM each).
- Host `npu-smi` is broken; NPU state is read via
  `docker exec qwen3.8 npu-smi info`.
- Live validation (all against `qwen3.8`): `api` full 43-task run = 43/43 pass
  (gen ~88 tok/s, TTFT p50 2.5 s); `dsh` 39/40 (one genuine model miss);
  `codex` agentic 5/5 (including fixing a failing unit test); `claude` smoke
  3/3 + agentic 5/5 — all CLIs fixed `buggy.py` and created exact-content
  files. vLLM 0.23.0 serves Anthropic `/v1/messages` natively (no bridge).
- **Definitive 4-backend baseline (2026-10-13, runs 20261013-1718…1742,
  sequential on the shared endpoint):** 163/163 task-runs passed
  (api 43 + dsh 40 + codex 40 + claude 40). Standout numbers: api TTFT avg
  4.7 s / gen 84 tok/s (content decode 27 tok/s); dsh is buffered (TTFT ≈
  total, ~12 s) with no token usage (estimated); codex has the lowest TTFT
  (0.36 s — events stream early) but a ~16 s per-task overhead; claude ~12 s
  per task. Prompt-token cost of the CLI harnesses dwarfs the prompt itself
  (codex ~593k, claude ~1.16M across 40 tasks vs api 7.8k) — expected, and
  visible in the `compare` table.
- The DSH file sandbox denies writes to `~/.dsh` and `~/.codex/tmp`; the CLI
  backends therefore need a one-shot `danger-full-access` escalation in this
  harness (on a normal shell they just run).
