# llmbench

Non-interactive benchmark harness for locally served models. Drives a model through a deterministic, offline task suite and records **quality** (pass/fail verdicts), **latency** (TTFT), **throughput** (tok/s), **token consumption**, and **resource usage** (accelerator HBM + core %, power, CPU, RAM, page cache, swap) — with both machine-readable (`run.json`) and human-readable (`report.md`) output.

Single file, Python 3.10+ stdlib only, zero dependencies.

## Quick start

```bash
# run the smoke suite against the local vLLM endpoint (OpenAI-compatible API)
# -- adjust --endpoint/--model to your server: any OpenAI-compatible engine
# works (vLLM, llama.cpp llama-server, Ollama /v1, cloud); the preflight
# line tells you before the first task whether the endpoint is up and the
# model is listed
python3 llmbench.py run --backend api --endpoint http://127.0.0.1:8000/v1 --model qwen3.8

# run suites through the agentic CLI backends instead
# (same endpoint/model, from flags or the LLMBENCH_* env vars)
python3 llmbench.py run --backend codex --suites agentic
python3 llmbench.py run --backend dsh --suites reasoning,spatial
python3 llmbench.py run --backend pi --suites reasoning,coding

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
| `sample.csv` | per-interval resource time series (accelerator + system columns; one row per interval, all tasks) |

## Backends

| backend | status | how |
|---|---|---|
| `api` | ✅ | direct streaming `/chat/completions` against an OpenAI-compatible endpoint — cleanest TTFT/tok/s/token numbers (ground truth); only backend for the `tools` suite (OpenAI `tools` parameter) |
| `dsh` | ✅ | `dsh --profile headless --patch <overlay>`; the per-run overlay defines a **self-contained provider** (`baseURL` from `--endpoint`, key from `--dsh-api-key-env`, model) plus default-model routing — so `--endpoint` is honored and `~/.dsh/settings.yaml` is never touched. Prints the final assistant message, no usage → tokens estimated (~4 chars/token, `token_source=estimate`) |
| `codex` | ✅ | `codex exec --json` with `-c` provider overrides pointing at the local endpoint; JSONL events, per-turn `usage` (server-side tokens) |
| `claude` | ✅ | `claude -p --output-format stream-json --verbose` → `ANTHROPIC_BASE_URL` = endpoint minus `/v1`. **No bridge needed: vLLM ≥ 0.23 serves the Anthropic `/v1/messages` protocol natively** (verified non-streaming + streaming SSE incl. `thinking` blocks). Server-side tokens from the `result` event |
| `pi` | ✅ | `pi --mode json --no-session --provider llmbench --model <m> --api-key <k>`; the per-run agent dir under the task scratch (`PI_CODING_AGENT_DIR`) holds a self-contained `models.json` provider (`baseUrl` from `--endpoint`, model, key) — `~/.pi/agent` is never touched, and a bare scratch dir triggers no trust prompt. **No trust/permission plumbing needed** (unlike claude): pi runs tools without per-call approval in one-shot mode. Server-side tokens from per-response `usage` events. `--pi-tools <list>` pins pi's toolset (isolation parity with the claude allowlist; default = pi's full toolset). `--pi-thinking <level>` sets pi's thinking level (parity knob; default = model default; on qwen-style vLLM models the level maps to `enable_thinking`: off disables, other levels force on), and thinking text is captured into the task `.md` Reasoning section (artifact parity with the api backend) |

CLI backends run the model as a **full agent**: they get a working directory (the per-task scratch dir) plus shell/file tools, which is what the `agentic` suite exercises. All five backends are validated live against `qwen3.8` (see *Host facts* below).

Claude-specific plumbing (non-interactive agentic as root):

- `--dangerously-skip-permissions` is refused for root, so instead each task gets a project-level allowlist (`<scratch>/.claude/settings.json` with `permissions.allow: Bash/Read/Write/Edit`) **and** a pre-accepted trust entry for the scratch path in `~/.claude.json` (`projects["<scratch>"].hasTrustDialogAccepted = true`) — the documented workaround for the untrusted-workspace gate. The trust file is read-modify-written atomically per task, and the entry is removed again when the task finishes. Caveat: the allowlist grants the benchmarked model full `Bash`/`Write` access in the scratch dir — fine on an isolated bench host, not on a machine with anything you care about.

## Using llmbench per backend — walkthrough & command reference

Every backend shares the same `run` command surface. What differs per backend: (a) the CLI invocation and environment the harness assembles, (b) which suites are admissible, (c) the backend-specific flags, and (d) which shared flags actually take effect on the wire. The shared table below lists every flag with its full value space once; each backend table then covers its specifics and the differences.

Current task inventory (59 tasks total): `smoke` 3 · `reasoning` 8 · `spatial` 8 · `ifeval` 8 · `coding` 8 · `tools` 8 · `agentic` 5 · `mullang` 8 · `longctx` 3 (opt-in). `list` prints them with verifiers, per-task backend gates, and — for the heavy tiers — the context window they need.

### Shared `run` flags (all backends)

| flag | values | default | notes |
|---|---|---|---|
| `--backend` | `api` \| `dsh` \| `codex` \| `claude` \| `pi` | `api` | |
| `--endpoint` | any `http(s)://` URL (validated; fail-fast exit 2 otherwise), e.g. `http://127.0.0.1:8000/v1` | `$LLMBENCH_ENDPOINT` or `http://127.0.0.1:8000/v1` | the API root the backend talks to |
| `--model` | any model id the endpoint serves | `$LLMBENCH_MODEL` or `qwen3.8` | for CLI backends also the model name handed to the CLI |
| `--api-key` | any string | `$LLMBENCH_API_KEY` or `llmbench` | vLLM ignores it, but it is still sent |
| `--suites` | comma-separated suite names, or `full` | all except `longctx` | `full` selects every suite, including heavy `longctx`; gated tasks on a non-admitted backend are logged as skipped, not errors |
| `--tasks` | comma-separated task ids (see `list`), e.g. `reasoning/knights-knaves,coding/caesar` | all in the selected suites | implies the tasks' suites — this is how `longctx` is reachable |
| `--repeat` | int ≥ 1 | `1` | aggregates with ≥ 2 samples gain 95 % CIs (`*_ci95`) |
| `--max-tokens` | int | `4096` | **effective on `api` and `pi` only** — the other CLIs have no output-budget knob wired. Also the output reservation the `--context-tokens` gate adds to a heavy prompt |
| `--temperature` | float | **unset** | **effective on `api` only.** Unset = the parameter is *omitted from the request*, so the endpoint's own configured sampling applies. Pass `--temperature 0` for greedy decoding (what the recorded baselines used). Endpoints that reject the key outright (OpenAI-style reasoning models) are only benchmarkable while it is unset |
| `--timeout` | float, seconds | `600` | per-task wall limit on every backend (`api` re-arms the socket timeout so a slow-but-trickling stream is still cut; CLIs kill the process tree); a timeout is an `error` (→ retried) |
| `--context-tokens` | int | `0` (= `$LLMBENCH_CONTEXT_TOKENS`, 0 = unknown) | the served window in tokens (prompt + output). When declared, tasks whose prompt + `--max-tokens` cannot fit are **skipped with a logged reason** instead of drawing a context-length error, and `pi` declares this window in its generated `models.json`. Estimates are worst-case (see the Context gating note): a skip line always shows its own arithmetic |
| `--retries` | int ≥ 0 | `1` | extra attempts for tasks that **error** only — a verifier fail is a real verdict and is never retried |
| `--concurrency` | int ≥ 1 | `1` | load mode: N tasks in flight, each keeping its own resource sampler; the shared endpoint is the load target — run backends sequentially, never in parallel |
| `--sample-interval` | float, seconds | `2.0` | resource-sampling cadence during each task |
| `--no-warmup` | flag (no value) | warmup runs | skip the single unscored warmup call (one chat request on `api`; a first-pass smoke task on CLI backends) |
| `--npu-container` | docker container name | `$LLMBENCH_NPU_CONTAINER`, else none | NPU state via `docker exec <c> npu-smi info` (NPU-in-container hosts) |
| `--out` | directory | `<script dir>/results` | run dirs are created under it |

### Other commands

| command | exact form | notes |
|---|---|---|
| `list` | `python3 llmbench.py list` | every suite/task with verifier and backend gates |
| `sample` | `python3 llmbench.py sample [--seconds S] [--interval I] [--container C]` | standalone resource monitor (defaults: 10 s, 2.0 s, env container) |
| `report` | `python3 llmbench.py report <run_dir>` | re-render `report.md` from an existing `run.json` |
| `compare` | `python3 llmbench.py compare <run_dir>... [--save FILE.md]` | side-by-side table (also printed); `--save` writes the file |

### Usage proxy — measuring true token cost for any backend

For CLIs that don't report usage (notably `dsh`), or to audit server-side usage of any backend, run a loopback usage-logging proxy between the harness and the engine:

```bash
# 1. start the proxy (forwards to 127.0.0.1:8000, appends one JSON record per request)
python3 tools/usage_proxy.py --port 8001 --log results/proxy-usage.jsonl &

# 2. run llmbench as usual, pointing the endpoint at the proxy
python3 llmbench.py run --backend dsh --endpoint http://127.0.0.1:8001/v1

# 3. dsh only: correlate the captured calls to tasks (prints per-task/suite totals,
#    writes <run_dir>/../dsh-proxy/measured.json)
python3 tools/dsh_usage_corr.py results/<run_dir> results/proxy-usage.jsonl
```

| item | value |
|---|---|
| what it does | transparent forwarder to `--upstream-host --upstream-port` (default `127.0.0.1:8000`); model names, auth, bodies and streaming pass through unchanged |
| usage capture | non-streaming: `usage` from the response body (included by default); streaming: injects `stream_options.include_usage` and captures the engine's final usage chunk — the client's stream is otherwise untouched |
| log records | one JSON line per request: epoch time, path, status, model, stream flag, prompt/completion/total tokens, and the last user message (`user_full` + head/tail) as a fingerprint for task correlation |
| overhead | loopback, per-connection; task latency in the measured dsh runs was indistinguishable from the direct runs |
| note | keep a single client per log — `dsh_usage_corr.py` attributes calls to tasks by prompt fingerprint + temporal order, which assumes the harness is the only caller. For other backends the raw JSONL is enough: records are time-ordered, one per internal API call |

Reference run: dsh through the proxy on 2026-10-14 (`results/20261014-160020_qwen3.8_dsh`) — 129 captured calls, 778.6k measured total tokens vs the harness's 8.6k visible-text estimate (see the dsh table above).

### api — reference backend

No CLI binary needed; streams `/chat/completions` directly.

```bash
python3 llmbench.py run --backend api --suites smoke      # 3 tasks — sanity
python3 llmbench.py run --backend api                    # 51 tasks (longctx opt-in; agentic is CLI-only)
python3 llmbench.py run --backend api --suites full      # all suites, including the 3 heavy longctx tasks
python3 llmbench.py run --backend api --suites longctx   # + the 3 heavy needle tasks
python3 llmbench.py run --backend api --suites longctx --context-tokens 131072  # gate to the served window
python3 llmbench.py run --backend api --tasks reasoning/knights-knaves,coding/caesar
python3 llmbench.py run --backend api --max-tokens 1024 --temperature 0.2 --suites smoke
```

| item | value |
|---|---|
| invocation | `POST <endpoint>/chat/completions` — streaming, `stream_options.include_usage`, `Bearer` auth. Works against any OpenAI-compatible engine (vLLM, llama.cpp `llama-server`, Ollama `:11434/v1`, cloud) — `/models` preflight warns if the endpoint is down or the model is unlisted |
| backend-specific flags | — (none) |
| reasoning | both delta field conventions captured — `reasoning` and `reasoning_content` (engines differ); lands in the task `.md` Reasoning section and sets `ttft_first_reasoning_s` |
| suites | 8 of 9 (all except CLI-only `agentic`) — 54 admissible, 51 by default (longctx opt-in); the only backend for `tools`, reference for `longctx` |
| shared flags effective | `--max-tokens` ✓, `--temperature` ✓ when passed (the only backend for both; omitted from the request when the flag is unset), `--context-tokens` ✓ (screens over-sized tasks before sending) |
| tokens / latency | server `usage` (final chunk); `ttft_s` = request → first content token — cleanest numbers, no harness overhead. If the engine reports no streamed usage, tokens fall back to the ~4 chars/token estimate (`token_source=estimate`, as on CLI backends) |

### dsh

Requires the `dsh` CLI on PATH.

```bash
python3 llmbench.py run --backend dsh --suites smoke
python3 llmbench.py run --backend dsh                       # 48 tasks; tools+longctx auto-skipped (logged)
python3 llmbench.py run --backend dsh --suites agentic
python3 llmbench.py run --backend dsh --dsh-api-key-env MY_OTHER_KEY_ENV --suites smoke
```

| item | value |
|---|---|
| invocation | `dsh --profile headless --patch <scratch>/llmbench.patch.yaml <prompt>` (one task per process; prompt as final argv) |
| per-run overlay | the patch defines a self-contained provider `llmbench` (`baseURL` = `--endpoint`, `api: openai-completions`, `apiKeyEnv`) plus default-model routing → `~/.dsh/settings.yaml` is never touched |
| env set | `<--dsh-api-key-env>` = `--api-key` |
| backend-specific flags | `--dsh-api-key-env <NAME>` — name of the env var dsh's per-run provider reads the key from; any non-empty string (default `MY_GATEWAY_API_KEY`) |
| suites | 7 (all except `tools` [api-only] and `longctx` [argv size limit]) = 48 tasks |
| shared flags effective | `--max-tokens` ✗, `--temperature` ✗ (no knobs wired for this CLI), `--context-tokens` ✓ (selection gate only) |
| tokens / latency | **estimated** (~4 chars/token of the prompt only, `token_source=estimate`) — dsh prints no usage; stdout-buffered (it *does* stream over the wire — SSE incl. thinking — but prints only the final message), so `ttft_model_s` = final message (TTFT ≈ total). True usage is measurable through the loopback usage-logging proxy `tools/usage_proxy.py` (+ `tools/dsh_usage_corr.py` for request→task correlation): measured 2026-10-14, run `results/20261014-160020_qwen3.8_dsh`, dsh's real cost is 778.6k total tokens / 48 tasks (~15.7k prompt/task — a ~170-token session-title call + ≥1 agent call with ~8.9k of dsh scaffolding each) |

### codex

Requires the `codex` CLI on PATH.

```bash
python3 llmbench.py run --backend codex --suites smoke
python3 llmbench.py run --backend codex                     # 48 tasks; tools+longctx auto-skipped
python3 llmbench.py run --backend codex --suites agentic    # file ops, shell, fixing a failing unit test
python3 llmbench.py run --backend codex --codex-provider vllm
```

| item | value |
|---|---|
| invocation | `codex exec --json --skip-git-repo-check --ephemeral -C <scratch> -m <model> -c model_provider=<p> -c model_providers.<p>.base_url=<endpoint> -c model_providers.<p>.name=<p> --dangerously-bypass-approvals-and-sandbox <prompt>` |
| env set | `OPENAI_API_KEY` = `--api-key` |
| backend-specific flags | `--codex-provider <NAME>` — the `model_providers` entry name; the harness sets its `base_url`/`name` inline via `-c`, so no pre-existing codex config entry is needed (any string; default `$LLMBENCH_CODEX_PROVIDER` or `vllm`) |
| suites | 7 (same as dsh) = 48 tasks |
| shared flags effective | `--max-tokens` ✗, `--temperature` ✗, `--context-tokens` ✓ (selection gate only) |
| tokens / latency | server `usage` from `turn.completed` (cumulative over the task's turns); `ttft_model_s` = first `item.*` event; JSONL stream — events surface in the task `.md` `CLI events` section |

### claude

Requires the `claude` CLI on PATH.

```bash
python3 llmbench.py run --backend claude --suites smoke
python3 llmbench.py run --backend claude                   # 48 tasks; tools+longctx auto-skipped
python3 llmbench.py run --backend claude --suites agentic
```

| item | value |
|---|---|
| invocation | `claude -p --output-format stream-json --verbose --model <m> --no-session-persistence <prompt>` (`+ --append-system-prompt <sys>` for tasks with a system prompt) |
| env set | `ANTHROPIC_BASE_URL` = `--endpoint` minus the trailing `/v1`; `ANTHROPIC_AUTH_TOKEN` = `ANTHROPIC_API_KEY` = `--api-key` |
| backend-specific flags | — (none) |
| suites | 7 (same as dsh) = 48 tasks |
| shared flags effective | `--max-tokens` ✗, `--temperature` ✗, `--context-tokens` ✓ (selection gate only) |
| tokens / latency | server `usage` from the `result` event; vLLM ≥ 0.23 serves the Anthropic `/v1/messages` protocol natively (no bridge) |
| notes | per-task trust plumbing is automatic (scratch allowlist + temporary `~/.claude.json` entry, removed after the task) — see the Backends section |

### pi

Requires the `pi` CLI on PATH.

```bash
python3 llmbench.py run --backend pi --suites smoke
python3 llmbench.py run --backend pi                        # 51 tasks; only tools auto-skipped
python3 llmbench.py run --backend pi --suites longctx        # 128k prompt rides pi's stdin
python3 llmbench.py run --backend pi --tasks longctx/needle-128k
python3 llmbench.py run --backend pi --pi-tools read,bash --suites smoke
python3 llmbench.py run --backend pi --pi-thinking off --suites smoke
```

| item | value |
|---|---|
| invocation | `pi --mode json --no-session --provider llmbench --model <m> --api-key <k>` (`+ --append-system-prompt <sys>`, `--tools <list>`, `--thinking <level>` as applicable) `-- <prompt>`; prompts > 96 000 chars are piped via **stdin** instead of argv (Linux per-argument limit), system prompt travels inside that text |
| per-run overlay | `PI_CODING_AGENT_DIR` = `<scratch>/.pi-agent` holding a self-contained `models.json` (provider `llmbench`, `api: openai-completions`, `maxTokens` mirroring `--max-tokens`, `contextWindow` from `--context-tokens` or 262 144 when undeclared) → `~/.pi/agent` is never touched, no trust prompt |
| backend-specific flags | `--pi-tools <list>` — pi's tool allowlist; entries are the built-in tool names `read,write,edit,bash,grep,find,ls` (glob patterns with `*` work; `+name`/`-name` add/remove against the default set; MCP tools stay enabled unless an entry starts with `mcp__`); default = pi's full default toolset. `--pi-thinking <level>` — `off`\|`minimal`\|`low`\|`medium`\|`high`\|`xhigh`\|`max`; on qwen-style vLLM models the level maps to `enable_thinking` (off = off, other = on); default = model default. Thinking text is captured in the task `.md` Reasoning section |
| suites | 8 (all except api-only `tools`; includes `longctx`) = 51 tasks |
| shared flags effective | `--max-tokens` ✓ (→ `maxTokens` in the generated `models.json`), `--temperature` ✗, `--context-tokens` ✓ (selection gate, incl. pi's own scaffolding, and declared as pi's `contextWindow`) |
| tokens / latency | server `usage` summed over the task's responses; `ttft_model_s` = first assistant event; fastest CLI observed, lowest prompt-token cost |

After any run: artifacts land in the run dir (see the table in Quick start), `report` re-renders, `compare` puts two runs side by side. And the standing rule: backends sharing one endpoint run **sequentially** — never in parallel.

## Metrics (see `metrics_legend` in run.json)

- `ttft_s` — `api`: request sent → first **content** token. `cli`: spawn → first stdout byte (includes CLI startup)
- `ttft_model_s` — request/spawn → first **model-originated** output (codex: first item event; claude: first assistant event; pi: first assistant event; dsh: final message, it buffers). Use this for cross-backend *model* latency — `ttft_s` on a CLI is mostly process startup
- `ttft_first_reasoning_s` — `api`: request sent → first **reasoning** token (both `reasoning` and `reasoning_content` delta fields count; reasoning models stream thinking before content). CLI backends: degenerate (= `ttft_s`)
- `gen_tps` — completion tokens / (end − `ttft_model_s`). For `api` this is the server decode rate (incl. reasoning); on CLI backends it also spans tool round-trips; suppressed when the window < 1 s (buffered CLIs)
- `decode_tps` — content chunks / (end − ttft) → content-only rate, `api` only (suppressed for windows < 250 ms)
- `overall_tps` — completion tokens / total latency (incl. TTFT)
- tokens — from server `usage` when the backend reports it (`token_source=server`: codex/claude/pi — cumulative over the task's internal API calls **including the harness system prompt**; api — the stream's usage chunk). Otherwise ~4 chars/token of the task prompt only (`token_source=estimate`, e.g. `dsh`, and `api` against engines with no streamed usage). `completion_tokens` **includes reasoning tokens**
- **`passed` vs `content_ok`** — two readings of every scored task. `passed` is the strict contract verdict (the answer line must *be* the value). `content_ok` answers a different question: was the value delivered at all, ignoring markdown, quotes, a unit or a prose lead-in — `ANSWER: **408**`, `ANSWER: 408 items`, `ANSWER: A is the knight` all deliver. Only answer-shaped verifiers (`exact`, `answer`, `math`, `gridpath`, a tool task's `answer`) split the two; constraint, unittest, file and post-run verdicts mirror `passed` because their form *is* the content. Aggregates: `totals.n_content` and `suites.<s>.content/content_rate`; the console summary and `report.md` show `content` only where it exceeds the strict pass count (the `report.md` scorecard always shows the column, so older runs render `—`), `compare` gets a `value delivered` row, and a task `.md` says `(value delivered, answer line not exact)`. Strict scoring is unchanged, so pre-0.2.0 baselines stay comparable. `content_ok` is a formatting signal, never a correctness claim — `ANSWER: A is the knave` reads as delivered

The `api` backend is the reference for pure server metrics; CLI numbers include harness overhead. **Run backends sequentially when they share one endpoint** (e.g. this host's single vLLM) — parallel runs contend on the NPU and corrupt each other's latency/throughput.

## Resource sampling

A background thread samples at `--sample-interval` (default 2 s) during each task:

- **Accelerator** (auto-detected, in order): `nvidia-smi` → `rocm-smi` → `docker exec <container> npu-smi info` (this host: container `qwen3.8`; override with `--npu-container` or `LLMBENCH_NPU_CONTAINER`). Slow on NPU (~1.7 s/call), so NPU readings run in their own loop and the latest reading is attached to each fast /proc sample.
- **System**: CPU %, RAM used/total, page cache (Cached+SReclaimable), swap.
- **Process tree** (CLI backends): CPU % and RSS of the CLI process and all its descendants — attached via `set_root(pid)` right after spawn, so the sample window covers the whole agent run.

Per-chip aggregates (avg/min/peak/p50/p95) land in each task's `resources` block and in the report's Resources section.

## Suites

All tasks are original and deterministic (no dataset downloads, no network needed). Select with `--suites` (comma-separated; default = all except `longctx`; use `--suites full` to include the heavy suite).

| suite | tasks | what's scored | verifier |
|---|---|---|---|
| `smoke` | 3 | pipeline sanity | exact / math / answer |
| `reasoning` | 8 | word problems, knights-&-knaves, syllogisms, sequences, logic fallacies | `ANSWER:` contract + numeric/string equivalence |
| `spatial` | 8 | seeded CLEVR-style 3D table scenes (count/compare/relational), grid shortest path (length + move string), map directions, rotation, cube counting | math / answer / move-string simulation on the grid |
| `ifeval` | 8 | programmatically checkable constraints (exact sentence/word counts, keyword include/exclude, JSON-only shape, line prefixes, casing, punctuation bans, first line) | constraint checkers (all must pass) |
| `coding` | 8 | original unit-tested problems (anagram, roman, intervals, moving avg, parens, flatten, caesar, top-k) | solution read from `scratch/solution.py` if the agent wrote one, else extracted from the first ` ```python ` block; unittest suite run (60 s timeout) |
| `tools` | 8 | OpenAI-style tool calls: correct function, argument subsets, arithmetic expressions safe-eval'd, multi-call, and a "don't over-call" task | tool-call matcher (api backend only) |
| `agentic` | 5 | CLI-only: create a file with exact content, sum a data file, shell-compute, script-and-run word count, fix a failing unit test in a provided repo | file-content / `ANSWER:` math / post-task command re-run (dsh, codex, claude, pi) |
| `mullang` | 8 | prompts in zh / ja / ko / ar / ru / de / es / hi — word problems, a grid path (ru), a 2-sentence constraint (de), a knights-&-knaves (es), exact word count in Devanagari (hi, `script` check) | existing verifiers (math / gridpath / constraints / answer) + script check |
| `longctx` | 3 | opt-in: needle retrieval in a deterministic filler log. Sizes are token-based and published per task as `min_context_tokens`: `needle-8k` ~8.7k tok, `multi-32k` ~34.5k tok (3 needles, order), `needle-128k` ~120.3k tok — the heavy tier is deliberately under 128k so prompt + the default `--max-tokens 4096` still fits a 128k-token window | `ANSWER:` exact/ordered (api + pi; prompts over the CLI per-argument limit ride pi's stdin) |

`python3 selftest.py` runs fast offline checks: the verifier self-tests plus characterization probes for the CLI backends (argv/env/overlay), the api streaming parser and the strict/content verdict split (no model, no network).

### What each suite actually scores

Of the 59 tasks, 41 decide on the **reply text** and 18 on **behavior** — which is why decoding temperature moves a pass rate at all (counts come from the verifier types, not from the suite names):

| verdict class | tasks | where | breaks when the model |
|---|---|---|---|
| value on the `ANSWER:` line (11 `answer` + 17 `math`) | 28 | smoke, reasoning, spatial, agentic, mullang, longctx | decorates or explains it → strict fail, `content_ok` pass |
| move string (`gridpath`) | 2 | spatial (`grid-path`), mullang (`ru-grid`) | writes `up, up, right` instead of `UURR` |
| whole-reply form (10 `constraints` + 1 `exact`) | 11 | ifeval (8), mullang (`de-constraints`, `hi-words`), `smoke/pong` | adds any prose: word/sentence/line counts, casing, banned chars, first line, JSON shape and the single-word reply all read the **entire** reply |
| workspace behavior (8 `code` + 1 `file` + 1 `run`) | 10 | coding (8), agentic (`file-create`, `fix-bug`) | nothing — phrasing is irrelevant, only what the code/file does |
| tool-call contract (`toolcall`) | 8 | tools | nothing textual — call name/args must match (a string arg is exact modulo case, so `Tokyo` ok, `Tokyo, Japan` not); `tools/no-call-needed` checks the answer line too |

Practical consequence: **greedy (`--temperature 0`) measures reasoning plus terse formatting; endpoint-default sampling measures reasoning plus willingness to obey a format contract under verbosity.** Both are real capability, so nothing is dropped — read the two numbers apart (`passed` vs `content`), and use `--repeat N` when you want adherence as a rate instead of a single coin flip.

## Run modes

- **Preflight** — before the first task, every `run` does a best-effort `GET <endpoint>/models`: a dead endpoint or an unlisted model surfaces as one warning line instead of N retried task errors. It never blocks the run (an engine may not implement `/models`, or may serve the model under an alias the list does not show).
- `--retries N` (default 1) — re-attempt a task that **errors** (transport/ crash), not one that merely fails its verifier (a fail is a real verdict). A deterministic client-side rejection (HTTP 4xx other than 408/425/429 — context too long, unsupported parameter) is **not** retried: replaying the same body only costs another round trip. Each task records its `attempts` and its `retryable` flag; the scratch dir is wiped between attempts so a crashed run's half-written `solution.py` can't leak into a retry.
- **Context gating** — with `--context-tokens N` declared, tasks that carry a prompt footprint (`longctx` tiers, see `list`) plus the `--max-tokens` reservation that cannot fit inside N are skipped up front, one logged line per task spelling out its own arithmetic (`prompt est + pi scaffold est + --max-tokens = need, > N declared, short by M`), and recorded the same way in `run.json` as `config.context_skipped`. No engine-specific probing, no error string parsing: the harness never guesses a server's window. The estimate divides chars by **3.8 with a 2 % margin**, calibrated on the reference deployment (served 3.79–3.82 chars/token) — tokenizers that pack this filler denser (gpt-oss-20b measured **4.4–4.7**, i.e. the estimate runs 21–26 % high there) can therefore be screened out while they would actually fit. The gate is a pessimist on purpose: leave it at `0` to send everything, lower `--max-tokens`, or declare the window you really serve.
- **No silent verdicts** — a response with no content and no tool call is scored, never passed by default: reasoning but no answer (`finish_reason= length`) is a `fail` with `truncated: true` and a hint to raise `--max-tokens`; a completely empty response is an `error`. Tasks whose verifier reads the workspace (`file`, `run`, `code`) are still scored when the agent said nothing but did the work.
- `--concurrency N` (default 1) — run N tasks in flight (load mode). Each task keeps its own resource sampler, so per-task NPU/CPU samples stay attributed; the shared endpoint is the thing under load.
- `--repeat N` — repeat every task N times. Aggregates then carry a 95% CI (`*_ci95` fields in `run.json`; `±` in the console summary) so single-run wiggles are distinguishable from signal.

## Notes / host facts

- **Defaults (0.2.0):** `--temperature` is unset by default, so the api backend omits the parameter and the endpoint's own sampling config is what gets measured; the recorded pre-0.2.0 baselines in `results/` were all greedy (`temperature: 0.0` in their `run.json`). Re-run with `--temperature 0` for a directly comparable number, or treat the recorded `temperature` field as the split. Same release: `longctx/needle-128k` was resized (512k → 448k filler chars) so it fits a 128k-token window.
- **Real-endpoint confirmation (2026-10-16, `results/local/20261016-233946_gpt-oss-20b_pi`):** 51/51 pass, 0 errors, every task one attempt, `temperature` omitted — served prompt tokens 7 364 / 27 894 / 95 847 for the three `longctx` tiers, so the resized heavy tier cost 95 847 prompt + 16 384 requested output = 112 231 tokens on that deployment (at that same measured density, the old 512k-char prompt would have cost ≈110k prompt tokens there — and 134 227 on the Qwen3.8 deployment, which is where the resize was actually needed)
- **Strict vs content (0.2.0):** measured across the recorded baselines and the two endpoint-sampled local runs — all 28 ANSWER-line tasks score a *correct* answer wrong when it is decorated or explained (`ANSWER: **408**`, `408 items`, `The answer is 408.`, `A is the knight`); `--temperature 0` hid this because greedy reliably emits the bare token. Observed for real: `reasoning/knights-knaves` in a qwen3.6-35b-a3b pi run (`exp=A got='A is the knight'`). All strict tasks are **kept** — contract adherence is capability, and `ifeval` measures nothing if its constraints are relaxed — and the verdict is split instead: each of the 30 answer-shaped tasks (28 ANSWER-line + 2 move strings) reports `content_ok` for those variants while `passed` stays strict; the 8 `toolcall` tasks mirror the call contract, which *is* the behaviour. Re-scoring 275 recorded verdicts with the new verifiers changes **zero** strict outcomes. `smoke/pong` is the one intentional strict change: trailing punctuation now passes (`pong.` ok, `The word is pong` still fails, and reads as delivered).
- **Two recorded non-sampling failures, for the record:** `ifeval/first-line-number` failed 4 runs because the reply said "The **42nd** Street" — the word-boundary check for `42` correctly rejects it, but the prompt's "contain the number 42" invites that phrasing; and `agentic/file-create` / `agentic/fix-bug` failed on claude at temp 0 with "write permission hasn't been granted yet" — CLI permission gating, not the model.
- Local model: `qwen3.8` = Qwen3.8-27B-w8a8 (W8A8, 262k ctx, MTP spec-decode) served by vLLM 0.23.0 on `0.0.0.0:8000` inside host-networked container `qwen3.8` (8× Ascend 910B4 NPUs, 64 GB HBM each).
- Host `npu-smi` is broken; NPU state is read via `docker exec qwen3.8 npu-smi info`.
- Live validation (all against `qwen3.8`): `api` full 43-task run = 43/43 pass (gen ~88 tok/s, TTFT p50 2.5 s); `dsh` 39/40 (one genuine model miss); `codex` agentic 5/5 (including fixing a failing unit test); `claude` smoke 3/3 + agentic 5/5 — all CLIs fixed `buggy.py` and created exact-content files. vLLM 0.23.0 serves Anthropic `/v1/messages` natively (no bridge).
- **Definitive 4-backend baseline (2026-10-13, runs 20261013-1718…1742, sequential on the shared endpoint):** 163/163 task-runs passed (api 43 + dsh 40 + codex 40 + claude 40). Standout numbers: api TTFT avg 4.7 s / gen 84 tok/s (content decode 27 tok/s); dsh is buffered (TTFT ≈ total, ~12 s) with no token usage (estimated); codex has the lowest TTFT (0.36 s — events stream early) but a ~16 s per-task overhead; claude ~12 s per task. Prompt-token cost of the CLI harnesses dwarfs the prompt itself (codex ~593k, claude ~1.16M across 40 tasks vs api 7.8k) — expected, and visible in the `compare` table.
- **pi backend added (2026-10-14, run 20261014-074156, same host/endpoint):** 48/48 task-runs (the 40 shared tasks + 8 `mullang` tasks, which postdate the other CLIs' baselines and are pi-only there) → 211/211 total across the five backends. pi is the fastest CLI backend (total avg 7.3 s vs claude 11.6 / dsh 11.8 / codex 16.9 s) and the most token-efficient (279k prompt tokens vs codex 593k / claude 1.16M; dsh 2.9k estimated) — see `results/pi-compare.md`.
- **dsh true-cost measurement (2026-10-14, run `20261014-160020_qwen3.8_dsh`):** loopback usage-logging proxy (`tools/usage_proxy.py`, injects `stream_options.include_usage`) captured 129 dsh internal calls — **752.9k prompt / 25.7k completion tokens over 48 tasks** (~15.7k prompt/task: a ~170-token session-title call echoing the task prompt + ≥1 agent call carrying ~8.9k of dsh scaffolding). The harness's visible-text estimate (8.6k) understated it ~91×. Verdicts identical to the raw run (47/48; `ifeval/first-line-number` miss reproduced in all three dsh runs at temperature 0).
- The DSH file sandbox denies writes to `~/.dsh` and `~/.codex/tmp`; the CLI backends therefore need a one-shot `danger-full-access` escalation in this harness (on a normal shell they just run).
