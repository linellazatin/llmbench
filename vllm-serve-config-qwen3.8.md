# vLLM Serve Configuration — `qwen3.8`

Capture date: 2026-10-13 · Host: `administrator` (10.199.139.164)
Source: live process `vllm serve` (PID 33021) inside Docker container **`qwen3.8`** (host network mode), launched Oct 10 from an interactive shell.

## Launch command (verbatim)

```bash
/usr/local/python3.12.13/bin/python3 /usr/local/python3.12.13/bin/vllm serve \
  /workspace/Qwen3.8-27B-w8a8/ \
  --host 0.0.0.0 --port 8000 \
  --data-parallel-size 4 --tensor-parallel-size 2 \
  --quantization ascend --served-model-name qwen3.8 \
  --max-num-seqs 64 --max-model-len 262144 --max-num-batched-tokens 32768 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --trust-remote-code --enable-prefix-caching --gpu-memory-utilization 0.90 \
  --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"enable_cpu_binding":true}'
```

## Runtime environment

| Item | Value |
|---|---|
| Binary | `/usr/local/python3.12.13/bin/vllm` (container-only, not on host) |
| vLLM version | `0.23.0+empty` (aarch64 wheel) |
| Ascend plugin | `vllm-ascend` 0.23.0 |
| Python | CPython 3.12.13 (`/usr/local/python3.12.13/`, container-local) |
| Ascend stack | CANN 9.1.0 toolkit, npu-smi 26.0.rc1 |
| Accelerators | 8× Huawei Ascend 910B4-1, 64 GB HBM each |
| Container | `qwen3.8` (image `6c183dfa86cf`), **host networking**, mounts: `/workspace`, `/usr/local/Ascend/driver`, `/usr/local/bin/npu-smi`, `/usr/local/dcmi`, `/usr/local/sbin`, `/etc/ascend_install.info`, `/etc/hccn.conf` |
| Endpoint | `http://10.199.139.164:8000/v1` (OpenAI-compatible), model id `qwen3.8` |

## Flags table

| Flag | Value | What it does |
|---|---|---|
| **Model** | `/workspace/Qwen3.8-27B-w8a8/` | 30 GB W8A8 Ascend-quantized checkpoint, 10 safetensors shards; arch `Qwen3_5ForConditionalGeneration` (hybrid linear+full attention) |
| **Host/Port** | `0.0.0.0:8000` | Binds all interfaces via host-net container |
| **served-model-name** | `qwen3.8` | Name exposed on `/v1/models` and in requests |
| **data-parallel-size** | `4` | 4 independent DP replicas (4 APIServers, 4 EngineCores) |
| **tensor-parallel-size** | `2` | Each replica sharded across 2 NPUs → **8 NPUs used total** |
| **quantization** | `ascend` | W8A8 weight-activation quantization path for Ascend |
| **max-num-seqs** | `64` | 64 in-flight requests per DP replica → **up to 256 concurrent requests cluster-wide** |
| **max-model-len** | `262144` | 256 K-token max sequence (prompt + generation) |
| **max-num-batched-tokens** | `32768` | Cap on tokens per scheduler step |
| **enable-prefix-caching** | on | Reuses KV cache for shared prefixes (system prompts, tool schemas) |
| **gpu-memory-utilization** | `0.90` | ~57.6 GB of each 64 GB card budgeted for weights + KV (observed ~60–62 GB HBM used at idle) |
| **speculative-config** | `{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}` | MTP speculative decoding: draft 3 tokens/step, verified by the main model |
| **compilation-config** | `{"cudagraph_mode": "FULL_DECODE_ONLY"}` | CUDA-graph capture for the decode phase only; prefill runs eager |
| **additional-config** | `{"enable_cpu_binding": true}` | Pins each worker's CPU threads to its NPU's NUMA socket (Ascend-specific) |
| **enable-auto-tool-choice** | on | Enables native function-calling / tool use |
| **tool-call-parser** | `qwen3_coder` | Parses tool calls from model output |
| **reasoning-parser** | `qwen3` | Splits `thinking`/reasoning blocks from final answer |
| **trust-remote-code** | on | Executes model-shipped code (required for the custom qwen3_5 hybrid architecture) |

## Topology

### Process tree (as observed on the host)

```
vllm serve            (PID 33021, /usr/local/python3.12.13/bin/python3)
├── VLLM::DPCoordinator                  (cross-replica coordination)
├── VLLM::APIServer_0 … APIServer_3      (4 HTTP front-ends, one per DP replica)
├── VLLM::EngineCore_DP0 … EngineCore_DP3 (4 scheduling/serving engines, one per replica)
│     ├── DP replica i
│     │     ├── VLLM::Worker_TP0  → NPU i
│     │     └── VLLM::Worker_TP1  → NPU i+1
```

- 4 DP replicas × 2 TP workers = **8 workers, one per NPU** (8× Ascend 910B4-1).
- Worker RSS ≈ 4.3–5.1 GB each (model weights are on HBM; CPU side is orchestrator buffers).

### Request flow

```
client → http://10.199.139.164:8000/v1/chat/completions  (model: "qwen3.8")
        → APIServer_k (k = 0..3, load-balanced per DP replica)
        → EngineCore_DPk  (scheduling, prefix-cache lookup, batching ≤32768 tok/step, ≤64 seqs)
        → Worker_TP0 + Worker_TP1  (TP=2 allreduce across a NPU pair)
        → MTP draft (3 tokens, eager) → main model verify (FULL_DECODE_ONLY cudagraphs)
        → qwen3_coder tool parser / qwen3 reasoning parser → SSE stream back
```

### Effective capacity summary

| Metric | Value |
|---|---|
| NPUs used | 8 / 8 (910B4-1) |
| HBM in use at idle | ~60–62 GB per card (of 64 GB) |
| Max context | 262,144 tokens |
| Max concurrent sequences | 256 (64 × 4 DP) |
| Max batch tokens/step | 32,768 |
| Speculative draft tokens | 3 (qwen3_5_mtp) |
| Quantization | W8A8 (Ascend) |

## Notes & caveats

- Launched **manually from a tty** (container `qwen3.8` entrypoint is `bash`); no systemd unit, no restart policy — killing the shell kills the server.
- No auth flag set: port 8000 is open on all interfaces (anyone reaching `10.199.139.164:8000` can query the model; the harness's `MY_GATEWAY_API_KEY` is checked by the client, not enforced by vLLM here).
- Host-side `npu-smi` is broken (dcmi init failure); use `docker exec qwen3.8 npu-smi info` for NPU health.
- Sibling artifact: `/workspace/6_xx/G620kv2_test/vllm_start.sh` documents an earlier `qwen3.5` deployment (port 8009, TP4/DP1, no prefix caching/tool parsers) — the current `qwen3.8` deployment was not started from a committed script.
