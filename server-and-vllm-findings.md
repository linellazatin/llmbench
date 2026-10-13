# Server & vLLM Findings — Host `administrator` (10.199.139.164)

Investigated 2026-10-13. All findings verified live on the host.

---

## 1. TL;DR

- **vLLM is not installed on the host.** It exists **only inside the Docker container `qwen3.8`**, at `/usr/local/python3.12.13/bin/vllm` (a container-local Python 3.12 install). That is why `which vllm` / `pip show vllm` come up empty on the host — the host's default PATH (`/usr/local/sbin:/usr/local/bin:/usr/bin:/bin…`) and system Python (3.10.12) simply don't contain it.
- The **`qwen3.8` model** is served by that in-container vLLM (OpenAI-compatible API) on **`0.0.0.0:8000`**, via the container's **host networking**.
- The **provider gateway** for this machine's AI harness is just a label: the DSH harness (`dsh`, DeepSeek Harness) is configured in `/root/.dsh/settings.yaml` with provider `my-gateway` → `baseURL: http://10.199.139.164:8000/v1`, model `qwen3.8`. `10.199.139.164` is this machine's own IP — so there is **no separate proxy process**; the "gateway" is vLLM itself, addressed by its own IP.

---

## 2. Where vLLM is installed

| Item | Value |
|---|---|
| Location | **Inside Docker container `qwen3.8`** (not on host) |
| Binary | `/usr/local/python3.12.13/bin/vllm` (container PATH) |
| vLLM version | `0.23.0+empty` (aarch64, custom wheel) |
| Ascend plugin | `vllm-ascend` **0.23.0** (vLLM Ascend backend) |
| Python | CPython **3.12.13**, custom install at `/usr/local/python3.12.13/` (container-only) |
| Site-packages | `/usr/local/python3.12.13/lib/python3.12/site-packages` |
| Ascend toolkit | **CANN 9.1.0** (`/usr/local/Ascend/cann-9.1.0`), `npu-smi 26.0.rc1` |
| Driver | Host driver `/usr/local/Ascend/driver` bind-mounted into the container |

How to reach it:

```bash
docker exec -it qwen3.8 bash
# inside: which vllm  ->  /usr/local/python3.12.13/bin/vllm
```

Not the running install (red herrings on the host filesystem):
- `/workspace/tlf/vllm-ascend/` — source checkout of the vllm-ascend repo (build artifacts, `vllm_ascend…aarch64…so`)
- `/workspace/vllm-ascend-v0.23.0-openeuler.tar`, `/workspace/6_xx/wx_test/vllm-ascend-v0.22.1rc1-openeuler-arm64.tar`, `/workspace/htp/vllm_0_22_rc1_910b_aisbench.tar` — packaging/tarballs from other test rounds
- `/root/.cache/vllm/` — host-side vLLM model-info cache (written by the container, shared volume)

## 3. How `qwen3.8` is served

**Container facts** (`docker inspect qwen3.8`):
- Image: `6c183dfa86cf` (built from an Ascend/vLLM base), created 2026-08-21, up since ~Oct 10
- **Network mode: `host`** (no published ports — the vLLM socket is directly the host's `0.0.0.0:8000`)
- Key bind mounts: `/workspace → /workspace` (model lives here), `/usr/local/Ascend/driver`, `/usr/local/bin/npu-smi`, `/usr/local/dcmi`, `/usr/local/sbin`, `/etc/ascend_install.info`, `/etc/hccn.conf`, TZ `Asia/Shanghai`

**Launch** — started **Oct 10 from an interactive shell** (pts/1) inside the container (parent is `bash`, PID 32849). No systemd unit, no docker CMD — it's a manual foreground process, kept alive by the open tty session. Full command:

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

**Process topology** (matches DP=4 × TP=2 over 8 NPUs):

```
vllm serve (PID 33021)
├── VLLM::DPCoordinator
├── VLLM::APIServer_0 … APIServer_3     (4 HTTP front-ends, one per DP replica)
├── VLLM::EngineCore_DP0 … DP3          (4 DP engines)
│    └── VLLM::Worker_TP0, Worker_TP1   (2 workers each → 8 workers, 1 NPU each)
```

**Model**: `Qwen3.8-27B-w8a8` at `/workspace/Qwen3.8-27B-w8a8/` (30 GB, 10-shard `quant_model_weights`, W8A8 Ascend quantization). Config: `Qwen3_5ForConditionalGeneration`, `model_type qwen3_5` — hybrid **linear + full attention** (full attention every 4th layer), hidden 5120, head_dim 256, bf16. Served as **`qwen3.8`** with MTP speculative decoding (3 tokens), prefix caching, 262 144-token context.

**Live verification** (performed during this investigation):

```
$ curl -s http://127.0.0.1:8000/v1/models
{"object":"list","data":[{"id":"qwen3.8","owned_by":"vllm",
  "root":"/workspace/Qwen3.8-27B-w8a8/","max_model_len":262144, ...}]}
```

## 4. The provider gateway

There is **no intermediate gateway service** on this box. The chain is:

```
DSH harness (node, PID 71333, profile "lines")
   │  provider config: /root/.dsh/settings.yaml
   │    provider "my-gateway"  api: openai-completions
   │    baseURL: http://10.199.139.164:8000/v1   ← this host's own IP
   │    apiKeyEnv: MY_GATEWAY_API_KEY, models: [qwen3.8]
   ▼
0.0.0.0:8000  (vLLM, host-networked container qwen3.8)
   ▼
8× Ascend 910B4 NPUs
```

`/root/.dsh/settings.yaml` (verbatim):

```yaml
llm-pi-ai:
  providers:
    my-gateway:
      apiKeyEnv: MY_GATEWAY_API_KEY
      baseURL: http://10.199.139.164:8000/v1
      api: openai-completions
      models:
        - id: qwen3.8
```

Other listening ports (all local): `0.0.0.0:22` (ssh), `127.0.0.1:6010/6011/6012` (+ `[::1]`) — DSH harness internal IPC; `*:37763 / *:38597 / *:40307 / *:47683` — wildcard listeners with no process attribution visible from this sandbox (count/shape consistent with vLLM worker RPC sockets in the host-net container). Outbound HTTP for this box is proxied via `http://10.199.139.162:3129` (set in `/workspace/http_proxy.sh`; same /20 subnet).

## 5. Full server specifications

### Compute
| Component | Specification |
|---|---|
| Architecture | aarch64 (ARM) |
| CPU | **2× HiSilicon Kunpeng-920**, 48 cores / 1 thread per core = **96 cores total**, 4 NUMA nodes (24 cpus each) |
| Accelerators | **8× Huawei Ascend 910B4-1** NPU, **64 GB HBM each (512 GB total HBM)**, PCIe `0000:0B–0C,0F–10,8B–90:00.0`; idle AICore 0%, ~60–62 GB HBM used per card by the loaded model |
| Memory | **471 GiB** RAM, 8 GiB swap |
| NPU software | CANN 9.1.0 toolkit, npu-smi 26.0.rc1; host `npu-smi` errors (`dcmi module initialize failed`) — NPU state only readable via the container |

### Storage
| Device | Size | Mount / use |
|---|---|---|
| `/dev/sda` (SATA SSD) | 446.6 GB | `/boot/efi` (1.1 G), `/boot` (2 G), LVM `ubuntu--vg-lv--0` → `/` — 436 G, **83 G used (21%)** |
| `/dev/nvme0n1` | 1.7 TB | Partitioned (1 G / 2 G / 1.7 T) — **currently unmounted** |
| `/dev/nvme1n1` | 1.8 TB | → `/workspace` (ext4) — **1.1 T used (64%)**; holds model weights, vllm-ascend source, test tarballs |

### OS & software
| Item | Value |
|---|---|
| OS | **Ubuntu 22.04 LTS (Jammy)**, aarch64 |
| Kernel | 5.15.0-190-generic (`#200-Ubuntu SMP`, built 2026) |
| Hostname | `administrator` |
| Timezone | Asia/Shanghai |
| Container runtime | **Docker 29.1.3** + containerd |
| Containers | `qwen3.8` (host-net, vLLM serving qwen3.8); `ais_bench_test` (AISBench v3.1 benchmark image, openEuler 24.03 / py3.11 / aarch64) |
| Host Python | 3.10.12 (system); Node v22.23.2 (nvm) for the DSH harness |
| AI harness | DeepSeek Harness `dsh` v0.8.5, profile `lines`, `~/.dsh`; session JSONL under `/root/.dsh/sessions/` |
| Services (systemd) | ssh, docker, containerd, networkd, irqbalance, cron, udisks2, snapd, unattended-upgrades |
| Outbound proxy | `http://10.199.139.162:3129` (squid on same subnet) |

### Networking
| Interface | State | Address |
|---|---|---|
| `enp125s0f1` | **UP** (Mellanox-class mlx5, 125G naming) | **10.199.139.164/20**, `fe80::b24f:a6ff:fec9:12c0/64` |
| `enp125s0f0`, `enp18s0f0np0/f1`, `enp19s0f0np0/f1` | DOWN | — |
| `docker0` | DOWN | 172.17.0.1/16 |
| `lo` | up | 127.0.0.1/8, ::1/128 |

### Summary line
> Dual-socket Kunpeng-920 (96-core ARM) Ubuntu 22.04 server with 8× Ascend 910B4 (512 GB HBM) and 471 GiB RAM, running Qwen3.8-27B-w8a8 (W8A8, 262k ctx, DP4×TP2, MTP spec-decode) via vLLM 0.23.0 + vllm-ascend 0.23.0 inside a host-networked Docker container on port 8000, which the local DSH harness consumes as its `my-gateway`/`qwen3.8` provider.

### Operational notes / caveats
1. **vLLM is a bare tty process** (no systemd, no restart policy) — it dies if the container shell session is killed; container `qwen3.8`'s entrypoint is just `bash`.
2. **Host `npu-smi` is broken** (dcmi init failure); NPU health is only visible via `docker exec qwen3.8 npu-smi info`.
3. Model weights sit on `/workspace` (64% full); a 30 GB W8A8 checkpoint.
4. This box doubles as an NPU benchmark lab: `/workspace/6_xx/…`, `/workspace/htp/…`, aisbench container, and start scripts (e.g. `vllm_start.sh` for an earlier `qwen3.5` TP4/DP1 deployment on port 8009) — the current `qwen3.8` deployment was not launched from a committed script.
