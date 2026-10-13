#!/usr/bin/env python3
"""llmbench — non-interactive local-model benchmark harness.

Runs a deterministic, offline benchmark suite against a locally served model
through one of several backends (direct OpenAI-compatible API, `dsh`,
`codex`, `claude code`), recording per-task quality verdicts plus latency
(TTFT), throughput (tok/s), token consumption, and resource usage (NPU/GPU
HBM + core utilization, CPU, RAM, page cache, swap).

Outputs per run, under results/<run-id>/:
  run.json          machine-readable results (schema v1)
  report.md         human-readable report
  tasks/<id>.md     per-task prompt/response/verdict
  tasks/<id>.samples.jsonl  per-task resource samples

Single-file tool, Python 3.10+ stdlib only.
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

VERSION = "0.1.0"
SCHEMA = 1

# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def percentile(values: list[float], p: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    k = (len(values) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    return values[f] + (values[c] - values[f]) * (k - f)


def stats(values: list[float]) -> dict:
    """mean/min/peak/p50/p95 for a list of numbers (zeros/empties ok)."""
    values = [v for v in values if v is not None]
    if not values:
        return {}
    return {
        "avg": round(statistics.fmean(values), 3),
        "min": round(min(values), 3),
        "peak": round(max(values), 3),
        "p50": round(percentile(values, 50), 3),
        "p95": round(percentile(values, 95), 3),
    }


def gb(mb: float) -> float:
    return round(mb / 1024.0, 3)


def fnum(x: Any, nd: int = 3) -> Optional[str]:
    return None if x is None else f"{x:.{nd}f}"


# ---------------------------------------------------------------------------
# host facts
# ---------------------------------------------------------------------------

def host_facts() -> dict:
    facts = {
        "hostname": platform.node(),
        "arch": platform.machine(),
        "os": platform.platform(),
        "cpu_cores": os.cpu_count(),
    }
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    facts["ram_total_gb"] = round(int(line.split()[1]) / 1048576.0, 1)
                    break
    except OSError:
        pass
    facts["accelerator"] = NPUProber.detect_hint()
    return facts


# ---------------------------------------------------------------------------
# resource sampling
# ---------------------------------------------------------------------------

NPU_CHIP_RE = re.compile(
    r"^\|\s*(\d+)\s+\|\s*[0-9A-Fa-f:.]{4,}\s*\|\s*(\d+(?:\.\d+)?)\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)\s*\|\s*$"
)
NPU_DEV_RE = re.compile(
    r"^\|\s*(\d+)\s+\S+\s+\|\s*(\S+)\s+\|\s*([\d.]+)\s+(-?\d+)\s+"
)


class NPUProber:
    """Detects and samples the accelerator (NPU/GPU).

    Backends, in detection order:
      nvidia  : nvidia-smi (fast, machine-readable)
      rocm    : rocm-smi (best effort)
      npu     : docker exec <container> npu-smi info (slow ~1.7s/call)
    """

    def __init__(self, container: Optional[str] = None):
        self.backend: Optional[str] = None
        self.container: Optional[str] = container
        self._err: Optional[str] = None
        self._detect()

    def _detect(self) -> None:
        if shutil.which("nvidia-smi"):
            self.backend = "nvidia"
            return
        if shutil.which("rocm-smi"):
            self.backend = "rocm"
            return
        candidates = [self.container, os.environ.get("LLMBENCH_NPU_CONTAINER"), "qwen3.8"]
        if shutil.which("docker"):
            for c in candidates:
                if not c:
                    continue
                try:
                    r = subprocess.run(
                        ["docker", "exec", c, "npu-smi", "-v"],
                        capture_output=True, timeout=10, check=False,
                    )
                    if r.returncode == 0:
                        self.backend = "npu"
                        self.container = c
                        return
                except Exception:
                    continue
        self._err = "no accelerator sampler found (tried nvidia-smi, rocm-smi, docker npu-smi)"

    @staticmethod
    def detect_hint() -> str:
        p = NPUProber()
        if p.backend:
            return f"{p.backend} ({p.container})" if p.container else p.backend
        return "none"

    def probe(self) -> Optional[list[dict]]:
        """Return list of per-chip dicts or None on failure. Slow on NPU."""
        try:
            if self.backend == "nvidia":
                out = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=6, check=False).stdout
                chips = []
                for line in out.strip().splitlines():
                    idx, util, used, total = (x.strip() for x in line.split(","))
                    chips.append({"chip": int(idx), "core_pct": float(util),
                                  "mem_used_mb": float(used), "mem_total_mb": float(total)})
                return chips
            if self.backend == "npu":
                out = subprocess.run(
                    ["docker", "exec", self.container, "npu-smi", "info"],
                    capture_output=True, text=True, timeout=30, check=False).stdout
                chips = []
                dev_by_npu: dict[int, dict] = {}
                cur_npu: Optional[int] = None
                for line in out.splitlines():
                    m = NPU_DEV_RE.match(line.strip())
                    if m:
                        cur_npu = int(m.group(1))
                        dev_by_npu[cur_npu] = {"health": m.group(2),
                                               "power_w": float(m.group(3)),
                                               "temp_c": int(m.group(4))}
                        continue
                    m = NPU_CHIP_RE.match(line.strip())
                    if m:
                        # group(1) on this line is the physical-chip index (always 0);
                        # the NPU id comes from the device line above it
                        npu = cur_npu if cur_npu is not None else int(m.group(1))
                        d = dev_by_npu.get(npu, {})
                        chips.append({
                            "chip": npu,
                            "core_pct": float(m.group(2)),
                            "mem_used_mb": float(m.group(3)),
                            "mem_total_mb": float(m.group(4)),
                            "power_w": d.get("power_w"),
                            "temp_c": d.get("temp_c"),
                            "health": d.get("health"),
                        })
                return chips or None
            if self.backend == "rocm":
                out = subprocess.run(
                    ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--csv"],
                    capture_output=True, text=True, timeout=10, check=False).stdout
                return self._parse_rocm(out)
        except Exception as e:  # noqa: BLE001
            self._err = f"{type(e).__name__}: {e}"
            return None
        return None

    @staticmethod
    def _parse_rocm(out: str) -> Optional[list[dict]]:
        try:
            chips = []
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4 and parts[0].isdigit():
                    chips.append({"chip": int(parts[0]), "core_pct": float(parts[1] or 0),
                                  "mem_used_mb": float(parts[2] or 0),
                                  "mem_total_mb": float(parts[3] or 0)})
            return chips or None
        except Exception:  # noqa: BLE001
            return None


def read_meminfo() -> dict:
    out: dict[str, float] = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key = line.split(":")[0]
                if key in ("MemTotal", "MemAvailable", "Cached", "SReclaimable",
                           "SwapTotal", "SwapFree"):
                    out[key] = int(line.split()[1])  # kB
    except OSError:
        pass
    if not out:
        return {}
    kb2gb = lambda v: v / 1048576.0  # noqa: E731
    out["ram_used_gb"] = round(kb2gb(out.get("MemTotal", 0) - out.get("MemAvailable", 0)), 3)
    out["ram_total_gb"] = round(kb2gb(out.get("MemTotal", 0)), 3)
    out["page_cache_gb"] = round(kb2gb(out.get("Cached", 0) + out.get("SReclaimable", 0)), 3)
    out["swap_used_gb"] = round(kb2gb(out.get("SwapTotal", 0) - out.get("SwapFree", 0)), 3)
    out["swap_total_gb"] = round(kb2gb(out.get("SwapTotal", 0)), 3)
    return out


class _ProcTree:
    """CPU% + RSS for a process and its descendants (Linux /proc)."""

    def __init__(self):
        self.hz = os.sysconf("SC_CLK_TCK")
        self._last_ticks: Optional[float] = None
        self._last_t: Optional[float] = None
        self._last_rss_kb: int = 0

    def _scan(self) -> tuple[float, int, list[int]]:
        total_ticks = 0.0
        rss_kb = 0
        pids: list[int] = []
        try:
            entries = [e for e in os.listdir("/proc") if e.isdigit()]
        except OSError:
            return 0.0, 0, []
        for pid in entries:
            path = f"/proc/{pid}"
            try:
                with open(f"{path}/stat", "rb") as f:
                    data = f.read()
            except OSError:
                continue
            rest = data.rsplit(b")", 1)
            if len(rest) != 2:
                continue
            parts = rest[1].split()
            if len(parts) < 13:
                continue
            try:
                ticks = (int(parts[11]) + int(parts[12])) / self.hz
            except ValueError:
                continue
            total_ticks += ticks
            pids.append(int(pid))
            try:
                with open(f"{path}/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            rss_kb += int(line.split()[1])
                            break
            except OSError:
                pass
        return total_ticks, rss_kb, pids

    def sample(self) -> Optional[dict]:
        ticks, rss_kb, _pids = self._scan()
        t = time.monotonic()
        if self._last_ticks is None or self._last_t is None:
            self._last_ticks, self._last_t, self._last_rss_kb = ticks, t, rss_kb
            return None
        dt = t - self._last_t
        if dt <= 0:
            return None
        cpu_pct = max(0.0, 100.0 * (ticks - self._last_ticks) / dt)
        out = {"cpu_pct": round(cpu_pct, 2), "rss_mb": round(rss_kb / 1024.0, 1)}
        self._last_ticks, self._last_t, self._last_rss_kb = ticks, t, rss_kb
        return out


class ResourceSampler:
    """Background sampler: NPU (slow prober thread) + system /proc (fast loop)."""

    def __init__(self, interval: float, prober: Optional[NPUProber] = None,
                 root_pid: Optional[int] = None):
        self.interval = max(0.2, interval)
        self.prober = prober or NPUProber()
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._latest_npu: Optional[list[dict]] = None
        self._tree = _ProcTree() if root_pid else None
        self._sys_last: Optional[tuple[int, int]] = None
        self._threads: list[threading.Thread] = []

    # -- NPU prober thread ---------------------------------------------------
    def _npu_loop(self) -> None:
        while not self._stop.is_set():
            reading = self.prober.probe()
            if reading is not None:
                self._latest_npu = reading
            self._stop.wait(max(0.5, self.interval * 0.6))

    # -- system CPU ----------------------------------------------------------
    def _sys_cpu(self) -> Optional[float]:
        try:
            with open("/proc/stat") as f:
                line = f.readline()
        except OSError:
            return None
        parts = line.split()
        if len(parts) < 5 or parts[0] != "cpu":
            return None
        vals = [int(x) for x in parts[1:]]
        idle = vals[3] + vals[4]
        total = sum(vals)
        if self._sys_last is None:
            self._sys_last = (total, idle)
            return None
        dtotal, didle = total - self._sys_last[0], idle - self._sys_last[1]
        self._sys_last = (total, idle)
        if dtotal <= 0:
            return None
        return 100.0 * (dtotal - didle) / dtotal

    # -- main loop -------------------------------------------------------------
    def _sample_once(self) -> Optional[dict]:
        mem = read_meminfo()
        if not mem:
            return None
        s: dict[str, Any] = {
            "ts": time.time(),
            "npu": self._latest_npu,
            "ram_used_gb": mem.get("ram_used_gb"),
            "ram_total_gb": mem.get("ram_total_gb"),
            "page_cache_gb": mem.get("page_cache_gb"),
            "swap_used_gb": mem.get("swap_used_gb"),
            "swap_total_gb": mem.get("swap_total_gb"),
            "sys_cpu_pct": self._sys_cpu(),
        }
        if self._tree is not None:
            tr = self._tree.sample()
            if tr:
                s.update(tr)
        return s

    def _loop(self) -> None:
        # prime system-CPU and proc-tree baselines
        self._sys_cpu()
        if self._tree is not None:
            self._tree.sample()
        next_t = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            if now >= next_t:
                s = self._sample_once()
                if s:
                    self.samples.append(s)
                next_t = now + self.interval
            self._stop.wait(min(0.1, max(0.005, next_t - time.monotonic())))

    def start(self) -> None:
        if self.prober.backend:
            th = threading.Thread(target=self._npu_loop, daemon=True)
            th.start()
            self._threads.append(th)
        th = threading.Thread(target=self._loop, daemon=True)
        th.start()
        self._threads.append(th)

    def stop(self) -> None:
        self._stop.set()
        for th in self._threads:
            th.join(timeout=35)


def aggregate_resources(samples: list[dict], accelerator: str = "none") -> dict:
    """Reduce raw samples to mean/peak aggregates (GB, %, MB)."""
    if not samples:
        return {"sample_count": 0}
    npu_by_chip: dict[int, dict] = {}
    for s in samples:
        for chip in s.get("npu") or []:
            d = npu_by_chip.setdefault(
                chip["chip"], {"core_pct": [], "mem_used_mb": [],
                               "power_w": [], "temp_c": []})
            if chip.get("core_pct") is not None:
                d["core_pct"].append(chip["core_pct"])
            if chip.get("mem_used_mb") is not None:
                d["mem_used_mb"].append(chip["mem_used_mb"])
            if chip.get("power_w") is not None:
                d["power_w"].append(chip["power_w"])
            if chip.get("temp_c") is not None:
                d["temp_c"].append(chip["temp_c"])
    chips_out = []
    for chip_id in sorted(npu_by_chip):
        d = npu_by_chip[chip_id]
        chips_out.append({
            "chip": chip_id,
            "core_pct": stats(d["core_pct"]),
            "mem_used_gb": {k: gb(v) for k, v in stats(d["mem_used_mb"]).items()},
            "power_w": stats(d["power_w"]),
            "temp_c": stats(d["temp_c"]),
        })
    agg = {
        "sample_count": len(samples),
        "accelerator": accelerator,
        "chips": chips_out or None,
        "sys_cpu_pct": stats([s.get("sys_cpu_pct") for s in samples]),
        "ram_used_gb": stats([s.get("ram_used_gb") for s in samples]),
        "ram_total_gb": (samples[0].get("ram_total_gb") if samples else None),
        "page_cache_gb": stats([s.get("page_cache_gb") for s in samples]),
        "swap_used_gb": stats([s.get("swap_used_gb") for s in samples]),
        "proc_cpu_pct": stats([s.get("cpu_pct") for s in samples if s.get("cpu_pct") is not None]),
        "proc_rss_mb": stats([s.get("rss_mb") for s in samples if s.get("rss_mb") is not None]),
    }
    return agg


# ---------------------------------------------------------------------------
# answer verification (phase 0: exact / math / answer-contract)
# ---------------------------------------------------------------------------

ANSWER_RE = re.compile(r"ANSWER:\s*([^\n]*)", re.IGNORECASE)


def extract_answer(text: str) -> str:
    matches = ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1].strip() if lines else ""


def _num(x: str) -> Optional[float]:
    x = x.strip().replace(",", "").replace("$", "").rstrip(".")
    m = re.search(r"^-?\d+(?:\.\d+)?$", x)
    if not m:
        m = re.search(r"^-?\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?$", x)
    if not m:
        return None
    try:
        if "/" in x:
            a, b = x.split("/")
            return float(a) / float(b)
        return float(x)
    except (ValueError, ZeroDivisionError):
        return None


def check_exact(response: str, spec: dict) -> dict:
    got = response.strip()
    want = spec["value"].strip()
    ok = got.lower() == want.lower()
    return {"passed": ok, "expected": want, "actual": got[:200]}


def check_answer(response: str, spec: dict) -> dict:
    """ANSWER: contract with optional numeric equivalence."""
    got = extract_answer(response)
    want = spec.get("value")
    if want is None:
        return {"passed": False, "expected": None, "actual": got[:200],
                "detail": "no expected value"}
    if spec.get("numeric"):
        a, b = _num(got), _num(str(want))
        if a is None or b is None:
            ok = got.lower() == str(want).lower()
        else:
            ok = math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)
    else:
        ok = got.lower() == str(want).lower()
    return {"passed": ok, "expected": want, "actual": got[:200]}


def check_math(response: str, spec: dict) -> dict:
    """ANSWER: contract, numeric equivalence only."""
    return check_answer(response, {**spec, "numeric": True})


VERIFIERS: dict[str, Callable[[str, dict], dict]] = {
    "exact": check_exact,
    "answer": check_answer,
    "math": check_math,
}


# ---------------------------------------------------------------------------
# backend: direct OpenAI-compatible API (ground-truth metrics)
# ---------------------------------------------------------------------------

@dataclass
class ChatResult:
    text: str = ""
    reasoning: str = ""
    ttft_first_content_s: Optional[float] = None   # request sent -> first content token
    ttft_first_reasoning_s: Optional[float] = None  # request sent -> first reasoning token
    total_s: float = 0.0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    content_chunks: int = 0      # ~content tokens (server streams 1 chunk/token)
    reasoning_chunks: int = 0    # ~reasoning tokens
    error: Optional[str] = None
    raw_last: Optional[dict] = None


class ApiBackend:
    name = "api"

    def __init__(self, endpoint: str, model: str, api_key: str = "llmbench",
                 max_tokens: int = 4096, temperature: float = 0.0,
                 timeout: float = 600.0):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

    def chat(self, messages: list[dict], tools: Optional[list] = None) -> ChatResult:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            body["tools"] = tools

        host, _, path = self.endpoint[7:].partition("/")
        path = "/" + path  # e.g. "/v1"
        req_path = path + "/chat/completions"
        conn = http.client.HTTPConnection(host, timeout=self.timeout)
        req_body = json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        res = ChatResult()
        t0 = time.monotonic()
        try:
            conn.request("POST", req_path, body=req_body, headers=headers)
            resp = conn.getresponse()
            if resp.status != 200:
                err = resp.read(4000).decode(errors="replace")
                res.error = f"HTTP {resp.status}: {err[:500]}"
                res.total_s = time.monotonic() - t0
                return res

            buf = b""
            while True:
                chunk = resp.read(1024)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.decode(errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        ev = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    choices = ev.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        rc = delta.get("reasoning")
                        if rc:
                            if res.ttft_first_reasoning_s is None:
                                res.ttft_first_reasoning_s = time.monotonic() - t0
                            res.reasoning += rc
                            res.reasoning_chunks += 1
                        c = delta.get("content")
                        if c:
                            if res.ttft_first_content_s is None:
                                res.ttft_first_content_s = time.monotonic() - t0
                            res.text += c
                            res.content_chunks += 1
                        if choices[0].get("finish_reason"):
                            res.finish_reason = choices[0]["finish_reason"]
                    usage = ev.get("usage")
                    if usage:
                        res.prompt_tokens = usage.get("prompt_tokens")
                        res.completion_tokens = usage.get("completion_tokens")
                        res.total_tokens = usage.get("total_tokens")
                        res.raw_last = usage
                    if payload == "[DONE]":
                        break
        except Exception as e:  # noqa: BLE001
            if not res.text:
                res.error = f"{type(e).__name__}: {e}"
        finally:
            conn.close()
            res.total_s = time.monotonic() - t0
        if res.ttft_first_content_s is None and res.ttft_first_reasoning_s is not None:
            res.ttft_first_content_s = res.ttft_first_reasoning_s
        return res

    def warmup(self) -> None:
        self.chat([{"role": "user", "content": "Reply with the single word: ready"}])


# ---------------------------------------------------------------------------
# benchmark suites (phase 0: smoke; more families arrive in phase 1)
# ---------------------------------------------------------------------------

def task(id: str, prompt: str, verify: dict, **meta) -> dict:
    return {"id": id, "suite": id.split("/")[0], "prompt": prompt,
            "verify": verify, **meta}


SMOKE_CONTRACT = (
    "\n\nShow your work briefly if needed. End your final reply with exactly one "
    "line in the form: ANSWER: <value>"
)

SMOKE_TASKS: list[dict] = [
    task(
        "smoke/pong",
        "Reply with the single word: pong",
        {"type": "exact", "value": "pong"},
    ),
    task(
        "smoke/math-1",
        "What is 17 * 24?" + SMOKE_CONTRACT,
        {"type": "math", "value": 408},
    ),
    task(
        "smoke/math-2",
        "A train leaves at 09:15 and travels for 2 hours and 40 minutes. "
        "At what time (24h, HH:MM) does it arrive?" + SMOKE_CONTRACT,
        {"type": "answer", "value": "11:55"},
    ),
]

SUITES: dict[str, list[dict]] = {"smoke": SMOKE_TASKS}


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    backend: str
    model: str
    endpoint: str
    api_key: str
    suites: list[str]
    tasks: list[str]
    repeat: int
    max_tokens: int
    temperature: float
    timeout: float
    sample_interval: float
    max_request_tokens: int
    warmup: bool
    npu_container: Optional[str]

    @property
    def out_id(self) -> str:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{stamp}_{self.model}_{self.backend}"


def selected_tasks(cfg: RunConfig) -> list[dict]:
    out: list[dict] = []
    for suite_name, tasks_ in SUITES.items():
        if cfg.suites and suite_name not in cfg.suites:
            continue
        for t in tasks_:
            if cfg.tasks and t["id"] not in cfg.tasks:
                continue
            out.append(t)
    return out


def console_line(r: dict) -> str:
    mark = {"pass": "✓", "fail": "✗", "error": "!", "timeout": "!"}.get(r["status"], "?")
    parts = [f"{mark} {r['id']}  {r['status']:<7}"]
    if r.get("total_s") is not None:
        parts.append(f"{r['total_s']:.2f}s")
    if r.get("ttft_s") is not None:
        parts.append(f"ttft {r['ttft_s']:.3f}s")
    if r.get("gen_tps") is not None:
        parts.append(f"{r['gen_tps']:.1f} tok/s")
    if r.get("decode_tps") is not None:
        parts.append(f"(content {r['decode_tps']:.1f})")
    if r.get("prompt_tokens") is not None:
        parts.append(f"(p {r['prompt_tokens']} / c {r.get('completion_tokens')})")
    if r.get("verdict") and r["verdict"].get("expected") is not None:
        parts.append(f"[want {str(r['verdict']['expected'])[:24]} got {str(r['verdict']['actual'])[:24]}]")
    if r.get("error"):
        parts.append(f"ERR {r['error'][:80]}")
    return "  ".join(parts)


def run_suite(cfg: RunConfig, out_dir: str, log: Callable[[str], None] = print) -> dict:
    prober = NPUProber(cfg.npu_container)
    log(f"[llmbench] accelerator sampler: {prober.backend or 'none'}"
        + (f" (container {prober.container})" if prober.container else ""))

    backend = build_backend(cfg)
    tasks = selected_tasks(cfg)
    if not tasks:
        log("[llmbench] no tasks matched; nothing to do")
        sys.exit(2)

    run: dict[str, Any] = {
        "schema": SCHEMA,
        "tool": f"llmbench {VERSION}",
        "started_at": now_iso(),
        "config": {
            "backend": cfg.backend, "model": cfg.model, "endpoint": cfg.endpoint,
            "suites": cfg.suites or sorted(SUITES), "task_filter": cfg.tasks,
            "repeat": cfg.repeat, "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature, "timeout_s": cfg.timeout,
            "sample_interval_s": cfg.sample_interval, "warmup": cfg.warmup,
        },
        "host": host_facts(),
        "metrics_legend": {
            "ttft_s": "request sent -> first content token (s)",
            "ttft_first_reasoning_s": "request sent -> first reasoning token (s)",
            "total_s": "request sent -> stream closed (s)",
            "decode_tps": "content chunks / (total - ttft) — content-only rate",
            "gen_tps": "completion tokens / (total - first any token) — incl. reasoning",
            "overall_tps": "completion tokens / total — incl. TTFT",
            "tokens": "from server usage; completion includes reasoning tokens",
            "resources": "sampled at --sample-interval; chip stats are avg/min/peak/p50/p95",
        },
        "tasks": [],
        "totals": {},
    }

    if cfg.warmup:
        log("[llmbench] warmup (unscored) ...")
        backend.warmup()
        time.sleep(0.5)

    for rep in range(cfg.repeat):
        for t in tasks:
            r = run_one_task(cfg, t, backend, prober, out_dir, rep, log)
            run["tasks"].append(r)
            log(console_line(r))

    run["finished_at"] = now_iso()
    run["totals"] = compute_totals(run["tasks"])
    return run


def build_backend(cfg: RunConfig):
    if cfg.backend == "api":
        return ApiBackend(cfg.endpoint, cfg.model, api_key=cfg.api_key,
                          max_tokens=cfg.max_tokens, temperature=cfg.temperature,
                          timeout=cfg.timeout)
    raise SystemExit(f"backend '{cfg.backend}' is not implemented yet (phase 2); "
                      f"use --backend api")


def run_one_task(cfg: RunConfig, t: dict, backend: Any, prober: NPUProber,
                 out_dir: str, rep: int, log: Callable[[str], None] = print) -> dict:
    rid = t["id"] if rep == 0 else f"{t['id']}#r{rep}"
    sampler = ResourceSampler(cfg.sample_interval, prober)
    sampler.start()
    result: ChatResult
    t_start = time.monotonic()
    if cfg.backend == "api":
        result = backend.chat([{"role": "user", "content": t["prompt"]}])
    else:  # pragma: no cover — phase 2
        raise NotImplementedError
    sampler.stop()
    wall = time.monotonic() - t_start

    res: dict[str, Any] = {
        "id": rid,
        "task_id": t["id"],
        "suite": t["suite"],
        "repeat": rep,
        "status": "pass" if not result.error else "error",
        "response": result.text,
        "ttft_first_reasoning_s": result.ttft_first_reasoning_s,
        "ttft_s": result.ttft_first_content_s,
        "total_s": round(result.total_s, 3),
        "wall_s": round(wall, 3),
        "finish_reason": result.finish_reason,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "content_chunks": result.content_chunks,
        "reasoning_chunks": result.reasoning_chunks,
        "error": result.error,
        "decode_tps": None,
        "gen_tps": None,
        "overall_tps": None,
    }
    # Throughput definitions (see run.json "metrics_legend"):
    #   ttft_s          request sent -> first content token
    #   decode_tps      content chunks / (end - first content token)  [content-only rate]
    #   gen_tps        completion tokens / (end - first any token)   [incl. reasoning]
    #   overall_tps    completion tokens / total latency             [incl. TTFT]
    first_any = min(filter(None, (result.ttft_first_reasoning_s,
                                 result.ttft_first_content_s)), default=None)
    if result.ttft_first_content_s is not None and \
            result.total_s > result.ttft_first_content_s and result.content_chunks:
        content_window = result.total_s - result.ttft_first_content_s
        if content_window >= 0.25:  # sub-250ms windows are burst noise
            res["decode_tps"] = round(result.content_chunks / content_window, 2)
    if first_any is not None and result.total_s > first_any and result.completion_tokens:
        res["gen_tps"] = round(result.completion_tokens / (result.total_s - first_any), 2)
    if result.completion_tokens and result.total_s > 0:
        res["overall_tps"] = round(result.completion_tokens / result.total_s, 2)

    # verdict
    if not result.error and result.text:
        verifier = VERIFIERS.get(t["verify"]["type"])
        if verifier is None:
            res["verdict"] = {"passed": False, "detail":
                              f"unimplemented verifier '{t['verify']['type']}' (phase 1)"}
        else:
            res["verdict"] = verifier(result.text, t["verify"])
            res["status"] = "pass" if res["verdict"]["passed"] else "fail"

    res["resources"] = aggregate_resources(
        sampler.samples,
        f"{prober.backend}" + (f" ({prober.container})" if prober.container else ""))

    # per-task artifacts
    task_dir = os.path.join(out_dir, "tasks")
    os.makedirs(task_dir, exist_ok=True)
    with open(os.path.join(task_dir, f"{rid.replace('/', '__')}.md"), "w") as f:
        r3 = lambda v: round(v, 3) if isinstance(v, (int, float)) else v  # noqa: E731
        f.write(f"# {rid}\n\n- suite: {t['suite']}\n- status: {res['status']}\n"
                f"- ttft: {r3(res['ttft_s'])}s (first reasoning: {r3(res.get('ttft_first_reasoning_s'))}s)"
                f"  total: {res['total_s']}s\n"
                f"- gen: {res.get('gen_tps')} tok/s  decode(content): "
                f"{res.get('decode_tps')} tok/s  overall: {res.get('overall_tps')} tok/s\n\n")
        f.write(f"## Prompt\n\n```\n{t['prompt']}\n```\n\n")
        if result.reasoning:
            f.write(f"## Reasoning\n\n```\n{result.reasoning}\n```\n\n")
        f.write(f"## Response\n\n```\n{result.text or '(empty)'}\n```\n\n")
        if res.get("verdict"):
            f.write(f"## Verdict\n\n```json\n{json.dumps(res['verdict'], indent=2)}\n```\n")
        if result.error:
            f.write(f"\n## Error\n\n```\n{result.error}\n```\n")
    with open(os.path.join(task_dir, f"{rid.replace('/', '__')}.samples.jsonl"), "w") as f:
        for s in sampler.samples:
            f.write(json.dumps(s) + "\n")
    return res


def compute_totals(task_results: list[dict]) -> dict:
    ok = [r for r in task_results if r.get("status") in ("pass", "fail")]
    by_suite: dict[str, list[dict]] = {}
    for r in task_results:
        by_suite.setdefault(r["suite"], []).append(r)
    suites_out = {}
    for suite, rs in sorted(by_suite.items()):
        scored = [r for r in rs if r.get("verdict")]
        suites_out[suite] = {
            "n": len(rs),
            "passed": sum(1 for r in scored if r["verdict"].get("passed")),
            "failed": sum(1 for r in scored if not r["verdict"].get("passed")),
            "errors": sum(1 for r in rs if r.get("error")),
            "pass_rate": round(sum(1 for r in scored if r["verdict"].get("passed"))
                              / len(scored), 3) if scored else None,
            "ttft_s": stats([r["ttft_s"] for r in rs]),
            "total_s": stats([r["total_s"] for r in rs]),
            "decode_tps": stats([r["decode_tps"] for r in rs]),
            "gen_tps": stats([r["gen_tps"] for r in rs]),
            "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in rs),
            "completion_tokens": sum(r.get("completion_tokens") or 0 for r in rs),
        }
    def flat(key: str) -> list:
        out = []
        for r in task_results:
            res = r.get("resources") or {}
            v = res.get(key)
            if isinstance(v, dict) and v.get("avg") is not None:
                out.append(v["avg"])
        return out

    chip_core: list[float] = []
    chip_hbm_peak: list[float] = []
    chip_ids: set[int] = set()
    for r in task_results:
        for c in (r.get("resources") or {}).get("chips") or []:
            chip_ids.add(c["chip"])
            if c.get("core_pct", {}).get("avg") is not None:
                chip_core.append(c["core_pct"]["avg"])
            if c.get("mem_used_gb", {}).get("peak") is not None:
                chip_hbm_peak.append(c["mem_used_gb"]["peak"])
    return {
        "n_tasks": len(task_results),
        "n_pass": sum(1 for r in task_results if r.get("status") == "pass"),
        "n_fail": sum(1 for r in task_results if r.get("status") == "fail"),
        "n_error": sum(1 for r in task_results if r.get("error")),
        "suites": suites_out,
        "ttft_s": stats([r["ttft_s"] for r in ok]),
        "total_s": stats([r["total_s"] for r in ok]),
        "decode_tps": stats([r["decode_tps"] for r in ok]),
        "gen_tps": stats([r["gen_tps"] for r in ok]),
        "overall_tps": stats([r["overall_tps"] for r in ok]),
        "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in ok),
        "completion_tokens": sum(r.get("completion_tokens") or 0 for r in ok),
        "total_tokens": sum(r.get("total_tokens") or 0 for r in ok),
        "resources_run": {
            "chip_count": len(chip_ids) or None,
            "chip_core_pct_avg": (round(statistics.fmean(chip_core), 2)
                                   if chip_core else None),
            "chip_hbm_used_peak_gb": max(chip_hbm_peak) if chip_hbm_peak else None,
            "sys_cpu_pct": stats(flat("sys_cpu_pct")),
            "ram_used_gb": stats(flat("ram_used_gb")),
            "page_cache_gb": stats(flat("page_cache_gb")),
            "swap_used_gb": stats(flat("swap_used_gb")),
        },
    }


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def write_outputs(run: dict, out_dir: str, log: Callable[[str], None] = print) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "run.json"), "w") as f:
        json.dump(run, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write(render_report_md(run))
    log(f"[llmbench] wrote {out_dir}/run.json and report.md")


def _ttft_reasoning_stats(run: dict) -> dict:
    return stats([t.get("ttft_first_reasoning_s") for t in run["tasks"]])


def render_report_md(run: dict) -> str:
    cfg, tot = run["config"], run["totals"]
    L: list[str] = []
    L.append(f"# llmbench report — {cfg['model']} via {cfg['backend']}\n")
    L.append(f"- started: {run['started_at']}  ·  finished: {run.get('finished_at')}")
    L.append(f"- endpoint: `{cfg['endpoint']}`  ·  repeat: {cfg['repeat']}  "
             f"·  max_tokens: {cfg['max_tokens']}  ·  temperature: {cfg['temperature']}")
    h = run["host"]
    L.append(f"- host: {h.get('hostname')} · {h.get('arch')} · {h.get('cpu_cores')} cores "
             f"· {h.get('ram_total_gb')} GB RAM · accel: {h.get('accelerator')}\n")

    L.append("## Scorecard\n")
    L.append("| suite | tasks | passed | failed | errors | pass rate |")
    L.append("|---|---|---|---|---|---|")
    for suite, s in sorted(tot["suites"].items()):
        L.append(f"| {suite} | {s['n']} | {s['passed']} | {s['failed']} | {s['errors']} "
                 f"| {fnum(s['pass_rate'] * 100, 1) if s['pass_rate'] is not None else '—'}% |")
    L.append("")

    def statline(name: str, d: dict, unit: str) -> str:
        if not d:
            return f"| {name} | — |"
        return (f"| {name} | avg {d['avg']}{unit} · p50 {d['p50']}{unit} "
                f"· p95 {d['p95']}{unit} · peak {d['peak']}{unit} |")

    L.append("## Latency & throughput\n")
    L.append("| metric | value |")
    L.append("|---|---|")
    L.append(statline("TTFT (req→first content token)", tot["ttft_s"], "s"))
    L.append(statline("TTFT (req→first reasoning token)", _ttft_reasoning_stats(run), "s"))
    L.append(statline("total latency", tot["total_s"], "s"))
    L.append(statline("gen tok/s (all completion tokens)", tot["gen_tps"], ""))
    L.append(statline("decode tok/s (content only)", tot["decode_tps"], ""))
    L.append(statline("overall tok/s (incl. TTFT)", tot["overall_tps"], ""))
    L.append("")
    L.append(f"tokens — prompt: **{tot['prompt_tokens']}** · completion: "
             f"**{tot['completion_tokens']}** · total: **{tot['total_tokens']}**\n")

    L.append("## Resources (mean/peak over run)\n")
    rc = tot.get("resources_run") or {}
    L.append("| metric | value |")
    L.append("|---|---|")
    chips = None
    for r in run["tasks"]:
        res = r.get("resources") or {}
        if res.get("chips"):
            chips = res["chips"]
            break
    if chips:
        for c in chips:
            L.append(f"| NPU {c['chip']} AICore % | "
                     f"{c['core_pct'].get('avg', '—')} avg · {c['core_pct'].get('peak', '—')} peak |")
            L.append(f"| NPU {c['chip']} HBM used | "
                     f"{c['mem_used_gb'].get('avg', '—')} GB avg · {c['mem_used_gb'].get('peak', '—')} GB peak |")
            if c.get("power_w"):
                L.append(f"| NPU {c['chip']} power | {c['power_w'].get('avg', '—')} W avg |")
    for key, label in (("sys_cpu_pct", "CPU % (system)"),
                       ("ram_used_gb", "RAM used (GB)"),
                       ("page_cache_gb", "page cache (GB)"),
                       ("swap_used_gb", "swap used (GB)")):
        d = rc.get(key) or {}
        if d:
            L.append(f"| {label} | {d.get('avg', '—')} avg · {d.get('peak', '—')} peak |")
    L.append("")

    L.append("## Per-task\n")
    L.append("| task | status | total s | ttft s | gen tok/s | dec tok/s | prompt tok | compl tok |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in run["tasks"]:
        ttft = r.get("ttft_s")
        L.append(f"| {r['id']} | {r['status']} | {r.get('total_s')} "
                 f"| {round(ttft, 3) if ttft is not None else None} "
                 f"| {r.get('gen_tps')} | {r.get('decode_tps')} | {r.get('prompt_tokens')} "
                 f"| {r.get('completion_tokens')} |")
    L.append("")
    return "\n".join(L) + "\n"


def console_summary(run: dict, log: Callable[[str], None] = print) -> None:
    tot = run["totals"]
    log("\n[llmbench] === summary ===")
    log(f"tasks: {tot['n_tasks']}  pass: {tot['n_pass']}  fail: {tot['n_fail']}  "
        f"error: {tot['n_error']}")
    for suite, s in sorted(tot["suites"].items()):
        pr = fnum(s["pass_rate"] * 100, 1) if s["pass_rate"] is not None else "—"
        log(f"  {suite:<14} {s['passed']}/{s['n']} pass ({pr}%)  "
            f"ttft avg {s['ttft_s'].get('avg')}s  total avg {s['total_s'].get('avg')}s  "
            f"gen {s['gen_tps'].get('avg')} tok/s (content {s['decode_tps'].get('avg')})")
    log(f"tokens: {tot['prompt_tokens']} prompt / {tot['completion_tokens']} completion "
        f"/ {tot['total_tokens']} total")


# ---------------------------------------------------------------------------
# sample subcommand (debug / manual monitoring)
# ---------------------------------------------------------------------------

def cmd_sample(args: argparse.Namespace) -> None:
    prober = NPUProber(args.container)
    print(f"[llmbench] sampling for {args.seconds}s every {args.interval}s "
          f"(accel: {prober.backend or 'none'})")
    sampler = ResourceSampler(args.interval, prober)
    sampler.start()
    try:
        time.sleep(args.seconds)
    except KeyboardInterrupt:
        pass
    sampler.stop()
    agg = aggregate_resources(
        sampler.samples,
        f"{prober.backend}" + (f" ({prober.container})" if prober.container else ""))
    print(json.dumps(agg, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llmbench",
        description="Non-interactive benchmark harness for locally served models.")
    p.add_argument("--version", action="version", version=f"llmbench {VERSION}")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run benchmark suite(s)")
    pr.add_argument("--backend", choices=["api", "dsh", "codex", "claude"], default="api")
    pr.add_argument("--endpoint", default=os.environ.get("LLMBENCH_ENDPOINT",
                                                          "http://127.0.0.1:8000/v1"))
    pr.add_argument("--model", default=os.environ.get("LLMBENCH_MODEL", "qwen3.8"))
    pr.add_argument("--api-key", default=os.environ.get("LLMBENCH_API_KEY", "llmbench"))
    pr.add_argument("--suites", default="",
                    help="comma-separated suite names (default: all)")
    pr.add_argument("--tasks", default="", help="comma-separated task ids (default: all)")
    pr.add_argument("--repeat", type=int, default=1)
    pr.add_argument("--max-tokens", type=int, default=4096)
    pr.add_argument("--temperature", type=float, default=0.0)
    pr.add_argument("--timeout", type=float, default=600.0)
    pr.add_argument("--sample-interval", type=float, default=2.0)
    pr.add_argument("--warmup", dest="warmup", action="store_true", default=True)
    pr.add_argument("--no-warmup", dest="warmup", action="store_false")
    pr.add_argument("--npu-container", default=os.environ.get("LLMBENCH_NPU_CONTAINER"))
    pr.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "results"))
    pr.set_defaults(func=cmd_run)

    pl = sub.add_parser("list", help="list available suites/tasks")
    pl.set_defaults(func=cmd_list)

    ps = sub.add_parser("sample", help="sample host/accelerator resources for N seconds")
    ps.add_argument("--seconds", type=float, default=10.0)
    ps.add_argument("--interval", type=float, default=2.0)
    ps.add_argument("--container", default=os.environ.get("LLMBENCH_NPU_CONTAINER"))
    ps.set_defaults(func=cmd_sample)

    prpt = sub.add_parser("report", help="re-render report.md from an existing run dir")
    prpt.add_argument("run_dir")
    prpt.set_defaults(func=cmd_report)
    return p


def cmd_list(_: argparse.Namespace) -> None:
    print(f"{'task id':<28} {'verifier':<10}")
    for suite_name, tasks_ in SUITES.items():
        print(f"\n[{suite_name}]")
        for t in tasks_:
            print(f"  {t['id']:<28} {t['verify']['type']:<10}")


def cmd_report(args: argparse.Namespace) -> None:
    path = os.path.join(args.run_dir, "run.json")
    with open(path) as f:
        run = json.load(f)
    os.makedirs(args.run_dir, exist_ok=True)
    with open(os.path.join(args.run_dir, "report.md"), "w") as f:
        f.write(render_report_md(run))
    console_summary(run)
    print(f"[llmbench] re-rendered {args.run_dir}/report.md")


def cmd_run(args: argparse.Namespace) -> None:
    cfg = RunConfig(
        backend=args.backend, model=args.model, endpoint=args.endpoint,
        api_key=args.api_key,
        suites=[s for s in args.suites.split(",") if s],
        tasks=[t for t in args.tasks.split(",") if t],
        repeat=args.repeat, max_tokens=args.max_tokens,
        temperature=args.temperature, timeout=args.timeout,
        sample_interval=args.sample_interval, max_request_tokens=0,
        warmup=args.warmup, npu_container=args.npu_container,
    )
    out_dir = os.path.join(args.out, cfg.out_id)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[llmbench] run {cfg.out_id} → {out_dir}")
    run = run_suite(cfg, out_dir)
    write_outputs(run, out_dir)
    console_summary(run)


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
