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
import ast
import fcntl
import http.client
import json
import math
import operator
import os
import platform
import queue
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

VERSION = "0.1.0"
BACKENDS = ["api", "dsh", "codex", "claude", "pi"]  # all known backends
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


def stats_ci95(values: list[float]) -> Optional[float]:
    """95% CI half-width over the (repeated) samples; None for n < 2."""
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    return round(1.96 * statistics.stdev(values) / math.sqrt(len(values)), 3)


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
    """CPU% + RSS of a root process and its descendants (Linux /proc).

    With root_pid set, only the process subtree is counted; the root can be
    attached later via the constructor (the sampler creates it once the
    backend has spawned its process).
    """

    def __init__(self, root_pid: Optional[int] = None):
        self.hz = os.sysconf("SC_CLK_TCK")
        self.root_pid = root_pid
        self._last_ticks: Optional[float] = None
        self._last_t: Optional[float] = None

    def _scan(self) -> tuple[float, int, list[int]]:
        pids_of: dict[int, int] = {}   # pid -> ppid
        ticks_of: dict[int, float] = {}
        rss_of: dict[int, int] = {}
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
                ppid = int(parts[1])
                ticks = (int(parts[11]) + int(parts[12])) / self.hz
            except ValueError:
                continue
            pids_of[int(pid)] = ppid
            ticks_of[int(pid)] = ticks
            try:
                with open(f"{path}/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            rss_of[int(pid)] = int(line.split()[1])
                            break
            except OSError:
                pass
        if self.root_pid is None:
            return sum(ticks_of.values()), sum(rss_of.values()), list(ticks_of)
        # walk the subtree from the root
        children: dict[int, list[int]] = {}
        for pid, ppid in pids_of.items():
            children.setdefault(ppid, []).append(pid)
        total_ticks, total_rss, seen = 0.0, 0, []
        stack = [self.root_pid]
        while stack:
            pid = stack.pop()
            if pid not in ticks_of:
                continue
            total_ticks += ticks_of[pid]
            total_rss += rss_of.get(pid, 0)
            seen.append(pid)
            stack.extend(children.get(pid, []))
        return total_ticks, total_rss, seen

    def sample(self) -> Optional[dict]:
        ticks, rss_kb, _pids = self._scan()
        t = time.monotonic()
        if self._last_ticks is None or self._last_t is None:
            self._last_ticks, self._last_t = ticks, t
            return None
        dt = t - self._last_t
        if dt <= 0:
            return None
        cpu_pct = max(0.0, 100.0 * (ticks - self._last_ticks) / dt)
        out = {"cpu_pct": round(cpu_pct, 2), "rss_mb": round(rss_kb / 1024.0, 1)}
        self._last_ticks, self._last_t = ticks, t
        return out


class ResourceSampler:
    """Background sampler: NPU (slow prober thread) + system /proc (fast loop)."""

    def __init__(self, interval: float, prober: Optional[NPUProber] = None):
        self.interval = max(0.2, interval)
        self.prober = prober or NPUProber()
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._latest_npu: Optional[list[dict]] = None
        self._tree: Optional[_ProcTree] = None
        self._sys_last: Optional[tuple[int, int]] = None
        self._threads: list[threading.Thread] = []

    def set_root(self, pid: int) -> None:
        """Attach the CLI process tree to sample (called right after spawn)."""
        self._tree = _ProcTree(pid)

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
        # fallback: trailing number, e.g. "7 × 8 = 56" or "the answer: 42."
        m = re.search(r"[-+]?\d+(?:\.\d+)?", x)
        if m:
            try:
                return float(m.group())
            except ValueError:
                return None
        return None


def check_exact(ctx: dict, spec: dict) -> dict:
    got = ctx["response"].strip()
    want = str(spec["value"]).strip()
    ok = got.lower() == want.lower()
    return {"passed": ok, "expected": want, "actual": got[:200]}


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).lower()


def _strip_end_punct(s: str) -> str:
    """Trailing sentence punctuation only ('B.' -> 'B', '11:55.' -> '11:55').
    Applied to both sides of a comparison so it can't collide distinct values."""
    return s.rstrip(".,;:!?'\")").strip()


def check_answer(ctx: dict, spec: dict) -> dict:
    """ANSWER: contract with optional numeric equivalence."""
    got = extract_answer(ctx["response"])
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
    elif spec.get("compact"):
        ok = _norm(_strip_end_punct(got)) == _norm(_strip_end_punct(str(want)))
    else:
        ok = (_strip_end_punct(got).lower() == _strip_end_punct(str(want)).lower())
    return {"passed": ok, "expected": want, "actual": got[:200]}


def check_math(ctx: dict, spec: dict) -> dict:
    """ANSWER: contract, numeric equivalence only."""
    return check_answer(ctx, {**spec, "numeric": True})


# --- safe arithmetic eval (tool-argument expressions) -----------------------

_AE_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
              ast.FloorDiv: operator.floordiv}


def safe_eval(expr: str):
    node = ast.parse(expr, mode="eval").body

    def ev(n: ast.AST):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _AE_BINOPS:
            return _AE_BINOPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        raise ValueError(f"unsupported in expression: {ast.dump(n)}")

    return ev(node)


# --- instruction-following constraints --------------------------------------

# a sentence ends at terminal punctuation followed by whitespace, a closing
# quote/bracket (…Licht." — the quote would otherwise hide the final period),
# or end-of-text
SENT_RE = re.compile(r"[.!?]+(?:\s|['\")\]]|$)")


def _word_re(w: str) -> re.Pattern:
    return re.compile(r"(?<![\w'])" + re.escape(w) + r"(?![\w'])", re.IGNORECASE)


def check_constraints(ctx: dict, spec: dict) -> dict:
    text = ctx["response"]
    checks: list[dict] = []
    for c in spec["checks"]:
        t = c["type"]
        if t == "sentences":
            n = len(SENT_RE.findall(text.strip()))
            if "count" in c:
                ok = n == c["count"]
                label = f"sentences=={c['count']}"
            elif "max" in c:
                ok = n <= c["max"]
                label = f"sentences<={c['max']}"
            elif "min" in c:
                ok = n >= c["min"]
                label = f"sentences>={c['min']}"
            else:
                ok, label = False, "sentences: bad spec"
            checks.append({"check": label, "passed": ok, "actual": n})
        elif t == "words":
            n = len(re.findall(r"\S+", text))
            if "count" in c:
                ok = n == c["count"]
            else:
                ok = c.get("min", 0) <= n <= c.get("max", 10 ** 9)
            checks.append({"check": f"words:{c}", "passed": ok, "actual": n})
        elif t == "include_all":
            missing = [w for w in c["words"] if not _word_re(w).search(text)]
            checks.append({"check": f"include {c['words']}", "passed": not missing,
                           "missing": missing})
        elif t == "exclude_any":
            present = [w for w in c["words"] if _word_re(w).search(text)]
            checks.append({"check": f"exclude {c['words']}", "passed": not present,
                           "present": present})
        elif t == "no_chars":
            found = [ch for ch in c["chars"] if ch in text]
            checks.append({"check": f"no_chars {c['chars']!r}", "passed": not found,
                           "found": found})
        elif t == "lines":
            lines = [ln for ln in text.strip().splitlines() if ln.strip()]
            ok = len(lines) == c["count"]
            if "prefix" in c:
                ok = ok and all(ln.lstrip().startswith(c["prefix"]) for ln in lines)
            checks.append({"check": f"lines:{c}", "passed": ok,
                           "actual": len(lines)})
        elif t == "first_line":
            lines = [ln for ln in text.strip().splitlines() if ln.strip()]
            got = lines[0].strip() if lines else ""
            ok = (got == c["line"]) if not c.get("ci") else (got.lower() == c["line"].lower())
            checks.append({"check": "first_line", "passed": ok, "actual": got[:120]})
        elif t == "json":
            obj, err = _extract_json(text)
            if obj is None:
                checks.append({"check": "json_parse", "passed": False, "detail": err})
            else:
                bad = []
                for key, want in c["shape"].items():
                    if key not in obj:
                        bad.append(f"missing {key}")
                        continue
                    val = obj[key]
                    wtype = want.get("type") if isinstance(want, dict) else want
                    wlen = want.get("len") if isinstance(want, dict) else None
                    type_ok = {"str": str, "int": int, "float": (int, float),
                               "list": list, "dict": dict, "bool": bool}.get(
                                   wtype, wtype)
                    if wtype in ("int", "float") and not isinstance(val, int):
                        type_ok = False
                    if type_ok is not False and not isinstance(val, type_ok):
                        bad.append(f"{key}: wrong type {type(val).__name__}")
                    elif isinstance(wlen, int) and not (isinstance(val, (list, str, dict))
                                                        and len(val) == wlen):
                        bad.append(f"{key}: len != {wlen}")
                checks.append({"check": "json_shape", "passed": not bad,
                               "detail": bad or None,
                               "actual": json.dumps(obj)[:200]})
        elif t == "script":
            # response must actually be written in the given script:
            # fraction of alphabetic chars inside `char_class` >= min_pct
            pat = re.compile(c.get("char_class", r"\w"))
            alpha = [ch for ch in text if ch.isalpha()]
            pct = (sum(1 for ch in alpha if pat.fullmatch(ch))
                   / len(alpha)) if alpha else 0.0
            min_pct = c.get("min_pct", 0.5)
            checks.append({"check": f"script>= {min_pct:.0%}", "passed":
                            bool(alpha) and pct >= min_pct,
                            "actual_pct": round(pct, 3),
                            "alpha_chars": len(alpha)})
        elif t == "case":
            alpha = [ch for ch in text if ch.isalpha()]
            if c["mode"] == "upper":
                ok = bool(alpha) and all(ch.isupper() for ch in alpha)
            else:
                ok = not alpha or all(ch.islower() for ch in alpha)
            checks.append({"check": f"all_{c['mode']}", "passed": ok})
        else:
            checks.append({"check": t, "passed": False,
                           "detail": f"unknown constraint type {t!r}"})
    passed = all(c["passed"] for c in checks)
    return {"passed": passed, "checks": checks}


def _extract_json(text: str):
    t = text.strip()
    try:
        return json.loads(t), None
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", t, re.S)
    if m:
        t2 = m.group(1).strip()
    else:
        a, b = t.find("{"), t.rfind("}")
        t2 = t[a:b + 1] if 0 <= a < b else None
    if t2:
        try:
            return json.loads(t2), None
        except json.JSONDecodeError as e:
            return None, str(e)
    return None, "no JSON object found"


# --- coding: extract solution, run unittest suite ---------------------------

CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)

TEST_BOILERPLATE = '''
if __name__ == "__main__":
    import json as _json, sys as _sys
    _runner = unittest.TextTestRunner(verbosity=0)
    _res = _runner.run(
        unittest.defaultTestLoader.loadTestsFromModule(_sys.modules[__name__]))
    _fails = [str(t).split("\\n")[0]
              for t, _tb in list(_res.failures) + list(_res.errors)]
    print(_json.dumps({"total": _res.testsRun,
                       "passed": _res.testsRun - len(_fails),
                       "failed": _fails}))
'''


def _extract_code(response: str) -> str:
    m = CODE_BLOCK_RE.search(response)
    return m.group(1) if m else response


def check_code(ctx: dict, spec: dict) -> dict:
    # CLI backends usually wrote the solution to a file in the scratch dir;
    # api backends require extraction from the response text.
    scratch = ctx.get("scratch")
    code, source = None, "response"
    if scratch:
        sol = os.path.join(scratch, "solution.py")
        if os.path.isfile(sol):
            try:
                with open(sol) as f:
                    code = f.read()
                source = "scratch/solution.py"
            except OSError:
                code = None
    if code is None:
        code = _extract_code(ctx["response"])
    work = os.path.join(ctx["out_dir"], "code", ctx["task_id"].replace("/", "__"))
    os.makedirs(work, exist_ok=True)
    with open(os.path.join(work, "solution.py"), "w") as f:
        f.write(code)
    with open(os.path.join(work, "test.py"), "w") as f:
        f.write(spec["tests"] + TEST_BOILERPLATE)
    try:
        proc = subprocess.run(
            [sys.executable, "test.py"], cwd=work, capture_output=True, text=True,
            timeout=spec.get("timeout", 60))
        out, err = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return {"passed": False, "detail": "test run timed out",
                "code_dir": work}
    summary = None
    for line in reversed(out.splitlines()):
        line = line.strip()
        if line.startswith('{"total"'):
            try:
                summary = json.loads(line)
            except json.JSONDecodeError:
                pass
            break
    if summary is None:
        return {"passed": False, "detail": "no test summary",
                "stdout_tail": out[-800:], "stderr_tail": err[-800:],
                "code_dir": work}
    return {"passed": summary["passed"] == summary["total"],
            "tests": summary, "source": source, "code_dir": work,
            "stderr_tail": err[-400:]}


def check_file(ctx: dict, spec: dict) -> dict:
    """Verify a file the agent created in the scratch dir."""
    p = os.path.join(ctx["scratch"], spec["path"])
    if not os.path.isfile(p):
        return {"passed": False, "expected": spec.get("content"),
                "actual": None, "detail": "file missing"}
    with open(p) as f:
        content = f.read()
    ok = True
    if spec.get("content") is not None:
        ok = ok and content == spec["content"]
    if "regexp" in spec:
        ok = ok and bool(re.search(spec["regexp"], content, re.S))
    return {"passed": ok, "expected": spec.get("content"),
            "actual": content[:200]}


def check_run(ctx: dict, spec: dict) -> dict:
    """Run a command in the scratch dir (post-task) and require all tests pass."""
    cmd = spec["cmd"]
    if isinstance(cmd, str):
        import shlex
        cmd = shlex.split(cmd)
    try:
        proc = subprocess.run(cmd, cwd=ctx["scratch"], capture_output=True,
                              text=True, timeout=spec.get("timeout", 120))
        out, err = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return {"passed": False, "detail": "post-task command timed out"}
    summary = None
    for line in reversed(out.splitlines()):
        line = line.strip()
        if line.startswith('{"total"'):
            try:
                summary = json.loads(line)
            except json.JSONDecodeError:
                pass
            break
    if summary is None:
        return {"passed": False, "detail": "no test summary found",
                "stdout_tail": out[-600:], "stderr_tail": err[-600:]}
    return {"passed": summary["passed"] == summary["total"], "tests": summary,
            "stdout_tail": out[-400:], "stderr_tail": err[-400:]}


# --- tool calling -------------------------------------------------------------

def _args_ok(args: dict, subset: dict) -> bool:
    for k, v in subset.items():
        if k not in args:
            return False
        a = args[k]
        if isinstance(v, str):
            if str(a).lower() != str(v).lower():
                return False
        elif a != v:
            return False
    return True


def check_toolcall(ctx: dict, spec: dict) -> dict:
    calls = ctx.get("tool_calls") or []
    checks: list[dict] = []
    if spec.get("no_calls"):
        checks.append({"check": "no_calls", "passed": not calls,
                       "actual_calls": len(calls)})
    for want in spec.get("calls", []):
        name, subset = want["name"], want.get("args_subset") or {}
        match = None
        for c in calls:
            if c["name"] == name and _args_ok(c["args"], subset):
                match = c
                break
        entry = {"check": f"call:{name}", "passed": match is not None,
                 "expected": {"name": name, "args_subset": subset},
                 "actual": match["args"] if match else
                         [c["args"] for c in calls]}
        if want.get("args_eval"):
            # e.g. {"key": "expression", "expect": 5888} — safe-eval an arg
            key, expect = want["args_eval"]["key"], want["args_eval"]["expect"]
            ev_ok = False
            for c in calls:
                if c["name"] != name or key not in c["args"]:
                    continue
                try:
                    ev_ok = math.isclose(safe_eval(str(c["args"][key])),
                                         float(expect), rel_tol=1e-9)
                except Exception:  # noqa: BLE001
                    ev_ok = False
            entry["check"] += f"[eval {key}=={expect}]"
            entry["passed"] = match is not None and ev_ok
        checks.append(entry)
    if "max_calls" in spec:
        checks.append({"check": "max_calls", "passed": len(calls) <= spec["max_calls"],
                       "actual": len(calls)})
    if "min_calls" in spec:
        checks.append({"check": "min_calls", "passed": len(calls) >= spec["min_calls"],
                       "actual": len(calls)})
    if "answer" in spec:
        sub = check_answer(ctx, {"value": spec["answer"],
                                 "numeric": spec.get("numeric"),
                                 "compact": spec.get("compact")})
        checks.append({"check": "answer", "passed": sub["passed"],
                       "expected": sub["expected"], "actual": sub["actual"]})
    return {"passed": all(c["passed"] for c in checks), "checks": checks,
            "calls": [{"name": c["name"], "args": c["args"]} for c in calls]}


def check_gridpath(ctx: dict, spec: dict) -> dict:
    """Verify a move-string (U/D/L/R) is a shortest valid path on the grid."""
    got = extract_answer(ctx["response"]).upper().replace(" ", "")
    grid, expect_len = spec["grid"], spec["expect_len"]
    start = goal = None
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == "S":
                start = (r, c)
            if ch == "G":
                goal = (r, c)
    if not re.fullmatch(r"[UDLR]+", got):
        return {"passed": False, "expected": f"{expect_len} moves",
                "actual": got[:60], "detail": "not a pure U/D/L/R move string"}
    r, c = start
    deltas = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
    bad = None
    for i, mv in enumerate(got):
        r += deltas[mv][0]
        c += deltas[mv][1]
        if not (0 <= r < len(grid) and 0 <= c < len(grid[0])):
            bad = f"move {i + 1} ({mv}) leaves the grid"
            break
        if grid[r][c] == "#":
            bad = f"move {i + 1} ({mv}) hits a wall"
            break
    if bad:
        return {"passed": False, "expected": f"{expect_len} moves",
                "actual": got[:60], "detail": bad}
    return {"passed": (r, c) == goal and len(got) == expect_len,
            "expected": f"{expect_len} moves ending at G",
            "actual": f"{got} ({len(got)} moves)"}


VERIFIERS: dict[str, Callable[[dict, dict], dict]] = {
    "exact": check_exact,
    "answer": check_answer,
    "math": check_math,
    "constraints": check_constraints,
    "code": check_code,
    "file": check_file,
    "gridpath": check_gridpath,
    "run": check_run,
    "toolcall": check_toolcall,
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
    tool_calls: list = field(default_factory=list)  # [{id, name, args}]
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
        done = False  # stream completed with [DONE]
        try:
            conn.request("POST", req_path, body=req_body, headers=headers)
            resp = conn.getresponse()
            if resp.status != 200:
                err = resp.read(4000).decode(errors="replace")
                res.error = f"HTTP {resp.status}: {err[:500]}"
                res.total_s = time.monotonic() - t0
                return res

            buf = b""
            tc_acc: dict[int, dict] = {}
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
                        done = True
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
                        for tcall in delta.get("tool_calls") or []:
                            idx = tcall.get("index", 0)
                            d = tc_acc.setdefault(idx, {"id": None, "name": "", "args": ""})
                            if tcall.get("id"):
                                d["id"] = tcall["id"]
                            fn = tcall.get("function") or {}
                            if fn.get("name"):
                                d["name"] += fn["name"]
                            if fn.get("arguments") is not None:
                                d["args"] += fn["arguments"]
                        if choices[0].get("finish_reason"):
                            res.finish_reason = choices[0]["finish_reason"]
                    usage = ev.get("usage")
                    if usage:
                        res.prompt_tokens = usage.get("prompt_tokens")
                        res.completion_tokens = usage.get("completion_tokens")
                        res.total_tokens = usage.get("total_tokens")
                        res.raw_last = usage
            for idx in sorted(tc_acc):
                d = tc_acc[idx]
                try:
                    args = json.loads(d["args"]) if d["args"].strip() else {}
                except json.JSONDecodeError:
                    args = {"_raw": d["args"]}
                res.tool_calls.append({"id": d["id"], "name": d["name"], "args": args})
        except Exception as e:  # noqa: BLE001
            if done:
                pass  # stream completed; a late local failure is harmless
            elif not res.text and not res.tool_calls:
                res.error = f"{type(e).__name__}: {e}"
            else:
                # partial text arrived but the stream never finished — a
                # transport interruption, not a model verdict (so it retries)
                res.error = f"stream interrupted ({type(e).__name__}: {e})"
        finally:
            conn.close()
            res.total_s = time.monotonic() - t0
        if res.ttft_first_content_s is None and res.ttft_first_reasoning_s is not None:
            res.ttft_first_content_s = res.ttft_first_reasoning_s
        return res

    def warmup(self) -> None:
        self.chat([{"role": "user", "content": "Reply with the single word: ready"}])

    def run(self, t: dict, scratch: str,
            set_root: Callable[[int], None]) -> "TaskOutcome":
        msgs = []
        if t.get("system"):
            msgs.append({"role": "system", "content": t["system"]})
        msgs.append({"role": "user", "content": t["prompt"]})
        r = self.chat(msgs, tools=t.get("tools"))
        o = TaskOutcome(
            text=r.text, reasoning=r.reasoning, ttft_s=r.ttft_first_content_s,
            ttft_model_s=r.ttft_first_content_s,
            ttft_first_reasoning_s=r.ttft_first_reasoning_s, total_s=r.total_s,
            prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens,
            total_tokens=r.total_tokens, token_source="server",
            finish_reason=r.finish_reason, error=r.error, tool_calls=r.tool_calls,
            content_chunks=r.content_chunks, reasoning_chunks=r.reasoning_chunks)
        return o


# ---------------------------------------------------------------------------
# unified task outcome + CLI backends (dsh / codex / claude)
# ---------------------------------------------------------------------------

def _est_tokens(s: str) -> int:
    """Rough token estimate when the CLI reports no usage (~4 chars/token)."""
    return max(1, len(s) // 4)


@dataclass
class TaskOutcome:
    text: str = ""
    reasoning: str = ""
    ttft_s: Optional[float] = None            # request sent -> first output
    ttft_model_s: Optional[float] = None      # -> first MODEL-originated output
    ttft_first_reasoning_s: Optional[float] = None
    total_s: float = 0.0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    token_source: str = "server"              # or "estimate"
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    exit_code: Optional[int] = None
    stderr_tail: str = ""
    tool_calls: list = field(default_factory=list)
    events: list = field(default_factory=list)  # compact event summaries (CLI)
    content_chunks: Optional[int] = None
    reasoning_chunks: Optional[int] = None


class CliBackend:
    """Base for CLI backends: spawn the CLI in a scratch dir and stream stdout.

    TTFT for CLI backends = time to first stdout byte (includes CLI startup;
    the api backend's TTFT is the server-side request->first-token metric).
    """

    name = "cli"

    def __init__(self, cfg: "RunConfig"):
        self.cfg = cfg
        self._prompt: str = ""

    # -- subclass hooks -------------------------------------------------------
    def build(self, t: dict, scratch: str) -> tuple[list[str], dict]:
        raise NotImplementedError

    def parse_line(self, outcome: TaskOutcome, line: str) -> None:  # noqa: B027
        pass

    def finalize(self, outcome: TaskOutcome) -> None:  # noqa: B027
        if outcome.completion_tokens is None:
            outcome.completion_tokens = _est_tokens(outcome.text)
            outcome.prompt_tokens = _est_tokens(self._prompt)
            outcome.total_tokens = (outcome.prompt_tokens or 0) + \
                                   (outcome.completion_tokens or 0)
            outcome.token_source = "estimate"
        elif outcome.total_tokens is None:
            outcome.total_tokens = (outcome.prompt_tokens or 0) + \
                                   (outcome.completion_tokens or 0)

    def warmup(self) -> None:
        with tempfile.TemporaryDirectory(prefix="llmbench-warm-") as wd:
            self.run({"prompt": "Reply with the single word: ready"}, wd,
                     lambda pid: None)

    # -- generic process runner ----------------------------------------------
    def run(self, t: dict, scratch: str,
            set_root: Callable[[int], None]) -> TaskOutcome:
        cfg = self.cfg
        cmd, env = self.build(t, scratch)
        outcome = TaskOutcome()
        proc = subprocess.Popen(
            cmd, cwd=scratch, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        set_root(proc.pid)
        t0 = time.monotonic()
        deadline = t0 + cfg.timeout

        q: "queue.Queue" = queue.Queue()
        err_chunks: list[bytes] = []

        def reader() -> None:
            assert proc.stdout is not None
            try:
                while True:
                    chunk = proc.stdout.read1(4096)
                    if not chunk:
                        break
                    q.put(chunk)
            finally:
                # sentinel even on a read error — without it the drain loop
                # would spin until the deadline on a dead pipe
                q.put(None)

        def err_drain() -> None:
            # drain stderr concurrently: a verbose CLI writing >64 KiB to
            # stderr would otherwise deadlock on the full pipe buffer
            assert proc.stderr is not None
            try:
                while True:
                    chunk = proc.stderr.read1(65536)
                    if not chunk:
                        break
                    if len(b"".join(err_chunks)) < 8192:
                        err_chunks.append(chunk)
            except (OSError, ValueError):
                pass

        th = threading.Thread(target=reader, daemon=True)
        th.start()
        eth = threading.Thread(target=err_drain, daemon=True)
        eth.start()
        buf = b""
        closed = False
        while not closed:
            drained = False
            while not drained:
                try:
                    chunk = q.get(timeout=0.1)
                except queue.Empty:
                    break
                drained = True
                if chunk is None:
                    closed = True
                    break
                if outcome.ttft_s is None and chunk.strip():
                    outcome.ttft_s = time.monotonic() - t0
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    self.parse_line(outcome, line.decode(errors="replace").rstrip("\r"))
            if time.monotonic() >= deadline:
                self._kill(proc)
                if not outcome.error:
                    outcome.error = f"timeout after {cfg.timeout:.0f}s"
                break
        if buf.strip():
            # a final event without a trailing newline must not be dropped
            self.parse_line(outcome, buf.decode(errors="replace").rstrip("\r"))
        th.join(timeout=5)
        eth.join(timeout=5)
        outcome.exit_code = proc.poll()
        if outcome.exit_code is None:  # not yet reaped (rare after pipe EOF)
            try:
                outcome.exit_code = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._kill(proc)
                outcome.exit_code = proc.wait()
        outcome.stderr_tail = b"".join(err_chunks)[-1200:].decode(errors="replace")
        outcome.total_s = time.monotonic() - t0
        if outcome.ttft_model_s is not None:  # parse_line stored a monotonic clock value
            outcome.ttft_model_s = round(outcome.ttft_model_s - t0, 3)
        self.finalize(outcome)
        self.after_run(scratch)
        return outcome

    def after_run(self, scratch: str) -> None:  # noqa: B027
        pass

    @staticmethod
    def _kill(proc: subprocess.Popen) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), 9)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except OSError:
                pass

    @staticmethod
    def full_prompt(t: dict) -> str:
        p = t["prompt"]
        return (t["system"] + "\n\n" + p) if t.get("system") else p


class DshBackend(CliBackend):
    """dsh --profile headless — one task, prints the final assistant message.

    The per-run `--patch` overlay defines a self-contained provider
    (baseURL/api/key-env/models) plus the default-model routing, so
    `--endpoint` is honored exactly like on the codex backend and the
    user's ~/.dsh/settings.yaml is never touched.
    """

    name = "dsh"

    def build(self, t: dict, scratch: str) -> tuple[list[str], dict]:
        cfg = self.cfg
        self._prompt = self.full_prompt(t)
        patch = os.path.join(scratch, "llmbench.patch.yaml")
        with open(patch, "w") as f:
            f.write(
                "- id: llm-pi-ai\n"
                "  name: '@deepseek-ai/dsh-llm-pi-ai'\n"
                "  config:\n"
                "    providers:\n"
                "      llmbench:\n"
                f"        baseURL: {cfg.endpoint}\n"
                "        api: openai-completions\n"
                f"        apiKeyEnv: {cfg.dsh_api_key_env}\n"
                "        models:\n"
                f"          - id: {cfg.model}\n"
                "- id: agent-default-model\n"
                "  name: '@deepseek-ai/dsh-agent-default-model'\n"
                "  config:\n"
                "    provider: llmbench\n"
                f"    model: {cfg.model}\n")
        cmd = ["dsh", "--profile", "headless", "--patch", patch, self._prompt]
        env = dict(os.environ)
        env[cfg.dsh_api_key_env] = cfg.api_key
        return cmd, env

    def parse_line(self, outcome: TaskOutcome, line: str) -> None:
        outcome.text += line + "\n"
        outcome.finish_reason = "stop"

    def finalize(self, outcome: TaskOutcome) -> None:
        outcome.text = outcome.text.strip()
        # dsh is fully buffered: its only output is the final model message,
        # so first output byte IS first model output
        if outcome.ttft_model_s is None:
            outcome.ttft_model_s = outcome.ttft_s
        super().finalize(outcome)
        if outcome.exit_code not in (0, None) and not outcome.text:
            outcome.error = f"dsh exit {outcome.exit_code}: {outcome.stderr_tail[-300:]}"


class CodexBackend(CliBackend):
    """codex exec --json — JSONL events incl. per-turn token usage."""

    name = "codex"

    def build(self, t: dict, scratch: str) -> tuple[list[str], dict]:
        cfg = self.cfg
        prov = cfg.codex_provider
        self._prompt = self.full_prompt(t)
        cmd = [
            "codex", "exec", "--json", "--skip-git-repo-check", "--ephemeral",
            "-C", scratch, "-m", cfg.model,
            "-c", f'model_provider="{prov}"',
            "-c", f'model_providers.{prov}.base_url="{cfg.endpoint}"',
            "-c", f'model_providers.{prov}.name="{prov}"',
            "--dangerously-bypass-approvals-and-sandbox", self._prompt,
        ]
        env = dict(os.environ)
        env["OPENAI_API_KEY"] = cfg.api_key
        return cmd, env

    def parse_line(self, outcome: TaskOutcome, line: str) -> None:
        line = line.strip()
        if not line.startswith("{"):
            return
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            return
        if ev.get("type") in ("item.started", "item.completed") \
                and outcome.ttft_model_s is None:
            outcome.ttft_model_s = time.monotonic()  # first model-originated action
        if ev.get("type") == "item.completed":
            item = ev.get("item") or {}
            it = item.get("type")
            if it == "agent_message":
                outcome.text = item.get("text", "")
                outcome.finish_reason = "stop"
            elif it == "error":
                outcome.events.append({"event": "error_item",
                                      "message": item.get("message")})
            elif it == "command_execution":
                outcome.events.append({
                    "event": "command",
                    "command": (item.get("command") or "")[:300],
                    "exit_code": item.get("exit_code")})
            else:
                outcome.events.append({"event": it})
        elif ev.get("type") == "turn.completed":
            usage = ev.get("usage") or {}
            outcome.prompt_tokens = usage.get("input_tokens")
            outcome.completion_tokens = usage.get("output_tokens")
            outcome.token_source = "server"
            outcome.events.append({"event": "turn.completed",
                                   "usage": usage})

    def finalize(self, outcome: TaskOutcome) -> None:
        outcome.text = outcome.text.strip()
        super().finalize(outcome)
        if not outcome.text and not outcome.error:
            if outcome.exit_code not in (0, None):
                outcome.error = (f"codex exit {outcome.exit_code}: "
                                 f"{outcome.stderr_tail[-300:]}")
            else:
                outcome.error = "no agent_message received"


@contextmanager
def _claude_trust_lock():
    """Serialize read-modify-write cycles on ~/.claude.json. Two claude tasks
    running concurrently (e.g. --concurrency > 1) must not interleave their
    trust-entry edits and lose one another's (or the user's) state."""
    lf = open(os.path.expanduser("~/.claude.json.lock"), "a")
    try:
        fcntl.flock(lf, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lf, fcntl.LOCK_UN)
        lf.close()


def _claude_trust_write(data: dict, trust: str) -> None:
    """Atomic replace with a unique temp name (a fixed name would let a
    crashed writer's leftover be misattributed by a later replace)."""
    tmp = f"{trust}.llmbench.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, trust)


def _claude_prepare(scratch: str) -> None:
    """Make a scratch dir usable for non-interactive agentic claude runs:
    1. project-level allowlist (<scratch>/.claude/settings.json)
    2. pre-accept the workspace-trust dialog in ~/.claude.json (documented
       workaround — claude refuses the allowlist in untrusted workspaces, and
       --dangerously-skip-permissions cannot be used as root)
    """
    os.makedirs(os.path.join(scratch, ".claude"), exist_ok=True)
    with open(os.path.join(scratch, ".claude", "settings.json"), "w") as f:
        json.dump({"permissions": {"allow": ["Bash", "Read", "Write", "Edit"]}},
                  f, indent=1)
    trust = os.path.expanduser("~/.claude.json")
    with _claude_trust_lock():
        data = {}
        if os.path.isfile(trust):
            try:
                with open(trust) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = {}
        entry = data.setdefault("projects", {}).setdefault(scratch, {})
        entry["hasTrustDialogAccepted"] = True
        _claude_trust_write(data, trust)


class ClaudeBackend(CliBackend):
    """claude -p --output-format stream-json (Anthropic-protocol endpoint;
    vLLM >= 0.23 serves /v1/messages natively, no bridge needed)."""

    name = "claude"

    def build(self, t: dict, scratch: str) -> tuple[list[str], dict]:
        cfg = self.cfg
        self._prompt = t["prompt"]
        _claude_prepare(scratch)
        root = cfg.endpoint[:-3] if cfg.endpoint.endswith("/v1") else cfg.endpoint
        cmd = ["claude", "-p", "--output-format", "stream-json", "--verbose",
              "--model", cfg.model,
              "--no-session-persistence", t["prompt"]]
        if t.get("system"):
            cmd += ["--append-system-prompt", t["system"]]
        env = dict(os.environ)
        env["ANTHROPIC_BASE_URL"] = root
        env["ANTHROPIC_AUTH_TOKEN"] = cfg.api_key
        env["ANTHROPIC_API_KEY"] = cfg.api_key
        return cmd, env

    def parse_line(self, outcome: TaskOutcome, line: str) -> None:
        line = line.strip()
        if not line.startswith("{"):
            if line:
                outcome.text += line + "\n"
            return
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            return
        if ev.get("type") == "assistant":
            if outcome.ttft_model_s is None:
                outcome.ttft_model_s = time.monotonic()  # first assistant turn
            for blk in (ev.get("message") or {}).get("content") or []:
                if blk.get("type") == "text" and blk.get("text"):
                    outcome.text += blk["text"]
                    outcome.finish_reason = "stop"
        elif ev.get("type") == "result":
            if ev.get("result"):
                outcome.text = ev["result"]
            u = ev.get("usage") or {}
            outcome.prompt_tokens = u.get("input_tokens")
            outcome.completion_tokens = u.get("output_tokens")
            if outcome.prompt_tokens is not None and \
                    outcome.completion_tokens is not None:
                outcome.total_tokens = (outcome.prompt_tokens
                                        + outcome.completion_tokens)
            outcome.token_source = "server"
            if ev.get("subtype"):
                outcome.finish_reason = ev["subtype"]
            outcome.events.append({"event": "result", "usage": u})

    def finalize(self, outcome: TaskOutcome) -> None:
        super().finalize(outcome)
        if not outcome.text and not outcome.error:
            outcome.error = (f"claude exit {outcome.exit_code}: "
                             f"{outcome.stderr_tail[-300:]}") or "no result event"

    def after_run(self, scratch: str) -> None:
        """Drop this run's trust entry from ~/.claude.json (it was only needed
        while the task ran; leaving them accumulates dead paths)."""
        trust = os.path.expanduser("~/.claude.json")
        if not os.path.isfile(trust):
            return
        with _claude_trust_lock():
            try:
                with open(trust) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                return
            if (data.get("projects") or {}).pop(scratch, None) is None:
                return
            try:
                _claude_trust_write(data, trust)
            except OSError:
                pass


class PiBackend(CliBackend):
    """pi --mode json — one-shot JSONL session events (assistant messages,
    usage, tool results). The per-run agent dir under scratch
    (PI_CODING_AGENT_DIR) carries a self-contained models.json defining the
    "llmbench" provider, so --endpoint/--model/--api-key are honored exactly
    like the dsh overlay and the user's ~/.pi/agent is never touched.

    No trust or permission plumbing is needed: a bare scratch dir triggers no
    trust prompt, and pi runs tools without per-call approval in one-shot mode.
    """

    name = "pi"

    def build(self, t: dict, scratch: str) -> tuple[list[str], dict]:
        cfg = self.cfg
        self._prompt = self.full_prompt(t)  # only used for token estimation
        agent_dir = os.path.join(scratch, ".pi-agent")
        os.makedirs(agent_dir, exist_ok=True)
        with open(os.path.join(agent_dir, "models.json"), "w") as f:
            json.dump({"providers": {"llmbench": {
                "baseUrl": cfg.endpoint,
                "api": "openai-completions",
                "apiKey": cfg.api_key,
                "models": [{"id": cfg.model,
                            "contextWindow": 262144, "maxTokens": 8192}],
            }}}, f, indent=1)
        cmd = ["pi", "--mode", "json", "--no-session",
               "--provider", "llmbench", "--model", cfg.model,
               "--api-key", cfg.api_key]
        if t.get("system"):
            cmd += ["--append-system-prompt", t["system"]]
        cmd += ["--", t["prompt"]]
        env = dict(os.environ)
        env["PI_CODING_AGENT_DIR"] = agent_dir
        return cmd, env

    def parse_line(self, outcome: TaskOutcome, line: str) -> None:
        line = line.strip()
        if not line.startswith("{"):
            return
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            return
        et = ev.get("type")
        if et == "message_start":
            # fresh assistant response: reset the per-response usage tracker
            outcome._pi_pending_usage = None
            if (ev.get("message") or {}).get("role") == "assistant" \
                    and outcome.ttft_model_s is None:
                outcome.ttft_model_s = time.monotonic()
        elif et == "message_update":
            u = ev.get("usage") or {}
            if u.get("input") or u.get("output") or u.get("totalTokens"):
                outcome._pi_pending_usage = u
        elif et == "message_end":
            m = ev.get("message") or {}
            if m.get("role") != "assistant":
                return
            # per-response cumulative usage: prefer the authoritative
            # message_end.message.usage, else the last message_update.usage
            u = m.get("usage") or outcome._pi_pending_usage
            if u:
                outcome.prompt_tokens = (outcome.prompt_tokens or 0) \
                    + u.get("input", 0)
                outcome.completion_tokens = (outcome.completion_tokens or 0) \
                    + u.get("output", 0)
                outcome.token_source = "server"
            text = "".join(b.get("text", "") for b in m.get("content") or []
                           if isinstance(b, dict) and b.get("type") == "text")
            if text:
                outcome.text = text
                outcome.finish_reason = m.get("stopReason") or "stop"
            outcome.events.append({"event": "assistant_message_end",
                                   "stopReason": m.get("stopReason")})
        elif et == "turn_end":
            for tr in (ev.get("toolResults") or [])[:10]:
                outcome.events.append({
                    "event": "tool_result",
                    "toolName": tr.get("toolName"),
                    "isError": tr.get("isError")})
        elif et in ("agent_end", "agent_settled"):
            # agent_settled is the true terminal (auto-retry/compaction can
            # follow agent_end) — the process exit still closes the run
            outcome.events.append({"event": et, "aborted": ev.get("aborted")})

    def finalize(self, outcome: TaskOutcome) -> None:
        outcome.text = outcome.text.strip()
        super().finalize(outcome)
        if not outcome.text and not outcome.error:
            if outcome.exit_code not in (0, None):
                outcome.error = (f"pi exit {outcome.exit_code}: "
                                  f"{outcome.stderr_tail[-300:]}")
            else:
                outcome.error = "no assistant message received"


def build_backend(cfg: "RunConfig"):
    if cfg.backend == "api":
        return ApiBackend(cfg.endpoint, cfg.model, api_key=cfg.api_key,
                          max_tokens=cfg.max_tokens, temperature=cfg.temperature,
                          timeout=cfg.timeout)
    if cfg.backend == "dsh":
        return DshBackend(cfg)
    if cfg.backend == "codex":
        return CodexBackend(cfg)
    if cfg.backend == "claude":
        return ClaudeBackend(cfg)
    if cfg.backend == "pi":
        return PiBackend(cfg)
    raise SystemExit(f"unknown backend '{cfg.backend}'")


# ---------------------------------------------------------------------------
# benchmark suites — all tasks are original and deterministic (no downloads)
# ---------------------------------------------------------------------------

def task(id: str, prompt: str, verify: dict, **meta) -> dict:
    return {"id": id, "suite": id.split("/")[0], "prompt": prompt,
            "verify": verify, **meta}


CONTRACT = (
    "\n\nShow your work briefly if needed. End your final reply with exactly one "
    "line in the form: ANSWER: <value>"
)

CODE_CONTRACT = (
    "\n\nWrite your complete solution to a file named solution.py in the current "
    "working directory (if you cannot write files, provide the complete solution "
    "as ONE ```python fenced code block instead). The file must contain the "
    "function(s) (imports allowed)."
)

SMOKE_TASKS: list[dict] = [
    task("smoke/pong", "Reply with the single word: pong",
         {"type": "exact", "value": "pong"}),
    task("smoke/math-1", "What is 17 * 24?" + CONTRACT,
         {"type": "math", "value": 408}),
    task("smoke/math-2",
         "A train leaves at 09:15 and travels for 2 hours and 40 minutes. "
         "At what time (24h, HH:MM) does it arrive?" + CONTRACT,
         {"type": "answer", "value": "11:55"}),
]

# --- reasoning ---------------------------------------------------------------

REASONING_TASKS: list[dict] = [
    task("reasoning/heads-legs",
         "A farm has only chickens and rabbits. Together they have 35 heads and "
         "94 legs. How many rabbits are there?" + CONTRACT,
         {"type": "math", "value": 12}),
    task("reasoning/markdown-up",
         "A $40 jacket is first marked up by 25%, and then the sale price is "
         "discounted by 20%. What is the final price in dollars?" + CONTRACT,
         {"type": "math", "value": 40}),
    task("reasoning/knights-knaves",
         "On an island every inhabitant is either a knight (always tells the "
         "truth) or a knave (always lies). A says: 'B is a knave.' B says: "
         "'A and I are of the same type.' Who is the knight?" + CONTRACT,
         {"type": "answer", "value": "A", "compact": True}),
    task("reasoning/syllogism",
         "All bloops are razzles. All razzles are lazzies. Must all bloops "
         "definitely be lazzies? Answer yes or no." + CONTRACT,
         {"type": "answer", "value": "yes", "compact": True}),
    task("reasoning/sequence",
         "What number comes next in the sequence: 2, 6, 12, 20, 30, ... ?"
         + CONTRACT,
         {"type": "math", "value": 42}),
    task("reasoning/wet-ground",
         "If it rains, then the ground gets wet. You observe that the ground "
         "is wet. Does it follow necessarily that it rained? Answer yes or no."
         + CONTRACT,
         {"type": "answer", "value": "no", "compact": True}),
    task("reasoning/pipes",
         "Pipe A alone fills a tank in 6 hours; pipe B alone fills it in 3 hours. "
         "With both pipes open, how many hours does it take to fill the empty "
         "tank?" + CONTRACT,
         {"type": "math", "value": 2}),
    task("reasoning/pages",
         "A book has 300 pages. On day 1 you read a quarter of the book. On "
         "day 2 you read a third of what remained. How many pages are left?"
         + CONTRACT,
         {"type": "math", "value": 150}),
]

# --- spatial (seeded generators, answers computed in code) --------------------

def _scene(seed: int, n: int = 6):
    rnd = random.Random(seed)
    shapes, colors, sizes = ["cube", "sphere", "cylinder"], ["red", "blue", "gray", "green"], ["small", "large"]
    cells = [(c, r) for c in (1, 2, 3) for r in (1, 2, 3)]
    objs = [{"col": c, "row": r, "shape": rnd.choice(shapes),
             "color": rnd.choice(colors), "size": rnd.choice(sizes)}
            for (c, r) in rnd.sample(cells, n)]
    desc = (f"On a 3x3 table there are {n} objects. Columns are numbered 1-3 "
            f"left to right, rows 1-3 front to back.\n"
            + "".join(f"Object {i}: a {o['size']} {o['color']} {o['shape']} "
                       f"at column {o['col']}, row {o['row']}.\n"
                       for i, o in enumerate(objs, 1)))
    return objs, desc.strip()


def _gen_scene_count() -> tuple[str, str, int]:
    from collections import Counter
    for seed in range(101, 400):
        objs, desc = _scene(seed)
        cnt = Counter((o["color"], o["shape"]) for o in objs)
        for (color, shape), n in cnt.items():
            if n >= 2:
                return desc, f"How many {color} {shape}s are on the table?", n
    raise RuntimeError("scene gen failed")


def _gen_scene_compare() -> tuple[str, str, str]:
    for seed in range(201, 500):
        objs, desc = _scene(seed)
        pairs = [(a, b) for i, a in enumerate(objs) for b in objs[i + 1:]
                 if a["size"] != b["size"]]
        if pairs:
            a, b = pairs[0]
            return desc, (f"Object A is at column {a['col']}, row {a['row']}. "
                           f"Object B is at column {b['col']}, row {b['row']}. "
                           f"Which one is larger, A or B?"), ("A" if a["size"] == "large" else "B")
    raise RuntimeError("scene gen failed")


def _gen_scene_right() -> tuple[str, str, int]:
    from collections import Counter
    for seed in range(301, 600):
        objs, desc = _scene(seed)
        cnt = Counter((o["color"], o["shape"]) for o in objs)
        for (color, shape), n in cnt.items():
            if n == 1:
                target = next(o for o in objs if o["color"] == color and o["shape"] == shape)
                n_right = sum(1 for o in objs if o is not target
                              and o["row"] == target["row"] and o["col"] > target["col"])
                return desc, (f"How many objects are in the same row as the "
                               f"{color} {shape} and strictly to its right?"), n_right
    raise RuntimeError("scene gen failed")


def _bfs(grid: list[str]):
    from collections import deque
    start = goal = None
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == "S":
                start = (r, c)
            if ch == "G":
                goal = (r, c)
    came = {start: None}
    q = deque([start])
    moves = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for mv, (dr, dc) in moves.items():
            nxt = (cur[0] + dr, cur[1] + dc)
            r, c = nxt
            if (0 <= r < len(grid) and 0 <= c < len(grid[0])
                    and grid[r][c] not in "#S" and nxt not in came):
                came[nxt] = (cur, mv)
                q.append(nxt)
    if goal not in came:
        raise RuntimeError(f"grid unsolvable: {grid}")
    path: list[str] = []
    node = goal
    while came[node] is not None:
        prev, mv = came[node]
        path.append(mv)
        node = prev
    path.reverse()
    return len(path), "".join(path)


GRID_B = ["S....", ".###.", ".#.#.", ".###.", "....G"]
_GRID_LEN, _GRID_PATH = _bfs(GRID_B)
_GRID_TEXT = "\n".join(GRID_B) + (
    "\n\nS is the start, G is the goal, # are walls, . are free cells. "
    "You may move one cell up (U), down (D), left (L) or right (R) per move."
)


def _mapwalk_trace():
    x = y = 0
    facing = "north"
    rot = {"north": "east", "east": "south", "south": "west", "west": "north"}
    def fwd(n):
        nonlocal x, y
        x += {"east": n, "west": -n}.get(facing, 0)
        y += {"north": n, "south": -n}.get(facing, 0)
    fwd(2)
    facing = rot[facing]
    fwd(3)
    facing = {"north": "south", "south": "north", "east": "west", "west": "east"}[facing]
    fwd(1)
    return f"{x},{y},{facing}"


MAPWALK_ANSWER = _mapwalk_trace()

_sc_count, _sc_compare, _sc_right = (_gen_scene_count(), _gen_scene_compare(),
                                      _gen_scene_right())

SPATIAL_TASKS: list[dict] = [
    task("spatial/scene-count", _sc_count[0] + "\n\n" + _sc_count[1] + CONTRACT,
         {"type": "math", "value": _sc_count[2]}),
    task("spatial/scene-compare", _sc_compare[0] + "\n\n" + _sc_compare[1] + CONTRACT,
         {"type": "answer", "value": _sc_compare[2], "compact": True}),
    task("spatial/scene-right", _sc_right[0] + "\n\n" + _sc_right[1] + CONTRACT,
         {"type": "math", "value": _sc_right[2]}),
    task("spatial/grid-length",
         _GRID_TEXT + f"\n\nWhat is the minimum number of moves from S to G?"
         + CONTRACT,
         {"type": "math", "value": _GRID_LEN}),
    task("spatial/grid-path",
         _GRID_TEXT + (
             "\n\nGive ONE shortest path from S to G as a string of moves "
             "using U, D, L, R (e.g. DRRU) with no spaces." + CONTRACT),
         {"type": "gridpath", "grid": GRID_B, "expect_len": _GRID_LEN}),
    task("spatial/mapwalk",
         "You start at position (0, 0) facing north. Perform these moves in "
         "order: move forward 2; turn right; move forward 3; turn around; move "
         "forward 1. Coordinates: east increases x, north increases y. Give "
         "your final position and facing as x,y,compass (e.g. 2,3,north)."
         + CONTRACT,
         {"type": "answer", "value": MAPWALK_ANSWER, "compact": True}),
    task("spatial/arrow-rotate",
         "Rotate each of these directions by 90 degrees clockwise: north, "
         "west, south. Give the three resulting directions separated by commas "
         "(e.g. east,north,west)." + CONTRACT,
         {"type": "answer", "value": "east,north,west", "compact": True}),
    task("spatial/cube-stack",
         "A sculpture is built from unit cubes. The bottom layer is a 3x3 "
         "square grid of cubes with the center cube missing. On top of it, a "
         "2x2 square layer of cubes sits centered on the bottom layer. How many "
         "unit cubes are used in total?" + CONTRACT,
         {"type": "math", "value": 12}),
]

# --- instruction following ----------------------------------------------------

IFEVAL_TASKS: list[dict] = [
    task("ifeval/sentences-keyword",
         "Write a short description of a lighthouse. Use exactly 2 sentences, "
         "and the word 'battery' must appear in your text.",
         {"type": "constraints", "checks": [
             {"type": "sentences", "count": 2},
             {"type": "include_all", "words": ["battery"]}]}),
    task("ifeval/five-words",
         "Name one color of the rainbow. Respond with exactly 5 words, no "
         "more, no less.",
         {"type": "constraints", "checks": [
             {"type": "words", "count": 5}]}),
    task("ifeval/json-only",
         "Respond with ONLY a JSON object (no code fences, no extra text) "
         "describing a fictional pet, with keys: \"name\" (string), \"age\" "
         "(integer), \"hobbies\" (list of exactly 3 strings).",
         {"type": "constraints", "checks": [
             {"type": "json", "shape": {
                 "name": "str", "age": "int", "hobbies": {"type": "list", "len": 3}}}]}),
    task("ifeval/include-exclude",
         "Write one sentence about a project launch. You must include the "
         "words 'alpha', 'beta' and 'gamma' in that sentence. Do NOT use the "
         "word 'delta' anywhere.",
         {"type": "constraints", "checks": [
             {"type": "include_all", "words": ["alpha", "beta", "gamma"]},
             {"type": "exclude_any", "words": ["delta"]}]}),
    task("ifeval/no-commas",
         "Write a definition of a bridge in at most 3 sentences. Do not use "
         "any commas in your entire reply.",
         {"type": "constraints", "checks": [
             {"type": "sentences", "max": 3},
             {"type": "no_chars", "chars": ","}]}),
    task("ifeval/bullet-lines",
         "List 4 tips for sleeping well. Each tip must be on its own line and "
         "every line must start with '- ' (a dash and a space).",
         {"type": "constraints", "checks": [
             {"type": "lines", "count": 4, "prefix": "- "}]}),
    task("ifeval/all-caps",
         "Shout the word 'fire' twice. Write only in uppercase letters (no "
         "lowercase letters at all), and use at most 4 words.",
         {"type": "constraints", "checks": [
             {"type": "case", "mode": "upper"},
             {"type": "words", "max": 4}]}),
    task("ifeval/first-line-number",
         "Write a summary of any movie you know. The very first line must be "
         "exactly 'SUMMARY:' and your text must contain the number 42.",
         {"type": "constraints", "checks": [
             {"type": "first_line", "line": "SUMMARY:"},
             {"type": "include_all", "words": ["42"]}]}),
]

# --- coding (unit-test verified) ----------------------------------------------

def _ct(name: str, prompt: str, tests: str) -> dict:
    return task(f"coding/{name}", prompt + CODE_CONTRACT,
                {"type": "code", "tests": tests})


CODING_TASKS: list[dict] = [
    _ct("anagram",
         "Write a Python function is_anagram(a: str, b: str) -> bool that "
         "returns True when `a` and `b` contain the same letters (anagrams). "
         "Ignore case and ignore all non-letter characters. Examples: "
         "is_anagram('listen', 'silent') is True; is_anagram('hello', 'world') "
         "is False.",
         """import unittest
from solution import is_anagram

class Test(unittest.TestCase):
    def test_basic(self):
        self.assertTrue(is_anagram("listen", "silent"))
    def test_case_and_spaces(self):
        self.assertTrue(is_anagram("Dormitory", "Dirty Room"))
    def test_false(self):
        self.assertFalse(is_anagram("hello", "world"))
    def test_empty(self):
        self.assertTrue(is_anagram("", ""))
    def test_digits_ignored(self):
        self.assertTrue(is_anagram("a1b2", "ba"))
"""),
    _ct("roman",
         "Write a Python function parse_roman(s: str) -> int that converts a "
         "Roman numeral string (values 1..3999, uppercase) to an integer. "
         "Example: parse_roman('MCMXCIV') is 1994.",
         """import unittest
from solution import parse_roman

class Test(unittest.TestCase):
    def test_one(self):
        self.assertEqual(parse_roman("I"), 1)
    def test_9(self):
        self.assertEqual(parse_roman("IX"), 9)
    def test_40(self):
        self.assertEqual(parse_roman("XL"), 40)
    def test_59(self):
        self.assertEqual(parse_roman("LIX"), 59)
    def test_1994(self):
        self.assertEqual(parse_roman("MCMXCIV"), 1994)
    def test_3999(self):
        self.assertEqual(parse_roman("MMMCMXCIX"), 3999)
    def test_444(self):
        self.assertEqual(parse_roman("CDXLIV"), 444)
"""),
    _ct("merge-intervals",
         "Write a Python function merge_intervals(intervals: list) -> list that "
         "takes a list of [start, end] intervals (ints) and returns the "
         "merged, non-overlapping intervals sorted by start. Intervals that "
         "touch (e.g. [1,4] and [4,5]) merge. Example: "
         "merge_intervals([[1,3],[2,6],[8,10]]) is [[1,6],[8,10]].",
         """import unittest
from solution import merge_intervals

class Test(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(merge_intervals([[1, 3], [2, 6], [8, 10]]), [[1, 6], [8, 10]])
    def test_touching(self):
        self.assertEqual(merge_intervals([[1, 4], [4, 5]]), [[1, 5]])
    def test_single(self):
        self.assertEqual(merge_intervals([[5, 9]]), [[5, 9]])
    def test_all_merge(self):
        self.assertEqual(merge_intervals([[1, 5], [2, 3], [4, 9]]), [[1, 9]])
    def test_none_overlapping(self):
        self.assertEqual(merge_intervals([[1, 2], [4, 5]]), [[1, 2], [4, 5]])
"""),
    _ct("moving-average",
         "Write a Python function moving_average(nums: list, k: int) -> list "
         "that returns the moving average of a sliding window of size k over "
         "the input list of numbers. If len(nums) < k return []. Values may be "
         "floats. Example: moving_average([1,2,3,4], 2) is [1.5, 2.5, 3.5].",
         """import unittest
from solution import moving_average

class Test(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(moving_average([1, 2, 3, 4], 2), [1.5, 2.5, 3.5])
    def test_window3(self):
        res = moving_average([1, 3, 5, 7, 9], 3)
        self.assertEqual(len(res), 3)
        for got, want in zip(res, [3.0, 5.0, 7.0]):
            self.assertAlmostEqual(got, want)
    def test_k1(self):
        self.assertEqual(moving_average([5, 6], 1), [5.0, 6.0])
    def test_too_short(self):
        self.assertEqual(moving_average([1, 2], 3), [])
    def test_k_equals_len(self):
        self.assertEqual(moving_average([1, 2, 3], 3), [2.0])
"""),
    _ct("parentheses",
         "Write a Python function valid_parentheses(s: str) -> bool that "
         "returns True if the string of round, square and curly brackets is "
         "properly balanced and nested. Example: valid_parentheses('([{}])') "
         "is True; valid_parentheses('([)]') is False. The empty string is "
         "valid.",
         """import unittest
from solution import valid_parentheses

class Test(unittest.TestCase):
    def test_ok(self):
        self.assertTrue(valid_parentheses("()"))
    def test_mixed(self):
        self.assertTrue(valid_parentheses("({[]})"))
    def test_wrong_order(self):
        self.assertFalse(valid_parentheses("([)]"))
    def test_unbalanced(self):
        self.assertFalse(valid_parentheses("((("))
    def test_empty(self):
        self.assertTrue(valid_parentheses(""))
    def test_close_first(self):
        self.assertFalse(valid_parentheses(")"))
"""),
    _ct("flatten",
         "Write a Python function flatten(x: list) -> list that recursively "
         "flattens arbitrarily nested lists (tuples are not present in the "
         "input). Example: flatten([1, [2, [3, [4]]], 5]) is [1, 2, 3, 4, 5]. "
         "Empty nested lists contribute nothing.",
         """import unittest
from solution import flatten

class Test(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(flatten([1, [2, [3, [4]]], 5]), [1, 2, 3, 4, 5])
    def test_flat(self):
        self.assertEqual(flatten([1, 2, 3]), [1, 2, 3])
    def test_empty_inner(self):
        self.assertEqual(flatten([[], [1], []]), [1])
    def test_all_empty(self):
        self.assertEqual(flatten([]), [])
    def test_strings(self):
        self.assertEqual(flatten(["a", ["b", ["c"]]]), ["a", "b", "c"])
"""),
    _ct("caesar",
         "Write a Python function caesar_cipher(text: str, shift: int) -> str "
         "that shifts every lowercase letter a-z forward by `shift` positions "
         "(wrapping around), leaving all other characters unchanged. The shift "
         "may be larger than 26. Non-lowercase letters are never modified. "
         "Example: caesar_cipher('abc', 1) is 'bcd' and caesar_cipher('xyz', 3) "
         "is 'abc'.",
         """import unittest
from solution import caesar_cipher

class Test(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(caesar_cipher("abc", 1), "bcd")
    def test_wrap(self):
        self.assertEqual(caesar_cipher("xyz", 3), "abc")
    def test_large_shift(self):
        self.assertEqual(caesar_cipher("a", 27), "b")
    def test_non_lower_untouched(self):
        self.assertEqual(caesar_cipher("Abc 123!", 1), "Acd 123!")
    def test_zero(self):
        self.assertEqual(caesar_cipher("hello", 0), "hello")
    def test_punctuation_kept(self):
        self.assertEqual(caesar_cipher("zoo, zoo!", 1), "app, app!")
"""),
    _ct("top-k",
         "Write a Python function top_k_frequent(nums: list, k: int) -> list "
         "that returns the k most frequent numbers in `nums`, in decreasing "
         "order of frequency; ties are broken by first appearance in `nums`. "
         "Example: top_k_frequent([1,1,2,2,3], 2) is [1, 2].",
         """import unittest
from solution import top_k_frequent

class Test(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(top_k_frequent([1, 1, 2, 2, 3], 2), [1, 2])
    def test_tie_first_seen(self):
        self.assertEqual(top_k_frequent([4, 2, 2, 1, 4], 2), [4, 2])
    def test_all_distinct(self):
        self.assertEqual(top_k_frequent([7, 8, 9], 2), [7, 8])
    def test_single(self):
        self.assertEqual(top_k_frequent([5], 1), [5])
    def test_k_3(self):
        self.assertEqual(top_k_frequent([3, 3, 2, 2, 1], 3), [3, 2, 1])
"""),
]

# --- tool calling (api backend) -----------------------------------------------

def tool_def(name: str, desc: str, props: dict, required: list) -> dict:
    return {"type": "function",
            "function": {"name": name, "description": desc,
                         "parameters": {"type": "object", "properties": props,
                                        "required": required}}}


T_WEATHER = tool_def("get_weather", "Get the current weather for a city",
                     {"city": {"type": "string", "description": "city name"}},
                     ["city"])
T_CALC = tool_def("calculate", "Evaluate an arithmetic expression",
                  {"expression": {"type": "string",
                                   "description": "arithmetic expression, e.g. '128*46'"}},
                  ["expression"])
T_SEARCH = tool_def("search_web", "Search the web for information",
                    {"query": {"type": "string", "description": "search query"}},
                    ["query"])
T_EMAIL = tool_def("send_email", "Send an email message",
                   {"to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"}},
                   ["to", "subject", "body"])
T_BOOK = tool_def("create_reservation", "Create a restaurant reservation",
                  {"name": {"type": "string"},
                   "date": {"type": "string", "description": "date, YYYY-MM-DD"},
                   "guests": {"type": "integer"}},
                  ["name", "date", "guests"])
T_STOCK = tool_def("get_stock_price", "Get the current price of a stock",
                   {"symbol": {"type": "string", "description": "ticker symbol"}},
                   ["symbol"])

TOOL_SYSTEM = ("You have tools available. You MUST use the appropriate tool(s) "
               "to fulfill the user's request — call the tool(s) with the "
               "correct arguments instead of answering from memory or with "
               "plain text.")

TOOLS_TASKS: list[dict] = [
    task("tools/weather", "What is the weather like in Tokyo?",
         {"type": "toolcall", "calls": [
             {"name": "get_weather", "args_subset": {"city": "tokyo"}}]},
         system=TOOL_SYSTEM, tools=[T_WEATHER]),
    task("tools/calc-eval", "What is 128 * 46?",
         {"type": "toolcall", "calls": [
             {"name": "calculate",
              "args_eval": {"key": "expression", "expect": 5888}}]},
         system=TOOL_SYSTEM, tools=[T_CALC]),
    task("tools/search", "Find out who the first person to walk on the moon was.",
         {"type": "toolcall", "calls": [{"name": "search_web"}]},
         system=TOOL_SYSTEM, tools=[T_SEARCH]),
    task("tools/email", "Email alice@example.com with subject 'Meeting moved' "
                        "and body 'The meeting is now at 3pm.'.",
         {"type": "toolcall", "calls": [
             {"name": "send_email",
              "args_subset": {"to": "alice@example.com",
                              "subject": "Meeting moved"}}]},
         system=TOOL_SYSTEM, tools=[T_EMAIL]),
    task("tools/booking", "Book a table for 4 under the name 'Bob Stone' on "
                         "2026-08-14.",
         {"type": "toolcall", "calls": [
             {"name": "create_reservation",
              "args_subset": {"name": "bob stone", "date": "2026-08-14",
                              "guests": 4}}]},
         system=TOOL_SYSTEM, tools=[T_BOOK]),
    task("tools/multi", "What's the weather in Paris, and what is 15 percent of 240?",
         {"type": "toolcall", "min_calls": 2, "calls": [
             {"name": "get_weather", "args_subset": {"city": "paris"}},
             {"name": "calculate", "args_eval": {"key": "expression", "expect": 36}}]},
         system=TOOL_SYSTEM, tools=[T_WEATHER, T_CALC]),
    task("tools/no-call-needed",
         "What is 7 * 8? (You know this one — no lookup needed.)" + CONTRACT,
         {"type": "toolcall", "no_calls": True, "answer": 56, "numeric": True},
         system=("You have tools available, but you may also answer directly "
                 "when no tool is needed."),
         tools=[T_WEATHER]),
    task("tools/two-stocks", "Get the current prices of AAPL and MSFT.",
         {"type": "toolcall", "min_calls": 2, "calls": [
             {"name": "get_stock_price", "args_subset": {"symbol": "aapl"}},
             {"name": "get_stock_price", "args_subset": {"symbol": "msft"}}]},
         system=TOOL_SYSTEM, tools=[T_STOCK]),
]

# --- agentic (CLI backends: real shell + file access) -----------------------

CLI_BACKENDS = ["dsh", "codex", "claude", "pi"]

BUGGY_PY = '''def double_total(nums):
    total = 0
    for n in nums:
        total += n
    return total
'''

BUGGY_TESTS = '''import unittest
from buggy import double_total


class Test(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(double_total([1, 2, 3]), 12)

    def test_empty(self):
        self.assertEqual(double_total([]), 0)

    def test_negative(self):
        self.assertEqual(double_total([-1, 2, -3]), -4)


if __name__ == "__main__":
    import json as _json, sys as _sys
    _runner = unittest.TextTestRunner(verbosity=0)
    _res = _runner.run(unittest.defaultTestLoader.loadTestsFromModule(_sys.modules[__name__]))
    _fails = [str(t).split("\\n")[0]
              for t, _tb in list(_res.failures) + list(_res.errors)]
    print(_json.dumps({"total": _res.testsRun,
                       "passed": _res.testsRun - len(_fails),
                       "failed": _fails}))
'''

AGENTIC_TASKS: list[dict] = [
    task("agentic/file-create",
         "Create a file called pong.txt in the current working directory "
         "containing exactly pong (no trailing newline). Then reply with the "
         "file's content.",
         {"type": "file", "path": "pong.txt", "content": "pong"},
         backends=CLI_BACKENDS),
    task("agentic/data-sum",
         "Read data.txt in the current working directory and sum all the "
         "numbers in it. End your final reply with exactly one line in the "
         "form: ANSWER: <value>",
         {"type": "math", "value": 130},
         setup={"data.txt": "3\n14\n15\n92\n6\n"},
         backends=CLI_BACKENDS),
    task("agentic/calc-shell",
         "Use a shell command (for example: python3 -c) in the current working "
         "directory to compute 128*46. Show the command you ran, then end your "
         "final reply with exactly one line in the form: ANSWER: <value>",
         {"type": "math", "value": 5888},
         backends=CLI_BACKENDS),
    task("agentic/poem-words",
         "Write and run a small script in the current working directory that "
         "counts the number of whitespace-separated words in poem.txt. End "
         "your final reply with exactly one line in the form: ANSWER: <value>",
         {"type": "math", "value": 10},
         setup={"poem.txt": "Twinkle twinkle little star\nHow I wonder what you are\n"},
         backends=CLI_BACKENDS),
    task("agentic/fix-bug",
         "tests.py in the current working directory is failing. Read buggy.py "
         "and tests.py, then fix buggy.py so that `python3 tests.py` passes all "
         "tests. Do not modify tests.py. Reply with a one-line summary of the "
         "fix.",
         {"type": "run", "cmd": ["python3", "tests.py"], "timeout": 120},
         setup={"buggy.py": BUGGY_PY, "tests.py": BUGGY_TESTS},
         backends=CLI_BACKENDS),
]

# --- multilingual (prompts in the target language; verifiers language-neutral)

MULLANG_TASKS: list[dict] = [
    task("mullang/zh-arith",
         "小明有 3 个苹果，又买了 5 袋苹果，每袋 4 个。他现在一共有多少个苹果？"
         + CONTRACT,
         {"type": "math", "value": 23}),
    task("mullang/ja-arith",
         "りんごが 7 個、みかんが 8 個あります。合わせていくつですか？"
         + CONTRACT,
         {"type": "math", "value": 15}),
    task("mullang/ko-arith",
         "사과가 6개 있고, 배가 7개 있습니다. 모두 합치면 몇 개입니까?"
         + CONTRACT,
         {"type": "math", "value": 13}),
    task("mullang/ar-math",
         "عندي 7 تفاحات و8 برتقالات. كم المجموع؟ أجب بالرقم الغربي (0-9) في سطر "
         "الإجابة." + CONTRACT,
         {"type": "math", "value": 15}),
    task("mullang/ru-grid",
         _GRID_TEXT + (
             "\n\nДайте КРАЙНЕ КОРОТКИЙ путь от S до G одной строкой ходов "
             "U, D, L, R (например DRRU), без пробелов." + CONTRACT),
         {"type": "gridpath", "grid": GRID_B, "expect_len": _GRID_LEN}),
    task("mullang/de-constraints",
         "Schreibe eine kurze Antwort auf Deutsch. Bedingung: deine Antwort "
         "besteht aus genau zwei Sätzen, und der zweite Satz enthält das Wort "
         "\"Licht\". Thema: ein Leuchtturm bei Nacht.",
         {"type": "constraints",
          "checks": [{"type": "sentences", "count": 2},
                      {"type": "include_all", "words": ["licht"]}]}),
    task("mullang/es-answer",
         "En una isla hay caballeros (siempre dicen la verdad) y traidores "
         "(siempre mienten). A dice: \"Ninguno de los dos es caballero\". "
         "¿Quién es el caballero? " + CONTRACT,
         {"type": "answer", "value": "B", "compact": True}),
    task("mullang/hi-words",
         "अपने जवाब में ठीक 5 शब्द लिखें। जवाब हिंदी में हो। कोई अन्य वाक्य, "
         "पंक्ति या संकेत न लिखें।",
         {"type": "constraints", "checks": [
             {"type": "words", "min": 5, "max": 5},
             # the answer must actually be in Devanagari, not English
             {"type": "script", "char_class": "[\\u0900-\\u097F]",
              "min_pct": 0.5}]}),
]

# --- longctx (opt-in — excluded from default runs; api backend only) --------
# CLI backends take the prompt as a process argument (Linux MAX_ARG_STRLEN is
# 128 KiB per argument), so 128k-token prompts cannot be delivered that way.

def _filler_log(seed: int, n_chars: int) -> str:
    """Deterministic machine-log filler of ~n_chars characters."""
    rnd = random.Random(seed)
    nouns = ["system", "cache", "queue", "thread", "buffer", "packet", "index",
             "vector", "matrix", "signal", "sensor", "relay", "socket", "module",
             "kernel", "driver", "tensor", "weight", "gradient", "batch", "shard",
             "chunk", "frame", "stream", "session", "gateway", "router", "probe"]
    lines, total, i = [], 0, 0
    while total < n_chars:
        w = " ".join(rnd.choices(nouns, k=rnd.randint(6, 14)))
        s = f"Log {i:07d}: the {w} was inspected and logged at node {i % 97:02d}."
        lines.append(s)
        total += len(s) + 1
        i += 1
    return "\n".join(lines)


def _needle_doc(seed: int, n_chars: int, codes: list[str],
                positions: list[float]) -> str:
    """Fisher-free: splice 'MAGIC CODE: X' lines into a filler doc at
    fractional positions (deterministic for a fixed seed)."""
    lines = _filler_log(seed, n_chars).split("\n")
    n_inserted = 0
    for code, pos in zip(codes, positions):
        idx = min(len(lines) - 1 - n_inserted, int(len(lines) * pos))
        lines.insert(idx + n_inserted,
                     f"MAGIC CODE: {code} — remember this code exactly.")
        n_inserted += 1
    return "\n".join(lines)


def _longctx(id: str, seed: int, n_chars: int, codes: list[str],
             positions: list[float], question: str, expected: str) -> dict:
    doc = _needle_doc(seed, n_chars, codes, positions)
    prompt = (f"Below is a long machine log.\n\n{doc}\n\n{question}\n"
              + CONTRACT)
    return task(id, prompt,
                {"type": "answer", "value": expected, "compact": True})


LONGCTX_TASKS: list[dict] = [
    _longctx("longctx/needle-8k", seed=101, n_chars=32_000,
             codes=["QX4V"], positions=[0.50],
             question="The log contains exactly one line of the form "
                      "'MAGIC CODE: <code>'. What is the magic code?",
             expected="QX4V"),
    _longctx("longctx/multi-32k", seed=202, n_chars=128_000,
             codes=["A1B2", "C3D4", "E5F6"], positions=[0.25, 0.50, 0.75],
             question="The log contains exactly three lines of the form "
                      "'MAGIC CODE: <code>'. List all three codes in the order "
                      "they appear, separated by commas (e.g. 'AB12, CD34, "
                      "EF56').",
             expected="A1B2, C3D4, E5F6"),
    _longctx("longctx/needle-128k", seed=303, n_chars=512_000,
             codes=["M3R9"], positions=[0.50],
             question="The log contains exactly one line of the form "
                      "'MAGIC CODE: <code>'. What is the magic code?",
             expected="M3R9"),
]

SUITES: dict[str, list[dict]] = {
    "smoke": SMOKE_TASKS,
    "reasoning": REASONING_TASKS,
    "spatial": SPATIAL_TASKS,
    "ifeval": IFEVAL_TASKS,
    "coding": CODING_TASKS,
    "tools": TOOLS_TASKS,
    "agentic": AGENTIC_TASKS,
    "mullang": MULLANG_TASKS,
    "longctx": LONGCTX_TASKS,
}
# the tools suite (OpenAI `tools` parameter) only makes sense on the api backend
for _t in TOOLS_TASKS:
    _t.setdefault("backends", ["api"])
# longctx prompts exceed the CLI per-argument length limit (see above)
for _t in LONGCTX_TASKS:
    _t.setdefault("backends", ["api"])
# heavy prompts: not part of a default run
DEFAULT_EXCLUDE = {"longctx"}


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
    retries: int
    concurrency: int
    warmup: bool
    npu_container: Optional[str]
    dsh_api_key_env: str
    codex_provider: str

    @property
    def out_id(self) -> str:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{stamp}_{self.model}_{self.backend}"


def selected_tasks(cfg: RunConfig) -> tuple[list[dict], int]:
    out: list[dict] = []
    skipped = 0
    if cfg.suites:
        active = set(cfg.suites)
    elif cfg.tasks:
        # an explicit task filter IS an explicit selection — the heavy-suite
        # exclusion applies only to default (unfiltered) runs
        active = set(SUITES)
    else:
        active = set(SUITES) - DEFAULT_EXCLUDE
    for suite_name, tasks_ in SUITES.items():
        if suite_name not in active:
            continue
        for t in tasks_:
            if cfg.tasks and t["id"] not in cfg.tasks:
                continue
            if t.get("backends") and cfg.backend not in t["backends"]:
                skipped += 1
                continue
            out.append(t)
    return out, skipped


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
    tasks, skipped = selected_tasks(cfg)
    if skipped:
        log(f"[llmbench] {skipped} task(s) skipped (backend '{cfg.backend}' not supported)")
    if not cfg.suites and DEFAULT_EXCLUDE:
        log(f"[llmbench] default run excludes heavy suite(s) "
            f"{sorted(DEFAULT_EXCLUDE)} — select explicitly with --suites")
    if not tasks:
        log("[llmbench] no tasks matched; nothing to do")
        sys.exit(2)

    run: dict[str, Any] = {
        "schema": SCHEMA,
        "tool": f"llmbench {VERSION}",
        "started_at": now_iso(),
        "config": {
            "backend": cfg.backend, "model": cfg.model, "endpoint": cfg.endpoint,
            "suites": cfg.suites or sorted(set(SUITES) - DEFAULT_EXCLUDE),
            "task_filter": cfg.tasks,
            "repeat": cfg.repeat, "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature, "timeout_s": cfg.timeout,
            "retries": cfg.retries, "concurrency": cfg.concurrency,
            "sample_interval_s": cfg.sample_interval, "warmup": cfg.warmup,
            "codex_provider": cfg.codex_provider,
        },
        "host": host_facts(),
        "metrics_legend": {
            "ttft_s": ("api: request sent -> first content token. "
                        "cli: spawn -> first stdout byte (includes CLI startup)"),
            "ttft_model_s": ("first MODEL-originated output (api: first token; "
                              "codex: first item event; claude: first assistant event; "
                              "pi: first assistant event; dsh: final message, since it "
                              "buffers) — use this for cross-backend "
                              "model-latency comparisons"),
            "ttft_first_reasoning_s": "api only: request sent -> first reasoning token",
            "total_s": "api: request sent -> stream closed. cli: spawn -> process exit",
            "decode_tps": "content chunks / (total - ttft) — content-only rate (api only)",
            "gen_tps": "completion tokens / (total - ttft_model_s); includes tool round-trips on cli backends",
            "overall_tps": "completion tokens / total — incl. TTFT",
            "tokens": ("server usage when reported (token_source=server: codex/claude/"
                       "pi — cumulative over the task's internal API calls, incl. the "
                       "harness system prompt), else ~4 chars/token estimate of the "
                       "task prompt only (token_source=estimate: dsh, or pi when the "
                       "provider reports no usage)"),
            "resources": ("sampled at --sample-interval; chip stats are avg/min/peak/p50/p95; "
                           "proc_cpu_pct/proc_rss_mb cover the CLI process tree on cli backends"),
        },
        "tasks": [],
        "totals": {},
    }

    if cfg.warmup:
        log("[llmbench] warmup (unscored) ...")
        try:
            backend.warmup()
        except Exception as e:  # noqa: BLE001 — warmup is best-effort
            log(f"[llmbench] warmup failed ({type(e).__name__}: {e}); continuing")
        time.sleep(0.5)

    def _one(t: dict, rep: int) -> dict:
        """Run one task, retrying only hard errors (a 'fail' is a real verdict)."""
        r = None
        attempt = 0
        for attempt in range(cfg.retries + 1):
            r = run_one_task(cfg, t, backend, prober, out_dir, rep, log)
            if r["status"] != "error":
                break
            if attempt < cfg.retries:
                log(f"[llmbench] retry {attempt + 1}/{cfg.retries} for {t['id']}: "
                    f"{(r.get('error') or 'error')[:80]}")
        r["attempts"] = attempt + 1
        return r

    for rep in range(cfg.repeat):
        if cfg.concurrency > 1 and len(tasks) > 1:
            # load mode: N tasks in flight; each keeps its own resource sampler
            results: list[Optional[dict]] = [None] * len(tasks)
            with ThreadPoolExecutor(max_workers=cfg.concurrency) as ex:
                futs = {ex.submit(_one, t, rep): i for i, t in enumerate(tasks)}
                for fut in as_completed(futs):
                    i = futs[fut]
                    results[i] = fut.result()
                    log(console_line(results[i]))
        else:
            results = []
            for t in tasks:
                r = _one(t, rep)
                results.append(r)
                log(console_line(r))
        for r in results:
            run["tasks"].append(r)

    run["finished_at"] = now_iso()
    run["totals"] = compute_totals(run["tasks"])
    return run





def run_one_task(cfg: RunConfig, t: dict, backend: Any, prober: NPUProber,
                 out_dir: str, rep: int, log: Callable[[str], None] = print) -> dict:
    rid = t["id"] if rep == 0 else f"{t['id']}#r{rep}"
    task_key = rid.replace("/", "__").replace("#", "_")
    scratch = os.path.join(out_dir, "work", task_key)
    # wipe stale artifacts (e.g. a half-written solution.py from a crashed
    # attempt must not leak into a retry's verdict)
    shutil.rmtree(scratch, ignore_errors=True)
    os.makedirs(scratch, exist_ok=True)
    for fname, content in (t.get("setup") or {}).items():
        with open(os.path.join(scratch, fname), "w") as f:
            f.write(content)

    sampler = ResourceSampler(cfg.sample_interval, prober)
    sampler.start()
    t_start = time.monotonic()
    try:
        outcome = backend.run(t, scratch, sampler.set_root)
    except Exception as e:  # noqa: BLE001 — one bad task must not kill the run
        outcome = TaskOutcome(error=f"{type(e).__name__}: {e}")
    finally:
        sampler.stop()
    wall = time.monotonic() - t_start
    result = outcome  # unified TaskOutcome

    res: dict[str, Any] = {
        "id": rid,
        "task_id": t["id"],
        "suite": t["suite"],
        "repeat": rep,
        "backend": cfg.backend,
        "status": "pass" if not result.error else "error",
        "response": result.text,
        "ttft_first_reasoning_s": result.ttft_first_reasoning_s,
        "ttft_s": result.ttft_s,
        "ttft_model_s": result.ttft_model_s,
        "total_s": round(result.total_s, 3),
        "wall_s": round(wall, 3),
        "finish_reason": result.finish_reason,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "token_source": result.token_source,
        "exit_code": result.exit_code,
        "content_chunks": result.content_chunks,
        "reasoning_chunks": result.reasoning_chunks,
        "tool_calls": result.tool_calls,
        "error": result.error,
        "decode_tps": None,
        "gen_tps": None,
        "overall_tps": None,
    }
    # Throughput definitions (see run.json "metrics_legend"):
    #   ttft_s          first output byte (cli) / first content token (api)
    #   ttft_model_s    first MODEL-originated output (cli startup excluded)
    #   decode_tps      content chunks / (end - ttft)  [api, content-only rate]
    #   gen_tps        completion tokens / (end - first model output)
    #   overall_tps    completion tokens / total latency             [incl. TTFT]
    if result.ttft_s is not None and result.total_s > result.ttft_s \
            and result.content_chunks:
        content_window = result.total_s - result.ttft_s
        if content_window >= 0.25:  # sub-250ms windows are burst noise
            res["decode_tps"] = round(result.content_chunks / content_window, 2)
    first_model = result.ttft_model_s if result.ttft_model_s is not None \
        else result.ttft_s
    # skip when the "first output" arrived at the very end (buffered CLIs like
    # dsh headless): the window is noise, not a rate
    if first_model is not None and result.total_s > first_model \
            and result.completion_tokens and result.total_s - first_model >= 1.0:
        res["gen_tps"] = round(result.completion_tokens
                               / (result.total_s - first_model), 2)
    if result.completion_tokens and result.total_s > 0:
        res["overall_tps"] = round(result.completion_tokens / result.total_s, 2)

    # verdict
    if not result.error and (result.text or result.tool_calls):
        try:
            verifier = VERIFIERS.get(t["verify"]["type"])
            if verifier is None:
                res["verdict"] = {"passed": False, "detail":
                                  f"unknown verifier '{t['verify']['type']}'"}
            else:
                ctx = {"response": result.text, "tool_calls": result.tool_calls,
                       "task_id": rid, "out_dir": out_dir, "scratch": scratch}
                res["verdict"] = verifier(ctx, t["verify"])
                res["status"] = "pass" if res["verdict"]["passed"] else "fail"
        except Exception as e:  # noqa: BLE001 — a verifier bug is not a model
            # verdict: record it, retry, and keep the run alive
            res["status"] = "error"
            res["error"] = (f"verifier {t['verify']['type']} crashed: "
                             f"{type(e).__name__}: {e}")

    res["resources"] = aggregate_resources(
        sampler.samples,
        f"{prober.backend}" + (f" ({prober.container})" if prober.container else ""))

    # per-task artifacts
    task_dir = os.path.join(out_dir, "tasks")
    os.makedirs(task_dir, exist_ok=True)
    try:
        r3 = lambda v: round(v, 3) if isinstance(v, (int, float)) else v  # noqa: E731
        with open(os.path.join(task_dir, f"{task_key}.md"), "w") as f:
            f.write(f"# {rid}\n\n- suite: {t['suite']}\n- backend: {cfg.backend}\n"
                    f"- status: {res['status']}\n"
                    f"- ttft: {r3(res['ttft_s'])}s (first reasoning: "
                    f"{r3(res.get('ttft_first_reasoning_s'))}s)  total: {res['total_s']}s\n"
                    f"- gen: {res.get('gen_tps')} tok/s  decode(content): "
                    f"{res.get('decode_tps')} tok/s  overall: {res.get('overall_tps')} tok/s\n"
                    f"- tokens ({res['token_source']}): prompt {res.get('prompt_tokens')} / "
                    f"completion {res.get('completion_tokens')} / total "
                    f"{res.get('total_tokens')}\n\n")
            f.write(f"## Prompt\n\n```\n{t['prompt']}\n```\n\n")
            if getattr(result, "reasoning", None):
                f.write(f"## Reasoning\n\n```\n{result.reasoning}\n```\n\n")
            f.write(f"## Response\n\n```\n{result.text or '(empty)'}\n```\n\n")
            if result.tool_calls:
                f.write("## Tool calls\n\n```json\n"
                         f"{json.dumps(result.tool_calls, indent=2)}\n```\n\n")
            if result.events:
                f.write("## CLI events\n\n```json\n"
                         f"{json.dumps(result.events, indent=2)[:6000]}\n```\n\n")
            if res.get("verdict"):
                f.write(f"## Verdict\n\n```json\n{json.dumps(res['verdict'], indent=2)}\n```\n")
            if result.stderr_tail:
                f.write(f"## CLI stderr (tail)\n\n```\n{result.stderr_tail}\n```\n")
            if result.error:
                f.write(f"\n## Error\n\n```\n{result.error}\n```\n")
        with open(os.path.join(task_dir, f"{task_key}.samples.jsonl"), "w") as f:
            for s in sampler.samples:
                f.write(json.dumps(s) + "\n")
    except OSError as e:
        # evidence store failed (disk full / not writable) — the verdict
        # stands, but flag the missing artifacts
        res["artifact_error"] = f"{type(e).__name__}: {e}"
    return res


def _token_source_label(task_results: list[dict]) -> str:
    srcs = {r.get("token_source") for r in task_results if r.get("token_source")}
    if len(srcs) == 1:
        s = next(iter(srcs))
        return "server" if s == "server" else "estimated"
    return "mixed" if srcs else "n/a"


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
            "ttft_s_ci95": stats_ci95([r["ttft_s"] for r in rs]),
            "ttft_model_s": stats([r.get("ttft_model_s") for r in rs]),
            "ttft_model_s_ci95": stats_ci95([r.get("ttft_model_s") for r in rs]),
            "total_s": stats([r["total_s"] for r in rs]),
            "total_s_ci95": stats_ci95([r["total_s"] for r in rs]),
            "decode_tps": stats([r["decode_tps"] for r in rs]),
            "gen_tps": stats([r["gen_tps"] for r in rs]),
            "gen_tps_ci95": stats_ci95([r["gen_tps"] for r in rs]),
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
        "ttft_s_ci95": stats_ci95([r["ttft_s"] for r in ok]),
        "ttft_model_s": stats([r.get("ttft_model_s") for r in ok]),
        "ttft_model_s_ci95": stats_ci95([r.get("ttft_model_s") for r in ok]),
        "token_source": _token_source_label(task_results),
        "total_s": stats([r["total_s"] for r in ok]),
        "total_s_ci95": stats_ci95([r["total_s"] for r in ok]),
        "decode_tps": stats([r["decode_tps"] for r in ok]),
        "gen_tps": stats([r["gen_tps"] for r in ok]),
        "gen_tps_ci95": stats_ci95([r["gen_tps"] for r in ok]),
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


def _fmt_ci(avg: Any, ci: Any) -> str:
    """'12.3 ±0.4' when a repeat-based CI exists, else just the value."""
    if ci is None:
        return str(avg if avg is not None else "—")
    return f"{avg if avg is not None else '—'} ±{ci}"


def console_summary(run: dict, log: Callable[[str], None] = print) -> None:
    tot = run["totals"]
    log("\n[llmbench] === summary ===")
    log(f"tasks: {tot['n_tasks']}  pass: {tot['n_pass']}  fail: {tot['n_fail']}  "
        f"error: {tot['n_error']}")
    for suite, s in sorted(tot["suites"].items()):
        pr = fnum(s["pass_rate"] * 100, 1) if s["pass_rate"] is not None else "—"
        log(f"  {suite:<14} {s['passed']}/{s['n']} pass ({pr}%)  "
            f"ttft avg {_fmt_ci(s['ttft_s'].get('avg'), s.get('ttft_s_ci95'))}s  "
            f"total avg {_fmt_ci(s['total_s'].get('avg'), s.get('total_s_ci95'))}s  "
            f"gen {_fmt_ci(s['gen_tps'].get('avg'), s.get('gen_tps_ci95'))} tok/s "
            f"(content {s['decode_tps'].get('avg')})")
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
    pr.add_argument("--backend", choices=BACKENDS, default="api")
    pr.add_argument("--endpoint", default=os.environ.get("LLMBENCH_ENDPOINT",
                                                          "http://127.0.0.1:8000/v1"))
    pr.add_argument("--model", default=os.environ.get("LLMBENCH_MODEL", "qwen3.8"))
    pr.add_argument("--api-key", default=os.environ.get("LLMBENCH_API_KEY", "llmbench"))
    pr.add_argument("--suites", default="",
                    help="comma-separated suite names (default: all except "
                         "longctx, which is opt-in)")
    pr.add_argument("--tasks", default="", help="comma-separated task ids (default: all)")
    pr.add_argument("--repeat", type=int, default=1)
    pr.add_argument("--max-tokens", type=int, default=4096)
    pr.add_argument("--temperature", type=float, default=0.0)
    pr.add_argument("--timeout", type=float, default=600.0)
    pr.add_argument("--retries", type=int, default=1,
                   help="extra attempts for tasks that ERROR (not for genuine fails); "
                        "default 1")
    pr.add_argument("--concurrency", type=int, default=1,
                   help="tasks in flight at once (load mode; >1 shares the endpoint)")
    pr.add_argument("--sample-interval", type=float, default=2.0)
    pr.add_argument("--warmup", dest="warmup", action="store_true", default=True)
    pr.add_argument("--no-warmup", dest="warmup", action="store_false")
    pr.add_argument("--npu-container", default=os.environ.get("LLMBENCH_NPU_CONTAINER"))
    pr.add_argument("--dsh-api-key-env", default="MY_GATEWAY_API_KEY",
                    help="env var name dsh's per-run provider reads the API key from")
    pr.add_argument("--codex-provider", default=os.environ.get("LLMBENCH_CODEX_PROVIDER",
                                                               "vllm"),
                    help="codex model_providers entry name (default: vllm)")
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

    pc = sub.add_parser("compare", help="compare multiple run dirs side by side")
    pc.add_argument("run_dirs", nargs="+")
    pc.add_argument("--save", default="", help="also write the table to this .md file")
    pc.set_defaults(func=cmd_compare)
    return p


def cmd_list(_: argparse.Namespace) -> None:
    print(f"{'task id':<28} {'verifier':<12} {'backends':<26}")
    for suite_name, tasks_ in SUITES.items():
        print(f"\n[{suite_name}]")
        for t in tasks_:
            bks = ",".join(t.get("backends", BACKENDS))
            print(f"  {t['id']:<28} {t['verify']['type']:<12} {bks:<26}")


def cmd_compare(args: argparse.Namespace) -> None:
    runs = []
    for d in args.run_dirs:
        try:
            with open(os.path.join(d, "run.json")) as f:
                run = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise SystemExit(f"[llmbench] cannot read {d}/run.json: {e}") from e
        cfg = run["config"]
        label = f"{cfg['backend']}/{cfg['model']}"
        runs.append((label, run))

    suites: list[str] = []
    for _, run in runs:
        for s in run["totals"]["suites"]:
            if s not in suites:
                suites.append(s)

    L: list[str] = []
    L.append("# llmbench compare\n")
    hdr = "| metric | " + " | ".join(label for label, _ in runs) + " |"
    L.append(hdr)
    L.append("|---|" + "---|" * len(runs))

    def row(name: str, vals: list[str]) -> None:
        L.append(f"| {name} | " + " | ".join(vals) + " |")

    for suite in suites:
        vals = []
        for _, run in runs:
            s = run["totals"]["suites"].get(suite)
            if not s:
                vals.append("—")
            elif s["pass_rate"] is None:
                vals.append(f"0/{s['n']} (no verdicts)")
            else:
                vals.append(f"{s['passed']}/{s['n']} "
                            f"({s['pass_rate'] * 100:.1f}%)")
        row(suite, vals)

    def avgp95(run: dict, key: str) -> str:
        d = run["totals"].get(key) or {}
        if not d:
            return "—"
        return f"avg {d.get('avg')} / p95 {d.get('p95')}"

    row("TTFT (s)", [avgp95(run, "ttft_s") for _, run in runs])
    row("TTFT-model (s)", [avgp95(run, "ttft_model_s") for _, run in runs])
    row("total latency (s)", [avgp95(run, "total_s") for _, run in runs])
    row("gen tok/s", [avgp95(run, "gen_tps") for _, run in runs])
    row("decode tok/s", [avgp95(run, "decode_tps") for _, run in runs])

    def src(run: dict) -> str:
        return run["totals"].get("token_source") or _token_source_label(run["tasks"])

    row("prompt tokens",
        [f"{run['totals']['prompt_tokens']} ({src(run)})" for _, run in runs])
    row("completion tokens",
        [f"{run['totals']['completion_tokens']} ({src(run)})" for _, run in runs])
    row("total tokens",
        [f"{run['totals']['total_tokens']} ({src(run)})" for _, run in runs])
    row("n tasks", [str(run["totals"]["n_tasks"]) for _, run in runs])

    L.append("")
    L.append("_TTFT-model excludes CLI startup (use for cross-backend model latency). "
             "Tokens: `server` = backend-reported (CLI backends include their harness "
             "system prompt, cumulative over the task's internal API calls); "
             "`estimated` = ~4 chars/token of the task prompt only (dsh)._")

    table = "\n".join(L)
    print(table)
    if args.save:
        with open(args.save, "w") as f:
            f.write(table + "\n")
        print(f"[llmbench] wrote {args.save}")


def cmd_report(args: argparse.Namespace) -> None:
    path = os.path.join(args.run_dir, "run.json")
    try:
        with open(path) as f:
            run = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"[llmbench] cannot read {path}: {e}") from e
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
        sample_interval=args.sample_interval,
        retries=args.retries, concurrency=args.concurrency,
        warmup=args.warmup, npu_container=args.npu_container,
        dsh_api_key_env=args.dsh_api_key_env,
        codex_provider=args.codex_provider,
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
