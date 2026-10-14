#!/usr/bin/env python3
"""Collate llmbench run.json files across backends into one analysis JSON."""
import json, statistics, glob, os

BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "results"))
runs = sorted(glob.glob(os.path.join(BASE, "2026*_qwen3.8_*", "run.json")))
out = {"runs": []}
for p in runs:
    d = json.load(open(p))
    c = d["config"]
    tasks = d["tasks"]
    totals = d.get("totals", {})
    row = {
        "dir": os.path.basename(os.path.dirname(p)),
        "backend": c["backend"],
        "model": c["model"],
        "started_at": d.get("started_at"),
        "finished_at": d.get("finished_at"),
        "n_tasks": len(tasks),
        "pass": sum(1 for t in tasks if t["status"] == "pass"),
        "fail": sum(1 for t in tasks if t["status"] == "fail"),
        "error": sum(1 for t in tasks if t["status"] == "error"),
        "requested_suites": c["suites"],
        "run_suites": sorted({t["suite"] for t in tasks}),
        "skipped_suites": sorted(set(c["suites"]) - {t["suite"] for t in tasks}),
        "prompt_tokens": sum(t.get("prompt_tokens") or 0 for t in tasks),
        "completion_tokens": sum(t.get("completion_tokens") or 0 for t in tasks),
        "total_tokens": sum(t.get("total_tokens") or 0 for t in tasks),
        "token_sources": sorted({t.get("token_source") for t in tasks}),
        "totals_suites": totals.get("suites", {}),
        "fails": [
            {"task": t["task_id"], "suite": t["suite"], "status": t["status"],
             "verdict": t.get("verdict"), "error": t.get("error"),
             "expected": (t.get("verdict") or {}).get("expected"),
             "actual": (t.get("verdict") or {}).get("actual")}
            for t in tasks if t["status"] != "pass"
        ],
        "tasks": [
            {"id": t["task_id"], "suite": t["suite"], "status": t["status"],
             "ttft_model_s": t.get("ttft_model_s"), "total_s": t.get("total_s"),
             "gen_tps": t.get("gen_tps"), "decode_tps": t.get("decode_tps"),
             "prompt_tokens": t.get("prompt_tokens"),
             "completion_tokens": t.get("completion_tokens"),
             "attempts": t.get("attempts"), "error": t.get("error")}
            for t in tasks
        ],
    }
    # per-task ttft_model/total aggregates (all tasks)
    for key, field in (("ttft_model", "ttft_model_s"), ("total_s", "total_s")):
        vals = [t[field] for t in tasks if t.get(field) is not None]
        row[key] = {"n": len(vals), "avg": statistics.mean(vals), "p50": statistics.median(vals),
                    "min": min(vals), "max": max(vals)} if vals else None
    # resource peaks: chip mem peak (max over chips), sys cpu peak, ram peak, proc rss peak
    res = {"suite_hbm_peak_gb": {}, "suite_power_peak_w": {}, "suite_sys_cpu_peak": {},
           "suite_rss_peak_mb": {}}
    for t in tasks:
        r = t.get("resources") or {}
        chips = r.get("chips") or []
        memp = [ (ch.get("mem_used_gb") or {}).get("peak") for ch in chips ]
        memp = [v for v in memp if v is not None]
        def bump(table, suite, val):
            cur = table.get(suite)
            table[suite] = val if cur is None else max(cur, val)
        if memp:
            bump(res["suite_hbm_peak_gb"], t["suite"], max(memp))
        pows = [ (ch.get("power_w") or {}).get("peak") for ch in chips ]
        pows = [v for v in pows if v is not None]
        if pows:
            bump(res["suite_power_peak_w"], t["suite"], max(pows))
        cpu = (r.get("sys_cpu_pct") or {}).get("peak")
        if cpu is not None:
            bump(res["suite_sys_cpu_peak"], t["suite"], cpu)
        rss = (r.get("proc_rss_mb") or {}).get("peak")
        if rss is not None:
            bump(res["suite_rss_peak_mb"], t["suite"], rss)
        pcpu = (r.get("proc_cpu_pct") or {}).get("peak")
        if pcpu is not None:
            bump(res.setdefault("suite_proc_cpu_peak", {}), t["suite"], pcpu)
    # also all-chip mem peak per task (max chip) for whole-run resource table
    row["resources"] = res
    row["host"] = {k: d.get("host", {}).get(k) for k in ("hostname", "arch", "cpu_cores", "ram_total_gb", "accelerator")}
    out["runs"].append(row)

json.dump(out, open(os.path.join(BASE, "analysis.json"), "w"), indent=1)
print("backends:", [(r["backend"], r["pass"], r["n_tasks"], r["fail"], r["error"]) for r in out["runs"]])
