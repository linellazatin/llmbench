#!/usr/bin/env python3
"""Correlate usage-proxy records to dsh llmbench tasks.

Usage:  python3 tools/dsh_usage_corr.py <run_dir> <proxy_usage.jsonl>

How dsh talks to the model (observed on the wire): each task first triggers
a small "session title" request whose last user message is

    Generate the session title from this JSON array of human messages:
    [{"seq":N,"text":"<escaped task prompt>"}]

followed by one or more agent requests whose last user message is dsh's own
scaffolding (skill reminders etc.). So correlation works by:
  1. fingerprinting the prompt visible in each title request — the proxy
     only logs a head (120 chars, of which ~88 are the fixed JSON prefix)
     and a tail (80 chars, of which ~4 are the fixed '"}]' suffix), so we
     recover a visible head fragment (~32 chars) and tail fragment
     (~76 chars) of the escaped prompt;
  2. matching tasks whose escaped prompt both starts with the head fragment
     and ends with the tail fragment;
  3. attributing every non-title request to the task of the most recent
     matched title request (the warmup's requests match nothing and have no
     predecessor, so they are dropped and reported).

Writes <run_dir>/../dsh-proxy/measured.json and prints per-task, per-suite
and overall measured token totals next to the run's own estimates.
"""
import json
import os
import re
import sys

TITLE_RE = re.compile(
    r'^Generate the session title from this JSON array of human messages: '
    r'\[\{"seq":\d+,"text":"')
SUFFIX_RE = re.compile(r'"\}\]$')


def norm(s: str) -> str:
    return " ".join((s or "").split())


def json_esc(s: str) -> str:
    """How the prompt appears inside dsh's session-title JSON payload."""
    return (s.replace("\\", "\\\\").replace('"', '\\"')
              .replace("\n", "\\n").replace("\t", "\\t"))


def load_task_prompts(run_dir: str):
    out = {}
    tdir = os.path.join(run_dir, "tasks")
    for fn in sorted(os.listdir(tdir)):
        if not fn.endswith(".md"):
            continue
        m = re.search(r"^## Prompt\n+```\n(.*?)\n```",
                     open(os.path.join(tdir, fn)).read(), re.S | re.M)
        if not m:
            continue
        task_id = fn[:-3].replace("__", "/")
        # escape FIRST (real newlines -> literal backslash-n), then normalize:
        # the text dsh sends keeps the JSON escape while its surrounding
        # whitespace is untouched
        out[task_id] = norm(json_esc(m.group(1)))
    return out


def prompt_fragments(rec):
    """(visible head, visible tail) of the escaped prompt, or None."""
    m = TITLE_RE.match(rec.get("user_head") or "")
    if not m:
        return None
    head = rec.get("user_head") or ""
    hp = head[m.end():]
    tp = SUFFIX_RE.sub("", rec.get("user_tail") or "")
    tp = tp.rsplit('"text":"', 1)[-1]
    return hp, tp


def prompt_full(rec):
    """Exact escaped prompt text from a full-message log (proxy v2+)."""
    u = rec.get("user_full") or ""
    m = TITLE_RE.match(u)
    if not m or not u.endswith('"}]'):
        return None
    return u[m.end():-len('"}]')]


def main():
    run_dir, usage_path = sys.argv[1], sys.argv[2]
    run = json.load(open(os.path.join(run_dir, "run.json")))
    prompts = load_task_prompts(run_dir)

    recs = [json.loads(l) for l in open(usage_path) if l.strip()]
    recs = [r for r in recs if r.get("method") == "POST" and r.get("status") == 200
            and str(r.get("path", "")).rstrip("/").endswith(
                ("chat/completions", "completions"))]
    recs.sort(key=lambda r: r["seq"])

    # match every title request to its task
    matched = {}
    ambiguous = []
    for r in recs:
        full = prompt_full(r)
        if full is not None:
            hits = [tid for tid, p in prompts.items() if p == full]
            if len(hits) == 1:
                matched[r["seq"]] = hits[0]
                continue
        fr = prompt_fragments(r)
        if not fr:
            continue
        hp, tp = fr
        hits = [tid for tid, p in prompts.items()
                if p.startswith(hp) and (not tp or p.endswith(tp) or tp in p)]
        if len(hits) == 1:
            matched[r["seq"]] = hits[0]
        elif len(hits) > 1:
            ambiguous.append((r["seq"], hits))
            matched[r["seq"]] = hits[0]
        # else: no match — treated as follow-up/warmup below

    per_task, warmup, unmatched = {}, 0, 0
    last = None
    for r in recs:
        tid = matched.get(r["seq"])
        if tid is None:
            if last is None:
                warmup += 1
                continue
            unmatched += 1
            tid = last
        last = tid
        pt = r.get("prompt_tokens") or 0
        ct = r.get("completion_tokens") or 0
        a = per_task.setdefault(tid, {"n_req": 0, "prompt": 0, "completion": 0,
                                      "total": 0, "streams": 0})
        a["n_req"] += 1
        a["prompt"] += pt
        a["completion"] += ct
        a["total"] += pt + ct
        a["streams"] += 1 if r.get("stream") else 0

    # side-by-side vs the run's own (estimated) numbers
    est = {t["task_id"]: t for t in run["tasks"]}
    missing = [tid for tid in est if tid not in per_task]
    print(f"{'task':32s} {'req':>3s} {'meas.prompt':>12s} {'est.prompt':>10s} "
          f"{'meas.compl':>11s} {'est.compl':>9s}")
    for tid in sorted(per_task):
        a = per_task[tid]
        e = est.get(tid, {})
        print(f"{tid:32s} {a['n_req']:3d} {a['prompt']:12d} "
              f"{e.get('prompt_tokens') or 0:10d} {a['completion']:11d} "
              f"{e.get('completion_tokens') or 0:9d}")

    suites = {}
    for tid, a in per_task.items():
        s = suites.setdefault(tid.split("/")[0], {"n": 0, "prompt": 0,
                                                  "completion": 0, "total": 0})
        s["n"] += a["n_req"]
        for k in ("prompt", "completion", "total"):
            s[k] += a[k]
    tot = {k: sum(a[k] for a in per_task.values())
           for k in ("prompt", "completion", "total")}
    n_req = sum(a["n_req"] for a in per_task.values())

    out = {"run_dir": run_dir, "warmup_requests": warmup,
           "followup_requests": unmatched,
           "ambiguous_title_requests": ambiguous,
           "tasks_missing_from_proxy": missing,
           "requests": n_req, "per_task": per_task, "per_suite": suites,
           "totals": tot}
    dest = os.path.normpath(os.path.join(run_dir, "..", "dsh-proxy",
                                         "measured.json"))
    json.dump(out, open(dest, "w"), indent=1)
    print(f"\nwarmup dropped: {warmup}  follow-ups attributed: {unmatched}  "
          f"ambiguous: {len(ambiguous)}  tasks missing from proxy: {missing or 'none'}")
    print(f"measured totals: prompt {tot['prompt']}  completion {tot['completion']}"
          f"  total {tot['total']}  (requests {n_req})")
    print("run.json estimates: "
          f"prompt {sum(t.get('prompt_tokens') or 0 for t in run['tasks'])}  "
          f"completion {sum(t.get('completion_tokens') or 0 for t in run['tasks'])}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
