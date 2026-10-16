#!/usr/bin/env bash
# Sequential llmbench sweep: all 5 backends x all 9 suites (incl. opt-in longctx),
# same model qwen3.8 from the single local vLLM provider, strictly sequential.
# --temperature 0 keeps the sweep greedy, like every baseline recorded before
# 0.2.0 (the harness default is now "omit the parameter" = endpoint's own value).
# LLMBENCH_CONTEXT_TOKENS (or --context-tokens) gates the heavy longctx tiers to
# the served window; 0 = declare nothing and send everything.
cd /root/workspace/lines/proj/llmbench || exit 1
SUITES="smoke,reasoning,spatial,ifeval,coding,tools,agentic,mullang,longctx"
CTX="${LLMBENCH_CONTEXT_TOKENS:-0}"
LOGDIR=results/logs-all
mkdir -p "$LOGDIR"
for B in api dsh codex claude pi; do
  echo "=== $(date -Is) backend=$B START ==="
  python3 llmbench.py run --backend "$B" --suites "$SUITES" \
    --temperature 0 --context-tokens "$CTX" 2>&1 | tee "$LOGDIR/$B.log"
  echo "=== $(date -Is) backend=$B END rc=${PIPESTATUS[0]} ==="
done
echo "=== $(date -Is) ALL DONE ==="
