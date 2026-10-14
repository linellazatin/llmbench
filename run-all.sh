#!/usr/bin/env bash
# Sequential llmbench sweep: all 5 backends x all 9 suites (incl. opt-in longctx),
# same model qwen3.8 from the single local vLLM provider, strictly sequential.
cd /root/workspace/lines/proj/llmbench || exit 1
SUITES="smoke,reasoning,spatial,ifeval,coding,tools,agentic,mullang,longctx"
LOGDIR=results/logs-all
mkdir -p "$LOGDIR"
for B in api dsh codex claude pi; do
  echo "=== $(date -Is) backend=$B START ==="
  python3 llmbench.py run --backend "$B" --suites "$SUITES" 2>&1 | tee "$LOGDIR/$B.log"
  echo "=== $(date -Is) backend=$B END rc=${PIPESTATUS[0]} ==="
done
echo "=== $(date -Is) ALL DONE ==="
