#!/usr/bin/env bash
# Drives Codex through PLAN.md one unchecked step at a time.
# After each Codex run this acts as the lifecycle hook: it runs pytest, halts on
# failure or on no progress, otherwise commits, pushes and moves to the next step.
# Usage: scripts/codex_loop.sh [max_runs]   (default 15)
set -uo pipefail

MAX_RUNS="${1:-15}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG_DIR="runs/codex_logs"
mkdir -p "$LOG_DIR"
STATUS="$LOG_DIR/STATUS"

next_step() { grep -m1 -E '^- \[ \]' PLAN.md; }
remaining() { grep -cE '^- \[ \]' PLAN.md; }

for ((i = 1; i <= MAX_RUNS; i++)); do
  if [ "$(remaining)" -eq 0 ]; then
    echo "DONE: all PLAN.md steps checked after $((i - 1)) runs." | tee "$STATUS"
    exit 0
  fi
  STEP="$(next_step)"
  echo "[run $i/$MAX_RUNS] $STEP" | tee "$STATUS"

  PROMPT="You are executing a hands-off, step-by-step automation chain governed by PLAN.md.
Read AGENTS.md first. Then:
1. Take the very first item marked '- [ ]' in PLAN.md. It is: $STEP
2. Implement only that step, with tests.
3. Run .venv/bin/pytest -q and fix failures caused by your change.
4. Mark that item '- [x]' in PLAN.md.
5. Append 3-6 lines to docs/reports/progress.md: step number, files touched, pytest summary line, deviations or open questions.
Do NOT run git commit; the driver commits after its own test run. Do not touch later steps."

  codex exec --sandbox workspace-write -C "$ROOT" \
    -o "$LOG_DIR/run_${i}_last.md" - <<<"$PROMPT" >"$LOG_DIR/run_${i}.log" 2>&1
  CODEX_RC=$?

  # Codex usage limit: wait for the reset time it reports, then retry the same
  # step (the run doesn't count against MAX_RUNS). Partial edits stay in place.
  RESET="$(grep -oE 'try again at [0-9]{1,2}:[0-9]{2} ?[AP]M' "$LOG_DIR/run_${i}.log" | tail -1 | sed -E 's/try again at //; s/ //')"
  if [ -n "$RESET" ]; then
    TARGET="$(date -j -f '%Y-%m-%d %I:%M%p' "$(date +%Y-%m-%d) $RESET" +%s 2>/dev/null)"
    if [ -n "$TARGET" ]; then
      [ "$TARGET" -le "$(date +%s)" ] && TARGET=$((TARGET + 86400))
      echo "WAIT: Codex usage limit; retrying at $RESET (+2 min)" | tee -a "$STATUS"
      while [ "$(date +%s)" -lt $((TARGET + 120)) ]; do sleep 60; done
      mv "$LOG_DIR/run_${i}.log" "$LOG_DIR/run_${i}_limited_$(date +%s).log"
      i=$((i - 1))
      continue
    fi
  fi

  # Hook: tests must pass.
  if ! .venv/bin/pytest -q >"$LOG_DIR/run_${i}_pytest.log" 2>&1; then
    echo "HALT: tests failed after run $i (codex rc=$CODEX_RC). See $LOG_DIR/run_${i}_pytest.log" | tee "$STATUS"
    exit 1
  fi
  # Hook: the step must have been checked off.
  if [ "$(next_step)" = "$STEP" ]; then
    echo "HALT: run $i made no progress on: $STEP (codex rc=$CODEX_RC). See $LOG_DIR/run_${i}.log" | tee "$STATUS"
    exit 2
  fi

  git add -A
  git commit -q -m "$(echo "$STEP" | sed -E 's/^- \[ \] //' | cut -c1-72)" || true
  # Push each step so GitHub tracks progress; a failed push is logged, not fatal.
  if ! git push -q -u origin HEAD >>"$LOG_DIR/push.log" 2>&1; then
    echo "WARN: push failed after run $i; see $LOG_DIR/push.log" | tee -a "$STATUS"
  fi
  echo "INSTRUCTION: Proceed to the next unchecked step. ($(remaining) remaining)" | tee -a "$STATUS"
done

echo "STOP: hit run cap $MAX_RUNS with $(remaining) steps remaining." | tee "$STATUS"
exit 0
