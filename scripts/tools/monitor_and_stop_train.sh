#!/bin/bash
# Wait for training to reach MAX_ITER then stop. No feet_air early stop.
# RUN_DIR: experiment folder name, or "auto" to follow the newest active events dir.
set -euo pipefail
RUN_DIR="${1:?run dir name e.g. 2026-06-09_02-03-16, or auto}"
LOG_ROOT="/home/elken/project/robot_lab/logs/rsl_rl/unitree_my_h1_rough"
PYTHON="/home/elken/miniconda3/envs/isaaclab/bin/python"
MONITOR="/home/elken/project/robot_lab/scripts/tools/tb_feet_air_monitor.py"
MAX_ITER="${2:-3500}"
POLL_SEC="${3:-120}"

get_train_pid() {
  ps -eo pid=,cmd= | grep 'isaaclab/bin/python.*train\.py.*My_H1' \
    | grep -v cursorsandbox | grep -v '/usr/bin/zsh' | awk '{print $1; exit}'
}

resolve_events_dir() {
  local run="$1"
  if [ "$run" = "auto" ]; then
    local latest
    latest=$(ls -t "${LOG_ROOT}"/*/events.out.tfevents.* 2>/dev/null | head -1 || true)
    if [ -z "$latest" ]; then
      echo ""
      return
    fi
    dirname "$latest"
    return
  fi
  echo "${LOG_ROOT}/${run}"
}

while true; do
  PID=$(get_train_pid)
  if [ -z "$PID" ]; then
    echo "STOP_REASON=train_exited"
    exit 0
  fi

  EVENTS_DIR=$(resolve_events_dir "$RUN_DIR")
  if [ -z "$EVENTS_DIR" ] || [ ! -d "$EVENTS_DIR" ]; then
    echo "$(date -Iseconds) status=no_events_dir run=${RUN_DIR}"
    sleep "$POLL_SEC"
    continue
  fi

  OUT=$("$PYTHON" "$MONITOR" "$EVENTS_DIR" 2>/dev/null || echo "status=no_data")
  echo "$(date -Iseconds) events_dir=${EVENTS_DIR} ${OUT}"
  status=$(echo "$OUT" | awk -F= '/^status=/{print $2}')
  last_step=$(echo "$OUT" | awk -F= '/^last_step=/{print $2}')

  if [ "${status:-}" = "no_data" ]; then
    sleep "$POLL_SEC"
    continue
  fi

  if [ "${last_step:-0}" -ge "$MAX_ITER" ]; then
    echo "STOP_REASON=max_iterations cap=${MAX_ITER}"
    kill -INT "$PID" 2>/dev/null || true
    break
  fi

  sleep "$POLL_SEC"
done

for _ in $(seq 1 60); do
  if [ -z "$(get_train_pid)" ]; then
    echo "train_stopped=yes"
    exit 0
  fi
  sleep 5
done

echo "train_stopped=timeout_sending_second_sigint"
PID=$(get_train_pid)
[ -n "$PID" ] && kill -INT "$PID" 2>/dev/null || true
sleep 30
