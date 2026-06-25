#!/bin/bash
# One training round: fresh train only (no --resume), run full max_iterations, play best ckpt.
# Usage:
#   ./scripts/tools/safe_round_train.sh [round_name] [max_iterations]
#
# Rules:
#   - Never --resume; each round is a new log dir
#   - Train full max_iterations (default 3500); no early stop
#   - Play uses best ckpt (feet_air Smoothed peak), NOT necessarily model_3500
set -euo pipefail

ROOT="/home/elken/project/robot_lab"
PYTHON="/home/elken/miniconda3/envs/isaaclab/bin/python"
TRAIN="${ROOT}/scripts/reinforcement_learning/rsl_rl/train.py"
PLAY="${ROOT}/scripts/reinforcement_learning/rsl_rl/play.py"
MONITOR_SH="${ROOT}/scripts/tools/monitor_and_stop_train.sh"
MONITOR_PY="${ROOT}/scripts/tools/tb_feet_air_monitor.py"
LOG_ROOT="${ROOT}/logs/rsl_rl/unitree_my_h1_rough"
ROUND_NAME="${1:-round}"
MAX_ITER="${2:-3500}"
POLL_SEC="${3:-90}"
LOG="${ROOT}/logs/${ROUND_NAME}_train.log"
MON_LOG="${ROOT}/logs/${ROUND_NAME}_monitor.log"

LOCK="${ROOT}/logs/.run13_train.lock"

get_running_train_pid() {
  ps -eo pid=,cmd= | grep 'isaaclab/bin/python.*train\.py.*My_H1' \
    | grep -v cursorsandbox | grep -v '/usr/bin/zsh' | awk '{print $1; exit}'
}

if [ -f "$LOCK" ] && [ -n "$(get_running_train_pid)" ]; then
  echo "ERROR: lock held and train running (lock=$LOCK)" >&2
  exit 1
fi
rm -f "$LOCK"

if [ -n "$(get_running_train_pid)" ]; then
  echo "ERROR: train.py already running; stop it before starting a new round." >&2
  exit 1
fi

echo "=== ${ROUND_NAME}: start fresh train (max_iterations=${MAX_ITER}, no resume) ==="
cd "$ROOT"
touch "$LOCK"
nohup "$PYTHON" "$TRAIN" \
  --task=RobotLab-Isaac-Velocity-Rough-Unitree-My_H1 \
  --headless --num_envs 1024 --max_iterations "$MAX_ITER" \
  >>"$LOG" 2>&1 &

sleep 15
for _ in $(seq 1 24); do
  if [ -n "$(get_running_train_pid)" ]; then
    break
  fi
  sleep 5
done
if [ -z "$(get_running_train_pid)" ]; then
  echo "ERROR: train failed to start; see ${LOG}" >&2
  exit 1
fi

echo "=== ${ROUND_NAME}: wait until ${MAX_ITER} steps (no early stop) ==="
"$MONITOR_SH" auto "$MAX_ITER" "$POLL_SEC" >"$MON_LOG" 2>&1 || true

echo "=== ${ROUND_NAME}: train stopped ==="
rm -f "$LOCK"
grep -E "STOP_REASON|train_stopped|events_dir=" "$MON_LOG" | tail -5 || true

LATEST_EVENTS=$(ls -t "${LOG_ROOT}"/*/events.out.tfevents.* 2>/dev/null | head -1 || true)
if [ -n "$LATEST_EVENTS" ]; then
  RUN_DIR=$(dirname "$LATEST_EVENTS")
  RUN_NAME=$(basename "$RUN_DIR")
  TB_OUT=$("$PYTHON" "$MONITOR_PY" "$RUN_DIR" 2>/dev/null || true)
  PEAK_STEP=$(echo "$TB_OUT" | awk -F= '/^peak_step=/{print $2}')
  PEAK_VAL=$(echo "$TB_OUT" | awk -F= '/^peak_smoothed=/{print $2}')
  PLAY_CKPT=$(echo "$TB_OUT" | awk -F= '/^play_ckpt=/{print $2}')
  if [ -n "$PLAY_CKPT" ] && [ -f "${RUN_DIR}/${PLAY_CKPT}" ]; then
    echo ""
    echo "=== Play best ckpt (feet_air peak @ step ${PEAK_STEP}, smoothed=${PEAK_VAL}) ==="
    echo "${PYTHON} ${PLAY} \\"
    echo "  --task=RobotLab-Isaac-Velocity-Rough-Unitree-My_H1 \\"
    echo "  --load_run ${RUN_NAME} \\"
    echo "  --checkpoint ${PLAY_CKPT} \\"
    echo "  --num_envs 1"
  else
    echo "WARN: could not resolve play ckpt; check TensorBoard for feet_air peak." >&2
  fi
fi

echo ""
echo "Next: run play above, then change ONE cfg item before another round."
echo "Do NOT use --resume between rounds."
