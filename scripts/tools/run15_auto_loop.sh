#!/bin/bash
# Run 15+ auto loop: wait train → play → eval → one cfg change → repeat.
# Usage: ./scripts/tools/run15_auto_loop.sh [start_round] [max_hours]
set -euo pipefail

ROOT="/home/elken/project/robot_lab"
PYTHON="/home/elken/miniconda3/envs/isaaclab/bin/python"
PLAY="${ROOT}/scripts/reinforcement_learning/rsl_rl/play.py"
MONITOR_PY="${ROOT}/scripts/tools/tb_feet_air_monitor.py"
SAFE_TRAIN="${ROOT}/scripts/tools/safe_round_train.sh"
LOG_ROOT="${ROOT}/logs/rsl_rl/unitree_my_h1_rough"
ROUNDS_LOG="${ROOT}/logs/run15_rounds.log"
LOOP_LOG="${ROOT}/logs/run15_auto_loop.log"
CFG="${ROOT}/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/humanoid/unitree_my_h1/rough_env_cfg.py"
MY_H1="${ROOT}/source/robot_lab/robot_lab/assets/my_h1.py"
START_ROUND="${1:-4}"
MAX_ROUND="${2:-12}"
MAX_HOURS="${3:-10}"
DEADLINE=$(($(date +%s) + MAX_HOURS * 3600))

exec >>"$LOOP_LOG" 2>&1
echo "=== run15_auto_loop start $(date -Iseconds) from R${START_ROUND} ==="

get_train_pid() {
  ps -eo pid=,cmd= | grep 'isaaclab/bin/python.*train\.py.*My_H1' \
    | grep -v cursorsandbox | grep -v '/usr/bin/zsh' | awk '{print $1; exit}'
}

wait_for_train() {
  while [ -n "$(get_train_pid)" ]; do sleep 45; done
}

py_check() {
  if ! "$PYTHON" -m py_compile "$MY_H1" "$CFG"; then
    echo "ERROR: py_compile failed"
    return 1
  fi
}

train_complete() {
  local rn="$1"
  grep -qE "Learning iteration (3499|3500)/3500" "${ROOT}/logs/${rn}_train.log" 2>/dev/null
}

# Returns: STATUS|metrics|tag  (tag drives next change)
eval_play() {
  local play_log="$1"
  local line100 line300 line500
  line100=$(grep '\[play debug\].*step=100 ' "$play_log" | tail -1 || true)
  line300=$(grep '\[play debug\].*step=300 ' "$play_log" | tail -1 || true)
  line500=$(grep '\[play debug\].*step=500 ' "$play_log" | tail -1 || true)
  if [ -z "$line500" ]; then
    echo "FAIL|no_debug|crash"
    return
  fi
  local fallen500 vx100 vx500 dx500 lc500 rc500
  fallen500=$(echo "$line500" | sed -n 's/.*fallen=\([^ ]*\).*/\1/p')
  vx100=$(echo "$line100" | sed -n 's/.*vx=\([-0-9.]*\).*/\1/p')
  vx500=$(echo "$line500" | sed -n 's/.*vx=\([-0-9.]*\).*/\1/p')
  dx500=$(echo "$line500" | sed -n 's/.*Δx=\([-0-9.]*\).*/\1/p')
  lc500=$(echo "$line500" | sed -n 's/.*left_contact=\([^ ]*\).*/\1/p')
  rc500=$(echo "$line500" | sed -n 's/.*right_contact=\([^ ]*\).*/\1/p')

  if grep -q 'fallen=True' "$play_log" || [ "$fallen500" = "True" ]; then
    echo "FAIL|fallen vx500=${vx500} dx=${dx500}|fall"
    return
  fi
  if [ -n "$vx100" ] && awk -v v="$vx100" 'BEGIN{exit !(v < -0.08)}'; then
    echo "FAIL|back_jump vx100=${vx100}|back_jump"
    return
  fi
  if awk -v dx="$dx500" -v vx="$vx500" 'BEGIN{exit !(dx > 0.3 && vx > 0.05)}'; then
    echo "PASS|dx=${dx500} vx=${vx500}|walk"
    return
  fi
  if [ "$lc500" = "True" ] && [ "$rc500" = "True" ] && awk -v dx="$dx500" 'BEGIN{exit !(dx < 0.15)}'; then
    echo "FAIL|planted dx=${dx500}|planted"
    return
  fi
  echo "FAIL|vx500=${vx500} dx=${dx500}|shuffle"
}

apply_next_change() {
  local next="$1"
  local tag="$2"
  case "$next" in
    5)
      case "$tag" in
        fall|back_jump)
          sed -i 's/".*_hip_pitch_joint": 6/".*_hip_pitch_joint": 7  # Run 15 R5/' "$MY_H1"
          sed -i 's/".*_knee_joint": 6/".*_knee_joint": 7/' "$MY_H1"
          echo "damping hip_pitch/knee 6→7 (after fall/back_jump)"
          ;;
        planted)
          sed -i 's/params\["threshold"\] = 0.6/params["threshold"] = 0.5  # Run 15 R5/' "$CFG"
          echo "feet_air threshold 0.6→0.5 (planted, keep damping 6)"
          ;;
        *)
          sed -i 's/".*_hip_pitch_joint": 6/".*_hip_pitch_joint": 7  # Run 15 R5/' "$MY_H1"
          sed -i 's/".*_knee_joint": 6/".*_knee_joint": 7/' "$MY_H1"
          echo "damping 6→7 (default after shuffle)"
          ;;
      esac
      ;;
    6)
      case "$tag" in
        fall|back_jump)
          sed -i 's/".*_hip_pitch_joint": 7/".*_hip_pitch_joint": 8/' "$MY_H1"
          sed -i 's/".*_knee_joint": 7/".*_knee_joint": 8/' "$MY_H1"
          echo "damping 7→8 revert stable"
          ;;
        planted)
          sed -i 's/params\["threshold"\] = 0.5/params["threshold"] = 0.5/' "$CFG" 2>/dev/null || \
          sed -i 's/params\["threshold"\] = 0.6/params["threshold"] = 0.5  # Run 15 R6/' "$CFG"
          echo "threshold 0.6→0.5"
          ;;
        *)
          sed -i 's/self.rewards.track_lin_vel_xy_exp.weight = 1.0/self.rewards.track_lin_vel_xy_exp.weight = 1.2  # Run 15 R6/' "$CFG"
          echo "track 1.0→1.2 slight forward"
          ;;
      esac
      ;;
    7)
      sed -i 's/self.rewards.feet_air_time.weight = 2.0/self.rewards.feet_air_time.weight = 2.5  # Run 15 R7/' "$CFG"
      echo "feet_air weight 2.0→2.5"
      ;;
    8)
      sed -i 's/self.rewards.joint_pos_penalty.weight = -0.5/self.rewards.joint_pos_penalty.weight = -0.3  # Run 15 R8/' "$CFG"
      echo "joint_pos_penalty -0.5→-0.3"
      ;;
    *)
      echo "NO_MORE"
      return 1
      ;;
  esac
}

NEXT_ROUND=$START_ROUND

while [ "$NEXT_ROUND" -le "$MAX_ROUND" ] && [ "$(date +%s)" -lt "$DEADLINE" ]; do
  RN="run15_r${NEXT_ROUND}"

  if [ -n "$(get_train_pid)" ] || [ -f "${ROOT}/logs/.run13_train.lock" ]; then
    echo "$(date -Iseconds) wait in-flight train for ${RN}"
    wait_for_train
  fi

  if ! train_complete "$RN"; then
    if ! py_check; then
      echo "$(date -Iseconds) cfg broken before ${RN}; stop loop"
      exit 3
    fi
    echo "$(date -Iseconds) start ${RN}"
    "$SAFE_TRAIN" "$RN" 3500 || true
    wait_for_train
  fi

  if ! train_complete "$RN"; then
    echo "R${NEXT_ROUND} | INCOMPLETE | train_crashed" >>"$ROUNDS_LOG"
    rm -f "${ROOT}/logs/.run13_train.lock"
    sleep 120
    continue
  fi

  LATEST_EVENTS=$(ls -t "${LOG_ROOT}"/*/events.out.tfevents.* 2>/dev/null | head -1)
  RUN_DIR=$(dirname "$LATEST_EVENTS")
  RUN_NAME=$(basename "$RUN_DIR")
  TB_OUT=$("$PYTHON" "$MONITOR_PY" "$RUN_DIR" 2>/dev/null || true)
  PLAY_CKPT=$(echo "$TB_OUT" | awk -F= '/^play_ckpt=/{print $2}')
  PEAK_STEP=$(echo "$TB_OUT" | awk -F= '/^peak_step=/{print $2}')
  PEAK_VAL=$(echo "$TB_OUT" | awk -F= '/^peak_smoothed=/{print $2}')
  CKPT_PATH="${RUN_DIR}/${PLAY_CKPT}"

  PLAY_LOG="${ROOT}/logs/${RN}_play.log"
  echo "$(date -Iseconds) play ${RUN_NAME}/${PLAY_CKPT} peak@${PEAK_STEP}=${PEAK_VAL}"
  cd "$ROOT"
  "$PYTHON" "$PLAY" \
    --task=RobotLab-Isaac-Velocity-Rough-Unitree-My_H1 \
    --num_envs 1 --headless \
    --checkpoint "$CKPT_PATH" \
    >"$PLAY_LOG" 2>&1 || true

  RESULT=$(eval_play "$PLAY_LOG")
  STATUS=$(echo "$RESULT" | cut -d'|' -f1)
  METRICS=$(echo "$RESULT" | cut -d'|' -f2)
  TAG=$(echo "$RESULT" | cut -d'|' -f3)

  echo "R${NEXT_ROUND} | ${RUN_NAME} | ${PLAY_CKPT} | peak=${PEAK_VAL} | ${METRICS} | ${STATUS} | tag=${TAG}" >>"$ROUNDS_LOG"

  if [ "$STATUS" = "PASS" ]; then
    echo "$(date -Iseconds) PASS at R${NEXT_ROUND}"
    exit 0
  fi

  NEXT_ROUND=$((NEXT_ROUND + 1))
  if [ "$NEXT_ROUND" -gt "$MAX_ROUND" ]; then
    break
  fi
  CHANGE=$(apply_next_change "$NEXT_ROUND" "$TAG" || echo "none")
  if ! py_check; then
    echo "$(date -Iseconds) apply failed py_compile; revert manual needed"
    exit 4
  fi
  echo "$(date -Iseconds) R${NEXT_ROUND} change: ${CHANGE}"
  sleep 10
done

echo "$(date -Iseconds) loop end"
exit 1
