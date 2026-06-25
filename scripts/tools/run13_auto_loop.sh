#!/bin/bash
# Run 13 overnight loop: train → play → log → one cfg change → repeat until pass or timeout.
set -euo pipefail

ROOT="/home/elken/project/robot_lab"
PYTHON="/home/elken/miniconda3/envs/isaaclab/bin/python"
PLAY="${ROOT}/scripts/reinforcement_learning/rsl_rl/play.py"
MONITOR_PY="${ROOT}/scripts/tools/tb_feet_air_monitor.py"
SAFE_TRAIN="${ROOT}/scripts/tools/safe_round_train.sh"
LOG_ROOT="${ROOT}/logs/rsl_rl/unitree_my_h1_rough"
ROUNDS_LOG="${ROOT}/logs/run13_rounds.log"
CFG="${ROOT}/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/humanoid/unitree_my_h1/rough_env_cfg.py"
MY_H1="${ROOT}/source/robot_lab/robot_lab/assets/my_h1.py"
MAX_HOURS="${1:-6}"
START_TS=$(date +%s)
DEADLINE=$((START_TS + MAX_HOURS * 3600))

get_train_pid() {
  ps -eo pid=,cmd= | grep 'isaaclab/bin/python.*train\.py.*My_H1' \
    | grep -v cursorsandbox | grep -v '/usr/bin/zsh' | awk '{print $1; exit}'
}

wait_for_train() {
  while [ -n "$(get_train_pid)" ]; do sleep 30; done
}

eval_play() {
  local play_log="$1"
  local last500 last300 last400 vx500 dx500 pass fail_reason
  last500=$(grep '\[play debug\].*step=500' "$play_log" | tail -1 || true)
  last300=$(grep '\[play debug\].*step=300' "$play_log" | tail -1 || true)
  last400=$(grep '\[play debug\].*step=400' "$play_log" | tail -1 || true)
  if [ -z "$last500" ]; then
    echo "FAIL|no_play_debug"
    return
  fi
  vx500=$(echo "$last500" | sed -n 's/.*vx=\([-0-9.]*\).*/\1/p')
  dx500=$(echo "$last500" | sed -n 's/.*Δx=\([-0-9.]*\).*/\1/p')
  # pass: Δx>0.5 and vx>0.1 on at least 2 of steps 300/400/500
  local high_vx=0
  for line in "$last300" "$last400" "$last500"; do
    vx=$(echo "$line" | sed -n 's/.*vx=\([-0-9.]*\).*/\1/p')
    if [ -n "$vx" ] && awk -v v="$vx" 'BEGIN{exit !(v > 0.1)}'; then high_vx=$((high_vx + 1)); fi
  done
  if awk -v dx="$dx500" -v hv="$high_vx" 'BEGIN{exit !(dx > 0.5 && hv >= 2)}'; then
    echo "PASS|vx500=${vx500} dx500=${dx500} high_vx_count=${high_vx}"
  else
    echo "FAIL|vx500=${vx500} dx500=${dx500} high_vx_count=${high_vx}|standing/jitter"
  fi
}

apply_change() {
  local n="$1"
  case "$n" in
    2)
      sed -i 's/self.rewards.track_lin_vel_xy_exp.weight = 1.5/self.rewards.track_lin_vel_xy_exp.weight = 1.0  # Run 13 R2/' "$CFG"
      echo "track_lin_vel weight 1.5→1.0"
      ;;
    3)
      sed -i 's/".*_hip_pitch_joint": 350/".*_hip_pitch_joint": 200,  # Run 13 R3 revert/' "$MY_H1"
      sed -i 's/".*_knee_joint": 350/".*_knee_joint": 200/' "$MY_H1"
      echo "hip_pitch/knee stiffness 350→200"
      ;;
    4)
      sed -i 's/stiffness=50,/stiffness=20,  # Run 13 R4 revert ankle/' "$MY_H1"
      sed -i 's/damping=10,/damping=2,/' "$MY_H1"
      echo "ankle PD revert to default"
      ;;
    5)
      python3 - <<'PY'
import re
p = "/home/elken/project/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/humanoid/unitree_my_h1/rough_env_cfg.py"
text = open(p).read()
text = re.sub(
    r'self\.actions\.joint_pos\.scale = 0\.25\n\s*# self\.actions\.joint_pos\.scale = UNITREE_My_H1_ACTION_SCALE',
    'self.actions.joint_pos.scale = UNITREE_My_H1_ACTION_SCALE  # Run 13 R5',
    text, count=1)
open(p, "w").write(text)
print("action scale 0.25→UNITREE_My_H1_ACTION_SCALE")
PY
      ;;
    6)
      sed -i 's/self.rewards.feet_air_time.weight = 1.0/self.rewards.feet_air_time.weight = 1.5  # Run 13 R6/' "$CFG"
      echo "feet_air weight 1.0→1.5"
      ;;
    7)
      sed -i 's/self.rewards.joint_pos_penalty.weight = -1.0/self.rewards.joint_pos_penalty.weight = -0.5  # Run 13 R7/' "$CFG"
      echo "joint_pos_penalty -1.0→-0.5"
      ;;
    *)
      echo "NO_MORE_CHANGES"
      return 1
      ;;
  esac
}

# Round queue: R1 cfg already applied before this script
NEXT_ROUND=1
if grep -q '^R1 ' "$ROUNDS_LOG" 2>/dev/null; then NEXT_ROUND=2; fi

round_train_complete() {
  local rn="$1"
  local log="${ROOT}/logs/${rn}_train.log"
  grep -q "Learning iteration 3499/3500\|Learning iteration 3500/3500" "$log" 2>/dev/null
}

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  RN="run13_r${NEXT_ROUND}"
  CHANGE_DESC=""
  case "$NEXT_ROUND" in
    1) CHANGE_DESC="feet_air threshold 0.6→0.4 (Run 13 R1)" ;;
  esac

  if grep -q "^R${NEXT_ROUND} " "$ROUNDS_LOG" 2>/dev/null; then
    NEXT_ROUND=$((NEXT_ROUND + 1))
    continue
  fi

  if [ -n "$(get_train_pid)" ] || [ -f "${ROOT}/logs/.run13_train.lock" ]; then
    echo "$(date -Iseconds) waiting for in-flight train (round ${NEXT_ROUND})..."
    wait_for_train
  elif ! round_train_complete "$RN"; then
    if [ "$NEXT_ROUND" -gt 1 ]; then
      CHANGE_DESC=$(apply_change "$NEXT_ROUND" || echo "none")
    fi
    if ! "$PYTHON" -m py_compile "$MY_H1" "$CFG" 2>>"${ROOT}/logs/run13_auto_loop.log"; then
      echo "$(date -Iseconds) ERROR: cfg syntax check failed before ${RN}; fix and restart loop" >>"${ROOT}/logs/run13_auto_loop.log"
      exit 3
    fi
    echo "$(date -Iseconds) === ${RN}: ${CHANGE_DESC} ==="
    mv "${ROOT}/logs/${RN}_train.log" "${ROOT}/logs/${RN}_train.log.bak.$(date +%s)" 2>/dev/null || true
    "$SAFE_TRAIN" "$RN" 3500
  fi

  wait_for_train

  if ! round_train_complete "$RN"; then
    echo "$(date -Iseconds) WARN: ${RN} train incomplete — retrying fresh train"
    mv "${ROOT}/logs/${RN}_train.log" "${ROOT}/logs/${RN}_train.log.bak.$(date +%s)" 2>/dev/null || true
    rm -f "${ROOT}/logs/.run13_train.lock"
    "$SAFE_TRAIN" "$RN" 3500
    wait_for_train
  fi

  if ! round_train_complete "$RN"; then
    echo "R${NEXT_ROUND} | ${CHANGE_DESC} | INCOMPLETE | - | train_crashed | FAIL | retry next loop" >>"$ROUNDS_LOG"
    sleep 60
    continue
  fi

  LATEST_EVENTS=$(ls -t "${LOG_ROOT}"/*/events.out.tfevents.* 2>/dev/null | head -1)
  RUN_DIR=$(dirname "$LATEST_EVENTS")
  RUN_NAME=$(basename "$RUN_DIR")
  TB_OUT=$("$PYTHON" "$MONITOR_PY" "$RUN_DIR" 2>/dev/null || true)
  PLAY_CKPT=$(echo "$TB_OUT" | awk -F= '/^play_ckpt=/{print $2}')
  PEAK_STEP=$(echo "$TB_OUT" | awk -F= '/^peak_step=/{print $2}')
  CKPT_PATH="${RUN_DIR}/${PLAY_CKPT}"

  PLAY_LOG="${ROOT}/logs/${RN}_play.log"
  echo "$(date -Iseconds) play ${RUN_NAME}/${PLAY_CKPT}"
  cd "$ROOT"
  "$PYTHON" "$PLAY" \
    --task=RobotLab-Isaac-Velocity-Rough-Unitree-My_H1 \
    --load_run "$RUN_NAME" \
    --checkpoint "$CKPT_PATH" \
    --num_envs 1 --headless \
    >"$PLAY_LOG" 2>&1 || true

  RESULT=$(eval_play "$PLAY_LOG")
  STATUS=$(echo "$RESULT" | cut -d'|' -f1)
  METRICS=$(echo "$RESULT" | cut -d'|' -f2)
  REASON=$(echo "$RESULT" | cut -d'|' -f3)

  NEXT_CHANGE=""
  if [ "$STATUS" = "PASS" ]; then
    NEXT_CHANGE="STOP — locomotion pass"
  elif [ "$NEXT_ROUND" -ge 7 ]; then
    NEXT_CHANGE="queue exhausted"
  else
    case $((NEXT_ROUND + 1)) in
      2) NEXT_CHANGE="track_lin_vel 1.5→1.0" ;;
      3) NEXT_CHANGE="hip/knee 350→200" ;;
      4) NEXT_CHANGE="ankle PD revert" ;;
      5) NEXT_CHANGE="action scale→UNITREE" ;;
      6) NEXT_CHANGE="feet_air weight 1.5" ;;
      7) NEXT_CHANGE="joint_pos_penalty -0.5" ;;
    esac
  fi

  echo "R${NEXT_ROUND} | ${CHANGE_DESC} | ${RUN_NAME} | ${PLAY_CKPT} (peak@${PEAK_STEP}) | ${METRICS} | ${STATUS} | next: ${NEXT_CHANGE}" >>"$ROUNDS_LOG"

  if [ "$STATUS" = "PASS" ]; then
    echo "$(date -Iseconds) PASS at R${NEXT_ROUND}"
    exit 0
  fi

  NEXT_ROUND=$((NEXT_ROUND + 1))
  if [ "$NEXT_ROUND" -gt 7 ]; then
    echo "$(date -Iseconds) change queue exhausted"
    exit 1
  fi
done

echo "$(date -Iseconds) deadline reached (${MAX_HOURS}h)"
exit 2
