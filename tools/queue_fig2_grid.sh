#!/usr/bin/env bash
# Figure 2 budget-tuning grid (SESiL-foundation only), per the professor's spec:
#   default (from Fig1) ......... exp2d  9+0.3  (exists)
#   x=6 finetune sweep .......... 6+0.5, 6+1, 6+1.5 (NEW: exp2f/g/h)
#                                 6+3 anchor = exp2b (exists)
#   low-x / low-y corner ........ 3+0.3  (NEW: exp2i)
#   low-x / high-y corner ....... exp2a  3+6    (exists)
#   high-x / low-y corner ....... exp2d  9+0.3  (exists, = default)
#
# Notation: x = SSL core full-set epochs (backbone_e<x>), y = TOTAL finetune
# budget in full-set units = 3S (10 agents x 0.3u x S subset-epochs each), so
#   y=0.5 -> S=1/6  (5 batches),  y=1 -> S=1/3 (10 batches),
#   y=1.5 -> S=1/2 (15 batches),  y=0.3 -> S=0.1 (3 batches).
# round(S x 30) makes each S land on its exact y. Same recipe as exp2a-d:
# permute, 25 gens, --init-lr 0.03, seed 0 (single run per the professor).
#
#   bash tools/queue_fig2_grid.sh

set -u
cd "$(dirname "$0")/.." || exit 1

# Pinned interpreter: a detached bash does not inherit the conda env (see
# queue_multiseed.sh header for the incident this prevents).
PY="/c/Users/32063/anaconda3/envs/sesil/python.exe"

SSL="runs_ssl/simsiam_e50_seed0"
QLOG="./runs_fig2_grid.queue.log"
POLL=180

log() { echo "[fig2grid $(date '+%F %T')] $*" | tee -a "$QLOG"; }

running_sesil() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*run_sesil.py*' }).Count" \
    2>/dev/null | tr -d '\r\n '
}

wait_idle() {
  local n
  n=$(running_sesil)
  if [ -n "$n" ] && [ "$n" != "0" ]; then
    log "run_sesil.py active; waiting"
    while [ "$(running_sesil)" != "0" ]; do sleep "$POLL"; done
  fi
}

# Completion = final-generation CSV has its 10 rows (robust to a missing Done.
# marker; see queue_multiseed.sh).
done_skip() {  # $1 = permute.log, $2 = final gen CSV, $3 = label
  if grep -aq '^Done\.' "$1" 2>/dev/null \
     || [ "$(grep -ac '^tensor(' "$2" 2>/dev/null || echo 0)" -ge 10 ]; then
    log "$3 already complete, skip"; return 0
  fi
  return 1
}

run_grid_point() {  # $1=out-root $2=backbone E $3=sub-epochs S $4=label
  local out=$1 E=$2 S=$3 label=$4
  done_skip "$out/seed0/permute.log" "$out/seed0/permute/csv/gen_25/configurations.csv" "$label" && return
  wait_idle
  mkdir -p "$out/seed0"
  log "starting $label (backbone_e$E, sub-epochs $S, 25 gens)"
  "$PY" run_sesil.py --seed 0 --method permute --generations 25 \
    --init-from "$SSL/backbone_e$E.pth.tar" --init-backbone-only \
    --sub-epochs "$S" --init-lr 0.03 --out-root "$out" \
    > "$out/seed0/permute.log" 2>&1
  log "$label exited rc=$?"
}

run_grid_point ./runs_exp2f 6 0.16667 "exp2f (6+0.5)"
run_grid_point ./runs_exp2g 6 0.33333 "exp2g (6+1)"
run_grid_point ./runs_exp2h 6 0.5     "exp2h (6+1.5)"
run_grid_point ./runs_exp2i 3 0.1     "exp2i (3+0.3, low-low corner)"

log "fig2 grid queue finished"
