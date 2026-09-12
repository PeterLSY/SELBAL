#!/usr/bin/env bash
# Figure 2, second series: finetune FIXED at y=1.5, SSL core x varying --
# the "impact of the unsupervised learning" sweep. With the first queue's
# exp2h (6+1.5) and the existing exp2c (7.5+1.5), the series is:
#   3+1.5 (NEW: exp2j)   6+1.5 (exp2h)   7.5+1.5 (exp2c)   9+1.5 (NEW: exp2k)
# y=1.5 -> S=0.5 -> 15 finetune batches, exact. Recipe identical to exp2a-d.
#
# WAITS for the whole first queue (tools/queue_fig2_grid.sh) to exit, not just
# for run_sesil.py to go idle -- the gap between that queue's sequential runs
# would otherwise let this one start and collide on the GPU.
#
#   bash tools/queue_fig2_xsweep.sh

set -u
cd "$(dirname "$0")/.." || exit 1

PY="/c/Users/32063/anaconda3/envs/sesil/python.exe"
SSL="runs_ssl/simsiam_e50_seed0"
QLOG="./runs_fig2_xsweep.queue.log"
POLL=180

log() { echo "[fig2xsweep $(date '+%F %T')] $*" | tee -a "$QLOG"; }

# Counts run_sesil.py pythons AND the first queue's bash processes. The string
# concat in the -like pattern stops the query's own powershell.exe command line
# from matching itself; the Name filter keeps it out too.
busy_count() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process | Where-Object { (\$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*run_sesil.py*') -or (\$_.Name -like '*bash*' -and \$_.CommandLine -like ('*queue_fig2_grid' + '.sh*')) }).Count" \
    2>/dev/null | tr -d '\r\n '
}

wait_idle() {
  local n
  n=$(busy_count)
  if [ -n "$n" ] && [ "$n" != "0" ]; then
    log "first queue / run_sesil.py active ($n procs); waiting"
    while [ "$(busy_count)" != "0" ]; do sleep "$POLL"; done
  fi
  log "idle confirmed"
}

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

run_grid_point ./runs_exp2j 3 0.5 "exp2j (3+1.5)"
run_grid_point ./runs_exp2k 9 0.5 "exp2k (9+1.5)"

log "fig2 x-sweep queue finished"
