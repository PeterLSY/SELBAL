#!/usr/bin/env bash
# Wait for the running exp1 (run_sesil.py) to exit, then run the two joint
# baselines back to back so the GPU never sits idle overnight.
#
#   bash tools/queue_baselines.sh
#
# Detection is by process, not by log content: the queue proceeds when no
# python.exe is running run_sesil.py any more, whether exp1 finished cleanly,
# crashed, or was Ctrl+C'd. The exit status of exp1 is reported but does NOT
# block the baselines -- if exp1 died, the GPU is free and the baselines are
# still worth having.
#
# NOTE: it keys on the string "run_sesil.py", so do not launch exp2 while this
# queue is waiting or it will keep waiting for that too.

set -u
cd "$(dirname "$0")/.." || exit 1

LOG_DIR="./runs_baseline"
mkdir -p "$LOG_DIR"
QLOG="$LOG_DIR/queue.log"
POLL=120

log() { echo "[queue $(date '+%F %T')] $*" | tee -a "$QLOG"; }

running_count() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*run_sesil.py*' }).Count" \
    2>/dev/null | tr -d '\r\n '
}

log "queue started; polling every ${POLL}s for run_sesil.py to exit"

n=$(running_count)
if [ -z "$n" ]; then
  log "WARNING: could not query processes; falling back to log-based detection"
  until grep -aqE "^Done\.|^Traceback" ./runs_weak/seed0/permute.log 2>/dev/null; do
    sleep "$POLL"
  done
else
  while [ "$(running_count)" != "0" ]; do
    sleep "$POLL"
  done
fi

log "run_sesil.py no longer running"

if grep -aq "^Done\." ./runs_weak/seed0/permute.log 2>/dev/null; then
  log "exp1 completed normally"
else
  log "WARNING: exp1 log has no 'Done.' line -- it may have crashed or been interrupted"
fi

# Let the GPU settle before claiming it.
sleep 20

# ---- baseline (a): budget-matched to exp1 pretrain -----------------------
log "starting joint baseline: 19 epochs"
python train_joint_baseline.py --epochs 19 --seed 0 --out-root "$LOG_DIR" \
  > "$LOG_DIR/joint_e19.log" 2>&1
rc=$?
log "19-epoch baseline exited rc=$rc"

# ---- baseline (c): budget-matched to exp1 total --------------------------
log "starting joint baseline: 520 epochs (save-every 10)"
python train_joint_baseline.py --epochs 520 --seed 0 --save-every 10 \
  --out-root "$LOG_DIR" > "$LOG_DIR/joint_e520.log" 2>&1
rc=$?
log "520-epoch baseline exited rc=$rc"

# ---- exp1b: half the pretrain budget ------------------------------------
# Fixed 3 epochs per expert, no --target-acc, so every expert gets exactly the
# same training regardless of how easy its class triple is:
#   10 experts x 3 epochs x 15000 samples = 450k = 9.0 full-set epochs
# against exp1's 64 subset-epochs = 19.2 full-set epochs.
mkdir -p ./runs_half/seed0
log "starting exp1b_half_budget (permute, 25 gens, fixed 3 epochs/expert)"
python run_sesil.py --seed 0 --method permute --generations 25 --epochs 3 \
  --out-root ./runs_half > ./runs_half/seed0/permute.log 2>&1
rc=$?
log "exp1b_half_budget exited rc=$rc"

log "queue finished"
