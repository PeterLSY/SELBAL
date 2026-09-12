#!/usr/bin/env bash
# Run the remaining exp2 budget points serially: exp2a -> exp2c -> exp2ref.
# All three read the already-trained SSL trajectory (runs_ssl/simsiam_e50_seed0)
# at different epochs, so no SSL retraining. ~12h each, ~36h total.
#
#   bash tools/queue_exp2_rest.sh
#
# Budget accounting (E full-set SSL epochs + 3S full-set-equiv finetune):
#   exp2a : E=3   S=2    = 3 + 6   = 9.0   backbone_e3
#   exp2c : E=7.5 S=0.5  = 7.5+1.5 = 9.0   backbone_e7.5 (mid-epoch, exact)
#   exp2ref: E=50 S=1    = 50 + 3  = 53    backbone_e50 (over-budget reference)

set -u
cd "$(dirname "$0")/.." || exit 1

SSL="runs_ssl/simsiam_e50_seed0"
QLOG="./runs_exp2_rest.queue.log"
POLL=120

log() { echo "[exp2rest $(date '+%F %T')] $*" | tee -a "$QLOG"; }

# If anything is still running run_sesil.py (e.g. a stray exp2b), wait it out so
# we don't double-book the GPU.
running_sesil() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*run_sesil.py*' }).Count" \
    2>/dev/null | tr -d '\r\n '
}

n=$(running_sesil)
if [ -n "$n" ] && [ "$n" != "0" ]; then
  log "another run_sesil.py is active; waiting for it to finish first"
  while [ "$(running_sesil)" != "0" ]; do sleep "$POLL"; done
fi

run_point() {
  local name=$1 E=$2 S=$3 outroot=$4
  local bb="$SSL/backbone_e${E}.pth.tar"
  if [ ! -f "$bb" ]; then
    log "ABORT $name: backbone $bb missing"
    return 1
  fi
  mkdir -p "$outroot/seed0"
  log "starting $name (E=$E backbone, S=$S finetune)"
  python run_sesil.py --seed 0 --method permute --generations 25 \
    --init-from "$bb" --init-backbone-only --sub-epochs "$S" --init-lr 0.03 \
    --out-root "$outroot" > "$outroot/seed0/permute.log" 2>&1
  log "$name exited rc=$?"
}

run_point exp2a   3    2   ./runs_exp2a
run_point exp2c   7.5  0.5 ./runs_exp2c
run_point exp2ref 50   1   ./runs_exp2ref

log "all exp2 budget points finished"
