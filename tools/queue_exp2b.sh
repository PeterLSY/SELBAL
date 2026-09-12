#!/usr/bin/env bash
# Wait for the SSL pretraining (train_ssl.py) to finish, then launch exp2b:
# evolution seeded from the SSL backbone at E=6, finetuned S=1 subset-epoch.
#
#   bash tools/queue_exp2b.sh
#
# Detection is by process (any python running train_ssl.py), so it proceeds
# whether SSL finished cleanly or died. It then checks the specific backbone
# checkpoint exists before starting -- if SSL crashed before epoch 6, exp2b
# cannot start and the queue says so instead of launching a broken run.

set -u
cd "$(dirname "$0")/.." || exit 1

QLOG="./runs_exp2b/queue.log"
mkdir -p ./runs_exp2b/seed0
POLL=120
BACKBONE="runs_ssl/simsiam_e50_seed0/backbone_e6.pth.tar"

log() { echo "[exp2b-queue $(date '+%F %T')] $*" | tee -a "$QLOG"; }

running_count() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*train_ssl.py*' }).Count" \
    2>/dev/null | tr -d '\r\n '
}

log "waiting for train_ssl.py to exit (poll ${POLL}s)"
n=$(running_count)
if [ -z "$n" ]; then
  log "WARNING: cannot query processes; falling back to checkpoint-existence wait"
  until [ -f "runs_ssl/simsiam_e50_seed0/backbone_e50.pth.tar" ]; do sleep "$POLL"; done
else
  while [ "$(running_count)" != "0" ]; do sleep "$POLL"; done
fi
log "train_ssl.py no longer running"

if grep -aq "History:" runs_ssl/simsiam_e50_seed0/ssl.log 2>/dev/null; then
  log "SSL completed normally"
else
  log "WARNING: SSL log has no 'History:' line -- may have crashed"
fi

if [ ! -f "$BACKBONE" ]; then
  log "ABORT: $BACKBONE not found; SSL did not reach epoch 6. exp2b not started."
  exit 1
fi

sleep 20  # let the GPU settle

# exp2b: E=6 (backbone) + S=1 (finetune) = 9.0 units, matched to exp1b.
# init-lr 0.03: the SSL backbone is good but the classifier is random, so it
# needs a moderate LR to learn the 3-class head without wrecking features.
# (Tunable -- see experiments.md.)
log "starting exp2b_ssl_E6S1 (permute, 25 gens)"
python run_sesil.py --seed 0 --method permute --generations 25 \
  --init-from "$BACKBONE" --init-backbone-only --sub-epochs 1 --init-lr 0.03 \
  --out-root ./runs_exp2b > ./runs_exp2b/seed0/permute.log 2>&1
rc=$?
log "exp2b exited rc=$rc"
log "queue finished"
