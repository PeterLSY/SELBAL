#!/usr/bin/env bash
# Wait for SSL pretraining to finish, then linear-probe every checkpoint.
# Runs alongside exp2b (both wait on train_ssl.py); the probe is light --
# frozen backbone, 1-epoch linear head, ~3 min for all eight checkpoints.
#
#   bash tools/queue_probe.sh

set -u
cd "$(dirname "$0")/.." || exit 1

RUN="runs_ssl/simsiam_e50_seed0"
QLOG="$RUN/probe_queue.log"
POLL=120

log() { echo "[probe-queue $(date '+%F %T')] $*" | tee -a "$QLOG"; }

running_count() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*train_ssl.py*' }).Count" \
    2>/dev/null | tr -d '\r\n '
}

log "waiting for train_ssl.py to exit (poll ${POLL}s)"
n=$(running_count)
if [ -z "$n" ]; then
  until [ -f "$RUN/backbone_e50.pth.tar" ]; do sleep "$POLL"; done
else
  while [ "$(running_count)" != "0" ]; do sleep "$POLL"; done
fi
log "train_ssl.py no longer running"

if [ ! -f "$RUN/backbone_e50.pth.tar" ]; then
  log "WARNING: backbone_e50 missing; probing whatever checkpoints exist"
fi

sleep 10
log "starting linear probe over e{1,3,6,7.5,10,20,30,50}"
python tools/linear_probe.py --run "$RUN" > "$RUN/probe.log" 2>&1
rc=$?
log "linear probe exited rc=$rc"
log "queue finished"
