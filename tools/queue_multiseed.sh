#!/usr/bin/env bash
# Multi-seed reproduction, queued AFTER the exp2 budget points (exp2a/c/ref).
# Priority order per the plan:
#   1. exp2b seed1, seed2 -- the 0.8538 over-shoot is the headline number and
#      needs error bars before it goes in the report. Run to 40 GENERATIONS (not
#      25): seed0 still climbed +0.010 from gen24->25, so 0.8538 may be a
#      truncation not a plateau. 40 gens gets the error bars AND the true
#      plateau in one run. (seed0's 25-gen run is left untouched as the
#      matched-generation baseline against the other runs.)
#   2. exp1 seed1, seed2  -- weak-init reproduction, lowest priority, 25 gens.
#
#   bash tools/queue_multiseed.sh
#
# Waits for any active run_sesil.py (the exp2-rest queue) to finish first.

set -u
cd "$(dirname "$0")/.." || exit 1

# Absolute path to the conda env python: a detached Start-Process bash does NOT
# inherit the activated conda env, so bare `python` resolves to base (no torch)
# and every run dies instantly at `import torch`. This bit exp2d/seed reps once
# (2026-07-26); pin the interpreter so it never depends on PATH again.
PY="/c/Users/32063/anaconda3/envs/sesil/python.exe"

SSL="runs_ssl/simsiam_e50_seed0"
QLOG="./runs_multiseed.queue.log"
POLL=180

log() { echo "[multiseed $(date '+%F %T')] $*" | tee -a "$QLOG"; }

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

# exp2d: the E-heavy endpoint of the budget ledger. The three points
# E3S2/E6S1/E7.5S0.5 rose monotonically (0.8426 -> 0.8538 -> 0.8581) as budget
# shifted toward SSL pretraining (E up, S down), so the natural close is E=9 with
# minimal finetune S=0.1 (=3 batches). Runs FIRST, before the multi-seed reps.
# Budget = 9 + 3*0.1 = 9.3 units (labelled "9.0+e").
# Skip a completed run. Judged by the FINAL-generation CSV having its 10 rows,
# NOT by the Done. marker: on 2026-07-28 an orphaned seed1 finished all 40 gens
# but exited before writing Done., so a marker-only check re-ran it and appended
# a second pass into the CSVs (gen1-21 got 20/11 rows). The final-CSV check is
# robust to that -- if the last generation is evaluated, the run is done.
done_skip() {  # $1 = permute.log, $2 = final gen CSV, $3 = label
  if grep -aq '^Done\.' "$1" 2>/dev/null \
     || [ "$(grep -ac '^tensor(' "$2" 2>/dev/null || echo 0)" -ge 10 ]; then
    log "$3 already complete, skip"; return 0
  fi
  return 1
}

run_exp2d() {
  local out="./runs_exp2d"
  done_skip "$out/seed0/permute.log" "$out/seed0/permute/csv/gen_25/configurations.csv" "exp2d" && return
  wait_idle
  mkdir -p "$out/seed0"
  log "starting exp2d (E=9 backbone, S=0.1 finetune, 25 gens)"
  "$PY" run_sesil.py --seed 0 --method permute --generations 25 \
    --init-from "$SSL/backbone_e9.pth.tar" --init-backbone-only \
    --sub-epochs 0.1 --init-lr 0.03 --out-root "$out" \
    > "$out/seed0/permute.log" 2>&1
  log "exp2d exited rc=$?"
}

# exp2b at a new seed: SSL backbone is seed-0-trained and shared; only the
# evolution seed varies (population data order, mating draws, mutation). 40 gens
# to reach the true plateau (see header).
run_exp2b_seed() {
  local s=$1 out="./runs_exp2b_seed$1"
  done_skip "$out/seed$s/permute.log" "$out/seed$s/permute/csv/gen_40/configurations.csv" "exp2b seed$s" && return
  wait_idle
  mkdir -p "$out/seed$s"
  log "starting exp2b seed$s (40 generations)"
  "$PY" run_sesil.py --seed "$s" --method permute --generations 40 \
    --init-from "$SSL/backbone_e6.pth.tar" --init-backbone-only \
    --sub-epochs 1 --init-lr 0.03 --out-root "$out" \
    > "$out/seed$s/permute.log" 2>&1
  log "exp2b seed$s exited rc=$?"
}

# exp2c reproduction: E=7.5 backbone, S=0.5 finetune, 40 gens (matches exp2b
# reps so the 40-gen plateau is comparable). Queued between exp2b seed2 and the
# exp1 reps.
run_exp2c_seed() {
  local s=$1 out="./runs_exp2c_seed$1"
  done_skip "$out/seed$s/permute.log" "$out/seed$s/permute/csv/gen_40/configurations.csv" "exp2c seed$s" && return
  wait_idle
  mkdir -p "$out/seed$s"
  log "starting exp2c seed$s (40 generations)"
  "$PY" run_sesil.py --seed "$s" --method permute --generations 40 \
    --init-from "$SSL/backbone_e7.5.pth.tar" --init-backbone-only \
    --sub-epochs 0.5 --init-lr 0.03 --out-root "$out" \
    > "$out/seed$s/permute.log" 2>&1
  log "exp2c seed$s exited rc=$?"
}

# exp2e: E=9 backbone + closed-form ridge head calibration, ZERO SGD. The strict
# S->0 endpoint of the budget ledger. 25 gens (matches exp2d). Queue tail.
run_exp2e() {
  local out="./runs_exp2e"
  done_skip "$out/seed0/permute.log" "$out/seed0/permute/csv/gen_25/configurations.csv" "exp2e" && return
  wait_idle
  mkdir -p "$out/seed0"
  log "starting exp2e (E=9 backbone, closed-form 0-SGD calibration, 25 gens)"
  "$PY" run_sesil.py --seed 0 --method permute --generations 25 \
    --init-from "$SSL/backbone_e9.pth.tar" --init-backbone-only \
    --calibrate closed_form --out-root "$out" \
    > "$out/seed0/permute.log" 2>&1
  log "exp2e exited rc=$?"
}

# exp1 at a new seed: weak supervised pretrain from scratch, target-acc 0.72.
run_exp1_seed() {
  local s=$1 out="./runs_weak_seed$1"
  done_skip "$out/seed$s/permute.log" "$out/seed$s/permute/csv/gen_25/configurations.csv" "exp1 seed$s" && return
  wait_idle
  mkdir -p "$out/seed$s"
  log "starting exp1 seed$s"
  "$PY" run_sesil.py --seed "$s" --method permute --generations 25 \
    --target-acc 0.72 --out-root "$out" \
    > "$out/seed$s/permute.log" 2>&1
  log "exp1 seed$s exited rc=$?"
}

run_exp2d          # budget-ledger endpoint (already done -> skipped)
run_exp2b_seed 1
run_exp2b_seed 2
run_exp2c_seed 1   # exp2c reps: after exp2b seed2, before exp1 reps
run_exp2c_seed 2
run_exp1_seed 1
run_exp1_seed 2
run_exp2e          # closed-form 0-SGD endpoint, queue tail

log "all multi-seed runs finished"
