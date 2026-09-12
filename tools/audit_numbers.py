"""Audit: every documented number must be recomputable from CSV / log.

Guards against expected/predicted values leaking into experiments.md,
weekly_cheatsheet.md, PROJECT_LOG.md as if they were real results. Run after
editing any doc.

Checks:
  1. gen25 of every completed single-seed run == the value written in the docs.
  2. expert start-point means == the pretrain-log 'mean' rows.
  3. no bare multi-seed number (spread / plateau) appears OUTSIDE an explicit
     "do not cite / pending / does not exist" honesty section.

Exit 0 if clean, 1 if any mismatch or stray number.

    python tools/audit_numbers.py
"""

import os
import re
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from plot_evolution import find_run_csvs, find_generation_csvs, parse_csv  # noqa: E402

# (name, kind, arg, documented gen25). Update the last field only from CSV.
GEN25 = [
    ("exp1", "run", "runs_weak/seed0/permute", 0.8048),
    ("exp1b", "run", "runs_half/seed0/permute", 0.8035),
    ("exp2a", "run", "runs_exp2a/seed0/permute", 0.8426),
    ("exp2b", "run", "runs_exp2b/seed0/permute", 0.8538),
    ("exp2c", "run", "runs_exp2c/seed0/permute", 0.8581),
    ("exp2d", "run", "runs_exp2d/seed0/permute", 0.8722),
    ("exp2ref", "run", "runs_exp2ref/seed0/permute", 0.8768),
    ("strong", "hist", "permute", 0.8404),
]

# (run, documented start-point mean) -- from the pretrain-summary 'mean' row.
STARTPOINTS = {
    "exp2a": 0.7663, "exp2b": 0.7312, "exp2c": 0.6273,
    "exp2d": 0.5571, "exp2ref": 0.8260,
}

DOCS = ["experiments.md", "weekly_cheatsheet.md", "PROJECT_LOG.md"]
DOC_DIRS = ["", "figures/"]

# Multi-seed numbers that must never appear as if real. Any occurrence must be
# within a line/section flagged by one of HONESTY_MARKERS.
STRAY_PATTERNS = [r"0\.8742", r"±0\.00\d", r"±0\.01\d"]
HONESTY_MARKERS = ["pending", "does not exist", "unverified", "must not", "prohibit",
                   "do not cite", "no source", "never", "reproduc", "boundary"]


def gen25_of(kind, arg):
    d = find_generation_csvs("./csvs", arg) if kind == "hist" else find_run_csvs(arg)
    if not d:
        return None, None
    g = max(d)
    return np.array([r["joint"] for r in parse_csv(d[g])[0]]).mean(), g


def start_mean_from_log(run):
    log = f"runs_{run}/seed0/permute.log"
    if not os.path.exists(log):
        return None
    with open(log, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    for i, ln in enumerate(lines):
        if "mean" in ln and "|" in ln:
            m = re.search(r"0\.\d{3,4}", ln)
            if m and any("Hash" in lines[j] for j in range(max(0, i - 13), i)):
                return float(m.group())
    return None


def find_doc(name):
    for d in DOC_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def main():
    fails = []

    print("== 1. gen25 (CSV vs doc) ==")
    for name, kind, arg, doc in GEN25:
        v, g = gen25_of(kind, arg)
        if v is None:
            print(f"  {name:>8}: NO CSV")
            continue
        ok = abs(v - doc) < 1e-4
        print(f"  {name:>8}: CSV={v:.4f}(g{g}) doc={doc} {'OK' if ok else 'MISMATCH'}")
        if not ok:
            fails.append(f"gen25 {name}: CSV {v:.4f} != doc {doc}")

    print("== 2. start-point means (log vs doc) ==")
    for run, doc in STARTPOINTS.items():
        v = start_mean_from_log(run)
        ok = v is not None and abs(v - doc) < 1e-4
        print(f"  {run:>8}: log={v} doc={doc} {'OK' if ok else 'MISMATCH/MISSING'}")
        if not ok:
            fails.append(f"start {run}: log {v} != doc {doc}")

    print("== 3. stray multi-seed numbers outside honesty sections ==")
    stray = 0
    for name in DOCS:
        p = find_doc(name)
        if not p:
            continue
        # Section-aware: a multi-seed number is fine if it sits under a heading
        # (or immediately after an intro line) that carries an honesty marker.
        # We track whether the current markdown section is a "honesty" section.
        in_honesty = False
        for lineno, line in enumerate(open(p, encoding="utf-8", errors="replace"), 1):
            if line.lstrip().startswith("#"):
                in_honesty = any(mk in line for mk in HONESTY_MARKERS)
            line_flagged = any(mk in line for mk in HONESTY_MARKERS)
            for pat in STRAY_PATTERNS:
                if re.search(pat, line) and not in_honesty and not line_flagged:
                    print(f"  STRAY {p}:{lineno}: {line.strip()[:70]}")
                    fails.append(f"stray {p}:{lineno}")
                    stray += 1
    if stray == 0:
        print("  none (all multi-seed numbers are inside honesty/pending sections)")

    print()
    if fails:
        print(f"AUDIT FAILED: {len(fails)} issue(s)")
        for f in fails:
            print("  -", f)
        return 1
    print("AUDIT CLEAN: every documented number recomputes from CSV/log.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
