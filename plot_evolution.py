"""
Plot SESiL population accuracy vs. generation (Figure-2a style).

Handles the actual CSV format produced by evolutionary_permute_training.py,
which mixes two row schemas in one file:
  - population rows:  tensor(J),tensor(PTA),name,[per-class list with commas],...
  - merge rows:       J,PTA,[numpy array no commas],TaskA,TaskB,SplitA,SplitB,...

Usage (from the SESiL repo root):
    python plot_evolution.py                     # auto-finds csvs/*/permute/*/
    python plot_evolution.py --method permute    # or wavg, zipit...
    python plot_evolution.py --root ./csvs       # custom csv root

Outputs: evolution_curve_<method>.png (300 dpi) and a summary table to stdout.
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TENSOR_RE = re.compile(r"tensor\(([-\d.eE]+)\)")
FLOAT_RE = re.compile(r"^-?\d*\.?\d+(?:[eE][-+]?\d+)?$")


def parse_csv(path):
    """Return (population_rows, merge_rows) for one generation's CSV.

    population row -> dict(joint, per_task_avg, model_name, n_classes)
    merge row      -> dict(joint, per_task_avg, task_a, task_b, split_a, split_b)
    """
    pop, merges = [], []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith("joint"):
                continue  # header / blank

            if line.startswith("tensor("):
                # Population row. The Per Class list contains commas, so we
                # extract by regex instead of naive splitting.
                tensors = TENSOR_RE.findall(line)
                if len(tensors) < 2:
                    continue
                joint, pta = float(tensors[0]), float(tensors[1])
                # model name = 3rd comma field (safe: first two fields are
                # tensor(...) with no internal commas)
                parts = line.split(",")
                name = parts[2].strip()
                n_classes = len(set(name.split("_"))) if "_" in name else None
                pop.append({
                    "joint": joint,
                    "per_task_avg": pta,
                    "model_name": name,
                    "n_classes": n_classes,
                })
            else:
                # Merge row. The per-class numpy array has NO commas, so a
                # plain split works. Guard: first field must be a float.
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 7 or not FLOAT_RE.match(parts[0]):
                    continue
                try:
                    merges.append({
                        "joint": float(parts[0]),
                        "per_task_avg": float(parts[1]),
                        "task_a": float(parts[3]),
                        "task_b": float(parts[4]),
                        "split_a": parts[5],
                        "split_b": parts[6],
                    })
                except ValueError:
                    continue
    return pop, merges


def find_generation_csvs(root, method):
    """Map generation number -> csv path, across all date folders."""
    pattern = os.path.join(root, "*", method, "*", "evolutionary_*_configurations.csv")
    gen_map = {}
    for path in glob.glob(pattern):
        gen_dir = os.path.basename(os.path.dirname(path))
        try:
            gen = int(gen_dir)
        except ValueError:
            continue
        # If the same generation appears under two dates, keep the newer file.
        if gen not in gen_map or os.path.getmtime(path) > os.path.getmtime(gen_map[gen]):
            gen_map[gen] = path
    return dict(sorted(gen_map.items()))


def _keep_newer(gen_map, gen, path):
    if gen not in gen_map or os.path.getmtime(path) > os.path.getmtime(gen_map[gen]):
        gen_map[gen] = path


def find_run_csvs(path, method=None):
    """Generation -> csv path for either layout.

    New layout (run_sesil.py):   <path>/csv/gen_<N>/configurations.csv
    Legacy layout (csvs root):   <path>/<date>/<method>/<N>/evolutionary_*.csv

    Tries the new layout first, then the legacy one, so a spec can point at
    either a run directory or the csvs root.
    """
    gen_map = {}

    # -- new layout ------------------------------------------------------
    for p in glob.glob(os.path.join(path, "csv", "gen_*", "configurations.csv")):
        name = os.path.basename(os.path.dirname(p))
        try:
            gen = int(name.split("_", 1)[1])
        except (ValueError, IndexError):
            continue
        _keep_newer(gen_map, gen, p)
    if gen_map:
        return dict(sorted(gen_map.items()))

    # -- legacy layout ---------------------------------------------------
    m = method or "*"
    pattern = os.path.join(path, "*", m, "*", "evolutionary_*_configurations.csv")
    for p in glob.glob(pattern):
        try:
            gen = int(os.path.basename(os.path.dirname(p)))
        except ValueError:
            continue
        _keep_newer(gen_map, gen, p)
    return dict(sorted(gen_map.items()))


def parse_spec(spec):
    """Parse a --compare spec: 'path[@method]=label'.

    '=' separates path from label ('=' cannot appear in a Windows path, while
    ':' can, so it is the safer delimiter). '@method' is only needed for the
    legacy csvs layout, where the method folder sits below a date folder.
    """
    if "=" not in spec:
        raise ValueError(
            f"bad --compare spec {spec!r}; expected 'path[@method]=label'"
        )
    src, label = spec.split("=", 1)
    method = None
    if "@" in src:
        src, method = src.rsplit("@", 1)
    return src.strip(), method, label.strip()


def series_for(path, method=None):
    """Per-generation population mean/min/max of Joint for one run."""
    gen_map = find_run_csvs(path, method)
    gens, mean, lo, hi = [], [], [], []
    for gen, csv_path in gen_map.items():
        pop, _ = parse_csv(csv_path)
        if not pop:
            continue
        joints = np.array([r["joint"] for r in pop])
        gens.append(gen)
        mean.append(joints.mean())
        lo.append(joints.min())
        hi.append(joints.max())
    return np.array(gens), np.array(mean), np.array(lo), np.array(hi)


def _budget_map(args):
    """Parse --budget 'start,step' entries, one per --compare spec.

    Returns a list of (start, step) or None if --budget was not given. When
    present, the x axis of curve i becomes start + step*(gen-1) -- gen 1 sits
    at the pretraining cost (start), and each generation adds `step` full-set
    equivalent epochs (see the budget-axis table in experiments.md).
    """
    if not args.budget:
        return None
    if len(args.budget) != len(args.compare):
        sys.exit(
            f"--budget needs one 'start,step' per --compare spec: "
            f"{len(args.budget)} given for {len(args.compare)} curves."
        )
    out = []
    for b in args.budget:
        try:
            start, step = (float(x) for x in b.split(","))
        except ValueError:
            sys.exit(f"bad --budget entry {b!r}; expected 'start,step'")
        out.append((start, step))
    return out


def run_compare(args):
    """Overlay the population-mean curve of several runs."""
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    budget = _budget_map(args)

    plotted = 0
    xhead = "Start.u" if budget else "Gens"
    print(f"{'Label':<34} | {xhead:>7} | {'First':>7} | {'Last':>7} | {'Best':>7}")
    print("-" * 78)
    for i, spec in enumerate(args.compare):
        src, method, label = parse_spec(spec)
        gens, mean, lo, hi = series_for(src, method)
        if len(gens) == 0:
            print(f"{label:<34} |  (no data found under {src})")
            continue
        if budget:
            start, step = budget[i]
            x = start + step * (gens - 1)
        else:
            x = gens
        color = colors[i % len(colors)]
        if args.band:
            ax.fill_between(x, lo, hi, alpha=0.12, color=color)
        ax.plot(x, mean, "o-", color=color, lw=2, ms=4, label=label)
        headval = f"{budget[i][0]:.1f}" if budget else f"{len(gens)}"
        print(
            f"{label:<34} | {headval:>7} | {mean[0]:>7.4f} | {mean[-1]:>7.4f} | "
            f"{mean.max():>7.4f}"
        )
        plotted += 1

    if not plotted:
        sys.exit("No runs produced any data - check the --compare paths.")

    for value, hlabel in args.hline or []:
        ax.axhline(float(value), ls="--", lw=1.3, color="0.35")
        ax.text(
            0.995, float(value) + 0.012, hlabel, transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8, color="0.25",
        )

    ax.set_xlabel("Compute (full-set equivalent epochs)" if budget else "Generation")
    ax.set_ylabel("CIFAR-10 accuracy (population mean, Joint)")
    ax.set_title(args.title or "SESiL evolution - run comparison")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    out = args.out or "evolution_compare.png"
    fig.savefig(out, dpi=300)
    print(f"\nSaved: {out}")


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
compare mode:
  python plot_evolution.py --compare \\
      "csvs@permute=permute (strong init 0.93)" \\
      "runs_weak/seed0/permute=permute (weak init 0.75)" \\
      --hline 0.92 "joint training" --out compare.png

  Each spec is 'path[@method]=label'. The path may be a run directory from
  run_sesil.py (<run>/csv/gen_N/) or the legacy csvs root (<root>/<date>/
  <method>/<N>/), in which case @method selects the method folder.
""",
    )
    ap.add_argument("--root", default="./csvs", help="csv root directory")
    ap.add_argument("--method", default="permute", help="merging method folder name")
    ap.add_argument("--out", default=None, help="output image path")
    ap.add_argument(
        "--compare", nargs="+", metavar="PATH[@METHOD]=LABEL",
        help="overlay population-mean curves of several runs",
    )
    ap.add_argument(
        "--budget", nargs="+", metavar="START,STEP",
        help="one per --compare spec; put curve i on a compute x-axis at "
             "start+step*(gen-1) full-set-equiv epochs (e.g. 19.2,20 for exp1)",
    )
    ap.add_argument(
        "--hline", nargs=2, action="append", metavar=("VALUE", "LABEL"),
        help="horizontal reference line (repeatable), e.g. --hline 0.92 'joint training'",
    )
    ap.add_argument("--band", action="store_true",
                    help="compare mode: also shade each run's population min-max")
    ap.add_argument("--title", default=None, help="compare mode: plot title")
    args = ap.parse_args()

    if args.compare:
        return run_compare(args)

    gen_map = find_generation_csvs(args.root, args.method)
    if not gen_map:
        sys.exit(f"No CSVs found under {args.root}/*/{args.method}/*/ - check paths.")

    gens, pop_mean, pop_min, pop_max = [], [], [], []
    pta_mean, pta_min, pta_max = [], [], []
    merge_mean, merge_min, merge_max = [], [], []
    print(f"{'Gen':>4} | {'#pop':>4} | {'Joint mean':>10} | {'Joint min':>9} | "
          f"{'Joint max':>9} | {'PTA mean':>8} | {'merged Joint':>12}")
    print("-" * 76)

    for gen, path in gen_map.items():
        pop, merges = parse_csv(path)
        if not pop:
            print(f"{gen:>4} | (no population rows parsed - skipped: {path})")
            continue
        joints = np.array([r["joint"] for r in pop])
        ptas = np.array([r["per_task_avg"] for r in pop])
        mjoints = np.array([m["joint"] for m in merges]) if merges else np.array([np.nan])

        gens.append(gen)
        pop_mean.append(joints.mean())
        pop_min.append(joints.min())
        pop_max.append(joints.max())
        pta_mean.append(ptas.mean())
        pta_min.append(ptas.min())
        pta_max.append(ptas.max())
        merge_mean.append(np.nanmean(mjoints))
        merge_min.append(np.nanmin(mjoints))
        merge_max.append(np.nanmax(mjoints))

        print(f"{gen:>4} | {len(pop):>4} | {joints.mean():>10.4f} | {joints.min():>9.4f} | "
              f"{joints.max():>9.4f} | {ptas.mean():>8.4f} | {np.nanmean(mjoints):>12.4f}")

    if not gens:
        sys.exit("No usable generations parsed.")

    gens = np.array(gens)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    ax.fill_between(gens, pop_min, pop_max, alpha=0.18, color="tab:blue",
                    label="population min-max")
    ax.plot(gens, pop_mean, "o-", color="tab:blue", lw=2,
            label="population mean (Joint, all 10 classes)")
    ax.fill_between(gens, pta_min, pta_max, alpha=0.12, color="tab:green")
    ax.plot(gens, pta_mean, "s--", color="tab:green", lw=1.5, ms=4,
            label="population mean (Per-Task Avg, known classes)")
    if not all(np.isnan(merge_mean)):
        ax.fill_between(gens, merge_min, merge_max, alpha=0.12, color="tab:orange")
        ax.plot(gens, merge_mean, "^:", color="tab:orange", lw=1.5, ms=5,
                label="merged offspring mean (pre-mutation)")

    for value, hlabel in args.hline or []:
        ax.axhline(float(value), ls="--", lw=1.3, color="0.35")
        ax.text(
            0.995, float(value) + 0.012, hlabel, transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8, color="0.25",
        )

    ax.set_xlabel("Generation")
    ax.set_ylabel("CIFAR-10 accuracy")
    ax.set_title(f"SESiL evolution - {args.method} merging (resnet20x4)")
    ax.set_xticks(gens)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    out = args.out or f"evolution_curve_{args.method}.png"
    fig.savefig(out, dpi=300)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()