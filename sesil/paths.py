"""Every output path for a run, derived from (out_root, seed, method).

The legacy scripts scattered hardcoded paths through their `__main__` blocks and
none of them included the seed, so two seeds would overwrite each other
(evolutionary_wavg_training.py:878 and :925). Everything is centralised here and
namespaced by seed.

Layout::

    runs/seed0/
      population/<hash>/resnet20x4_v0.pth.tar   <- shared by every --method
      wavg/gen_1/<hash>/resnet20x4_v0.pth.tar
      wavg/csv/gen_1/configurations.csv
      wavg/run.json
"""

import os

# Directories the pipeline must never write into. Existing experiment output and
# the reference population live here.
PROTECTED = (
    os.path.normpath("./checkpoints/cifar10_evolution/initial"),
    os.path.normpath("./checkpoints/cifar10_evolution/wavg"),
    os.path.normpath("./checkpoints/cifar10_evolution/permute"),
    os.path.normpath("./checkpoints/cifar10_evolution/zipit"),
    os.path.normpath("./csvs"),
)


class RunPaths:
    def __init__(self, out_root, seed, method, model_width=4):
        self.out_root = os.path.normpath(out_root)
        self.seed = seed
        self.method = method
        self.model_width = model_width
        self.seed_dir = os.path.join(self.out_root, f"seed{seed}")
        # Pretrained population sits at the seed level, not under the method, so
        # `--method permute` reuses the weights `--method wavg` already trained.
        self.population_dir = os.path.join(self.seed_dir, "population")
        self.method_dir = os.path.join(self.seed_dir, method)
        self.csv_dir = os.path.join(self.method_dir, "csv")
        self.run_json = os.path.join(self.method_dir, "run.json")
        self.summary_csv = os.path.join(self.method_dir, "summary.csv")
        self._assert_not_protected()

    # -- guards ----------------------------------------------------------
    def _assert_not_protected(self):
        target = os.path.abspath(self.seed_dir)
        for prot in PROTECTED:
            p = os.path.abspath(prot)
            if target == p or target.startswith(p + os.sep):
                raise ValueError(
                    f"refusing to run: output dir {target} is inside protected "
                    f"path {p}. Pick a different --out-root."
                )

    # -- population ------------------------------------------------------
    def member_dir(self, hash_id):
        return os.path.join(self.population_dir, hash_id)

    def member_ckpt(self, hash_id):
        return os.path.join(
            self.member_dir(hash_id), f"resnet20x{self.model_width}_v0.pth.tar"
        )

    # -- evolution -------------------------------------------------------
    def generation_dir(self, gen):
        return os.path.join(self.method_dir, f"gen_{gen}")

    def offspring_dir(self, gen, hash_id):
        return os.path.join(self.generation_dir(gen), hash_id)

    def csv_for(self, gen):
        return os.path.join(self.csv_dir, f"gen_{gen}", "configurations.csv")

    def makedirs(self):
        os.makedirs(self.population_dir, exist_ok=True)
        os.makedirs(self.method_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)

    def __repr__(self):
        return f"<RunPaths seed={self.seed} method={self.method} root={self.seed_dir}>"
