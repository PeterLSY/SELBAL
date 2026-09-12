"""Run configuration for the unified SESiL pipeline.

Every value that the legacy scripts hardcoded in their `__main__` block lives
here instead, so a whole run is described by one object and one CLI call.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Which matching function each --method selects. The legacy scripts passed these
# strings literally to run_auxiliary_experiment():
#   evolutionary_wavg_training.py:903     -> 'match_tensors_identity'
#   evolutionary_permute_training.py:928  -> 'match_tensors_permute'
# 'zipit' is wired up because it is the same code path, but it is NOT covered by
# the equivalence check in tools/check_equivalence.py.
METHOD_MERGING_FN: Dict[str, str] = {
    "wavg": "match_tensors_identity",
    "permute": "match_tensors_permute",
    "zipit": "match_tensors_zipit",
}

VERIFIED_METHODS = ("wavg", "permute")


@dataclass
class RunConfig:
    """One SESiL run: pretrain a population, then evolve it."""

    # --- identity -------------------------------------------------------
    seed: int = 0
    method: str = "wavg"

    # --- dataset --------------------------------------------------------
    # Key into sesil.data.DATASETS. Decides the data, the head width, the
    # full-set-equivalent epoch, and (via the spec's evo_name) which
    # datasets/configs.py entry the evolution stage loads.
    dataset: str = "cifar10"
    # Population shape. The reference population fixes these for CIFAR-10
    # (10 agents x 3 classes); they only take effect when no reference
    # population exists for this dataset and assignments are drawn instead.
    n_agents: int = 10
    classes_per_agent: int = 3

    # Optimizer steps per FULL-SET-equivalent epoch, for every SGD stage of the
    # run (pretrain finetune and mutation). The epoch is the x-axis unit of the
    # figures, so it is defined by update count rather than by a pass over the
    # data -- otherwise a stage's updates per plotted epoch would depend on its
    # dataset size and batch size. A subset loader gets its proportional share
    # (15000/50000 -> 0.3x), so the y = 3S budget ledger is unchanged.
    # 0 = legacy, one full pass per epoch (100 updates full-set / 30 subset).
    updates_per_epoch: int = 20

    # --- evolution ------------------------------------------------------
    generations: int = 15
    start_from: int = 0
    config_name: str = "cifar_evolution_resnet20"
    model_width: int = 4

    # Merging node config. Legacy value: evolutionary_wavg_training.py:866
    #   {'stop_node': 21, 'params': {'a': .0001, 'b': .075}}
    stop_node: Optional[int] = 21
    merge_a: float = 0.0001
    merge_b: float = 0.075

    # --- mating ---------------------------------------------------------
    # build_score_matrix(..., mode="threshold", threshold=0.5) at wavg:369
    mate_mode: str = "threshold"
    mate_threshold: float = 0.5
    mate_weight_extra: float = 1.0
    mate_weight_common: float = 0.1
    mate_max_retries: int = 100

    # --- mutation (finetune of merged offspring) ------------------------
    # finetune_merged_model(..., epochs=2) default at wavg:823
    mutation_epochs: int = 2
    mutation_lr: float = 0.001

    # --- pretrain -------------------------------------------------------
    pretrain_epochs: int = 60
    pretrain_lr: float = 0.1
    pretrain_momentum: float = 0.9
    pretrain_weight_decay: float = 5e-4
    pretrain_batch_size: int = 500
    pretrain_num_workers: int = 0
    target_acc: Optional[float] = None
    force_pretrain: bool = False

    # Shared-pretraining entry point (exp2): initialise every expert from one
    # checkpoint instead of from scratch, then finetune it to target_acc.
    # Must be a 10-wide-head raw state_dict -- e.g. any epoch_*.pth.tar from
    # train_joint_baseline.py.
    init_from: Optional[str] = None
    # Separate LR for that finetune. Reusing pretrain_lr (0.1) on already
    # converged weights destroys the features, so this defaults much lower.
    init_lr: float = 0.01
    # exp2: the --init-from checkpoint is a bare backbone (no linear.*). Load the
    # backbone via strict=False and leave each expert's classifier random.
    # None => auto-detect from the checkpoint; True/False => force.
    init_backbone_only: Optional[bool] = None
    # exp2: fractional finetune length in SUBSET epochs (S). When set, the
    # finetune runs round(S * batches_per_epoch) batches instead of an integer
    # number of epochs. Only consulted on the --init-from path; the integer
    # pretrain_epochs path (exp1/exp1b) is untouched.
    finetune_sub_epochs: Optional[float] = None
    # exp2e: closed-form head calibration with ZERO SGD. 'closed_form' fills the
    # classifier rows for the subset classes by ridge least-squares on frozen
    # backbone features. The strict S->0 endpoint of the budget ledger. Only on
    # the --init-from backbone path; takes precedence over finetune_sub_epochs.
    calibrate: Optional[str] = None
    calibrate_ridge: float = 1e-2

    # --- io -------------------------------------------------------------
    out_root: str = "./runs"
    # Source of the 10 class assignments. Read-only, never written to.
    reference_population: str = "./checkpoints/cifar10_evolution/initial"
    mapping_file: str = "mapping.json"
    device: str = "cuda"

    # --- misc -----------------------------------------------------------
    skip_evolution: bool = False
    skip_pretrain: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.method not in METHOD_MERGING_FN:
            raise ValueError(
                f"unknown --method {self.method!r}; "
                f"choose from {sorted(METHOD_MERGING_FN)}"
            )
        self.spec  # resolve now, so a bad --dataset fails before any training

    @property
    def spec(self):
        """The resolved DatasetSpec for this run."""
        from .data import get_spec

        return get_spec(self.dataset)

    @property
    def num_classes(self) -> int:
        """Head width: every model spans the dataset's whole label space."""
        return self.spec.num_classes

    @property
    def merging_fn(self) -> str:
        return METHOD_MERGING_FN[self.method]

    @property
    def node_config(self) -> Dict[str, Any]:
        """The dict the legacy code called `node_config`."""
        return {
            "stop_node": self.stop_node,
            "params": {"a": self.merge_a, "b": self.merge_b},
        }

    def describe(self) -> str:
        lines = [
            f"seed             : {self.seed}",
            f"dataset          : {self.dataset} ({self.num_classes} classes, "
            f"{self.n_agents} agents x {self.classes_per_agent})",
            f"method           : {self.method} -> {self.merging_fn}",
            f"generations      : {self.generations} (start_from={self.start_from})",
            f"pretrain epochs  : {self.pretrain_epochs}"
            + (f"  (early stop at acc>={self.target_acc})" if self.target_acc else ""),
            f"pretrain init    : "
            + (f"{self.init_from}  (lr={self.init_lr})" if self.init_from else "from scratch"),
            f"mutation epochs  : {self.mutation_epochs}",
            f"updates/epoch    : {self.updates_per_epoch or 'full pass'}",
            f"merge node       : stop_node={self.stop_node} a={self.merge_a} b={self.merge_b}",
            f"device           : {self.device}",
        ]
        return "\n".join("  " + ln for ln in lines)
