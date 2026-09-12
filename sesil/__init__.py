"""SESiL unified pipeline.

A single entry point (`run_sesil.py`) that runs pretrain + evolution for one
seed. The legacy `training_scripts/evolutionary_*_training.py` scripts are left
untouched and remain the reference implementation.
"""

__all__ = ["config", "paths", "seeding", "pipeline"]
