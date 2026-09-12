"""GLOBA merge operator for SESiL -- implements docs/globa_operator_spec.md.

Standalone: nothing in sesil/ imports this package and this package imports
nothing from sesil/. It is a pure function on state dicts.
"""

from .operator import (
    MergeConfig,
    PRESETS,
    TYPES,
    classify,
    decompose,
    merge,
    prune,
)

__all__ = ["MergeConfig", "PRESETS", "TYPES", "classify", "decompose", "merge", "prune"]
