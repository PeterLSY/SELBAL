"""Safety net around the global mapping.json.

Facts established by inspection:

* `mapping.json` is a SINGLE GLOBAL file, not one per population directory.
  `utils.encode_labels`/`utils.decode_labels` default to the bare relative path
  ``"mapping.json"`` (utils.py:1139, utils.py:1168), so it resolves against the
  process CWD. Every caller in the repo uses that default, which is why the
  pipeline chdirs to the repo root before doing anything.

* Append logic already exists and is correct: `encode_labels` reads the whole
  file, sets exactly one key (``mapping[hash_id] = labels``, utils.py:1160) and
  writes it back (utils.py:1162-1163). Other entries survive. Because the hash
  is derived from the label list itself, re-encoding an existing assignment
  rewrites an identical value -- idempotent, no conflict.

Two real hazards remain, which this module handles without touching utils.py:

1. The write is NOT atomic -- `json.dump` truncates the real file in place. A
   crash mid-write destroys all 552 entries.
2. `FileNotFoundError` is swallowed (utils.py:1156-1157). Run from the wrong
   CWD and you silently get a brand new 1-entry mapping.json in the wrong place.

`mapping_guard` snapshots the file, verifies after the fact that the mapping
only ever grew, and restores from the snapshot if it did not.
"""

import json
import os
import shutil
from contextlib import contextmanager


class MappingCorruption(RuntimeError):
    pass


def load_mapping(mapping_file="mapping.json"):
    with open(mapping_file, "r") as f:
        return json.load(f)


@contextmanager
def mapping_guard(mapping_file="mapping.json", required=True):
    """Guarantee mapping.json only gains entries inside this block.

    Args:
        mapping_file: path to the mapping.
        required: if True, the file must already exist. This is the guard
            against a wrong-CWD run silently creating a fresh mapping.
    """
    mapping_file = os.path.abspath(mapping_file)
    if not os.path.exists(mapping_file):
        if required:
            raise FileNotFoundError(
                f"{mapping_file} not found. utils.encode_labels would silently "
                f"create a new empty mapping here (utils.py:1156-1157) and every "
                f"decode_labels lookup would then fail. Run from the repo root."
            )
        before = {}
    else:
        before = load_mapping(mapping_file)

    backup = mapping_file + ".bak"
    if before:
        shutil.copy2(mapping_file, backup)

    try:
        yield before
    finally:
        after = load_mapping(mapping_file) if os.path.exists(mapping_file) else {}

        lost = [k for k in before if k not in after]
        changed = [k for k in before if k in after and after[k] != before[k]]

        if lost or changed:
            if os.path.exists(backup):
                shutil.copy2(backup, mapping_file)
            raise MappingCorruption(
                f"mapping.json lost {len(lost)} entries and altered {len(changed)}; "
                f"restored from {backup}. lost={lost[:5]} changed={changed[:5]}"
            )

        added = [k for k in after if k not in before]
        if added:
            print(f"[mapping] appended {len(added)} new entries: {added[:10]}")
        if os.path.exists(backup):
            os.remove(backup)


def assignments_from_dir(population_dir, mapping_file="mapping.json"):
    """Read the class assignment of every model folder in `population_dir`.

    Returns a list of (hash_id, [class ids]) sorted by hash_id so the order is
    stable across machines -- `os.listdir` order is not guaranteed, and the
    order determines population_info order, which feeds the mating draws.
    """
    mapping = load_mapping(mapping_file)
    out = []
    for name in sorted(os.listdir(population_dir)):
        if not os.path.isdir(os.path.join(population_dir, name)):
            continue
        if name not in mapping:
            raise KeyError(
                f"folder {name!r} under {population_dir} has no entry in "
                f"{mapping_file}; cannot recover its class assignment."
            )
        out.append((name, [int(c) for c in mapping[name]]))
    return out
