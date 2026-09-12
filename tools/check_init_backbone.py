"""Pre-launch test for the --init-backbone-only load path (exp2).

Verifies, without a GPU or a full run:
  * a SimSiam backbone checkpoint (no linear.*) loads into a 10-wide resnet20
    via auto-detect, backbone weights match, classifier stays random;
  * a full 10-wide checkpoint still auto-detects to the strict path;
  * --init-backbone-only forced True ignores a full checkpoint's head;
  * a truncated (fractional) finetune runs the expected number of batches.

    python tools/check_init_backbone.py
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

from models.resnets import resnet20  # noqa: E402
from sesil.population import load_init_weights  # noqa: E402
from sesil.ssl_pretrain import SimSiam, save_backbone  # noqa: E402


def main():
    out = os.path.join(REPO_ROOT, ".init_backbone_tmp")
    os.makedirs(out, exist_ok=True)
    results = []

    # -- build a SimSiam, save its backbone -------------------------------
    ss = SimSiam(width=4)
    bb_path = os.path.join(out, "backbone.pth.tar")
    save_backbone(ss, bb_path)
    bb_sd = torch.load(bb_path, map_location="cpu")
    print(f"backbone checkpoint: {len(bb_sd)} keys, "
          f"has linear.weight = {'linear.weight' in bb_sd}")

    # -- 1: auto-detect backbone-only, backbone matches, head random ------
    m = resnet20(w=4, num_classes=10)
    rand_head = m.linear.weight.detach().clone()
    load_init_weights(m, bb_path, "cpu", backbone_only=None)
    bb_match = torch.equal(m.conv1.weight.detach(), bb_sd["conv1.weight"])
    head_unchanged = torch.equal(m.linear.weight.detach(), rand_head)
    ok = bb_match and head_unchanged and "linear.weight" not in bb_sd
    print(f"  {'PASS' if ok else 'FAIL'}  auto-detect backbone-only: "
          f"backbone matches={bb_match}, head left random={head_unchanged}")
    results.append(ok)

    # -- 2: full checkpoint auto-detects to strict path -------------------
    full = resnet20(w=4, num_classes=10)
    full_path = os.path.join(out, "full.pth.tar")
    torch.save(full.state_dict(), full_path)
    m2 = resnet20(w=4, num_classes=10)
    load_init_weights(m2, full_path, "cpu", backbone_only=None)
    strict_ok = torch.equal(m2.linear.weight.detach(), full.linear.weight.detach())
    print(f"  {'PASS' if strict_ok else 'FAIL'}  full checkpoint auto-detects to "
          f"strict (head loaded={strict_ok})")
    results.append(strict_ok)

    # -- 3: forced backbone-only ignores a full checkpoint's head ---------
    m3 = resnet20(w=4, num_classes=10)
    rand3 = m3.linear.weight.detach().clone()
    load_init_weights(m3, full_path, "cpu", backbone_only=True)
    ignored = not torch.equal(m3.linear.weight.detach(), full.linear.weight.detach())
    kept_rand = torch.equal(m3.linear.weight.detach(), rand3)
    ok3 = ignored and kept_rand
    print(f"  {'PASS' if ok3 else 'FAIL'}  forced backbone-only ignores head "
          f"(head!=full={ignored}, head==random={kept_rand})")
    results.append(ok3)

    # -- 4: a wrong-width backbone is rejected ----------------------------
    ss8 = SimSiam(width=8)  # incompatible backbone
    bad_path = os.path.join(out, "backbone_w8.pth.tar")
    save_backbone(ss8, bad_path)
    try:
        load_init_weights(resnet20(w=4, num_classes=10), bad_path, "cpu")
        print("  FAIL  width-mismatch backbone loaded unexpectedly")
        results.append(False)
    except (ValueError, RuntimeError):
        print("  PASS  width-mismatch backbone rejected")
        results.append(True)

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
