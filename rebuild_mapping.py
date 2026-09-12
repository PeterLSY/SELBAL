import hashlib, json, os
from itertools import permutations

# 1. Collect all hash-folder names from every initial population
targets = set()
for root, dirs, files in os.walk("checkpoints"):
    if os.path.basename(root) == "initial":
        targets.update(dirs)
print(f"Found {len(targets)} unique hash folders to decode")

# 2. Brute-force label permutations (CIFAR-10 classes 0-9, group sizes seen in experiments)
found = {}
for k in (2, 3, 7):  # add other sizes if needed
    for perm in permutations(range(10), k):
        name = "_".join(map(str, perm))
        h = hashlib.md5(name.encode()).hexdigest()[:8]
        if h in targets:
            found[h] = list(perm)

print(f"Decoded {len(found)} / {len(targets)}")
missing = targets - set(found) 
# Some targets belong to CIFAR-100 experiments and are already in mapping.json
existing = json.load(open("mapping.json"))
still_missing = [h for h in missing if h not in existing]
print(f"Still unknown (not in mapping.json either): {still_missing}")

# 3. Merge into mapping.json (never overwrite existing entries)
added = 0
for h, labels in found.items():
    if h not in existing:
        existing[h] = labels
        added += 1
json.dump(existing, open("mapping.json", "w"))
print(f"Added {added} new entries to mapping.json")