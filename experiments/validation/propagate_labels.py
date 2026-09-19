"""
V9 Step 0: Propagate corrected labels from expanded_pairs.json to
subclassified_pairs.json and faithful_unfaithful_pairs.json.

The v9 fix updated expanded_pairs.json (3,229 BBH examples reclassified).
This script copies the corrected label_3class, subtype, subtype_reason,
judge_label, and behavior fields to the other data files.
"""

import json
import shutil
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"


def main():
    # Load corrected data
    with open(DATA / "expanded_pairs.json") as f:
        expanded = json.load(f)

    # Build lookup by composite key
    lookup = {}
    for d in expanded:
        key = (d["id"], d.get("perturbation_strategy", ""), d.get("perturbation_point", ""))
        lookup[key] = {
            "label_3class": d.get("label_3class"),
            "subtype": d.get("subtype"),
            "subtype_reason": d.get("subtype_reason"),
            "judge_label": d.get("judge_label"),
            "behavior": d.get("behavior"),
        }

    # Update subclassified_pairs.json
    sub_path = DATA / "subclassified_pairs.json"
    if sub_path.exists():
        shutil.copy2(sub_path, DATA / "subclassified_pairs_pre_v9.json")
        with open(sub_path) as f:
            sub = json.load(f)

        updated = 0
        for d in sub:
            key = (d["id"], d.get("perturbation_strategy", ""), d.get("perturbation_point", ""))
            if key in lookup:
                corrected = lookup[key]
                old_label = d.get("label_3class")
                new_label = corrected["label_3class"]
                if old_label != new_label:
                    updated += 1
                for field, value in corrected.items():
                    if value is not None:
                        d[field] = value

        with open(sub_path, "w") as f:
            json.dump(sub, f, indent=2)
        print(f"subclassified_pairs.json: updated {updated} labels ({len(sub)} total)")

        # Verify
        labels = Counter(d.get("label_3class", "") for d in sub if d.get("source") == "bbh")
        print(f"  BBH labels: {dict(labels)}")

    # Update faithful_unfaithful_pairs.json
    ff_path = DATA / "faithful_unfaithful_pairs.json"
    if ff_path.exists():
        shutil.copy2(ff_path, DATA / "faithful_unfaithful_pairs_pre_v9.json")
        with open(ff_path) as f:
            ff = json.load(f)

        updated = 0
        for d in ff:
            key = (d["id"], d.get("perturbation_strategy", ""), d.get("perturbation_point", ""))
            if key in lookup:
                corrected = lookup[key]
                old_label = d.get("label_3class")
                new_label = corrected["label_3class"]
                if old_label != new_label:
                    updated += 1
                for field, value in corrected.items():
                    if value is not None:
                        d[field] = value

        with open(ff_path, "w") as f:
            json.dump(ff, f, indent=2)
        print(f"faithful_unfaithful_pairs.json: updated {updated} labels ({len(ff)} total)")

        labels = Counter(d.get("label_3class", "") for d in ff if d.get("source") == "bbh")
        print(f"  BBH labels: {dict(labels)}")

    print("\nDone. Corrected labels propagated to all data files.")


if __name__ == "__main__":
    main()
