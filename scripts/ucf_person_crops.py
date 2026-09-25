"""Cut person crops out of UCF frames, for person-level probe training.

    python -m scripts.ucf_person_crops --manifest data/ucf/frames.jsonl \
        --out data/ucf_person --per-category 1200

WHY A SEPARATE PASS

The frame probes were fitted on whole frames, which is what the frame-level
checks in anomalies.txt see. The checks in person_checks.txt run on PERSON
CROPS — a padded box around one person, cut by the proposer. A probe fitted
on whole frames would be scored on a crop it never saw during training.

So this runs the real proposer, keeps only boxes that categorise as a
person, and cuts them with the same roi_pad naming.py uses. Same
distribution in, same distribution out.

ONE IMPORTANT DIFFERENCE FROM THE FRAME PASS

There, "Normal" frames were trusted negatives without asking: a normal
video contains no fight, so every frame is a no.

That does NOT hold here. Normal footage is full of people in uniform,
people carrying large objects, people whose faces are not visible. Marking
those zero would teach the probe that a security guard is not in uniform.
So Normal crops are INCLUDED and must be asked like any other.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from PIL import Image

from jana2.config import CFG


def pad_crop(image, box, pad):
    """Same framing naming.py uses, so the probe sees what the check sees."""
    W, H = image.size
    x0, y0, x1, y1 = box
    pw, ph = (x1 - x0) * pad, (y1 - y0) * pad
    return image.crop((max(0, x0 - pw), max(0, y0 - ph),
                       min(W, x1 + pw), min(H, y1 + ph)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/ucf/frames.jsonl")
    ap.add_argument("--out", default="data/ucf_person")
    ap.add_argument("--per-category", type=int, default=1200)
    ap.add_argument("--min-side", type=int, default=48,
                    help="drop crops smaller than this — a 30px person is "
                         "not something any encoder can read attributes off")
    ap.add_argument("--max-per-frame", type=int, default=3,
                    help="a crowd frame should not dominate the set")
    a = ap.parse_args()

    from jana2.proposals import Proposer
    from jana2.postprocess import category
    prop = Proposer()

    rows = [json.loads(l) for l in open(a.manifest) if l.strip()]
    out = Path(a.out)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    man = open(out / "person_crops.jsonl", "w")

    counts, n = {}, 0
    for i, r in enumerate(rows, 1):
        cat = r["category"]
        if counts.get(cat, 0) >= a.per_category:
            continue
        try:
            img = Image.open(r["path"]).convert("RGB")
        except Exception:                                     # noqa: BLE001
            continue

        try:
            boxes, scores, srcs, labels, _ = prop(img)
        except Exception:                                     # noqa: BLE001
            continue

        kept = 0
        for b, s, lab in zip(boxes, scores, labels or [None] * len(boxes)):
            if counts.get(cat, 0) >= a.per_category or kept >= a.max_per_frame:
                break
            name = lab or ""
            if category(name) != "person":
                continue
            c = pad_crop(img, b, CFG.roi_pad)
            if min(c.size) < a.min_side:
                continue
            d = out / "crops" / cat
            d.mkdir(parents=True, exist_ok=True)
            fp = d / f"{Path(r['path']).stem}_p{kept}.jpg"
            c.save(fp, quality=92)
            man.write(json.dumps({"path": str(fp), "category": cat,
                                  "frame": r["path"],
                                  "score": round(float(s), 3)}) + "\n")
            counts[cat] = counts.get(cat, 0) + 1
            kept += 1
            n += 1

        if i % 500 == 0:
            print(f"[person] {i}/{len(rows)} frames -> {n} crops {counts}",
                  flush=True)

    man.close()
    print(f"\n[person] {n} person crops -> {out}/person_crops.jsonl")
    for k in sorted(counts):
        print(f"  {k:<16}{counts[k]:>7}")
    if n < 5000:
        print("[warn] under ~5k crops the probes will be thin. Raise "
              "--per-category or lower --min-side.")


if __name__ == "__main__":
    main()
