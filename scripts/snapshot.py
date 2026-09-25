"""Regression harness — no ground truth needed.

You cannot label 300 frames, so we do the next best thing: freeze the output
on a fixed set of images, and after every change show exactly what moved.

    # once, after a build you are happy with
    python -m scripts.snapshot save --images data/eval --tag baseline

    # after any change
    python -m scripts.snapshot diff --tag baseline

It will not tell you whether the caption is CORRECT. It will tell you that
lowering rpn_min_score added 40 boxes and dropped the machete, which is the
question you actually keep asking.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from PIL import Image

from jana2.config import CFG

IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
STORE = Path(CFG.out_dir) / "snapshots"


def images(folder, limit=None):
    fs = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG)
    return fs[:limit] if limit else fs


def run_all(paths):
    from jana2.proposals import Proposer
    from jana2.naming import Namer, build_evidence
    from jana2 import readout

    proposer, namer = Proposer(), Namer()
    out = {}
    for i, p in enumerate(paths, 1):
        img = Image.open(p).convert("RGB")
        boxes, scores, srcs, labels, stats = proposer(img)
        ev = build_evidence(p, img, boxes, scores, srcs, labels, namer)
        out[str(p)] = {
            "caption": readout.caption(ev),
            "objects": [{"name": o.name, "imp": o.importance,
                         "score": round(o.clip_names[0][1], 3) if o.clip_names else 0,
                         "src": o.source, "pos": o.position,
                         "checks": sorted((o.checks or {}).keys())}
                        for o in ev.objects[:12]],
            "relations": [f"{r['subject_name']} {r['predicate']} "
                          f"{r['object_name']}" for r in (ev.relations or [])],
            "alerts": [a["name"] for a in (ev.anomalies or [])
                       if a.get("triggered")],
            "counts": stats.as_dict(),
        }
        print(f"  {i}/{len(paths)}  {p.name}", flush=True)
    return out


def cmd_save(a):
    paths = images(a.images, a.limit)
    if not paths:
        raise SystemExit(f"no images under {a.images}")
    print(f"[snap] {len(paths)} images")
    data = run_all(paths)
    STORE.mkdir(parents=True, exist_ok=True)
    f = STORE / f"{a.tag}.json"
    f.write_text(json.dumps(data, indent=2))
    print(f"[snap] saved -> {f}")


def _obj_set(rec):
    return {(o["name"], o["pos"]) for o in rec["objects"]}


def cmd_diff(a):
    f = STORE / f"{a.tag}.json"
    if not f.exists():
        raise SystemExit(f"{f} not found — run 'save' first")
    old = json.loads(f.read_text())
    paths = [Path(p) for p in old if Path(p).exists()]
    print(f"[snap] re-running {len(paths)} images")
    new = run_all(paths)

    changed = 0
    for p, o in old.items():
        n = new.get(p)
        if n is None:
            continue
        lines = []

        if o["caption"] != n["caption"]:
            lines.append(f"  caption -\n    OLD {o['caption']}\n    NEW {n['caption']}")

        gone = _obj_set(o) - _obj_set(n)
        added = _obj_set(n) - _obj_set(o)
        if gone:
            lines.append("  LOST objects:  " + ", ".join(f"{a}@{b}" for a, b in sorted(gone)))
        if added:
            lines.append("  NEW objects:   " + ", ".join(f"{a}@{b}" for a, b in sorted(added)))

        rg = set(o["relations"]) - set(n["relations"])
        ra = set(n["relations"]) - set(o["relations"])
        if rg:
            lines.append("  LOST relations: " + "; ".join(sorted(rg)))
        if ra:
            lines.append("  NEW relations:  " + "; ".join(sorted(ra)))

        ag = set(o["alerts"]) - set(n["alerts"])
        aa = set(n["alerts"]) - set(o["alerts"])
        if ag:
            lines.append("  LOST alerts:    " + ", ".join(sorted(ag)))
        if aa:
            lines.append("  NEW alerts:     " + ", ".join(sorted(aa)))

        if o["counts"].get("final") != n["counts"].get("final"):
            lines.append(f"  final boxes:    {o['counts'].get('final')} -> "
                         f"{n['counts'].get('final')}")

        if lines:
            changed += 1
            print(f"\n{'='*70}\n{Path(p).name}")
            print("\n".join(lines))

    print(f"\n{'='*70}\n{changed}/{len(old)} images changed")
    if a.update:
        f.write_text(json.dumps(new, indent=2))
        print(f"[snap] baseline updated -> {f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("save")
    s.add_argument("--images", required=True)
    s.add_argument("--limit", type=int, default=40)
    s.add_argument("--tag", default="baseline")
    s.set_defaults(fn=cmd_save)

    d = sub.add_parser("diff")
    d.add_argument("--tag", default="baseline")
    d.add_argument("--update", action="store_true",
                   help="accept the new output as the baseline")
    d.set_defaults(fn=cmd_diff)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
