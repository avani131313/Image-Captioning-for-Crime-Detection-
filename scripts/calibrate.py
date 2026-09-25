"""Set thresholds from your own footage instead of guessing.

    # normal frames only — controls the false-alarm rate
    python -m scripts.calibrate --normal data/eval/normal

    # with event frames too — also reports what you would CATCH
    python -m scripts.calibrate --normal data/normal --events data/events

    # write the chosen thresholds back into the checks files
    python -m scripts.calibrate --normal data/normal --write

The rule: a threshold is a false-alarm budget. If one false alert per 500
frames is acceptable, the threshold is the 99.8th percentile of scores on
NORMAL footage. Everything below is bookkeeping for that one idea.

Feed it real frames from the real cameras. Thresholds tuned on stock photos
do not survive a rainy night on an actual gate camera.
"""
from __future__ import annotations
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from jana2.config import CFG

IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def images(folder, limit=None):
    fs = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG)
    return fs[:limit] if limit else fs


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


# --------------------------------------------------------------------------- #
def collect(paths, proposer, namer, label=""):
    """Run the pipeline and gather every score we might threshold on."""
    from jana2.naming import build_evidence
    from jana2.postprocess import category

    frame = defaultdict(list)      # check -> [max prob per frame]
    person = defaultdict(list)     # check -> [prob per person crop]
    names, objness, imps = [], [], []

    for i, p in enumerate(paths, 1):
        try:
            img = Image.open(p).convert("RGB")
        except Exception:                                     # noqa: BLE001
            continue
        boxes, scores, srcs, _ = proposer(img)
        ev = build_evidence(p, img, boxes, scores, srcs, namer)

        for a in ev.anomalies:
            frame[a["name"]].append(a["prob"])

        for o in ev.objects:
            if o.clip_names:
                names.append(o.clip_names[0][1])
            objness.append(o.objectness)
            imps.append(o.importance)

        # person checks: re-score to get ALL values, not just the ones that fired
        if namer.person_checks is not None:
            pidx = [j for j, o in enumerate(ev.objects)
                    if category(o.name) == "person"]
            if pidx:
                embs = namer.embed_rois(img, [ev.objects[j].box for j in pidx])
                for row in namer.person_checks.scan_embs_full(embs):
                    for k, v in row.items():
                        person[k].append(v)

        if i % 25 == 0:
            print(f"  [{label}] {i}/{len(paths)}", flush=True)

    return {"frame": frame, "person": person,
            "names": names, "objectness": objness, "importance": imps}


def report(tag, data, budget):
    """budget = allowed false alarms per 1000 frames."""
    q = 100.0 * (1.0 - budget / 1000.0)
    print(f"\n{'='*74}\n{tag}  (suggesting thresholds at p{q:.2f} of normal, "
          f"= {budget} false alarms per 1000)")
    print(f"{'check':<22}{'p50':>8}{'p90':>8}{'p99':>8}{'max':>8}"
          f"{'suggest':>10}")
    out = {}
    for k in sorted(data):
        a = np.array(data[k])
        if not len(a):
            continue
        s = min(0.95, max(0.30, round(pct(a, q) + 0.02, 2)))
        out[k] = s
        print(f"{k:<22}{pct(a,50):>8.3f}{pct(a,90):>8.3f}{pct(a,99):>8.3f}"
              f"{a.max():>8.3f}{s:>10.2f}")
    return out


def recall_at(events, thresholds):
    if not events:
        return
    print(f"\n{'='*74}\nWhat those thresholds would CATCH on the event frames")
    print(f"{'check':<22}{'thr':>8}{'caught':>10}{'rate':>8}")
    for k, thr in sorted(thresholds.items()):
        a = np.array(events.get(k, []))
        if not len(a):
            continue
        hit = int((a >= thr).sum())
        print(f"{k:<22}{thr:>8.2f}{hit:>10}{hit/len(a):>8.1%}")


def rewrite(txt_path: Path, thresholds: dict):
    """Update threshold= values in a checks file, leaving everything else."""
    lines = Path(txt_path).read_text().splitlines()
    out, n = [], 0
    for line in lines:
        m = re.match(r"^\[([^\]]+)\](.*)$", line.strip())
        if m and m.group(1).strip() in thresholds:
            name = m.group(1).strip()
            rest = m.group(2)
            new = re.sub(r"threshold=[\d.]+",
                         f"threshold={thresholds[name]:.2f}", rest)
            if "threshold=" not in rest:
                new = rest.rstrip() + f" threshold={thresholds[name]:.2f}"
            out.append(f"[{name}]{new}")
            n += 1
        else:
            out.append(line)
    Path(txt_path).write_text("\n".join(out) + "\n")
    print(f"  wrote {n} thresholds -> {txt_path}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal", required=True,
                    help="folder of ORDINARY frames (no events)")
    ap.add_argument("--events", default=None,
                    help="optional folder of frames that SHOULD alert")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--budget", type=float, default=2.0,
                    help="acceptable false alarms per 1000 frames")
    ap.add_argument("--write", action="store_true",
                    help="write suggested thresholds into the checks files")
    a = ap.parse_args()

    from jana2.proposals import Proposer
    from jana2.naming import Namer
    proposer, namer = Proposer(), Namer()

    norm_paths = images(a.normal, a.limit)
    print(f"[calib] {len(norm_paths)} normal frames from {a.normal}")
    if not norm_paths:
        raise SystemExit("no images found")
    normal = collect(norm_paths, proposer, namer, "normal")

    events = None
    if a.events:
        ev_paths = images(a.events, a.limit)
        print(f"[calib] {len(ev_paths)} event frames from {a.events}")
        events = collect(ev_paths, proposer, namer, "events")

    ft = report("FRAME CHECKS (anomalies.txt)", normal["frame"], a.budget)
    pt = report("PERSON CHECKS (person_checks.txt)", normal["person"], a.budget)

    if events:
        recall_at(events["frame"], ft)
        recall_at(events["person"], pt)

    print(f"\n{'='*74}\nDISTRIBUTIONS on normal footage — for the caption cutoffs")
    for k in ("names", "objectness", "importance"):
        arr = np.array(normal[k])
        if not len(arr):
            continue
        print(f"{k:<14} p10={pct(arr,10):.3f}  p50={pct(arr,50):.3f}  "
              f"p90={pct(arr,90):.3f}  p99={pct(arr,99):.3f}")
    print("\nSet caption_min_score near the p50 of 'names' and "
          "caption_min_objectness near the p50 of 'objectness' — that keeps "
          "roughly the better half of detections and drops the noise.")

    if a.write:
        print()
        rewrite(CFG.anomalies_txt, ft)
        rewrite(CFG.person_checks_txt, pt)
        print("  now rerun: python -m scripts.build_checks")


if __name__ == "__main__":
    main()
