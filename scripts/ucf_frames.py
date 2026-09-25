"""Sample frames from UCF-Crime videos, keeping the category label.

    python -m scripts.ucf_frames --root data/ucf_raw --out data/ucf
    python -m scripts.ucf_frames --root data/ucf_raw --out data/ucf \
        --stride 30 --per-category 4000

IMPORTANT — these labels are VIDEO level, not frame level. A four minute
"Fighting" clip may contain twenty seconds of fighting and three minutes of
an empty corridor. Every frame here inherits the video's category, so the
event folders are only CANDIDATES. scripts/moondream_verify.py turns them
into real frame-level labels.

"Normal" is the exception: every frame of a normal video really is normal,
so those are trusted negatives with no verification needed. That class is
worth as much as the event classes — hundreds of videos of real CCTV doing
nothing, which is the ordinary-scene distribution the hand-written
negatives kept failing to anchor.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

VID = {".mp4", ".avi", ".mkv", ".mov", ".mpg", ".mpeg"}

CANON = {
    "abuse": "Abuse", "arrest": "Arrest", "arson": "Arson",
    "assault": "Assault", "burglary": "Burglary", "explosion": "Explosion",
    "fighting": "Fighting", "roadaccidents": "RoadAccidents",
    "roadaccident": "RoadAccidents", "robbery": "Robbery",
    "shooting": "Shooting", "shoplifting": "Shoplifting",
    "stealing": "Stealing", "vandalism": "Vandalism",
}


def category_of(path: Path) -> str | None:
    """Infer the category from any ancestor folder name.

    Anything containing "normal" collapses to Normal, so both
    Testing_Normal_Videos_Anomaly and z_Normal_Videos_event merge into one
    trusted-negative pool.
    """
    for part in reversed(path.parts):
        k = re.sub(r"[^a-z]", "", part.lower())
        if k in CANON:
            return CANON[k]
        if "normal" in k:
            return "Normal"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder of UCF videos")
    ap.add_argument("--out", default="data/ucf")
    ap.add_argument("--stride", type=int, default=30,
                    help="keep every Nth frame (~1/sec at 30fps)")
    ap.add_argument("--per-category", type=int, default=4000)
    ap.add_argument("--min-side", type=int, default=120)
    a = ap.parse_args()

    try:
        import cv2
    except ImportError:
        raise SystemExit("pip install opencv-python")

    out = Path(a.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    manifest = open(out / "frames.jsonl", "w")

    vids = sorted(p for p in Path(a.root).rglob("*") if p.suffix.lower() in VID)
    print(f"[ucf] {len(vids)} videos under {a.root}", flush=True)
    if not vids:
        raise SystemExit("no videos found — check --root")

    counts: dict[str, int] = {}
    kept = 0
    for vi, v in enumerate(vids, 1):
        cat = category_of(v)
        if cat is None:
            continue
        if counts.get(cat, 0) >= a.per_category:
            continue

        cap = cv2.VideoCapture(str(v))
        i = 0
        while counts.get(cat, 0) < a.per_category:
            ok, fr = cap.read()
            if not ok:
                break
            if i % a.stride == 0 and min(fr.shape[:2]) >= a.min_side:
                d = out / "frames" / cat
                d.mkdir(parents=True, exist_ok=True)
                fp = d / f"{v.stem}_{i:06d}.jpg"
                cv2.imwrite(str(fp), fr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                manifest.write(json.dumps({
                    "path": str(fp), "category": cat, "video": v.stem,
                    "frame": i}) + "\n")
                counts[cat] = counts.get(cat, 0) + 1
                kept += 1
            i += 1
        cap.release()
        if vi % 50 == 0:
            print(f"[ucf] {vi}/{len(vids)} videos | {kept} frames | {counts}",
                  flush=True)

    manifest.close()
    print(f"\n[ucf] {kept} frames -> {out}/frames.jsonl")
    for k in sorted(counts):
        print(f"  {k:<16}{counts[k]:>7}")


if __name__ == "__main__":
    main()
