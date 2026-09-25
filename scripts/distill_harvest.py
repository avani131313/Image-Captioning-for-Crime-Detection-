"""Build the distillation training set: crops, not photographs.

The student is only ever asked to embed what the pipeline actually feeds it —
object crops from the proposer, and whole-frame / tile views for scene and
anomaly checks. Training it on tidy full images would optimise a
distribution it never sees at inference.

So this walks a folder of surveillance frames, runs the REAL proposer over
each one, and saves:

    crops/obj_*.jpg     padded object crops, exactly as naming.py cuts them
    crops/tile_*.jpg    full frame + the anomaly grid tiles

No labels. No annotation. Just frames in, crops out.

    python -m scripts.distill_harvest --frames /data/cctv --out data/distill
    python -m scripts.distill_harvest --frames /data/cctv --out data/distill \
        --max-frames 20000 --stride 5

On video files, --stride N keeps every Nth frame; consecutive CCTV frames
are near-duplicates and add cost without adding information.
"""
from __future__ import annotations
import argparse
import random
from pathlib import Path

from PIL import Image

from jana2.config import CFG

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXT = {".mp4", ".avi", ".mkv", ".mov"}


def iter_frames(root: Path, stride: int, limit: int):
    """Yield PIL frames from a folder of images and/or videos."""
    n = 0
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)
    for p in files:
        if n >= limit:
            return
        try:
            yield Image.open(p).convert("RGB")
            n += 1
        except Exception:                                     # noqa: BLE001
            continue

    vids = sorted(p for p in root.rglob("*") if p.suffix.lower() in VID_EXT)
    if not vids:
        return
    try:
        import cv2
    except ImportError:
        print("[harvest] opencv not installed — skipping video files")
        return
    for v in vids:
        cap = cv2.VideoCapture(str(v))
        i = 0
        while n < limit:
            ok, fr = cap.read()
            if not ok:
                break
            if i % stride == 0:
                yield Image.fromarray(fr[:, :, ::-1])          # BGR -> RGB
                n += 1
            i += 1
        cap.release()


def pad_crop(image, box, pad):
    """Identical to naming._crop — the student must see the same framing."""
    W, H = image.size
    x0, y0, x1, y1 = box
    pw, ph = (x1 - x0) * pad, (y1 - y0) * pad
    return image.crop((max(0, x0 - pw), max(0, y0 - ph),
                       min(W, x1 + pw), min(H, y1 + ph)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="folder of images/videos")
    ap.add_argument("--out", default="data/distill")
    ap.add_argument("--max-frames", type=int, default=20000)
    ap.add_argument("--stride", type=int, default=5, help="video frame stride")
    ap.add_argument("--min-side", type=int, default=32,
                    help="drop crops smaller than this")
    ap.add_argument("--tiles", action="store_true", default=True,
                    help="also save whole-frame and grid tiles")
    a = ap.parse_args()

    out = Path(a.out) / "crops"
    out.mkdir(parents=True, exist_ok=True)

    from jana2.proposals import Proposer
    from jana2.anomaly import tiles as grid_tiles
    prop = Proposer()

    n_obj = n_tile = n_frame = 0
    for img in iter_frames(Path(a.frames), a.stride, a.max_frames):
        n_frame += 1
        try:
            boxes, _, _, _, _ = prop(img)
        except Exception as e:                                # noqa: BLE001
            print(f"[harvest] proposer failed on frame {n_frame}: {e}")
            continue

        for b in boxes:
            c = pad_crop(img, b, CFG.roi_pad)
            if min(c.size) < a.min_side:
                continue
            c.save(out / f"obj_{n_obj:07d}.jpg", quality=92)
            n_obj += 1

        if a.tiles:
            for crop, _, _ in grid_tiles(img, CFG.anomaly_grids):
                if min(crop.size) < a.min_side:
                    continue
                crop.save(out / f"tile_{n_tile:07d}.jpg", quality=92)
                n_tile += 1

        if n_frame % 200 == 0:
            print(f"[harvest] {n_frame} frames -> {n_obj} objs, {n_tile} tiles")

    files = sorted(p.name for p in out.glob("*.jpg"))
    random.Random(0).shuffle(files)
    cut = max(1, int(0.02 * len(files)))          # 2% held out
    (Path(a.out) / "val.txt").write_text("\n".join(files[:cut]))
    (Path(a.out) / "train.txt").write_text("\n".join(files[cut:]))

    print(f"\n[harvest] {n_frame} frames -> {len(files)} crops "
          f"({n_obj} objects, {n_tile} tiles)")
    print(f"[harvest] train {len(files) - cut} | val {cut}")
    print(f"[harvest] wrote {a.out}/")
    if len(files) < 20000:
        print("[harvest] WARNING: under ~20k crops the student will overfit "
              "the few scenes it saw. More frames from more cameras beats "
              "more epochs on the same ones.")


if __name__ == "__main__":
    main()
