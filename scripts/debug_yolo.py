"""Is YOLO underperforming, and if so why? Run the same image several ways.

    python -m scripts.debug_yolo data/eval/factory.jpg

Checks, in order of how likely they are to be the problem:

  1. RGB vs BGR — ultralytics treats a numpy array as BGR (OpenCV order).
     We pass np.array(pil_image), which is RGB. If the BGR row detects
     noticeably more, that swap is the bug.
  2. model size — yolo26n is the nano model, chosen only because rpn_v2's
     first layer needs neck widths 64/128/256. Nothing stops us running a
     larger model for DETECTION and keeping nano for RPN features.
  3. input resolution — small/distant objects need imgsz above 640.
  4. confidence floor.
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from jana2.config import CFG      # sets YOLO_AUTOINSTALL=False


def summarise(r, label):
    if r.boxes is None or not len(r.boxes):
        print(f"{label:<34} 0 boxes")
        return 0
    conf = r.boxes.conf.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    names = Counter(r.names[c] for c in cls)
    top = ", ".join(f"{k}x{v}" for k, v in names.most_common(5))
    print(f"{label:<34} {len(conf):>3} boxes   "
          f"conf max={conf.max():.2f} mean={conf.mean():.2f}   {top}")
    return len(conf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--conf", type=float, default=0.10)
    a = ap.parse_args()

    from ultralytics import YOLO
    img = Image.open(a.image).convert("RGB")
    rgb = np.array(img)
    bgr = rgb[:, :, ::-1].copy()
    print(f"\n{a.image}  {img.size}  conf={a.conf}\n")

    # ---- 1. colour order -------------------------------------------------
    print("--- 1. COLOUR ORDER (ultralytics reads ndarray as BGR) ---")
    m = YOLO(CFG.yolo_weights)
    n_pil = summarise(m.predict(img, conf=a.conf, verbose=False)[0],
                      "PIL image (correct RGB)")
    n_rgb = summarise(m.predict(rgb, conf=a.conf, verbose=False)[0],
                      "ndarray RGB  (what we do now)")
    n_bgr = summarise(m.predict(bgr, conf=a.conf, verbose=False)[0],
                      "ndarray BGR  (what it expects)")
    if n_pil > n_rgb or n_bgr > n_rgb:
        print("  >> confirmed: passing RGB ndarray loses detections. "
              "Pass the PIL image instead.")
    else:
        print("  >> colour order is not the problem here.")

    # ---- 2. model size ---------------------------------------------------
    print("\n--- 2. MODEL SIZE (nano is only needed for RPN features) ---")
    for w in ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo11x.pt"]:
        try:
            summarise(YOLO(w).predict(img, conf=a.conf, verbose=False)[0], w)
        except Exception as e:                                # noqa: BLE001
            print(f"{w:<34} unavailable ({type(e).__name__})")

    # ---- 3. resolution ---------------------------------------------------
    print("\n--- 3. INPUT SIZE (small/distant objects need more pixels) ---")
    for sz in [640, 960, 1280, 1600]:
        summarise(m.predict(img, conf=a.conf, imgsz=sz, verbose=False)[0],
                  f"imgsz={sz}")

    # ---- 4. confidence ---------------------------------------------------
    print("\n--- 4. CONFIDENCE FLOOR ---")
    for c in [0.25, 0.10, 0.05, 0.01]:
        summarise(m.predict(img, conf=c, verbose=False)[0], f"conf={c}")

    # ---- 5. what our own code produces -----------------------------------
    print("\n--- 5. THROUGH OUR PIPELINE (after all filtering) ---")
    from jana2.proposals import Proposer
    p = Proposer()
    boxes, scores, srcs, stats = p(img)
    print(stats.render())


if __name__ == "__main__":
    main()
