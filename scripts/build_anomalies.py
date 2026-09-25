"""Encode assets/anomalies.txt into assets/anomalies.npz (run once).

    python -m scripts.build_anomalies
"""
from __future__ import annotations
import argparse
from pathlib import Path

from jana2.config import CFG
from jana2.anomaly import build, parse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt", type=Path, default=CFG.anomalies_txt)
    ap.add_argument("--out", type=Path, default=CFG.anomalies_path)
    a = ap.parse_args()

    for c in parse(a.txt):
        print(f"   {c.name:<20} sev={c.severity:<6} thr={c.threshold:.2f} "
              f"+{len(c.pos)} / -{len(c.neg)}")
    build(a.txt, a.out)


if __name__ == "__main__":
    main()
