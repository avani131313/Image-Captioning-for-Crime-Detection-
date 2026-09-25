"""Encode assets/attributes.txt into assets/attributes.npz (run once).

    python -m scripts.build_attributes
    python -m scripts.build_attributes --txt assets/attributes.txt
"""
from __future__ import annotations
import argparse
from pathlib import Path

from jana2.config import CFG
from jana2.attributes import build, parse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt", type=Path, default=CFG.attributes_txt)
    ap.add_argument("--out", type=Path, default=CFG.attributes_path)
    a = ap.parse_args()

    groups = parse(a.txt)
    print(f"[attr] {a.txt}")
    for g in groups:
        shown = sorted(g.applies_to)[:4]
        more = "..." if len(g.applies_to) > 4 else ""
        print(f"   {g.name:<20} {g.n:>3} options   applies_to={shown}{more}")
    build(a.txt, a.out)


if __name__ == "__main__":
    main()
