"""Encode both yes/no check files into .npz (run once, and after any edit).

    python -m scripts.build_checks
"""
from __future__ import annotations
from jana2.config import CFG
from jana2.anomaly import build, parse


def one(txt, out, label):
    print(f"\n[{label}] {txt}")
    for c in parse(txt):
        print(f"   {c.name:<20} sev={c.severity:<6} thr={c.threshold:.2f} "
              f"+{len(c.pos)} / -{len(c.neg)}")
    build(txt, out)


def main():
    one(CFG.anomalies_txt, CFG.anomalies_path, "frame")
    one(CFG.person_checks_txt, CFG.person_checks_path, "person")


if __name__ == "__main__":
    main()
