"""Build the scene vocabulary in the CURRENT CLIP space — same embedding
logic as build_vocab.py, just pointed at assets/scenes.txt.

    python -m scripts.build_scenes
"""
from __future__ import annotations
from pathlib import Path

from jana2.config import CFG
from jana2.clip_space import build_vocab


def read_names(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text().splitlines()
            if l.strip() and not l.startswith("#")]


def main() -> None:
    if not Path(CFG.scenes_txt).exists():
        raise SystemExit(f"[scenes] missing {CFG.scenes_txt} — create it "
                          f"first, one scene phrase per line.")
    names = list(dict.fromkeys(read_names(Path(CFG.scenes_txt))))
    print(f"[scenes] source = {CFG.scenes_txt}")
    print(f"[scenes] {len(names)} unique scene labels | first 10: {names[:10]}")
    if not names:
        raise SystemExit(f"[scenes] {CFG.scenes_txt} is empty — add scene "
                          f"phrases before building.")
    build_vocab(names, CFG.scenes_path)
    print(f"[scenes] wrote {CFG.scenes_path}")


if __name__ == "__main__":
    main()
