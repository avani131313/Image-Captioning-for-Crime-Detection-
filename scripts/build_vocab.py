"""Build the naming vocabulary in the CURRENT CLIP space.

Name-list resolution order:
    1. --names-from FILE        (.txt one-per-line, or .npz)
    2. assets/vocab.txt         (the normal path)
    3. built-in 54-name FALLBACK (placeholder only — never the real vocab)

The *embeddings* are computed once here and cached to assets/vocab.npz with
the encoder identity stamped inside. Naming never re-encodes text per image.

    python -m scripts.build_vocab
    python -m scripts.build_vocab --names-from data/lvis/naming_vocab.npz
    python -m scripts.build_vocab --names-from assets/vocab.txt --extra assets/extra.txt
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

from jana2.config import CFG
from jana2.clip_space import build_vocab

FALLBACK = [
    "person", "man", "woman", "child", "cat", "kitten", "dog", "puppy", "bird",
    "horse", "cow", "sheep", "car", "truck", "bus", "bicycle", "motorcycle",
    "chair", "table", "sofa", "bed", "umbrella", "backpack", "handbag",
    "bottle", "cup", "laptop", "phone", "book", "clock", "tree", "flower",
    "building", "road", "sky", "grass", "beach", "sand", "sea", "mountain",
    "machine", "conveyor belt", "pipe", "helmet", "safety vest", "worker",
    "scarf", "sash", "stole", "gown", "shirt", "jacket", "hat", "glasses",
]


def read_names(path: Path) -> list[str]:
    if path.suffix == ".txt":
        return [l.strip() for l in path.read_text().splitlines()
                if l.strip() and not l.startswith("#")]
    z = np.load(path, allow_pickle=True)
    for key in ("names", "vocab", "labels", "classes"):
        if key in z.files:
            return [str(x) for x in z[key]]
    raise KeyError(f"no name array in {path}; keys = {z.files}")


def resolve(args) -> tuple[list[str], str]:
    if args.names_from:
        return read_names(Path(args.names_from)), str(args.names_from)
    if Path(CFG.vocab_txt).exists():
        return read_names(Path(CFG.vocab_txt)), str(CFG.vocab_txt)
    return FALLBACK, "built-in FALLBACK (placeholder!)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names-from", default=None)
    ap.add_argument("--extra", default=None,
                    help="additional .txt merged in (domain terms)")
    ap.add_argument("--out", type=Path, default=CFG.vocab_path)
    ap.add_argument("--save-txt", action="store_true",
                    help="also write the resolved list to assets/vocab.txt")
    a = ap.parse_args()

    names, src = resolve(a)
    if a.extra:
        names += read_names(Path(a.extra))

    names = [n.strip() for n in names if n and n.strip()]
    names = list(dict.fromkeys(names))
    print(f"[vocab] source = {src}")
    print(f"[vocab] {len(names)} unique names | first 10: {names[:10]}")
    if len(names) < 200:
        print("[vocab] WARNING: small vocabulary. Naming recall is capped by "
              "the names present here — a forklift cannot be named 'forklift' "
              "if 'forklift' is not in this list.")

    if a.save_txt:
        Path(CFG.vocab_txt).parent.mkdir(parents=True, exist_ok=True)
        Path(CFG.vocab_txt).write_text("\n".join(names))
        print(f"[vocab] wrote {CFG.vocab_txt}")

    build_vocab(names, a.out)


if __name__ == "__main__":
    main()
