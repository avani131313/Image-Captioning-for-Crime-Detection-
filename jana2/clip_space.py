"""Single source of truth for the CLIP encoder + the vocabulary bound to it.

The bug this file exists to prevent: a vocabulary encoded with encoder A,
loaded and compared against ROI embeddings from encoder B. Cosine similarity
still returns confident-looking numbers — they are simply wrong. So the
encoder identity is written into the vocab file and asserted on load.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import open_clip

from .config import CFG

_CACHE: dict = {}


def load_clip():
    """Returns (model, preprocess, tokenizer, space_id). Cached per process."""
    if "clip" in _CACHE:
        return _CACHE["clip"]

    avail = {(m, p) for m, p in open_clip.list_pretrained()}
    want = (CFG.clip_model, CFG.clip_pretrained)
    if want not in avail:
        print(f"[clip] {want} unavailable in this open_clip build; "
              f"falling back to ({CFG.clip_fallback_model}, {CFG.clip_fallback_pretrained})")
        want = (CFG.clip_fallback_model, CFG.clip_fallback_pretrained)

    model, _, preprocess = open_clip.create_model_and_transforms(
        want[0], pretrained=want[1])
    model = model.to(CFG.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = open_clip.get_tokenizer(want[0])
    space_id = f"{want[0]}/{want[1]}"
    print(f"[clip] space = {space_id}")

    _CACHE["clip"] = (model, preprocess, tok, space_id)
    return _CACHE["clip"]


# --------------------------------------------------------------------------- #
#  vocabulary
# --------------------------------------------------------------------------- #
PROMPTS = (
    "a photo of a {}.",
    "a close-up photo of a {}.",
    "a cropped photo of a {}.",
    "a photo of the {} in a scene.",
)


@torch.no_grad()
def build_vocab(names: list[str], out_path: Path, batch: int = 256) -> None:
    """Encode `names` with prompt-ensembling into the current CLIP space."""
    model, _, tok, space_id = load_clip()
    names = [n.strip() for n in names if n and n.strip()]
    names = list(dict.fromkeys(names))                     # dedupe, keep order

    embs = []
    for i in range(0, len(names), batch):
        chunk = names[i:i + batch]
        per_prompt = []
        for tpl in PROMPTS:
            t = tok([tpl.format(n) for n in chunk]).to(CFG.device)
            e = model.encode_text(t).float()
            per_prompt.append(e / e.norm(dim=-1, keepdim=True))
        e = torch.stack(per_prompt).mean(0)                # ensemble
        embs.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy())
        print(f"  encoded {min(i+batch, len(names))}/{len(names)}", end="\r")

    embs = np.concatenate(embs).astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, names=np.array(names, dtype=object),
             embs=embs, clip_space=space_id)
    print(f"\n[vocab] {len(names)} names -> {out_path}  (space {space_id})")


def load_vocab(path: Path | None = None):
    """Returns (names[list[str]], embs[N,D] float32 unit-norm). Asserts space."""
    path = Path(path or CFG.vocab_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run: python -m scripts.build_vocab")
    z = np.load(path, allow_pickle=True)
    _, _, _, space_id = load_clip()
    stored = str(z["clip_space"])
    if stored != space_id:
        raise RuntimeError(
            f"vocab/encoder mismatch: vocab was built in '{stored}' but the "
            f"active encoder is '{space_id}'. Rebuild the vocab.")
    return list(z["names"]), z["embs"].astype(np.float32)
