"""Is the CLIP preprocessing wrong? Compare models on the same crop.

    python -m scripts.debug_clip data/eval/person.jpg
    python -m scripts.debug_clip img.jpg --box 100 50 300 500

The suspicion: MobileCLIP in open_clip is trained on raw [0,1] pixels —
mean=(0,0,0), std=(1,1,1) — NOT the usual CLIP mean/std. Applying the wrong
normalisation leaves coarse naming roughly intact ("person" still wins) while
destroying the fine distinctions attributes depend on. Which is exactly the
symptom: names fine, attributes wrong.

This prints each model's ACTUAL transform, then runs the same crop through
several models and shows the attribute answers side by side.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
from PIL import Image

CANDIDATES = [
    ("ViT-B-16-SigLIP2", "webli"),
    ("ViT-B-16", "laion2b_s34b_b88k"),
    ("MobileCLIP-S1", "datacompdr"),
    ("MobileCLIP-S2", "datacompdr"),
    ("MobileCLIP-B", "datacompdr_lt"),
]

PROBES = {
    "upper_colour": ["a person wearing a black top",
                     "a person wearing a white top",
                     "a person wearing a red top",
                     "a person wearing a blue top",
                     "a person wearing a yellow top",
                     "a person wearing a green top"],
    "gender": ["a photo of a man", "a photo of a woman"],
    "carrying": ["a person carrying a backpack",
                 "a person carrying nothing",
                 "a person holding a mobile phone",
                 "a person holding a large knife"],
}


def describe_transform(pp):
    """Pull size and normalisation out of a torchvision Compose."""
    size, mean, std = None, None, None
    for t in getattr(pp, "transforms", []):
        n = type(t).__name__
        if n in ("Resize", "RandomResizedCrop", "CenterCrop") and size is None:
            size = getattr(t, "size", None)
        if n == "Normalize":
            mean = tuple(round(float(x), 4) for x in t.mean)
            std = tuple(round(float(x), 4) for x in t.std)
    return size, mean, std


CLIP_MEAN = (0.4815, 0.4578, 0.4082)
ZERO_MEAN = (0.0, 0.0, 0.0)


def verdict(model_name, mean, std):
    if mean is None:
        return "no Normalize step — raw [0,1] input"
    if "MobileCLIP" in model_name:
        if max(abs(m) for m in mean) < 1e-6 and all(abs(s - 1) < 1e-6 for s in std):
            return "OK — zero mean / unit std, as MobileCLIP expects"
        return ("!! MISMATCH — MobileCLIP expects mean=(0,0,0) std=(1,1,1) "
                "but this transform normalises. Attributes will be wrong.")
    return "standard CLIP normalisation"


@torch.no_grad()
def probe(model, pp, tok, crop, device):
    img = pp(crop).unsqueeze(0).to(device)
    e = model.encode_image(img).float()
    e = e / e.norm(dim=-1, keepdim=True)
    scale = getattr(model, "logit_scale", None)
    scale = scale.detach().exp() if scale is not None else torch.tensor(1.0)

    out = {}
    for group, phrases in PROBES.items():
        t = tok(phrases).to(device)
        te = model.encode_text(t).float()
        te = te / te.norm(dim=-1, keepdim=True)
        p = ((e @ te.T) * scale).softmax(-1)[0]
        k = int(p.argmax())
        srt = p.sort(descending=True).values
        out[group] = (phrases[k], float(srt[0]), float(srt[0] - srt[1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--box", nargs=4, type=float, default=None,
                    help="x0 y0 x1 y1 — crop before testing")
    a = ap.parse_args()

    import open_clip
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    img = Image.open(a.image).convert("RGB")
    crop = img.crop(tuple(a.box)) if a.box else img
    print(f"\n{a.image}  crop size {crop.size}  device {dev}")

    have = {(m, p) for m, p in open_clip.list_pretrained()}
    print("\nMobileCLIP variants in this build:")
    for m, p in sorted(have):
        if "MobileCLIP" in m:
            print(f"   {m} / {p}")

    for name, pre in CANDIDATES:
        if (name, pre) not in have:
            print(f"\n{'='*72}\n{name}/{pre}  — not in this open_clip build")
            continue
        print(f"\n{'='*72}\n{name}/{pre}")
        try:
            model, _, pp = open_clip.create_model_and_transforms(
                name, pretrained=pre)
            model = model.to(dev).eval()
            tok = open_clip.get_tokenizer(name)
        except Exception as e:                                # noqa: BLE001
            print(f"  load failed: {e}")
            continue

        size, mean, std = describe_transform(pp)
        print(f"  input size {size}")
        print(f"  mean {mean}")
        print(f"  std  {std}")
        print(f"  -> {verdict(name, mean, std)}")

        res = probe(model, pp, tok, crop, dev)
        for g, (label, p, margin) in res.items():
            print(f"  {g:<14} {label:<38} p={p:.3f} margin={margin:.3f}")

        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

    print(f"\n{'='*72}")
    print("Read the margins, not the probabilities. A model with the wrong\n"
          "normalisation gives near-zero margins on colour groups while still\n"
          "getting gender roughly right — coarse survives, fine collapses.")


if __name__ == "__main__":
    main()
