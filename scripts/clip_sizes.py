"""Measure every candidate CLIP: parameters, speed, and attribute quality.

    python -m scripts.clip_sizes
    python -m scripts.clip_sizes --probe data/eval/factory.jpg --box 400 250 750 680

Size alone is the wrong basis for this choice. A model that is half the size
and cannot tell a black top from a dark blue one has not saved you anything,
because attributes are where most of your caption detail comes from.

So this reports three things per model:
  params    image tower only — the text tower runs once at build time
  ms        time to embed one crop, after warmup
  margins   how confidently it separates attribute options. This is the
            number that matters. A weak model does not give WRONG colours,
            it gives colours with near-zero margins, which the margin test
            then drops — so the caption quietly loses its detail.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import torch
from PIL import Image

CANDIDATES = [
    ("MobileCLIP-S0", "datacompdr"),
    ("MobileCLIP-S1", "datacompdr"),
    ("MobileCLIP-S2", "datacompdr"),
    ("MobileCLIP-B", "datacompdr_lt"),
    ("TinyCLIP-ViT-40M-32-Text-19M", "laion400m_e32"),
    ("ViT-B-32", "laion2b_s34b_b79k"),
    ("ViT-B-16", "laion2b_s34b_b88k"),
    ("ViT-B-16-SigLIP", "webli"),
    ("ViT-B-16-SigLIP2", "webli"),
]

# groups whose separation we actually care about
PROBES = {
    "gender": ["a photo of a man", "a photo of a woman"],
    "upper_colour": ["a person wearing a black top",
                     "a person wearing a white top",
                     "a person wearing a red top",
                     "a person wearing a blue top",
                     "a person wearing a dark blue top",
                     "a person wearing a yellow top"],
    "carrying": ["a person carrying a backpack",
                 "a person carrying nothing",
                 "a person holding a mobile phone",
                 "a person holding a large knife"],
    "headwear": ["a person wearing a helmet",
                 "a person wearing a cap",
                 "a person with an uncovered head"],
}


@torch.no_grad()
def probe(model, pp, tok, crop, device):
    """Returns {group: (winner, margin)} — margin is what matters."""
    e = model.encode_image(pp(crop).unsqueeze(0).to(device)).float()
    e = e / e.norm(dim=-1, keepdim=True)
    ls = getattr(model, "logit_scale", None)
    scale = ls.detach().exp() if ls is not None else torch.tensor(1.0)

    out = {}
    for g, phrases in PROBES.items():
        t = tok(phrases).to(device)
        te = model.encode_text(t).float()
        te = te / te.norm(dim=-1, keepdim=True)
        p = ((e @ te.T) * scale).softmax(-1)[0]
        srt, idx = p.sort(descending=True)
        out[g] = (phrases[int(idx[0])], float(srt[0] - srt[1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default=None, help="image to test attributes on")
    ap.add_argument("--box", nargs=4, type=float, default=None,
                    help="x0 y0 x1 y1 — crop a person out first")
    a = ap.parse_args()

    import open_clip
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    have = {(m, p) for m, p in open_clip.list_pretrained()}

    crop = None
    if a.probe and Path(a.probe).exists():
        img = Image.open(a.probe).convert("RGB")
        crop = img.crop(tuple(a.box)) if a.box else img
        print(f"probing on {a.probe} crop {crop.size}")

    print(f"\n{'model':<32}{'image':>9}{'text':>9}{'total':>9}{'ms':>8}   normalisation")
    print("-" * 96)

    results = []
    for name, pre in CANDIDATES:
        if (name, pre) not in have:
            print(f"{name:<32}{'not in this open_clip build':>40}")
            continue
        try:
            model, _, pp = open_clip.create_model_and_transforms(name, pretrained=pre)
            model = model.to(dev).eval()
            tok = open_clip.get_tokenizer(name)
        except Exception as e:                                # noqa: BLE001
            print(f"{name:<32}  load failed: {type(e).__name__}")
            continue

        tot = sum(p.numel() for p in model.parameters())
        vis = sum(p.numel() for p in model.visual.parameters())

        mean = None
        for t in getattr(pp, "transforms", []):
            if type(t).__name__ == "Normalize":
                mean = tuple(round(float(x), 3) for x in t.mean)
        norm = "raw 0-1" if (mean and max(abs(m) for m in mean) < 1e-6) \
            else (str(mean) if mean else "none")

        ms = float("nan")
        if crop is not None:
            x = pp(crop).unsqueeze(0).to(dev)
            with torch.no_grad():
                for _ in range(3):
                    model.encode_image(x)
                if dev == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(10):
                    model.encode_image(x)
                if dev == "cuda":
                    torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) * 100

        print(f"{name:<32}{vis/1e6:>8.1f}M{(tot-vis)/1e6:>8.1f}M"
              f"{tot/1e6:>8.1f}M{ms:>8.1f}   {norm}")

        if crop is not None:
            res = probe(model, pp, tok, crop, dev)
            results.append((name, vis / 1e6, res))
            for g, (win, margin) in res.items():
                flag = "  <-- cannot separate" if margin < 0.05 else ""
                print(f"      {g:<14}{win:<40}margin {margin:.3f}{flag}")

        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

    if results:
        print(f"\n{'='*96}\nATTRIBUTE SEPARATION — mean margin across groups "
              f"(higher is better)\n")
        print(f"{'model':<32}{'image M':>10}{'mean margin':>14}"
              f"{'margin per M':>14}")
        for name, mp, res in sorted(
                results, key=lambda r: -sum(m for _, m in r[2].values())):
            mm = sum(m for _, m in res.values()) / len(res)
            print(f"{name:<32}{mp:>10.1f}{mm:>14.3f}{mm/mp*100:>14.2f}")
        print("\nThe last column is margin per million parameters — which model "
              "gives you\nthe most discrimination for its size. Pick on that, "
              "not on size alone.")


if __name__ == "__main__":
    main()
