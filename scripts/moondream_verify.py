"""Verify candidate frames with Moondream3 via vLLM, producing frame labels.

    export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/root/aniket/git_clones/image-captioning/.venv-vllm/lib/python3.10/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
    export VLLM_USE_FLASHINFER_SAMPLER=0
    /root/aniket/git_clones/image-captioning/.venv-vllm/bin/python \
        scripts/moondream_verify.py --manifest data/ucf/frames_sample.jsonl \
        --out data/ucf/verified.jsonl

WHY THIS STEP EXISTS

UCF-Crime says a VIDEO contains a fight. It does not say WHICH FRAMES. Train
on every frame of a Fighting video and most "positives" are an empty
corridor; the probe learns a blur of both and fails for a reason invisible
in the loss curve. Moondream sorts each frame into yes/no.

Nothing is discarded. A no-fight frame from a Fighting video is a HARD
NEGATIVE — same camera, same place, same lighting, no fight — which is the
most valuable negative there is.

THREE THINGS LEARNED THE HARD WAY, ALL LOAD-BEARING

1. BATCH. vLLM measured 43 q/s batched versus ~1 q/s one at a time. The
   whole point of vLLM is continuous batching; querying per frame throws it
   away.

2. NO STRUCTURED OUTPUTS. Constraining generation to choice=["yes","no"]
   produced "no" for EVERY question, including ones the model answers
   correctly when free. Clean format, wrong content — the worst kind of
   failure because it looks fine.

3. WARM UP. Triton JIT-compiles on first inference and the requests caught
   in that window come back empty.

Anything unparseable is retried once, then written as null and EXCLUDED
from training. A wrong label is worse than a missing one.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

from PIL import Image

TMPL = "<|endoftext|><image><|md_reserved_0|>{q}<|md_reserved_1|>"

QUESTIONS = {
    "fight":        "Are two or more people physically fighting or attacking each other?",
    "accident":     "Is there a crashed, damaged or overturned vehicle?",
    "fire":         "Are there visible flames or a fire burning?",
    "smoke":        "Is there thick smoke visible?",
    "vandalism":    "Is someone damaging property, breaking something, or spraying graffiti?",
    "snatching":    "Is someone stealing or snatching an object from another person?",
    "weapon_scene": "Is anyone holding or brandishing a weapon such as a gun, knife or stick?",
    "person_down":  "Is a person lying on the ground or collapsed?",
    "climbing":     "Is someone climbing over a wall, fence or gate?",
}

ASK = {
    "Fighting":      ["fight", "weapon_scene", "person_down"],
    "Assault":       ["fight", "weapon_scene", "person_down"],
    "Abuse":         ["fight", "person_down"],
    "RoadAccidents": ["accident", "person_down"],
    "Vandalism":     ["vandalism", "fight"],
    "Robbery":       ["snatching", "weapon_scene", "fight"],
    "Stealing":      ["snatching"],
    "Shoplifting":   ["snatching"],
    "Arson":         ["fire", "smoke"],
    "Explosion":     ["fire", "smoke"],
    "Shooting":      ["weapon_scene", "person_down", "fight"],
    "Burglary":      ["climbing", "vandalism", "snatching"],
    "Arrest":        ["fight", "person_down"],
}

NEG = ("no", "none", "not ", "nothing", "nobody", "neither", "absent")
POS = ("yes", "there is", "there are", "someone is", "appears to")


def parse_yes_no(text):
    """1, 0, or None. None means retry or drop — never guess."""
    t = (text or "").strip().lower()
    if not t:
        return None
    if t.startswith("yes"):
        return 1
    if t.startswith("no"):
        return 0
    for p in NEG:
        if p in t:
            return 0
    for p in POS:
        if p in t:
            return 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/ucf/frames_sample.jsonl")
    ap.add_argument("--out", default="data/ucf/verified.jsonl")
    ap.add_argument("--model", default="moondream/moondream3-preview")
    ap.add_argument("--chunk", type=int, default=256,
                    help="frames per batch — bigger is faster, more RAM")
    ap.add_argument("--gpu-util", type=float, default=0.75)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--no-raw", action="store_true",
                    help="skip storing raw text (raw is cheap and auditable)")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.manifest) if l.strip()]
    if a.max_frames:
        rows = rows[:a.max_frames]

    outp = Path(a.out)
    done = set()
    if outp.exists():
        for l in open(outp):
            try:
                done.add(json.loads(l)["path"])
            except Exception:                                 # noqa: BLE001
                pass
        print(f"[resume] {len(done)} frames already done")

    todo = [r for r in rows if r["path"] not in done and r["category"] in ASK]
    trusted = [r for r in rows if r["path"] not in done
               and r["category"] not in ASK]
    print(f"[plan] {len(todo)} to query | {len(trusted)} trusted negatives")

    f = open(outp, "a")
    for r in trusted:              # Normal video frames: no query needed
        f.write(json.dumps({"path": r["path"], "category": r["category"],
                            "labels": {k: 0 for k in QUESTIONS},
                            "source": "trusted"}) + "\n")
    f.flush()
    if not todo:
        f.close()
        print("[done] nothing to query")
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, tokenizer="moondream/starmie-v1",
              trust_remote_code=True, max_model_len=4096,
              gpu_memory_utilization=a.gpu_util,
              limit_mm_per_prompt={"image": 1}, disable_log_stats=True)
    sp = SamplingParams(max_tokens=24, temperature=0.0)

    # warmup — absorbs the Triton JIT spike that returns empty answers
    warm = Image.open(todo[0]["path"]).convert("RGB")
    llm.generate([{"prompt": TMPL.format(q="Is this an image?"),
                   "multi_modal_data": {"image": warm}}], sp)
    print("[warmup] done")

    t0, n_done, n_q, n_null = time.perf_counter(), 0, 0, 0
    for i in range(0, len(todo), a.chunk):
        block = todo[i:i + a.chunk]
        reqs, keys, imgs = [], [], {}
        for bi, r in enumerate(block):
            try:
                img = Image.open(r["path"]).convert("RGB")
            except Exception:                                 # noqa: BLE001
                continue
            imgs[bi] = img
            for c in ASK[r["category"]]:
                reqs.append({"prompt": TMPL.format(q=QUESTIONS[c]),
                             "multi_modal_data": {"image": img}})
                keys.append((bi, c))

        if not reqs:
            continue
        outs = llm.generate(reqs, sp)
        n_q += len(reqs)

        res = {}
        retry_reqs, retry_keys = [], []
        for (bi, c), o in zip(keys, outs):
            txt = o.outputs[0].text
            v = parse_yes_no(txt)
            if v is None:                     # one retry before giving up
                retry_reqs.append({"prompt": TMPL.format(
                    q=QUESTIONS[c] + " Answer yes or no."),
                    "multi_modal_data": {"image": imgs[bi]}})
                retry_keys.append((bi, c))
            res[(bi, c)] = (v, txt)

        if retry_reqs:
            for (bi, c), o in zip(retry_keys, llm.generate(retry_reqs, sp)):
                txt = o.outputs[0].text
                res[(bi, c)] = (parse_yes_no(txt), txt)
            n_q += len(retry_reqs)

        for bi, r in enumerate(block):
            if bi not in imgs:
                continue
            labels, raw = {}, {}
            for c in ASK[r["category"]]:
                v, txt = res.get((bi, c), (None, ""))
                labels[c] = v
                raw[c] = txt
                if v is None:
                    n_null += 1
            rec = {"path": r["path"], "category": r["category"],
                   "labels": labels, "source": "moondream"}
            if not a.no_raw:
                rec["raw"] = raw
            f.write(json.dumps(rec) + "\n")
            n_done += 1
        f.flush()

        el = time.perf_counter() - t0
        rate = n_q / max(el, 1e-9)
        left = (len(todo) - n_done) * (n_q / max(n_done, 1)) / max(rate, 1e-9)
        print(f"[{n_done}/{len(todo)}] {rate:.1f} q/s | "
              f"null={n_null} | ETA {left/60:.1f} min", flush=True)

    f.close()
    print(f"\n[done] {n_done} frames, {n_q} queries, {n_null} unparseable "
          f"in {(time.perf_counter()-t0)/60:.1f} min")
    if n_null:
        print(f"[warn] {n_null} checks written as null — excluded from "
              f"training, not guessed")


if __name__ == "__main__":
    main()
