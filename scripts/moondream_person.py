"""Ask Moondream the PERSON-level questions about person crops.

    export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/root/aniket/git_clones/image-captioning/.venv-vllm/lib/python3.10/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
    export VLLM_USE_FLASHINFER_SAMPLER=0
    /root/aniket/git_clones/image-captioning/.venv-vllm/bin/python \
        scripts/moondream_person.py --manifest data/ucf_person/person_crops.jsonl \
        --out data/ucf_person/verified.jsonl

DIFFERENT FROM THE FRAME PASS IN ONE IMPORTANT WAY

There, frames from Normal videos were trusted negatives without asking — a
normal video contains no fight. That shortcut is WRONG here. Normal footage
is full of people in uniform, carrying large objects, with faces turned
away. Marking those zero would teach the probe that a security guard is not
in uniform.

So every crop is asked, including Normal. Roughly double the queries, and
not optional.

Which questions get asked still depends on the source category — asking
"is this person climbing a wall" of every shopper wastes time on a question
whose answer is almost always no. But the descriptive checks (uniform,
face_hidden, carrying_large) are asked EVERYWHERE, because they occur
everywhere.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

from PIL import Image

TMPL = "<|endoftext|><image><|md_reserved_0|>{q}<|md_reserved_1|>"

QUESTIONS = {
    "weapon_held":     "Is this person holding a weapon such as a gun, knife, stick or iron rod?",
    "aggressive":      "Is this person being physically aggressive, raising a fist, or about to strike someone?",
    "on_ground":       "Is this person lying on the ground or collapsed?",
    "face_hidden":     "Is this person's face covered or hidden from view?",
    "carrying_large":  "Is this person carrying a large or heavy object?",
    "climbing_person": "Is this person climbing over a wall, fence or gate?",
    "fleeing":         "Is this person running away in a hurry?",
    "uniform":         "Is this person wearing a uniform, such as a security guard, police or delivery uniform?",
}

# Descriptive checks occur in ordinary footage, so they are asked of every
# crop regardless of where it came from.
ALWAYS = ["uniform", "face_hidden", "carrying_large"]

# Incident checks are only worth asking where they could plausibly occur.
EXTRA = {
    "Fighting":      ["aggressive", "weapon_held", "on_ground"],
    "Assault":       ["aggressive", "weapon_held", "on_ground"],
    "Abuse":         ["aggressive", "on_ground"],
    "Arrest":        ["aggressive", "on_ground"],
    "Shooting":      ["weapon_held", "on_ground", "fleeing"],
    "Robbery":       ["weapon_held", "aggressive", "fleeing"],
    "Stealing":      ["fleeing"],
    "Shoplifting":   ["fleeing"],
    "Burglary":      ["climbing_person", "fleeing"],
    "Vandalism":     ["aggressive"],
    "RoadAccidents": ["on_ground"],
    "Normal":        [],
}

NEG = ("no", "none", "not ", "nothing", "nobody", "neither", "absent")
POS = ("yes", "there is", "there are", "someone is", "appears to",
       "this person is", "the person is")


def parse_yes_no(text):
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
    ap.add_argument("--manifest", default="data/ucf_person/person_crops.jsonl")
    ap.add_argument("--out", default="data/ucf_person/verified.jsonl")
    ap.add_argument("--model", default="moondream/moondream3-preview")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--gpu-util", type=float, default=0.75)
    ap.add_argument("--max-crops", type=int, default=0)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.manifest) if l.strip()]
    if a.max_crops:
        rows = rows[:a.max_crops]

    outp = Path(a.out)
    done = set()
    if outp.exists():
        for l in open(outp):
            try:
                done.add(json.loads(l)["path"])
            except Exception:                                 # noqa: BLE001
                pass
        print(f"[resume] {len(done)} crops already done")
    todo = [r for r in rows if r["path"] not in done]
    print(f"[plan] {len(todo)} crops to query")
    if not todo:
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, tokenizer="moondream/starmie-v1",
              trust_remote_code=True, max_model_len=4096,
              gpu_memory_utilization=a.gpu_util,
              limit_mm_per_prompt={"image": 1}, disable_log_stats=True)
    sp = SamplingParams(max_tokens=24, temperature=0.0)

    warm = Image.open(todo[0]["path"]).convert("RGB")
    llm.generate([{"prompt": TMPL.format(q="Is this an image?"),
                   "multi_modal_data": {"image": warm}}], sp)
    print("[warmup] done")

    f = open(outp, "a")
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
            asks = ALWAYS + EXTRA.get(r["category"], [])
            for c in asks:
                reqs.append({"prompt": TMPL.format(q=QUESTIONS[c]),
                             "multi_modal_data": {"image": img}})
                keys.append((bi, c))
        if not reqs:
            continue

        outs = llm.generate(reqs, sp)
        n_q += len(reqs)
        res, retry_r, retry_k = {}, [], []
        for (bi, c), o in zip(keys, outs):
            txt = o.outputs[0].text
            v = parse_yes_no(txt)
            if v is None:
                retry_r.append({"prompt": TMPL.format(
                    q=QUESTIONS[c] + " Answer yes or no."),
                    "multi_modal_data": {"image": imgs[bi]}})
                retry_k.append((bi, c))
            res[(bi, c)] = (v, txt)
        if retry_r:
            for (bi, c), o in zip(retry_k, llm.generate(retry_r, sp)):
                res[(bi, c)] = (parse_yes_no(o.outputs[0].text),
                                o.outputs[0].text)
            n_q += len(retry_r)

        for bi, r in enumerate(block):
            if bi not in imgs:
                continue
            asks = ALWAYS + EXTRA.get(r["category"], [])
            labels, raw = {}, {}
            for c in asks:
                v, txt = res.get((bi, c), (None, ""))
                labels[c] = v
                raw[c] = txt
                if v is None:
                    n_null += 1
            f.write(json.dumps({"path": r["path"], "category": r["category"],
                                "labels": labels, "raw": raw,
                                "source": "moondream"}) + "\n")
            n_done += 1
        f.flush()

        el = time.perf_counter() - t0
        rate = n_q / max(el, 1e-9)
        left = (len(todo) - n_done) * (n_q / max(n_done, 1)) / max(rate, 1e-9)
        print(f"[{n_done}/{len(todo)}] {rate:.1f} q/s | null={n_null} | "
              f"ETA {left/60:.1f} min", flush=True)

    f.close()
    print(f"\n[done] {n_done} crops, {n_q} queries, {n_null} unparseable "
          f"in {(time.perf_counter()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
