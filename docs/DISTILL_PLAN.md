# Shrinking the encoder by distillation

## The number we are attacking

| component | params |
|---|---|
| yolo26n | ~3M |
| open-vocab detector (world-s or yoloe-26n) | ~4–13M |
| **SigLIP2 ViT-B-16 image tower** | **93M** |
| total | ~100–109M |

The detector choice moves single-digit millions. **SigLIP2 is 85–90% of the
budget.** It is the only component worth attacking.

## Why distillation is defensible here

SigLIP2 was trained to embed the entire internet — artwork, food, celebrities,
animals, memes. This system sees Indian roads, society gates, mall entries,
vehicles, people, and weapons. Most of that 93M is capacity for a
distribution we never encounter.

A student trained only on the target distribution can be much smaller *for
the same quality on our data*. That is a real argument, not premature
optimisation.

Target: **MobileCLIP-S2 (~36M) or MobileCLIP-S1 (~21M)**, taking the system
to roughly **28–43M total** — comfortably under the 100M ceiling with room
to raise the detector back up.

## The three decisions

**1. Inherit, never start from scratch.**
Random init needs millions of images. Starting from a pretrained small CLIP
visual tower and fine-tuning it to imitate SigLIP2 (TinyCLIP's own recipe)
needs tens of thousands. This is the difference between a feasible project
and an infeasible one.

**2. Replace the image tower only. Keep SigLIP2's text tower.**
The student is trained to land in the teacher's embedding space, so:

- all five `.npz` caches keep working
- `vocab.txt`, `anomalies.txt`, `person_checks.txt`, `attributes.txt` unchanged
- thresholds, gating, `threat.py`, the composer — all untouched

Swap the encoder, keep the system.

**3. Optimise the quantity the pipeline actually reads.**
This is the one that is easy to get wrong. Nothing downstream consumes a raw
embedding. Everything consumes

```
margin = max(sim to positive phrases) − max(sim to negative phrases)
```

and those margins live in a **0.02–0.05 band** — we measured that the hard
way while chasing the "weapon held 61%" bug. A student can hit 0.97 cosine
agreement with the teacher and still compress that band to nothing, at which
point every check goes silent and the app *looks* fine while detecting
nothing.

So the loss has three terms, and the task term carries the highest weight:

| loss | what it protects | weight |
|---|---|---|
| direct | absolute position in the space | 1.0 |
| structure | crop-to-crop relative geometry | 1.0 |
| **task** | similarity distribution over *our* phrases | **2.0** |

## Pipeline

```bash
# 1. harvest — crops from real frames, using the real proposer.
#    No labels needed. Videos accepted; --stride skips near-duplicates.
python -m scripts.distill_harvest --frames /path/to/cctv --out data/distill

# 2. train
python -m scripts.distill_train --data data/distill --student MobileCLIP-S2

# 3. the decision — margin preservation, not agreement
python -m scripts.distill_eval --ckpt checkpoints_distill/student_best.pt
```

## The adoption gate

`distill_eval.py` prints a **ratio** per check: student 95th-percentile
margin ÷ teacher 95th-percentile margin.

- **≈ 1.0** — the check behaves identically. Adopt.
- **0.8–1.0** — usable, but thresholds need re-tuning against the new scale.
- **< 0.8** — that check has lost discriminating power.

If the mean ratio is well below 0.8, the honest conclusion is that the
student is not good enough and we stay on SigLIP2. Build the measurement so
that a negative result is visible rather than hidden.

## Data requirement — the real constraint

| crops | expectation |
|---|---|
| < 20k | overfits the few scenes it saw |
| 50–100k | workable for domain specialisation |
| 200k+ | comfortable |

Frames are free if you have camera access — hours of CCTV, sampled every
Nth frame, no annotation. **Breadth beats volume**: 50k crops from twenty
cameras is worth more than 200k from two, because the student will inherit
whatever bias the source scenes have.

## What this does not fix

Distillation copies the teacher, including its mistakes. If SigLIP2 already
struggles to separate "a person holding a large knife" from "a person
holding an umbrella" — and the 0.02 margins suggest it does — the student
inherits that. Distillation buys **size**, not accuracy.

If the margins turn out to be the real problem, the fix is different work:
rewriting the check phrases so they are visually distinctive rather than
grammatically parallel. Worth knowing which problem we have before spending
weeks on the other one.

## Integration, once a student passes the gate

Not built yet, deliberately — no point wiring in a model that may not clear
the bar. When one does, the change is small:

1. add `clip_student_ckpt: Path | None = None` to `config.py`
2. in `clip_space.load_clip()`, if that is set, load the student and return
   it in place of the teacher's visual tower, keeping the teacher's
   tokenizer and `space_id` (the caches must still validate)
3. rebuild nothing — the `.npz` files are text-side and unchanged
