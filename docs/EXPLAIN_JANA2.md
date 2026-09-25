# JANA2 — full system description, for ChatGPT to explain

Paste everything below the line into ChatGPT. The questions are at the end.

---

I have built an image captioning / surveillance analysis system. I want you to
explain it back to me thoroughly — the architecture, why each design decision
was made, what the trade-offs are, and where the weaknesses are. I know
PyTorch and general deep learning. Don't pad with basics I already have, but
don't skip steps either.

# 1. What it is for

Indian video surveillance: housing society gates, roads, mall entrances.
The output should describe a frame accurately enough that a security operator
can act on it, and be auditable enough to defend in an incident review.

Right now it works on single images. No video, no tracking.

# 2. History that shaped the design

A previous system (JANA-VLM, ~77M params) had this architecture:

    image → YOLO+MobileViT trunk → object memory slots → transformer decoder → caption

It failed in a specific, instructive way. The **memory was correct** — a glass-box
inspector proved the slot for a cat contained `cat` — but the **decoder wrote
"dog"**. Language priors overrode visual evidence. Five separate interventions
were tried over months (a name-expert channel, caption regeneration, training
data diversity, a binding loss, an anchor-only retrain). None fixed it.

Two lessons carried into the rebuild:

1. **A free-form language decoder trained to maximise caption likelihood will
   drift toward what sentences usually say.** That is not a bug you fix with
   more training; it's the objective doing its job.
2. **The metrics couldn't see the failure.** Validation cross-entropy is
   computed under teacher forcing, over non-masked tokens. It never measured
   generation, and a suspected `pad == bos == eos` collision meant the real
   end-of-sequence token was masked out of the loss entirely — so the decoder
   may never have learned to stop. Captions ran to the 256-token cap and
   degenerated into repeated phrases.

So JANA2 has **no language model anywhere.** Every word in the output is
produced by deterministic rules over structured evidence.

# 3. Pipeline

    image
      │
      ├─ PROPOSALS  "where is something?"
      │    yolo   : COCO detector (yolo26m @ 1280), 80 classes, labels KEPT
      │    world  : YOLO-World, open vocabulary from a text file
      │    rpn    : rpn_v2, a class-agnostic objectness head
      │    → geometric cleanup: min-area, NMS, part-suppression
      │
      ├─ NAMING     "what is each region?"
      │    CLIP (ViT-B-16-SigLIP2) embeds each crop
      │    cosine vs ~725 precomputed text embeddings, softmax over vocabulary
      │    a confident detector label overrides CLIP
      │
      ├─ ATTRIBUTES per object, grouped
      │    ~25 mutually-exclusive groups (gender, age, upper_colour, carrying,
      │    pose…), softmax WITHIN each group, omitted when the margin is small
      │
      ├─ PERSON CHECKS  yes/no on each person crop
      │    contrastive pairs: "a person holding a machete" vs
      │    "a person with empty hands" / "holding a phone" / "holding a tool"
      │
      ├─ MERGE      collapse duplicate boxes into one entity
      ├─ ANOMALIES  yes/no over whole frame + tiles, gated on context
      ├─ IMPORTANCE explicit weighted policy, not learned
      ├─ RELATIONS  geometry: wearing, holding, riding, at, has_part
      │
      └─ COMPOSE    evidence → caption, deterministic

# 4. Modules

| file | role |
|---|---|
| `config.py` | every setting, one dataclass |
| `clip_space.py` | CLIP loader; builds/loads text-embedding caches |
| `proposals.py` | the three detectors + geometric cleanup |
| `rpn_head.py` | vendored objectness head (inference only) |
| `naming.py` | orchestrates every pass, produces the evidence object |
| `attributes.py` | grouped attribute scoring |
| `anomaly.py` | contrastive yes/no checks, frame and per-object |
| `postprocess.py` | categories, duplicate merging, importance |
| `threat.py` | weapon taxonomy and threat assessment |
| `relations.py` | geometric relations between objects |
| `readout.py` | operator lines, evidence tables |
| `compose.py` | the caption |
| `evidence.py` | the dataclasses everything reads and writes |

Vocabularies live in text files, not code: `vocab.txt` (~725 object names),
`scenes.txt` (~40 scene labels), `attributes.txt`, `person_checks.txt`,
`anomalies.txt`, `world_classes.txt` (~140 detector prompts). Each is encoded
once into a `.npz` cache.

# 5. Design decisions and the reasoning behind each

## 5.1 One CLIP space, enforced

Every text embedding is computed in exactly one encoder's space, and the
encoder identity is written inside the cache file:

```python
np.savez(out_path, names=..., embs=embs, clip_space=space_id)

def load_vocab(path):
    z = np.load(path, allow_pickle=True)
    _, _, _, space_id = load_clip()
    if str(z["clip_space"]) != space_id:
        raise RuntimeError("vocab/encoder mismatch — rebuild")
```

The previous project had four vocabulary files in three different embedding
spaces (MiniLM, MobileCLIP, plain CLIP). Pairing the wrong one with the wrong
encoder still returns confident-looking cosine similarities — they are simply
wrong. This makes that impossible rather than unlikely.

## 5.2 Three different readouts for three different questions

This turned out to be the subtlest part of the whole system.

**Naming — softmax over the vocabulary.** The question is "which one of these
725 names fits?" That is a competition, so softmax.

**Attributes — softmax within a group.** "Which colour?" is a competition among
colours only, not against every name.

**Yes/no checks — contrastive pairs with a small temperature.** "Is there
fire?" is not a competition among 725 nouns. It is one claim against its
negation.

I originally used SigLIP's native sigmoid for naming. SigLIP is trained with a
sigmoid objective: `sigmoid(cos·exp(logit_scale) + logit_bias)` answers "does
this image match THIS ONE caption?", independently per name. Across 725
candidates almost every answer sits near zero, so I got `man 0.0012` and
`car 0.2248` for equally good matches. Ranking was fine; the numbers were
meaningless, and the top1−top2 margin was useless as a confidence signal.

Then the same parameter caused the opposite failure in the checks. With
`logit_scale ≈ 100`, a cosine gap of 0.05 between a positive and negative
phrase becomes a logit gap of 5, which softmaxes to 99%. On a studio portrait
of one woman, the system reported `no helmet 99%`, `crowd surge 87%`,
`darkness 92%`. It did not think there was a motorcycle — it just preferred
one meaningless sentence to another by a hair, and the scale turned that into
certainty.

Fixes: a modest fixed temperature (20 instead of ~100), plus a **raw cosine
margin floor**, so a check cannot fire when both phrases match badly:

```python
sp = (embs @ pos.T).max(-1).values     # best positive
sn = (embs @ neg.T).max(-1).values     # best negative
margin = sp - sn
prob = torch.sigmoid(margin * temperature)
fired = prob >= threshold and margin >= min_cos_margin
```

## 5.3 Gating: don't ask questions the scene can't answer

```
[no_helmet] severity=low threshold=0.70 requires=motorcycle,scooter,bicycle
[crowd_surge] severity=medium threshold=0.78 requires_people=8
[darkness] severity=low threshold=0.80 tiles=no
```

`requires=` means the check does not run unless that object was detected.
`tiles=no` restricts a check to the whole frame — one dark tile in a bright
image is a dark jacket, not a dark scene.

## 5.4 Detector labels beat CLIP zero-shot

COCO's person/car/motorcycle heads are heavily supervised. CLIP zero-shot on a
60-pixel crop is not close. So above a confidence threshold the detector's own
label is kept and CLIP only names what the detectors couldn't. The CLIP
alternatives are still recorded.

## 5.5 The detector was accidentally crippled two ways

`rpn_v2` (from the old project) consumes three feature maps from inside a YOLO
neck, with channel widths 64/128/256 — which are nano-scale. Because I first
used one model for both detection and RPN features, **the detector was forced
to be nano.** Splitting them lets a medium detector run at 1280 while a nano
model runs purely to feed the RPN.

Separately: **ultralytics reads a numpy array as BGR** (OpenCV convention). I
was passing `np.array(pil_image)`, which is RGB. Every frame had its red and
blue channels swapped. Passing the PIL image directly fixes it.

## 5.6 Merging is geometry-first

```python
if iou >= same_object_iou:        # 0.55
    merge                          # same object, whatever the labels say
elif same_canonical_name and (iou > 0.30 or containment > 0.70):
    merge
```

Requiring matching names meant YOLO's `person`, World's `woman` and CLIP's
`girl` on one body stayed three entities and the caption said "three people".
Two boxes sitting on top of each other are one object regardless of what the
labels claim. Containment merging still requires matching names, so a knife
inside a person box is never absorbed into the person.

## 5.7 Part-suppression must not eat held objects

The original rule dropped any box mostly inside a bigger box. A knife held by
a person is 100% inside the person box — so the rule deleted the single most
important object in a crime frame, by construction.

New rule: drop the child only if it is *both* mostly inside the parent *and*
close to the parent's size, i.e. a near-duplicate. A knife is a few percent of
a person's area and survives.

## 5.8 Importance is an explicit policy, not a learned score

```python
base  = 0.25*area + 0.20*centrality + 0.30*objectness + 0.25*name_confidence
score = CATEGORY_WEIGHT[category(name)] * base   (+ bonus if in an alert tile)
```

Every term is one line and inspectable. The old system had a learned
importance head that needed labels and could not be argued with. When this
ranks something wrongly you can see which term did it.

## 5.9 Parts are not entities

A caption listing "a person, a salwar kameez, a chain and a face" treats a
face as a peer of the person wearing it. So detections are categorised:

- `part` — face, hand, hair. Evidence only, never spoken.
- `garment` / `accessory` — only ever "a woman wearing X".
- entities — people, vehicles, bags, animals, weapons. These get listed.

Relations do the attaching, which makes relations load-bearing: a garment that
fails to produce a `wearing` relation silently disappears from the caption
instead of being listed.

## 5.10 Weapons are tiered by how much context they need

A flat weapon list ranked a misdetected yellow guard rail ("wooden stick",
weight 2.0) above an actual worker.

| tier | idle weight | held weight | examples |
|---|---|---|---|
| firearm | 3.00 | 3.50 | pistol, revolver, country-made pistol |
| blade | 2.20 | 3.00 | machete, sword, talwar, khukri |
| incendiary | 1.30 | 2.60 | petrol can, acid bottle |
| dual_use | 0.55 | 2.00 | knife, axe, sickle, hammer, crowbar |
| improvised | 0.40 | 1.60 | lathi, iron rod, sariya, cricket bat, brick |

Most objects that can be weapons are ordinary things. What distinguishes them
is not appearance but whose hand they are in. Person-level checks
(`aggressive`, `face_hidden`) escalate further. Only `confirmed`/`probable`
raise alerts; `possible`/`context` stay in the table without interrupting
anyone.

## 5.11 Abstention everywhere

Attribute groups include explicit "none" options ("a person carrying nothing",
"a fully visible face"), and a group is omitted entirely when top1 and top2
are within a margin. The previous system had no way to say "I don't see much"
— given a headshot with almost no objects, it confidently invented a black
umbrella. Sparse evidence must produce a short caption, not a fluent
fabrication.

## 5.12 Depth from geometry, in the composer

Small and high in the frame usually means far away, because cameras look
outward and down. So a wide society shot reads "several people are visible
further back" instead of "eight people". This is a heuristic and it breaks on
steeply downward-mounted cameras.

# 6. What the output looks like

Operator view, one line per entity:

    person · woman · young adult · yellow top · centre · 0.71   !! weapon held 0.83
    car · black · middle right · 0.44

Caption:

    A residential building compound. A young man in a dark shirt stands in the
    centre, holding a long blade. Two cars are on the left. Several people are
    visible further back.

# 7. Known weaknesses

- **No evaluation set.** No labelled ground truth, so no measured accuracy.
  There is a regression harness that freezes outputs on fixed images and diffs
  them after changes — it catches changes, not errors.
- **~3.9 s per image.** The anomaly pass alone is 14 CLIP crops.
- **CLIP still misnames things.** On a cat/kitten image it produced `dog` and
  `puppy` — the same failure mode as the old system, in a completely different
  architecture.
- **Thresholds are guesses.** A calibration script exists but needs real
  footage.
- **Composer failure mode is silence.** If a relation doesn't fire, the object
  vanishes rather than being described wrongly.

---

# Questions

Please answer these in order, referring to the specifics above.

1. **Explain the whole pipeline** end to end, as if to a new engineer joining
   the project. What happens to one image, at each stage, with tensor shapes.

2. **The three readouts.** Explain properly why softmax-over-vocabulary,
   softmax-within-group and contrastive-pair-with-temperature are the right
   choices for their respective questions, and what specifically goes wrong
   when you use the wrong one. Cover SigLIP's sigmoid objective, `logit_scale`
   and `logit_bias`, and why a large logit scale turns near-ties into
   near-certainties.

3. **Is the "no language model" decision correct?** The previous system failed
   because a decoder overrode correct evidence. Is a fully deterministic
   composer the right response, or an overcorrection? What is actually lost?

4. **Copy-constrained decoding.** I am considering a small transformer that can
   only emit function words plus tokens copied from the evidence. Explain
   pointer-generator networks, whether the copy constraint genuinely prevents
   hallucination, and what failure modes remain.

5. **Critique the importance formula.** Is a hand-weighted linear combination
   of area, centrality, objectness and name confidence reasonable? What would
   a learned alternative need, and would it be better?

6. **Critique the threat tiering.** Is idle-versus-held the right axis? What
   other context should modify a threat assessment? Where will this produce
   false negatives?

7. **The parts-vs-entities design** makes relations load-bearing, so a missing
   relation silently deletes an object from the caption. Is there a safer
   design with the same benefit?

8. **Evaluating without ground truth.** I cannot label a test set. Beyond a
   freeze-and-diff regression harness, what can be measured? Are there
   reference-free caption metrics (CLIPScore and similar) worth using, and
   what are their blind spots?

9. **The cat called a dog.** CLIP names a cat crop `dog` at 0.56. Give me a
   ranked list of causes — crop quality, ROI padding, resolution, prompt
   ensembling, vocabulary competition between near-synonyms, model size — and
   how to test each.

10. **What would you do next**, given the goal is accurate description of
    surveillance frames, no video, and no labelled data?

Use concrete numbers and small examples. Define jargon once, then use it
freely. Where you disagree with a decision, say so plainly and explain why.
