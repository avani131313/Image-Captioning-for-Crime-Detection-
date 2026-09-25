# JANA2, explained backwards

A study document. We start at the finished caption and work back to the raw
pixels, asking at each step: *where did that come from?*

Reading forwards tells you what the code does. Reading backwards tells you
why each piece has to exist.

New terms are defined in boxes like this, at the point they first matter:

> **Term** — plain definition, then why it matters here.

Full glossary at the end.

---

# 0. The finished output

Here is what the system produced for a frame from a society gate:

```
The entrance gate of a housing society. A young man wearing a helmet is
riding a scooter in the centre, carrying a delivery bag. A boom barrier is
on the left. Several people are visible further back.
```

Alongside it, the operator view:

```
person · young adult · man · helmet · center · 0.71
scooter · center · 0.66
boom barrier · middle left · 0.52
```

Every single word above traces to a number. Our job in this document is to
follow each one back to where it was produced.

Three things are worth noticing before we start.

**Nothing is invented.** "Helmet" is in the output because a box was named
`helmet`. If no box had been named that, no amount of the scene looking like
a delivery rider would put the word there.

**Nothing is hedged.** The caption does not say "possibly wearing a helmet".
Either the evidence cleared the bar and the word appears, or it did not and
the word is absent. There is no middle register.

**Words are missing on purpose.** The man's shirt colour is not mentioned.
That is not an oversight — the colour question was asked, the answer was too
close to call, and it was dropped. Silence is a designed output.

---

# 1. Layer 7 — the composer

**File:** `jana2/compose.py`
**Models used:** none

The composer receives a finished data structure and turns it into English. It
has no vocabulary of its own beyond function words — *a, the, is, in,
wearing, carrying, and*. Every noun and adjective is copied from the data.

## 1.1 What it receives

A `StructuredEvidence` object:

```python
StructuredEvidence(
    image_path = "gate_0412.jpg",
    width = 1920, height = 1080,
    objects = [ObjectSlot, ObjectSlot, ...],
    scene = {"clip_names": [("the entrance gate of a housing society", 0.41),
                            ("a residential building compound", 0.22), ...]},
    relations = [{"subject": 1, "predicate": "riding", "object": 2, ...}, ...],
    anomalies = [{"name": "fire", "prob": 0.12, "triggered": False, ...}, ...],
    timings_ms = {...},
)
```

and each `ObjectSlot`:

```python
ObjectSlot(
    idx = 1,
    box = (820.0, 340.0, 1090.0, 910.0),   # x0, y0, x1, y1 in pixels
    objectness = 0.94,
    source = "yolo",
    named_by = "detector",
    area_frac = 0.074,
    position = "center",
    clip_names = [("person", 0.94), ("man", 0.31), ("pedestrian", 0.12)],
    attributes = {"gender": {"label": "man", "prob": 0.68, "margin": 0.36},
                  "age":    {"label": "young adult", "prob": 0.54, "margin": 0.19}},
    checks = {},
    importance = 0.81,
    merged_count = 2,
)
```

> **Data structure vs model** — everything above is an ordinary Python object
> holding numbers. No neural network is involved from here to the end of the
> caption. This is the point at which the system stops perceiving and starts
> reporting.

## 1.2 The four decisions it makes

**Which objects to speak about.** It drops anything below the score
thresholds, anything whose category is `part` or `scenery`, and — importantly
— anything that already appears as the *object* of an attaching relation.

That last rule is why "helmet" does not appear as its own item. There is a
relation `person 1 —wearing→ helmet 4`, so helmet 4 is marked as spoken-for
and will be mentioned through the person instead.

**Which one leads.** The highest `importance` becomes the grammatical
subject, and gets described in full. Everything else is context.

**Foreground or background.** Small area plus high position in the frame is
treated as far away:

```python
if area_frac >= 0.12:            foreground
elif area_frac <= 0.015 and centre_y < 0.55*H:   background
elif area_frac >= 0.04:          foreground
else:                            midground
```

Background entities are counted, not described. "Several people are visible
further back" instead of eight separate clauses.

> **Heuristic** — a rule that is usually right rather than provably right.
> This one assumes a camera looking outward and slightly down. On a camera
> mounted high and pointing steeply down it will be wrong, and it will be
> wrong *confidently*, which is worse than being absent.

**Which verb.** Chosen from the evidence, not from a language model:

| evidence | verb |
|---|---|
| a `riding` relation exists | "is riding" |
| attribute `pose = walking` | "is walking" |
| attribute `pose = lying on the ground` | "is lying on the ground" |
| person, nothing else | "stands" |
| vehicle | "is parked" |

## 1.3 Building the noun phrase

For our man:

```
attributes: age="young adult", gender="man"
relations:  wearing → helmet,  riding → scooter,  holding → delivery bag
```

becomes

```
adjectives  = ["young adult"]
head noun   = "man"                        (gender attribute overrides "person")
worn        = ["helmet"]                   (from the wearing relation)
held        = ["delivery bag"]             (from the holding relation)
mount       = "scooter"                    (from the riding relation)

→ "a young man wearing a helmet"  +  "is riding"  +  "in the centre"
  +  ", carrying a delivery bag"
```

**Working backwards, the questions are now:** where did `relations` come
from, where did `importance` come from, and where did `attributes` come
from?

---

# 2. Layer 6 — the reasoning layer

**Files:** `jana2/relations.py`, `jana2/threat.py`, `jana2/postprocess.py`
**Models used:** none

Still no neural network. This layer only reads box coordinates and names.

## 2.1 Relations — pure geometry

The claim "he is riding the scooter" is not a harder perception problem than
"there is a man" and "there is a scooter". It is those two boxes in a
particular arrangement. So we measure the arrangement.

Three helper quantities, all from coordinates:

```python
inside_frac(a, b) = area(a ∩ b) / area(a)     # how much of A is inside B
h_overlap(a, b)   = width(a ∩ b) / min(width(a), width(b))
v_overlap(a, b)   = height(a ∩ b) / min(height(a), height(b))
```

> **∩ (intersection)** — the overlapping rectangle between two boxes. If they
> do not overlap, its area is zero.

The rules:

| relation | condition |
|---|---|
| `wearing` | B is a person, A is a garment or accessory, `inside_frac(A,B) > 0.55` |
| `has_part` | B is a person, A is a body part, `inside_frac(A,B) > 0.6` |
| `holding` | B is a person, A is carryable, `inside_frac(A,B) > 0.55` **and** `area(A)/area(B) < 0.35` |
| `riding` | A is a person, B is a two-wheeler, `h_overlap > 0.5` and `v_overlap > 0.15` |
| `inside` | A is a person, B is a car or bus, `inside_frac(A,B) > 0.75` |
| `at` | A is a person, B is a gate or counter, gap < 3% of the image diagonal |

The `holding` rule needs both clauses. Without the size clause, the person's
own jacket would qualify as held — it is 100% inside the person box. The
0.35 limit says a held thing is small relative to its holder.

> **Why these rules are deliberately strict.** A relation that fires when it
> should not becomes a sentence asserting something that never happened. When
> the geometry is ambiguous, we emit nothing and the object falls back to
> being listed separately. Silence is recoverable; a false statement is not.

**The cost of this design:** relations are load-bearing. If the `wearing`
rule fails to fire on a garment, the composer still marks that garment as
attachable, so it is neither attached nor listed — it vanishes. The `spoken`
column in the evidence table exists to catch exactly this.

## 2.2 Threat — context, not appearance

A flat weapon list ranked a misdetected yellow guard rail above an actual
worker, because "wooden stick" was in the weapons set with a high weight.

The fix is to recognise that **most objects that can be weapons are ordinary
things**, and what distinguishes them is whose hand they are in.

| tier | idle weight | held weight | examples |
|---|---|---|---|
| firearm | 3.00 | 3.50 | pistol, revolver, country-made pistol |
| blade | 2.20 | 3.00 | machete, sword, talwar, khukri |
| incendiary | 1.30 | 2.60 | petrol can, acid bottle |
| dual_use | 0.55 | 2.00 | knife, axe, sickle, hammer, crowbar |
| improvised | 0.40 | 1.60 | lathi, iron rod, sariya, cricket bat, brick |

The *held* column is only used when a `holding` relation points at the object
— which is why threat assessment must run after relations, not before.

Person-level flags escalate further. A knife held by someone the person-check
pass flagged as `aggressive` becomes `confirmed`; the same knife held by
someone with no flags is `possible`. Only `confirmed` and `probable` raise
alerts.

## 2.3 Importance — an explicit policy

```python
base  = 0.25·area + 0.20·centrality + 0.30·objectness + 0.25·name_confidence
score = CATEGORY_WEIGHT[category] · base   (+0.60 if inside an alert tile)
```

where

```python
area        = min(1, sqrt(area_frac) · 2.2)
centrality  = 1 − (distance of box centre from image centre) / 0.707
```

> **Why square-root the area.** Raw area is dominated by one large object. The
> square root compresses the range so a person at 7% of the frame and a wall
> at 40% are not two orders of magnitude apart.

> **Why 0.707.** That is √2/2, the distance from the centre of a unit square
> to its corner. Dividing by it puts the corner at exactly 1.0.

Category weights: person 1.00, vehicle 0.75, carried 0.70, animal 0.55,
access 0.45, scenery 0.05, part 0.00. Threat weights come from the table
above.

**This is deliberately not learned.** A learned importance head would need
labelled data — someone marking what matters in thousands of frames — and
when it ranked something wrongly you could not ask it why. Here you can point
at the term that caused it and change that number.

## 2.4 Merge — before all of the above

Merging runs *before* ranking and relations, because both operate on
`idx` values that renumbering would invalidate.

```python
if IoU(a, b) >= 0.55:
    merge — same object, whatever the two labels say
elif canonical_name(a) == canonical_name(b) and (IoU > 0.30 or contained > 0.70):
    merge
```

> **IoU (Intersection over Union)** — overlap area divided by combined area.
> 0 means no overlap, 1 means identical boxes. The standard measure of "are
> these the same box".

The first rule matters more than it looks. YOLO says `person`, YOLO-World
says `woman`, CLIP names a third box `girl`. All three sit on one body. If
merging required matching names they would survive as three entities and the
caption would say "three people". Geometry has to win.

> **Canonical name** — a lookup that collapses synonyms: man, woman, worker,
> pedestrian, crowd → `person`. Used for counting and for the second merge
> rule. Note the caption still prints the *specific* name when a group has
> one member, so canonicalising does not cost you detail.

**Working backwards, the question is now:** where did `attributes`,
`checks` and `clip_names` come from?

---

# 3. Layer 5 — the whole-image passes

**File:** `jana2/anomaly.py`
**Model used:** SigLIP2 image encoder

Two things are asked of the whole frame rather than of individual boxes.

## 3.1 Scene

The full image is embedded and compared against 40 scene labels from
`scenes.txt`, softmax across them. Highest wins, if it beats second place by
more than 0.05.

This is a *separate* vocabulary from the object list, and that separation was
learned the hard way. Scoring the whole image against object names produced
"a worker scene" — because a list of objects can only answer with an object.

## 3.2 Anomalies

The frame is cut into 1 + 4 + 9 = **14 tiles**: the whole image, a 2×2 grid,
and a 3×3 grid. All 14 are embedded in one batch.

> **Why tiles.** A fire in one corner barely moves the embedding of the whole
> image — it is a small part of a large picture. Scoring each tile separately
> lets a local event register, and tells you *where* it registered.

Each check then runs the contrastive-pair maths (section 4.3) against all 14
and takes the highest. Checks marked `tiles=no` only look at tile 0.

**Gating happens first.** A check declared as

```
[no_helmet] threshold=0.70 requires=motorcycle,scooter,bicycle
```

does not run at all unless one of those names is among the detected objects.
Without gating, a studio portrait produced `no helmet 99%` — the model did not
think there was a motorcycle, it merely preferred one meaningless sentence to
another by a hair, and the maths turned that into apparent certainty.

---

# 4. Layer 4 — three readouts, one embedding

**Files:** `jana2/naming.py`, `jana2/attributes.py`, `jana2/anomaly.py`

This is the conceptual heart of the system, so it is worth going slowly.

We have, for each of ~48 boxes, a vector of 768 numbers. We also have
pre-computed vectors for every word and phrase in the text files. Everything
in this layer is comparing those vectors.

> **Embedding** — a list of numbers representing something, arranged so that
> similar things get similar lists. SigLIP2 produces 768 numbers for an image
> and 768 numbers for a piece of text, *in the same space* — which is what
> lets you compare a picture to a word.

> **Unit-normalised** — every vector is divided by its own length, so they all
> sit on the surface of a sphere. This means the dot product between two of
> them equals the cosine of the angle between them.

> **Cosine similarity** — a number from −1 to 1 measuring how aligned two
> vectors are. 1 = same direction, 0 = unrelated, −1 = opposite. For
> unit-normalised vectors it is just `a · b`, one matrix multiply.

Three different questions are asked of these vectors, and each needs
different arithmetic. Using the wrong one caused two of the worst bugs in
this project.

## 4.1 Naming — "which one of 725?"

```python
sims   = embs @ vocab.T             # [48, 768] × [768, 725] → [48, 725]
logits = sims * exp(logit_scale)    # logit_scale is learned; exp ≈ 100
probs  = softmax(logits, dim=-1)    # across the 725 names
```

> **Softmax** — turns a list of scores into a probability distribution:
> exponentiate each, divide by the total. The results sum to 1. It expresses
> *competition* — if one option gains, the others must lose.

Softmax is right here because the question genuinely is a competition. The
crop is one thing; exactly one of the 725 names should win.

> **logit_scale** — a learned scalar inside CLIP-style models, typically
> around `log(100)`. Raw cosine similarities are bunched between roughly 0.1
> and 0.3; multiplying by ~100 spreads them out before softmax, otherwise the
> distribution would be nearly flat.

**The bug this replaced.** SigLIP's native readout is
`sigmoid(logits + logit_bias)`.

> **Sigmoid** — squashes one number to the range 0-1, independently of any
> other number. Expresses *independent presence*, not competition.

SigLIP is trained with a sigmoid objective, so `sigmoid(cos·scale + bias)`
answers "does this image match **this one** caption?" — independently, per
name. Across 725 candidates almost every answer is near zero. That produced
`man 0.0012` and `car 0.2248` for two equally good matches. The ranking was
still correct, but the numbers were meaningless and the top1−top2 gap was
useless as a confidence signal.

Softmax asks the question we actually have, and the numbers become
comparable across objects.

**The override.** If a detector supplied a label above its trust threshold,
that label goes into position 0 and CLIP's answers become the alternatives.
COCO's `person` head has seen millions of labelled people; zero-shot CLIP on a
60-pixel crop has not. The specialist wins where it applies.

> **Zero-shot** — using a model on categories it was never explicitly trained
> to classify, by describing them in text. Flexible, but weaker than a model
> trained directly on those categories.

## 4.2 Attributes — "which colour?"

```python
sub    = attr_embs[41:59]        # the 18 rows belonging to upper_colour
sims   = roi_emb @ sub.T         # [18]
probs  = softmax(sims * scale)   # across those 18 only
if probs[0] - probs[1] < 0.10:
    omit the group entirely
```

The competition is restricted to one group. Colours compete with colours.
"A person wearing a black top" is never scored against "a photo of a man",
because those answer different questions.

Two design details carry weight:

**Groups have explicit "none" options.** `carrying` includes "a person
carrying nothing"; `face_cover` includes "a person with a fully visible face".
Without them the group is forced to pick *some* bag, and you get invented
evidence.

**The margin test.** If first and second place are within 0.10, nothing is
recorded. In a crime report an omitted shirt colour is fine; a guessed one is
evidence that never existed.

> **Margin** — the gap between the best and second-best score. A good proxy
> for confidence: a large margin means the model discriminated clearly, a tiny
> one means it effectively could not tell.

## 4.3 Checks — "yes or no?"

```python
sp     = (emb @ positives.T).max()    # best-matching positive phrase
sn     = (emb @ negatives.T).max()    # best-matching negative phrase
margin = sp - sn                       # raw cosine gap
prob   = sigmoid(margin * 20)
fires if prob >= threshold and margin >= 0.02
```

The question is one claim against its negation, so neither softmax-over-725
nor softmax-within-group fits. We compare two sides directly.

**`max`, not `mean`.** One strongly-matching positive should not be diluted
by five weak ones. If the image matches "a person holding a machete" strongly,
that is the signal, regardless of how "a person holding a sword" scored.

**The negatives are the important half.** Without something to compare
against, "a person holding a knife" always scores *something* and you cannot
tell whether that is a lot or a little. The negatives establish a baseline.

**Temperature 20, not 100.** This was the second bug. At `exp(logit_scale)` ≈
100, a cosine gap of 0.05 becomes a logit gap of 5, which sigmoids to 99%. So
when *neither* phrase matches the image, whichever is fractionally ahead comes
back as near-certain. At 20 the same gap gives 73%.

> **Temperature** — a divisor (or here a multiplier) controlling how sharply
> scores are converted to probabilities. High temperature → confident,
> near-binary outputs. Low → soft, hedged ones. Choosing it badly makes a
> model look certain about things it cannot distinguish.

**The cosine floor.** `margin >= 0.02` means two equally poor matches produce
nothing at all, whatever the temperature does. Preferring one bad answer over
another bad answer is not evidence.

---

# 5. Layer 3 — the embedding

**File:** `jana2/clip_space.py`
**Model:** SigLIP2 ViT-B-16, image tower only, ~86M parameters, frozen

Every box is expanded 8% on each side and cropped from the original image.

> **ROI padding** — deliberately including a margin of context around the box.
> A tightly-cropped face is harder to identify than a face with some shoulders
> visible. 8% is a compromise: too little and there is no context, too much
> and the neighbouring object starts influencing the answer.

Each crop goes through the model's own preprocessing — resize to 224×224,
convert to tensor, normalise with that model's mean and standard deviation.

> **Normalisation** — subtracting a mean and dividing by a standard deviation
> per colour channel, so inputs match the statistics the model saw in
> training. **These values differ per model.** MobileCLIP, for instance, uses
> mean 0 and std 1 — raw 0-1 pixels — while most CLIP variants use ImageNet-ish
> statistics. Applying the wrong ones leaves coarse recognition roughly intact
> while destroying fine distinctions, which shows up as correct names but
> wrong attributes.

This is why the transform always comes from `create_model_and_transforms()`
and is never written by hand.

All crops are stacked into `[48, 3, 224, 224]` and pushed through in **one**
forward pass, producing `[48, 768]`, then unit-normalised.

**This tensor is computed once and used three times** — naming, attributes,
person checks. The second and third uses are a single matrix multiply each.
That is why adding the attribute pass and the weapon check barely changed the
runtime.

> **ViT (Vision Transformer)** — an image model that cuts the picture into
> fixed patches (16×16 pixels here, hence "B-16"), treats each patch as a
> token, and runs transformer attention over them. "B" is the base size.

> **Frozen** — the weights never change. There is no training in this system;
> every model is used exactly as downloaded.

---

# 6. Layer 2 — geometric cleanup

**File:** `jana2/proposals.py`
**Models used:** none

Roughly 55 raw boxes arrive from three detectors. Four filters reduce them.

**Specks.** Drop anything under `min_area_frac` (0.0015) of the frame. On
1920×1080 that is about 3100 px² — a 55×55 box.

**NMS, with priority.**

> **NMS (Non-Maximum Suppression)** — sort boxes by score, keep the best, and
> delete every remaining box overlapping it by more than a threshold. Repeat.
> The standard way to remove duplicate detections.

Before sorting, each score gets a bump by source: yolo +3, world +2, rpn +1.
So when a yolo box and an rpn box overlap past 0.6, yolo survives regardless
of raw scores. The bump exists only for ordering and is discarded after.

This is necessary because **the three sources' scores are not comparable**.
YOLO's confidence is a class probability from a supervised head. rpn_v2's is
a sigmoid objectness from a completely different objective. Sorting them
together as if they meant the same thing would let one source silently starve
another.

**Part-suppression.** Drop box A if all three hold:

```
inside_frac(A, B) >= 0.90        A is almost entirely inside B
area(A)/area(B)   >= 0.55        A is a similar size to B  → a duplicate
area(B)/frame     <= 0.60        B is an object, not scenery
```

The middle condition is the one that matters. **A knife held by a person is
100% inside the person's box.** An earlier version dropped anything mostly
inside a bigger box, and therefore deleted the single most important object in
a crime frame, by construction. The 0.55 size test distinguishes a duplicate
box of the same object (~90% of the parent) from a small distinct thing (~4%).

**Budget.** Sort by source priority then score, keep 48.

---

# 7. Layer 1 — the detectors

**File:** `jana2/proposals.py`, `jana2/rpn_head.py`

Three sources, answering "where is something?" — never "what is it?" in the
semantic sense we care about.

## 7.1 yolo26 — the specialist

Trained on COCO: 80 categories, millions of labelled boxes. Excellent on
person, car, motorcycle, bus, truck, bicycle — which is most of what appears
in surveillance. Blind to everything else.

Prediction letterboxes 1920×1080 to 1280×736, runs the network, applies
internal NMS, and maps boxes back to original coordinates. Returns
`xyxy [N,4]`, `conf [N]`, `cls [N]`.

> **Letterboxing** — resizing to fit the target while preserving aspect ratio,
> padding the remainder. Distinct from squashing, which distorts. Ultralytics
> letterboxes; our RPN path squashes, because that is what the RPN's training
> assumed.

> **A bug worth remembering.** Ultralytics interprets a numpy array as **BGR**
> (OpenCV's channel order). We were passing `np.array(pil_image)`, which is
> **RGB** — so every frame had red and blue swapped. Passing the PIL image
> directly fixes it, because PIL is understood as RGB.

## 7.2 YOLO-World — the generalist

> **Open-vocabulary detection** — a detector that takes its class list as
> text at runtime rather than having it fixed at training time.

At startup, `set_classes()` runs your 140 lines from `world_classes.txt`
through a CLIP text encoder and writes the resulting vectors into the
detection head. The model's final layer literally contains your class list.

This is why that file is short. An open-vocabulary detector loses precision as
the prompt list grows, and near-synonyms split the score between them. 140
carefully chosen prompts beat 700 sloppy ones.

It also accepts *phrases*, not only nouns: `motorcycle rider without helmet`,
`unattended bag on the ground`. It can detect a situation directly, which is
often easier than detecting two objects and inferring the relation.

## 7.3 rpn_v2 — the objectness head

> **Class-agnostic proposals** — boxes for "something is here" with no
> attempt to say what. Useful for things no vocabulary contains.

This is 0.21M parameters inherited from the previous project, and it does not
take an image. It takes three feature maps from partway inside a YOLO network:

```
P3  [1, 64, 80, 80]     stride 8      small objects
P4  [1, 128, 40, 40]    stride 16     medium
P5  [1, 256, 20, 20]    stride 32     large
```

> **Feature map** — the intermediate output of a convolutional layer. Early
> layers hold edges and textures; later ones hold shapes and object parts. A
> stride-8 map has one cell per 8×8 pixel region.

> **Feature pyramid (P3/P4/P5)** — several maps at different resolutions, so
> small objects can be found on the fine map and large ones on the coarse map.

Those channel widths — 64, 128, 256 — are nano-scale, which is why the
feature network must be a nano YOLO. **This constraint silently crippled the
detector for a while**, because one model was serving both roles, forcing
detection to be nano too. Splitting them let a medium detector run at 1280
while a nano runs purely to feed the RPN.

The head itself: a 1×1 convolution projecting each level to 128 channels, a
shared 3×3 tower, then two output heads — one objectness logit per grid
position, and four distances (left, top, right, bottom) per position, forced
positive by softplus.

Decoding at grid cell `(r, c)` on P3: the pixel centre is
`((c+0.5)·8, (r+0.5)·8)`, and the box is `centre ± ltrb·stride`. Sigmoid the
objectness. Top 300 per level, concatenate, NMS at 0.6, keep 32.

> **Softplus** — a smooth function that is always positive. Used here because
> a distance cannot be negative.

**Whether it earns its slot is an open question.** rpn_v2's boxes arrive with
no name, so CLIP must guess from a bare crop with no detector prior — and
that is where `dog`, `puppy` and `wooden stick` came from on real images.
Running with `sources = ("yolo", "world")` and comparing is the experiment
that settles it.

---

# 8. Layer 0 — the image, and the build step

## 8.1 The image

A JPEG, opened as PIL RGB. That is all.

## 8.2 What happened before any image arrived

The text files became `.npz` caches, once.

Take `vocab.txt`, 725 lines. Each name is wrapped in four templates:

```
"a photo of a scooter."
"a close-up photo of a scooter."
"a cropped photo of a scooter."
"a photo of the scooter in a scene."
```

> **Prompt ensembling** — embedding several phrasings of the same concept and
> averaging them. Cancels the quirks of any single phrasing and measurably
> improves zero-shot accuracy.

Each template batch is tokenised to `[725, 64]` and pushed through SigLIP2's
**text** encoder → `[725, 768]`. Each row is normalised. The four are averaged
and normalised again.

Saved as `vocab.npz`: the `[725, 768]` array, the 725 strings, and the string
`"ViT-B-16-SigLIP2/webli"`.

> **Why store the encoder's name.** A vocabulary only means anything in the
> space it was built in. Pairing embeddings from one encoder with images from
> another still returns confident-looking cosine similarities — they are
> simply wrong. The previous project had four vocabulary files across three
> different spaces. `load_vocab` now refuses a mismatch rather than silently
> degrading.

The text encoder is never loaded again. Every later comparison is a matrix
multiply against these cached numbers.

| file | shape of cache | compared against |
|---|---|---|
| `vocab.txt` | `[725, 768]` | each box crop |
| `scenes.txt` | `[40, 768]` | the whole frame |
| `attributes.txt` | `[N, 768]` + group offsets | person and vehicle crops |
| `person_checks.txt` | `[N, 768]` + ± offsets | person crops |
| `anomalies.txt` | `[N, 768]` + ± offsets | 14 tiles |
| `world_classes.txt` | never cached | goes into the detector head |

---

# 9. The shape of the whole thing

Read backwards, the system is four layers of decreasing cleverness:

```
CAPTION            no model — string assembly
REASONING          no model — arithmetic on boxes and names
COMPARISONS        one model — cosine similarity against cached text
DETECTION          three models — boxes, and names where available
IMAGE
```

Only two model families exist: YOLO variants that produce boxes, and one
SigLIP2 image tower that produces meaning. Roughly half the system's
behaviour — merging, ranking, relations, threat, composition — contains no
learned parameters at all.

**Why it is built this way.** The previous system put a trained transformer
decoder at the top. Its memory contained the correct evidence — a glass-box
inspector proved the slot held `cat` — and it wrote "dog", because a language
model trained to maximise sentence likelihood drifts toward what sentences
usually say. Five interventions over several months did not fix it. It is not
a bug; it is the training objective working as designed.

So the top of this system cannot write. It can only read a list out loud.

**What that costs.** The captions are plainer than a language model would
produce. Word order is templated. There is no world knowledge — the system
cannot infer that a person in that uniform beside that vehicle is probably a
delivery rider, because inference of that kind is exactly the faculty that
also invents umbrellas.

**What it buys.** Every word traces to a number you can look up. When the
output is wrong, the wrong number is findable, and it is usually a threshold
in a text file rather than a weight in a checkpoint.

---

# Glossary

**Canonical name** — the result of collapsing synonyms (man, woman, worker →
person). Used for counting and merging; the specific name is still printed
when unambiguous.

**Class-agnostic** — producing boxes without attempting to name them.

**Confidence vs margin** — confidence is the top score; margin is the gap to
second place. Margin is usually the better signal, because a high score with a
tiny margin means the model liked several answers equally.

**Contrastive pair** — a yes/no question posed as a positive statement and its
negation, scored by comparing which the image resembles more.

**Cosine similarity** — the cosine of the angle between two vectors; for
unit-normalised vectors, their dot product. Ranges −1 to 1.

**Embedding** — a vector representing something, arranged so similar things
get similar vectors.

**Feature map** — the intermediate output of a convolutional layer.

**Feature pyramid** — several feature maps at different resolutions, so
objects of different sizes can each be detected on a suitable scale.

**Frozen** — weights that never change; no training.

**Gating** — refusing to run a check unless a precondition is met, e.g. no
helmet check without a motorcycle.

**Heuristic** — a rule that is usually right rather than provably right.

**IoU** — Intersection over Union; overlap area divided by combined area.

**Letterboxing** — resizing while preserving aspect ratio and padding the
remainder, as opposed to squashing.

**logit_bias** — a learned offset used in SigLIP's sigmoid readout.

**logit_scale** — a learned multiplier that spreads cosine similarities before
softmax. Typically around 100 after exponentiation.

**NMS** — Non-Maximum Suppression; keep the best box, delete everything
overlapping it, repeat.

**Normalisation (image)** — subtracting a per-channel mean and dividing by a
standard deviation to match training statistics. Model-specific.

**Open-vocabulary detection** — detection where the class list is supplied as
text at runtime.

**Prompt ensembling** — averaging embeddings of several phrasings of one
concept.

**ROI** — Region Of Interest; a box, or the crop taken from it.

**Sigmoid** — squashes one number to 0-1 independently of others. Expresses
presence.

**Softmax** — converts a list of scores into a probability distribution
summing to 1. Expresses competition.

**Softplus** — a smooth always-positive function.

**Stride** — how many input pixels one cell of a feature map corresponds to.

**Temperature** — a scaling applied before softmax or sigmoid, controlling how
sharp the resulting probabilities are.

**Unit-normalised** — divided by its own length, so it lies on the unit
sphere.

**ViT** — Vision Transformer; an image model operating on fixed-size patches.

**Zero-shot** — classifying categories the model was never explicitly trained
on, by describing them in text.

---

# Appendix — which file controls what

| I want to change… | edit | then run |
|---|---|---|
| what objects can be named | `assets/vocab.txt` | `build_vocab` |
| what scenes can be named | `assets/scenes.txt` | `build_vocab --names-from … --out scenes.npz` |
| what attributes are asked | `assets/attributes.txt` | `build_attributes` |
| per-person yes/no questions | `assets/person_checks.txt` | `build_checks` |
| frame-level alerts | `assets/anomalies.txt` | `build_checks` |
| what the detector looks for | `assets/world_classes.txt` | restart only |
| which model, thresholds, budgets | `jana2/config.py` | restart |
| what counts as a weapon | `jana2/threat.py` | restart |
| what counts as a part or garment | `jana2/postprocess.py` | restart |
| how relations are decided | `jana2/relations.py` | restart |
| how sentences are built | `jana2/compose.py` | restart |
