# JANA2 — what happens when an image arrives

A step-by-step trace, in order, from the moment a file is opened to the moment
a caption exists. Every step shows the actual function called, the shape of
the data, and what the numbers look like.

Running example: **a 1920×1080 frame from a housing society gate.** A delivery
rider on a scooter is coming through the gate. A boom barrier is on the left.
Three people are walking near the far building.

---

# Before the image — what already exists in memory

The process started earlier. Five things were loaded and are sitting in RAM,
never reloaded per image.

**Three detection models:**

```
yolo26m.pt            ~20M params    on GPU, eval mode
yolov8x-worldv2.pt    ~72M params    on GPU, your 140 class names baked into
                                     its final layer by set_classes()
yolo26n.pt            ~2.6M params   on GPU — NOT for detecting. Forward hooks
                                     are attached to layers 16, 19 and 22.
rpn_v2 (rpn_ep2.pt)   0.21M params   on GPU
```

**One embedding model:**

```
SigLIP2 ViT-B-16 image tower   ~86M params
```

Its *text* tower is not loaded. It was used once during the build step and is
not needed again.

**Five caches of pre-computed text vectors:**

```
vocab.npz          [725, 768]   object names
scenes.npz         [40, 768]    scene labels
attributes.npz     [~420, 768]  attribute phrases + a table of group offsets
person_checks.npz  [~70, 768]   phrases + which are positive/negative per check
anomalies.npz      [~120, 768]  same, plus gating rules per check
```

Every one of those arrays is unit-normalised — each row has length exactly 1.
That matters, because it means a dot product between any two rows *is* the
cosine of the angle between them.

---

# Step 1 — the file is opened

```python
img = Image.open("gate_0412.jpg").convert("RGB")
```

**State now:** a PIL image, 1920×1080, three channels, values 0-255.

`.convert("RGB")` is not decoration. A PNG might arrive as RGBA, a scan as
greyscale. Everything downstream assumes exactly three channels in red-green-
blue order.

---

# Step 2 — the proposal stage begins

```python
boxes, scores, srcs, labels, stats = proposer(img)
```

This one call runs three detectors and four filters. We will follow all seven.

## 2.1 — yolo26 runs

```python
r = self.det.predict(img, conf=0.10, imgsz=1280, verbose=False)[0]
```

Inside ultralytics:

1. The PIL image is converted to a tensor. **We pass PIL deliberately** —
   ultralytics reads a raw numpy array as BGR, so passing `np.array(img)`
   would swap red and blue on every frame.
2. Letterboxed from 1920×1080 to 1280×736 — scaled by 0.667 to preserve
   aspect ratio, then padded to a multiple of 32.
3. Forward pass. The network outputs, for every anchor position, 4 box
   numbers and 80 class scores.
4. Anything whose best class score is under 0.10 is discarded.
5. Internal NMS at IoU 0.7 removes duplicates.
6. Boxes are mapped back to original 1920×1080 coordinates.

**What comes back:**

```python
r.boxes.xyxy   # [6, 4]  float, original pixel coords
r.boxes.conf   # [6]     0.10 to 1.0
r.boxes.cls    # [6]     integers 0-79
```

For our frame, six detections:

```
person       0.94   (820, 340, 1090, 910)
motorcycle   0.88   (760, 620, 1160, 980)
person       0.51   (1520, 400, 1580, 560)
person       0.44   (1590, 405, 1650, 565)
person       0.38   (1650, 410, 1705, 570)
backpack     0.36   (900, 400, 1020, 560)
```

Then in our code:

```python
labels = [COCO_ALIAS.get(r.names[c], r.names[c]) for c in cls]
```

`r.names[0]` is `"person"`, `r.names[3]` is `"motorcycle"`. `COCO_ALIAS`
rewrites a handful of COCO's odd names — `cell phone` → `mobile phone`,
`couch` → `sofa`.

Finally the trust test:

```python
labels = [(l if s >= 0.40 else None) for l, s in zip(yl, ys)]
```

Above 0.40 the label is kept as truth. Below, the **box survives but the name
is thrown away** — CLIP will name it later. Here the three distant people at
0.51/0.44/0.44 keep their labels; the backpack at 0.36 loses its name.

**State: 6 boxes, 5 with trusted labels.**

## 2.2 — YOLO-World runs

```python
r = self.world.predict(img, conf=0.08, imgsz=1280, verbose=False)[0]
```

Mechanically the same, but `cls` indexes **your** 140 names from
`world_classes.txt`, not COCO's 80. That happened at startup:

```python
self.world.set_classes(names)   # ran a CLIP text encoder over your 140 lines
                                # and wrote the vectors into the detection head
```

The model's final layer now literally contains your class list. This is what
"open vocabulary" means in practice — the classes are data, not architecture.

**What comes back for our frame:**

```
scooter                  0.61   (760, 615, 1165, 985)
helmet                   0.55   (880, 330, 1000, 430)
boom barrier             0.44   (120, 500, 640, 590)
person                   0.41   (825, 345, 1085, 905)
guard booth              0.30   (60, 380, 300, 700)
food delivery bag        0.22   (900, 395, 1025, 565)
gate                     0.19   (100, 300, 700, 720)
number plate             0.12   (900, 900, 1010, 950)
```

Trust threshold here is lower — `world_trust_conf = 0.15` — because
open-vocabulary scores run lower than a supervised COCO head. Comparing them
on the same scale would be a mistake.

**State: 14 boxes total.**

## 2.3 — rpn_v2 runs

Completely different path.

```python
S = 640
t = TF.to_tensor(image.resize((S, S))).unsqueeze(0).to(device)   # [1,3,640,640]
```

Note `.resize((640,640))` — a **squash**, not a letterbox. The 16:9 frame is
distorted to square. That is deliberate: the RPN was trained that way, and
the boxes come back normalised 0-1 so the distortion reverses exactly when we
scale by the original width and height.

```python
feats = self.feat.feats(t)
```

Inside: `yolo26n` runs a forward pass. We do not want its detections — we want
what is happening in the middle. Forward hooks placed at startup capture the
outputs of layers 16, 19 and 22, which are the three feature maps that
normally feed YOLO's detection head:

```
P3   [1,  64, 80, 80]    stride 8    one cell per 8×8 pixels
P4   [1, 128, 40, 40]    stride 16
P5   [1, 256, 20, 20]    stride 32
```

Those widths — 64, 128, 256 — are why the feature net must be nano. `rpn_v2`'s
first layer is `Conv2d(64→128)`, `Conv2d(128→128)`, `Conv2d(256→128)`, shaped
for exactly these.

```python
bx, sc = self.rpn.proposals(feats, 640, pre_topk=300, iou=0.6, topk=32)
```

Inside `proposals()`, for each of the three levels:

```python
t   = tower(proj(f))                # 1×1 conv to 128, then 3×3 conv + GroupNorm + ReLU
obj = obj_head(t)                   # [1, 1, H, W]  one objectness logit per cell
ltrb = softplus(box_head(t))        # [1, 4, H, W]  four distances, forced positive
```

Decoding at cell `(r, c)` on P3:

```
centre_x = (c + 0.5) × 8
centre_y = (r + 0.5) × 8
x0 = centre_x − left×8     y0 = centre_y − top×8
x1 = centre_x + right×8    y1 = centre_y + bottom×8
score = sigmoid(obj[r, c])
```

P3 alone has 80×80 = 6400 cells, so 6400 candidate boxes. Take the top 300 by
score. Same for P4 (1600 cells) and P5 (400). Concatenate → 900 boxes. NMS at
0.6. Keep the top 32.

Boxes come out as fractions of 640, so:

```python
box_pixels = (x0×1920, y0×1080, x1×1920, y1×1080)
```

Then anything under `rpn_min_score = 0.20` is dropped, leaving ~18 boxes —
**all nameless**.

**State: ~32 boxes. 13 have names, ~19 do not.**

## 2.4 — the three lists are combined

```python
for name in ("yolo", "world", "rpn"):        # priority order
    budget = getattr(cfg, f"{name}_budget")   # 32, 40, 32
    for bb, ss, ll in list(zip(b, s, l))[:budget]:
        boxes.append(bb); scores.append(ss)
        srcs.append(name); labels.append(ll if ss >= trust else None)
```

Four parallel lists, all the same length:

```python
boxes  = [(820,340,1090,910), (760,620,1160,980), ...]
scores = [0.94, 0.88, ..., 0.61, 0.55, ..., 0.43, 0.38, ...]
srcs   = ["yolo","yolo",...,"world","world",...,"rpn","rpn",...]
labels = ["person","motorcycle",...,"scooter","helmet",..., None, None,...]
```

**`stats.raw = 32`, `by_source_raw = {"yolo": 6, "world": 8, "rpn": 18}`**

## 2.5 — filter one: specks

```python
keep = [i for i in range(len(boxes))
        if _area(boxes[i]) / (1920*1080) >= 0.0015]
```

`0.0015 × 2,073,600 ≈ 3110 px²`, so a box smaller than about 55×55 is gone.
Three tiny rpn boxes on wall texture disappear.

**`stats.after_area = 29`**

## 2.6 — filter two: NMS, with source priority

```python
pri = {"yolo": 3, "world": 2, "rpn": 1}
rank_s = [scores[i] + pri[srcs[i]] for i in keep]
keep = tv_nms(boxes_kept, rank_s, 0.6)
```

The bump is the important detail. **The three sources' scores are not
comparable** — YOLO's 0.94 is a supervised class probability, rpn_v2's 0.43 is
a sigmoid objectness from an entirely different objective. Sorting them
together as though they meant the same thing would let one source silently
delete another's boxes.

So yolo's person becomes 3.94, world's person becomes 2.41, an rpn box on the
same body becomes 1.43. They overlap past 0.6, and yolo wins. The bump is
discarded immediately after.

Here the world `person` (0.41) and two rpn boxes on the rider are removed as
duplicates of yolo's `person` (0.94). The world `scooter` and yolo's
`motorcycle` overlap at IoU 0.97 — yolo wins, so we keep `motorcycle`. (The
synonym table maps them together later anyway.)

**`stats.after_nms = 21`**

## 2.7 — filter three: part-suppression

```python
suppress_parts(boxes, thresh=0.90, frame_area, max_parent=0.60,
               min_child_ratio=0.55, protected=yolo_and_world_indices)
```

Sort largest first. Drop box A only if **all three** conditions hold against
some kept box B:

```
inside_frac(A,B) >= 0.90     A is almost entirely inside B
area(A)/area(B)  >= 0.55     A is a similar size to B → a duplicate
area(B)/frame    <= 0.60     B is an object, not scenery
```

Worked through for our frame:

- `helmet` inside `person`: inside_frac 1.0 ✓, but area ratio is
  `(120×100)/(270×570)` = **0.078**, well under 0.55 → **kept**.
- `food delivery bag` inside `person`: ratio 0.13 → **kept**.
- An rpn box covering the rider's torso: inside_frac 1.0, ratio 0.62 → over
  0.55 → **dropped as a duplicate**.

**That middle condition is the one that matters.** An earlier version dropped
anything mostly inside a bigger box, which deleted the helmet, the bag, and —
on a different image — a machete held in someone's hand. A weapon in a hand
is always 100% inside the person's box. The rule deleted the most important
object in a crime frame by construction.

**`stats.after_containment = 17`**

## 2.8 — filter four: budget

```python
keep.sort(key=lambda i: (-pri[srcs[i]], -scores[i]))
keep = keep[:48]
```

17 boxes, under the cap, so nothing is lost.

**Returned:** `boxes` (17), `scores` (17), `srcs` (17), `labels` (17, some
`None`), and `stats`.

**Elapsed: roughly 900 ms.** Three detector forward passes dominate.

---

# Step 3 — evidence building begins

```python
ev = build_evidence(path, img, boxes, scores, srcs, labels, namer)
```

Everything from here happens inside this one function.

## 3.1 — the crops

```python
crops = [preprocess(_crop(image, b, 0.08)) for b in boxes]
```

For each box, expand 8% on every side and cut it out of the **original**
1920×1080 image — not the resized one, so we get full resolution.

For the rider: `(820,340,1090,910)` becomes `(798,283,1112,967)` after
padding.

> Why pad: a tightly-cropped object is harder to identify than one with a
> little context. 8% is a compromise — too much and the neighbour starts
> influencing the answer.

Then SigLIP2's own preprocessing on each crop: resize to 224×224, convert to
tensor, normalise with **that model's** mean and standard deviation.

> The normalisation values differ per model. MobileCLIP uses mean 0 and std 1
> — raw 0-1 pixels. Most CLIP variants use ImageNet-ish statistics. Applying
> the wrong ones leaves coarse recognition roughly intact while destroying
> fine distinctions — which shows up as correct names but wrong attributes.
> This is why the transform always comes from `create_model_and_transforms()`
> and is never hand-written.

```python
batch = torch.stack(crops).to(device)    # [17, 3, 224, 224]
e = model.encode_image(batch).float()    # [17, 768]
embs = e / e.norm(dim=-1, keepdim=True)  # unit length
```

**One forward pass for all 17 crops.** This tensor is now used three separate
times. That is why adding the attribute pass and the weapon check barely
changed the runtime — each is one matrix multiply against an array already in
memory.

**Elapsed: ~120 ms.**

## 3.2 — naming

```python
sims   = embs @ vocab_t.T              # [17,768] × [768,725] → [17, 725]
logits = sims * logit_scale.exp()      # scalar, ≈ 100
probs  = logits.softmax(dim=-1)        # across the 725 names
vals, idxs = probs.topk(5, dim=-1)
```

Because both sides are unit-length, `embs @ vocab_t.T` **is** cosine
similarity. Raw values sit around 0.10-0.30. Multiplying by ~100 spreads them
before softmax; without it the distribution over 725 options would be nearly
flat.

For the rider crop:

```
scooter          0.31
motorcycle       0.24
person           0.11
delivery rider   0.07
auto rickshaw    0.03
```

(The crop contains both rider and scooter, so both score.)

> **Softmax, not sigmoid.** SigLIP's native readout is
> `sigmoid(logits + logit_bias)`, which answers "does this crop match *this
> one* caption?" independently per name. Across 725 candidates almost
> everything lands near zero — that produced `man 0.0012` and `car 0.2248`
> for two equally good matches. The ranking was fine; the numbers were
> meaningless, and the top1−top2 gap was useless as confidence. Softmax asks
> the question we actually have: *which one of these 725*.

## 3.3 — ObjectSlot objects are created

```python
for i, (b, s, src) in enumerate(zip(boxes, scores, srcs)):
    lab = labels[i]
    if lab:                                   # trusted detector label
        alt   = [x for x in clip_names[i] if x[0] != lab][:3]
        names = [(lab, s)] + alt
        named_by = "detector"
    else:
        names = clip_names[i]
        named_by = "clip"
```

The rider had `labels[i] = "person"` from yolo at 0.94, so:

```python
clip_names = [("person", 0.94), ("scooter", 0.31), ("motorcycle", 0.24), ...]
named_by   = "detector"
```

COCO's person head has seen millions of labelled people; zero-shot CLIP on a
270×570 crop has not. The specialist wins where it applies, and CLIP's answers
are preserved underneath as alternatives.

**State: 17 `ObjectSlot` objects**, each with box, score, source, position
bucket, area fraction, top-5 names, and empty attributes/checks.

## 3.4 — attributes

```python
for i, o in enumerate(objs):
    o.attributes = namer.attrs.describe(o.name, embs[i])
```

Inside `describe`, for each group whose `applies_to` contains this object's
name:

```python
sub   = self.embs[g.start : g.start + g.n]   # e.g. rows 41-58, upper_colour
sims  = roi_emb @ sub.T                      # [18]
probs = (sims * scale).softmax(-1)           # across those 18 only
p1, p2 = top two
if (p1 - p2) < 0.10:  skip this group entirely
```

For the rider:

| group | winner | prob | margin | kept? |
|---|---|---|---|---|
| gender | a photo of a man | 0.68 | 0.36 | yes |
| age | a young adult | 0.54 | 0.19 | yes |
| upper_colour | a person wearing a red top | 0.21 | **0.04** | **dropped** |
| headwear | a person wearing a motorcycle helmet | 0.62 | 0.31 | yes |
| carrying | a person carrying a delivery bag | 0.49 | 0.22 | yes |
| pose | a person sitting | 0.31 | **0.06** | **dropped** |

Two groups dropped. That is why the final caption never mentions his shirt
colour — the question was asked, the answer was a coin flip between red and
orange, so nothing was recorded.

> Note the group is only asked at all if the object's name is in its
> `applies_to` list. No point asking a boom barrier what colour shirt it is
> wearing.

Labels are shortened for display: `"a person wearing a motorcycle helmet"` →
`"motorcycle helmet"`.

**Elapsed: ~15 ms** — 25 small matrix multiplies against an embedding we
already have.

## 3.5 — person checks

```python
pidx = [i for i, o in enumerate(objs) if category(o.name) == "person"]
hits = namer.person_checks.scan_embs(embs[pidx])
```

Different maths again. For `weapon_held`, with 6 positive and 7 negative
phrases:

```python
sp     = (emb @ pos.T).max()      # best-matching positive
sn     = (emb @ neg.T).max()      # best-matching negative
margin = sp - sn
prob   = sigmoid(margin * 20)
fires if prob >= 0.60 and margin >= 0.02
```

For our rider: best positive is "a person holding a large knife" at 0.14;
best negative is "a person holding a bag" at 0.23. Margin is **−0.09**,
sigmoid gives 0.14. Does not fire. Correct.

> **`max`, not `mean`.** One strongly-matching positive should not be diluted
> by five weak ones.

> **Temperature 20, not ~100.** At the model's own `logit_scale`, a cosine gap
> of 0.05 becomes a logit gap of 5 → 99%. So when *neither* phrase matches,
> whichever is fractionally ahead returns as near-certain. That is exactly how
> a studio portrait once produced `no helmet 99%`. At 20 the same gap gives
> 73%, and the `margin >= 0.02` floor means two equally poor matches produce
> nothing at all.

## 3.6 — merging duplicates

```python
objs = merge_duplicates(objs, iou_thresh=0.30, contain_thresh=0.70,
                        same_object_iou=0.55)
```

Sorted by strength (`name_score × objectness`), then pairwise:

```python
if IoU(a, b) >= 0.55:
    merge — same object, whatever the labels say
elif canonical_name(a) == canonical_name(b) and (IoU > 0.30 or contained > 0.70):
    merge
```

The first rule is the important one. YOLO said `person`, World said `person`,
an rpn box also landed on the rider. If merging required matching names, an
rpn box named `man` by CLIP would survive as a second entity and the caption
would say "two people".

Survivors take the **union box** and the **higher objectness**, and record
`merged_count`.

**17 → 12 objects.**

## 3.7 — anomalies, gated

```python
ctx = {"names": {o.name.lower() for o in objs} | {canon(o.name) for o in objs},
       "n_people": sum(1 for o in objs if category(o.name) == "person")}
anomalies = namer.frame_checks.scan_frame(image, ctx)
```

`ctx` here is `{"person", "motorcycle", "scooter", "helmet", "boom barrier",
"guard booth", "food delivery bag", "gate", ...}` and `n_people = 4`.

Gating runs **first**, per check:

```
[no_helmet]   requires=motorcycle,scooter,bicycle   → motorcycle present → RUNS
[crowd_surge] requires_people=8                     → only 4 → SKIPPED
[accident]    requires=car,truck,bus,motorcycle,…   → RUNS
[flooding]    tiles=no                              → whole frame only
[fire]        no requirement                        → RUNS
```

Skipped checks return `prob = 0.0` and `where = "gated"`, so you can see the
gate working rather than guessing.

For the checks that survive, the image is cut into **14 tiles** — the whole
frame, a 2×2 grid, and a 3×3 grid — all embedded in one batch `[14, 768]`.
Each check runs the same pair maths against all 14 and takes the highest.

> **Why tiles.** A fire in one corner barely moves the embedding of the whole
> frame — it is a small part of a large picture. Per-tile scoring lets a local
> event register, and tells you which tile it was in.

`no_helmet` runs and scores 0.31 — the rider is wearing one. Nothing fires.

**Elapsed: ~1400 ms.** This is the single most expensive stage. Dropping
`anomaly_grids` to `(1, 2)` takes it from 14 crops to 5.

## 3.8 — importance ranking

```python
objs = rank(objs, W, H, hot_tiles)
```

For each object:

```python
area       = min(1, sqrt(area_frac) × 2.2)
centrality = 1 − hypot(cx−0.5, cy−0.5) / 0.707
base  = 0.25·area + 0.20·centrality + 0.30·objectness + 0.25·name_conf
score = CATEGORY_WEIGHT[category] × base
```

The rider: area_frac 0.074 → `sqrt(0.074)×2.2 = 0.60`. Centre at (0.50, 0.58)
→ centrality 0.89. Objectness 0.94. Name confidence 0.94.

```
base  = 0.25(0.60) + 0.20(0.89) + 0.30(0.94) + 0.25(0.94) = 0.85
score = 1.00 × 0.85 = 0.85          (person weight = 1.00)
```

The boom barrier: large but off-centre and low confidence, category `access`
weight 0.45 → **0.31**.

Sorted descending, then **renumbered** `idx = 1, 2, 3...`. That renumbering is
why relations must be derived after this point, not before.

## 3.9 — relations

```python
rels = derive(objs, W, H, max_relations=16)
```

An O(n²) loop over box pairs, pure coordinate arithmetic:

```
helmet    inside person  = 1.00  → wearing    (garment/accessory rule)
bag       inside person  = 0.95, area ratio 0.13 < 0.35 → holding
person    h_overlap with motorcycle = 0.87, v_overlap = 0.42 → riding
person    gap to gate = 1.8% of diagonal → at
```

Four relations. Each stores the numbers that produced it, so a wrong one
traces to a threshold rather than to an argument.

## 3.10 — scene

```python
e = encode_image(preprocess(whole_image))     # [1, 768]
probs = (e @ scenes_t.T × scale).softmax(-1)  # across 40 labels
```

```
the entrance gate of a housing society   0.41
a residential building compound          0.22
a parking area                           0.09
```

Top beats second by 0.19, comfortably over the 0.05 requirement, so it is
kept.

## 3.11 — the evidence object is returned

```python
StructuredEvidence(
    width=1920, height=1080,
    objects=[12 ObjectSlots, ranked],
    scene={"clip_names": [("the entrance gate of a housing society", 0.41), ...]},
    relations=[4 relations],
    anomalies=[18 checks, none triggered],
    timings_ms={"clip_roi": 121, "naming": 4, "attributes": 15,
                "person_checks": 6, "merge": 2, "anomaly": 1403,
                "relations": 3, "proposals": 912},
    clip_space="ViT-B-16-SigLIP2/webli",
)
```

**Total elapsed: ~2.5 s.**

---

# Step 4 — the caption

```python
caption = compose.compose(ev)
```

No model runs here. String assembly only.

**Select.** Drop anything under the score thresholds, anything categorised
`part` or `scenery`, and anything that is the *object* of an attaching
relation. The helmet, the bag and the motorcycle are all attached to the
rider, so they will be spoken through him rather than listed.

12 objects → 4 speakable entities: the rider, the boom barrier, and two of
the distant people.

**Depth.** Rider area 0.074 → foreground. Boom barrier 0.078 → foreground.
Distant people 0.004 and high in frame → background.

**Subject.** Highest importance is the rider at 0.85.

**Noun phrase.**

```
adjectives  ["young adult"]         from the age attribute
head noun   "man"                   gender attribute overrides "person"
worn        ["motorcycle helmet"]   from the wearing relation
held        ["food delivery bag"]   from the holding relation
mount       "motorcycle"            from the riding relation
```

**Verb.** A `riding` relation exists → "is riding".

**Assembled:**

```
"The entrance gate of a housing society."            ← scene
"A young adult man wearing a motorcycle helmet is riding a motorcycle
 in the centre, carrying a food delivery bag."       ← subject
"A boom barrier is on the left."                     ← other foreground
"Several people are visible further back."           ← background, counted
```

No alerts fired, so no alert sentence.

---

# The whole thing, in order

| # | stage | model | time |
|---|---|---|---|
| 1 | open image | — | 5 ms |
| 2.1 | yolo26 detect | yolo26m | ~350 ms |
| 2.2 | YOLO-World detect | yolov8x-world | ~450 ms |
| 2.3 | rpn_v2 propose | yolo26n + rpn_v2 | ~110 ms |
| 2.5-2.8 | four geometric filters | — | 2 ms |
| 3.1 | crop and embed 17 boxes | SigLIP2 | 121 ms |
| 3.2 | name against 725 words | — | 4 ms |
| 3.4 | attributes, 25 groups | — | 15 ms |
| 3.5 | person checks | — | 6 ms |
| 3.6 | merge duplicates | — | 2 ms |
| 3.7 | 14 tiles + anomaly checks | SigLIP2 | 1403 ms |
| 3.8 | importance | — | 1 ms |
| 3.9 | relations | — | 3 ms |
| 3.10 | scene | SigLIP2 | 25 ms |
| 4 | compose | — | 1 ms |

**~2.5 s total.** Detection is 36%, the anomaly tiles are 56%, and everything
else together is under 8%.

Two observations worth carrying away.

**The expensive parts are the two places we run a model many times** — three
detectors on a big image, and fourteen tiles through the encoder. Every stage
that reuses an existing embedding costs single-digit milliseconds.

**Every stage after 3.2 is arithmetic.** Naming, attributes, checks, merging,
ranking, relations and composition are all operations on numbers we already
have. Half the system's behaviour, and none of its parameters.
