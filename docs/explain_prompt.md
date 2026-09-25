# Prompt for ChatGPT — explain this image-captioning "probe"

Copy everything below the line into ChatGPT.

---

I am building an object-centric image captioning system from scratch. I know
PyTorch and general deep learning, but I want a careful, concrete explanation of
one script and its output. Do not skip steps and do not assume I know the
jargon — but also do not pad with generic ML background I already have.

## Background on the project

The end goal is a system that looks at an image and describes it. The design
philosophy is **object-centric**: instead of feeding the whole image into one
big vision-language model, the pipeline is:

1. Find candidate object regions (bounding boxes) in the image.
2. Give each region a *name* using open-vocabulary matching against a list of
   possible names, rather than a fixed classifier.
3. Later stages (not built yet) will turn those named, located objects into a
   caption.

A previous version of this system had a serious failure: the region evidence was
correct (the memory clearly contained "cat"), but the language decoder wrote
"dog" anyway, because language priors overrode the visual evidence. So this
rebuild deliberately starts with **just perception**, and verifies the evidence
is right before any language model is added.

## What has been built so far

Three components, plus a script called `probe.py`:

- **Proposals** (`proposals.py`): runs YOLO (an object detector) on the image to
  get bounding boxes. YOLO's *class labels are deliberately thrown away* — only
  the boxes and confidence scores are kept. Then geometric cleanup: drop tiny
  boxes, NMS to remove duplicates, and "containment suppression" to drop boxes
  that sit almost entirely inside a bigger box (e.g. a wheel inside a car),
  because those fragments inflate the object count.

- **Naming** (`naming.py`): crops each box, encodes the crop with a CLIP-style
  image encoder (specifically `ViT-B-16-SigLIP2`), and compares that embedding
  by cosine similarity against a precomputed matrix of text embeddings — one row
  per candidate name. It returns the **top-5 names with scores**, not just the
  best one.

- **Evidence** (`evidence.py`): a dataclass holding the result — image size, a
  list of object slots (box, score, name candidates, position, area), a scene
  label, and timings.

Key code, `naming.py`:

```python
@torch.no_grad()
def embed_rois(self, image, boxes):
    crops = [self.preprocess(_crop(image, b, self.cfg.roi_pad)) for b in boxes]
    batch = torch.stack(crops).to(self.cfg.device)
    e = self.model.encode_image(batch).float()
    return e / e.norm(dim=-1, keepdim=True)      # unit-normalise

@torch.no_grad()
def topk(self, embs, k=5):
    sims = embs @ self.vocab_t.T                 # cosine sim, both unit-norm
    vals, idxs = sims.topk(k, dim=-1)
    return [[(self.names[j], float(v)) for v, j in zip(vr, ir)]
            for vr, ir in zip(vals.cpu(), idxs.cpu())]
```

And each object exposes a `margin` property:

```python
@property
def margin(self):
    """top1 - top2 score."""
    return self.clip_names[0][1] - self.clip_names[1][1]
```

`probe.py` just glues it together: load image → get boxes → name them → print a
table. It is a diagnostic CLI, not part of the final product.

## The command I ran and the output I got

```
$ python -m scripts.probe data/eval/factory.jpg --no-rpn

[clip] space = ViT-B-16-SigLIP2/webli
========================================================================
data/eval/factory.jpg   1024x682   space=ViT-B-16-SigLIP2/webli
2 objects (2 yolo / 0 rpn)
scene: worker 0.14, machine 0.09, helmet 0.09

| # | name    | score | margin | objness | src  | area  | pos      |
|---|---------|-------|--------|---------|------|-------|----------|
| 1 | worker  | 0.139 | 0.046  | 0.83    | yolo | 0.414 | center   |
| 2 | machine | 0.139 | 0.019  | 0.17    | yolo | 0.014 | top-left |

top-k per object (the number that matters):
  #1  worker:0.139  helmet:0.093  safety vest:0.091
  #2  machine:0.139  conveyor belt:0.120  worker:0.118

timings (ms): {'clip_roi': 407.8, 'naming': 7.0, 'scene': 173.1, 'proposals': 1341.4}
```

Note: the candidate name list currently contains only **54 hand-written names**
(person, cat, dog, machine, worker, conveyor belt, …) as a placeholder. The real
list has thousands of names.

## What I want you to explain

Please answer these in order, with concrete reference to the numbers above:

1. **What is a "probe" here, and why would someone build this before building
   the captioning model at all?** What class of bugs does it catch early?

2. **Walk through what happens to `factory.jpg`, step by step**, from file on
   disk to the printed table. Be specific about tensor shapes and what each
   stage produces.

3. **Explain every column** of the output table: `name`, `score`, `margin`,
   `objness`, `src`, `area`, `pos`. Where does each number come from?

4. **What exactly is the `score` 0.139?** It's a cosine similarity between an
   image embedding and a text embedding. Explain what that means geometrically,
   why the value is so small in absolute terms, and whether a small absolute
   value means the match is bad.

5. **SigLIP specifically**: I'm told SigLIP models are trained with a sigmoid
   loss and have a learned `logit_scale` and `logit_bias`, and that raw cosine
   similarity is not the calibrated way to read them. Explain what that means,
   what `sigmoid(cos_sim * logit_scale.exp() + logit_bias)` would give me
   instead, and whether it changes the *ranking* or only the *scale*.

6. **Why is `margin` (top1 − top2) interesting?** How could it be used to make
   the system say "I'm not sure" instead of confidently guessing?

7. **Only 2 objects were found in a factory photo.** Explain why a COCO-trained
   detector like YOLO would behave this way, what "class-agnostic proposals"
   means, and what the standard approaches are for getting proposals for objects
   outside a detector's training classes.

8. **Interpret the timings.** Why might the first run be slow, and which of
   these numbers would you expect to change on a second run?

9. Finally: **what would you check next**, and what would a good result look
   like versus a bad one?

Use plain language, concrete numbers, and small examples. Where a term is
standard jargon (ROI, NMS, embedding space, open-vocabulary detection), define
it once in one sentence and then use it freely.
