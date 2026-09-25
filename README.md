# JANA2 — object-centric surveillance captioning

An explainable captioning pipeline for surveillance frames. Instead of a black-box language model, JANA2 detects objects, builds structured evidence (attributes, relations, threat and anomaly checks), and composes a deterministic, auditable caption, with a distilled 13.4M-parameter encoder option for lightweight deployment.

```
jana2/      the pipeline
scripts/    build, distill, label, train, debug
assets/     word lists (the .npz caches are NOT here — rebuild them)
docs/       report, plans, study notes
app.py      streamlit UI
```

## What each file does

### jana2/

| file | role |
|---|---|
| `config.py` | every tunable in one dataclass; env vars `JANA2_STUDENT`, `JANA2_PROBES` |
| `clip_space.py` | loads the encoder, stamps caches with its identity, wraps the distilled student |
| `proposals.py` | yolo26n + YOLO-World, NMS, part suppression |
| `naming.py` | orchestrates the whole evidence build |
| `postprocess.py` | categories, synonyms, geometry-first merging, importance ranking |
| `threat.py` | weapon tiering — idle vs held, escalated by person flags |
| `anomaly.py` | yes/no checks; uses a learned probe where one exists, phrases otherwise |
| `attributes.py` | grouped softmax with a margin test |
| `relations.py` | wearing / holding / riding from geometry |
| `evidence.py` | the structured evidence dataclasses |
| `readout.py` | auditable operator lines + alerts |
| `compose.py` | the caption — deterministic, no language model |
| `rpn_head.py` | vendored from an earlier project; **no longer used** |

### scripts/

| stage | scripts |
|---|---|
| build caches | `build_vocab` `build_scenes` `build_checks` `build_attributes` |
| distillation | `distill_harvest` → `distill_projector` (or `distill_train`) → `distill_eval` |
| labelling | `ucf_frames` → `moondream_verify` → `train_probes` |
| person probes | `ucf_person_crops` → `moondream_person` → `train_probes` |
| measurement | `calibrate` `snapshot` `batch_report` `why` |
| debugging | `probe` `debug_clip` `debug_yolo` `clip_sizes` |

`why.py` is the one to reach for first — dumps everything about a single
image, including which mechanism produced each check.

## Standard workflow

```bash
source venv/bin/activate

# caches — needed after ANY edit to assets/*.txt, and after changing encoder
python -m scripts.build_vocab && python -m scripts.build_scenes && \
python -m scripts.build_attributes && python -m scripts.build_checks

# baseline (teacher, phrase checks)
streamlit run app.py

# student + probes
JANA2_PROBES=1 JANA2_STUDENT=checkpoints_distill/projector_best.pt streamlit run app.py
```

Labelling uses Moondream3 served through vLLM, in its own virtualenv:

```bash
python scripts/moondream_verify.py \
  --manifest data/ucf/frames_sample.jsonl --out data/ucf/verified.jsonl
```

## Gotchas

**CUDA versions.** vLLM and Moondream3 need a CUDA 13 build. On a machine
with an older driver, a cu130 build reports `cuda: False` until the CUDA
forward-compatibility library is on `LD_LIBRARY_PATH`.

**`.npz` caches are encoder-stamped** and refuse to load against a different
encoder. Change `clip_model` → rebuild all four.

**Streamlit loads models lazily.** An empty log doesn't mean failure; open the
page first.

**`st.dataframe` / `st.table` can segfault** via pyarrow on some hosts; the app
avoids them.

## What's not here

- `.npz` caches — rebuild with the build scripts
- `checkpoints_distill/` and `assets/probes__*.npz` — trained artifacts, not committed
- `data/` — ~160 GB (UCF-Crime, Visual Genome, Places365, harvested crops)
- Virtual environments

## Current state

| | |
|---|---|
| encoder | SigLIP2 93M, or distilled MobileCLIP2-S0 + projector at 13.4M |
| frame checks | 8 learned probes (AP 0.80–0.95), `weapon_scene` rejected at AP 0.34 |
| person checks | all 8 still phrase-based — `weapon_held` misfires on children |
