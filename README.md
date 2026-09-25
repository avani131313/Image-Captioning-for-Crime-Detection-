# JANA2 — object-centric surveillance captioning

Copy of the code as it stood at the end of the session. Server copy of record:
`/root/interns/avani/git_clones/image-captioning` (bitbucket `research_experiments`).

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
| `rpn_head.py` | vendored from the old project; **no longer used** |

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
cd /root/interns/avani/git_clones/image-captioning && source venv/bin/activate

# caches — needed after ANY edit to assets/*.txt, and after changing encoder
python -m scripts.build_vocab && python -m scripts.build_scenes && \
python -m scripts.build_attributes && python -m scripts.build_checks

# baseline (teacher, phrase checks)
nohup python -u -m streamlit run app.py --server.port 8616 --server.address 0.0.0.0 > teacher.log 2>&1 & disown

# student + probes
JANA2_PROBES=1 JANA2_STUDENT=checkpoints_distill/projector_best.pt \
nohup python -u -m streamlit run app.py --server.port 8617 --server.address 0.0.0.0 > student.log 2>&1 & disown
```

Labelling (needs the vLLM venv and the CUDA compat path):

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/root/aniket/git_clones/image-captioning/.venv-vllm/lib/python3.10/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
/root/aniket/git_clones/image-captioning/.venv-vllm/bin/python scripts/moondream_verify.py \
  --manifest data/ucf/frames_sample.jsonl --out data/ucf/verified.jsonl
```

## Gotchas — the things that cost time

**CUDA.** Driver is 550.163.01 / CUDA 12.4. Only cu124 builds work directly.
A cu130 build reports `cuda: False` until you export
`LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:...` — that forward-compat
library is what makes vLLM and Moondream3 work at all. Not set by default.

**Wrong venv.** `numpy was built with baseline optimizations (X86_V2)` means
you're in `.venv-1` or another user's environment. Re-activate your own.

**`.npz` caches are encoder-stamped** and refuse to load against a different
encoder. Change `clip_model` → rebuild all four.

**Heredocs get mangled** on long pastes — files arrive truncated or empty.
Anything over ~50 lines, paste in an editor and check `wc -l`.

**VS Code root.** If it's open on `/root/aniket/...`, files land in his repo.
Happened twice.

**`&` scoping.** `cd x && nohup y &` runs the `cd` inside the subshell — your
shell stays put and the log isn't where you look for it. Separate them.

**Streamlit loads models lazily.** An empty log doesn't mean failure; hit the
page first.

**Never `st.dataframe`/`st.table`** — pyarrow segfaults on this host.

**Ports.** 8616 and 8617. 8601 is taken.

## What's not here

- `.npz` caches — rebuild with the build scripts
- `checkpoints_distill/` and `assets/probes__*.npz` — on the server
- `data/` — ~160 GB (UCF-Crime, VG, places365, harvested crops)
- The `.venv`s

## State at handover

| | |
|---|---|
| encoder | SigLIP2 93M, or distilled MobileCLIP2-S0 + projector at 13.4M |
| frame checks | 8 learned probes (AP 0.80–0.95), `weapon_scene` rejected at AP 0.34 |
| person checks | all 8 still phrase-based — `weapon_held` misfires on children |
| thresholds | calibrated on 400 real CCTV frames, 2 false alarms / 1000 |
| open | person-crop probes, recall measurement, attribute floors |
