# Task brief: get Moondream3-preview running for offline batch labelling

## Objective

Produce a **working, repeatable way to run `moondream/moondream3-preview`
locally on this server** and answer yes/no questions about ~20,000 JPEG
images in a batch job.

Deliverable is a script at
`/root/interns/avani/git_clones/image-captioning/scripts/moondream_verify.py`
that:

- reads a manifest `data/ucf/frames_sample.jsonl` where each line is
  `{"path": "...jpg", "category": "Fighting", "video": "...", "frame": 123}`
- for each frame, asks 1–3 yes/no questions (mapping below)
- appends to `data/ucf/verified.jsonl`, one line per frame:
  `{"path": ..., "category": ..., "labels": {"fight": 1, "weapon_scene": 0}, "source": "moondream"}`
- is **resume-safe** — rerunning skips frames already in the output
- prints throughput (frames/sec) and an ETA

Speed matters: batching is strongly preferred over one-image-at-a-time.

---

## Machine facts (verified today, do not re-derive)

| | |
|---|---|
| GPU | NVIDIA A100 80GB PCIe, reported as MIG instance `nvidia_a100_80gb_pcie_mig_7g_80gb`, **sm80** |
| Driver | `550.163.01`, **CUDA 12.4** |
| CUDA 13 forward-compat lib | `/usr/local/cuda-13.0/compat/libcuda.so.580.173.02` |
| Default `LD_LIBRARY_PATH` | points at `/usr/local/cuda-11.8/lib64` (stale, likely unhelpful) |

**Critical discovery:** with

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
```

a **cu130 PyTorch reports `cuda: True`** despite the 12.4 driver. Without
it, the same interpreter reports `False`. Any solution must set this (or an
equivalent) explicitly — it is not set by default in a fresh shell.

---

## Environments available

Do **not** `pip install` into anyone else's venv. Read-only use is fine.
Creating new venvs under `/root/interns/avani/` is fine.

| venv | torch | transformers | vllm | cuda avail |
|---|---|---|---|---|
| `/root/aniket/git_clones/image-captioning/.venv-vllm` | 2.11.0+cu130 | – | **0.25.0** | True *only with compat path* |
| `/root/aniket/git_clones/image-captioning/.venv` | 2.6.0+cu124 | 5.13.1 | – | True |
| `/root/khushi/venv` | 2.7.1+cu126 | 4.55.0 | 0.10.0 | True |
| `/root/abhijay/git_clones/video-surveillance/.venv` | 2.6.0+cu124 | 5.8.1 | – | True |
| `/root/interns/avani/venv_md` | 2.13.0+cu126 | – | – | False |
| `/root/interns/avani/venv_md2` | 2.6.0+cu124 | 4.44.2 | – | True |

Models already in the HF cache (no download needed):

```
moondream/moondream3-preview      29.0 GB   (17.26 GiB of weights)
moondream/moondream3.1-9B-A2B     10.5 GB
vikhyatk/moondream2                3.9 GB
moondream/starmie-v1               (tokenizer, tiny)
```

---

## What has already been tried, and exactly how it failed

Please do not repeat these without a specific reason.

**1. `transformers` + `vikhyatk/moondream2`** — in a venv with
transformers 5.15:

```
AttributeError: 'HfMoondream' object has no attribute 'all_tied_weights_keys'.
Did you mean: '_tied_weights_keys'?
```

Model's bundled remote code predates the transformers 5.x rename. Untested
on transformers 4.x, which may well work.

**2. `transformers` + `moondream/moondream3.1-9B-A2B`**

```
ValueError: Unrecognized model in moondream/moondream3.1-9B-A2B.
Should have a `model_type` key in its config.json
```

Not loadable via `AutoModelForCausalLM`.

**3. `moondream` pip package, Photon/kestrel backend, in `venv_md`**
(`md.vl(local=True, ...)`):

```
RuntimeError: Photon needs CUDA, but PyTorch could not initialize CUDA.
Installed PyTorch is built for CUDA 13.0. The NVIDIA driver reports CUDA 12.4.
```

then after installing a cu126 torch:

```
cute_runtime_shim: missing symbol cudaLibraryLoadData in the CUDA runtime.
This kestrel-kernels wheel requires a CUDA runtime with the CUDA Library API
(CUDA 12.8 or newer).
```

**4. Same Photon path in aniket's `.venv-vllm` WITH the compat lib exported**
— got much further, loaded the model, then:

```
UserWarning: KV scales found in checkpoint but FP8 KV cache currently requires
SM87/SM89/SM90/... decode kernels; falling back to standard KV cache.
RuntimeWarning: No exact CuTe MoE config for GPU
'nvidia_a100_80gb_pcie_mig_7g_80gb' on sm80; using closest config
Illegal instruction (core dumped)
```

**Reading: kestrel's CuTe MoE kernels appear to crash on sm80.** This is the
most informative failure — Photon may simply not support A100.

**5. `vllm serve` from `.venv-vllm`, with and without the compat path:**

```
OSError: libnvrtc.so.13: cannot open shared object file: No such file or directory
  (raised from torchcodec/_internally_replaced_utils.py)
```

`torchcodec` is a **video** decoder and should be irrelevant to still images,
but it is imported at startup. Unclear whether the server died from this or
from something later — it never bound port 8000. `libnvrtc.so.13` was not
found under `/usr/local/cuda-13.0/lib64`; it is probably inside the venv's
bundled `nvidia/` packages, and that directory may just need adding to
`LD_LIBRARY_PATH`.

---

## Strong evidence this DID work on this machine

`/root/aniket/git_clones/image-captioning/moondream_vllm.log`, dated
**07-12**, shows a **complete successful run** of moondream3-preview on this
A100 producing 55,325 captions
(`/root/interns/avani/git_clones/image-captioning/data/captions/mircaps_moondream.jsonl`).

It ran through **vLLM, not kestrel**. Exact non-default args from that log:

```python
{'model': 'moondream/moondream3-preview',
 'tokenizer': 'moondream/starmie-v1',        # <-- separate tokenizer repo, easy to miss
 'trust_remote_code': True,
 'max_model_len': 4096,
 'gpu_memory_utilization': 0.85,
 'limit_mm_per_prompt': {'image': 1},
 'disable_log_stats': True}
```

It reported `Resolved architecture: HfMoondream`, loaded 17.27 GiB in ~5s,
and `vllm 0.25.0` ships a native implementation at
`.venv-vllm/lib/python3.10/site-packages/vllm/model_executor/models/moondream3.py`.

The generating script is
`/root/aniket/git_clones/image-captioning/scripts/generate_moondream_captions.py`
— worth reading in full; it has both a Photon backend and a transformers
backend, and something in that repo drove the vLLM path.

**vLLM in-process (`from vllm import LLM`) is likely the answer** rather than
`vllm serve`, since it avoids the HTTP layer and batches natively. But an
OpenAI-compatible server is also fine if that proves easier — the calling
script would then need only `requests`.

---

## Questions to ask per frame

```python
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

ASK = {   # which questions are worth asking, by manifest category
  "Fighting": ["fight","weapon_scene","person_down"],
  "Assault":  ["fight","weapon_scene","person_down"],
  "Abuse":    ["fight","person_down"],
  "RoadAccidents": ["accident","person_down"],
  "Vandalism":["vandalism","fight"],
  "Robbery":  ["snatching","weapon_scene","fight"],
  "Stealing": ["snatching"],
  "Shoplifting":["snatching"],
  "Arson":    ["fire","smoke"],
  "Explosion":["fire","smoke"],
  "Shooting": ["weapon_scene","person_down","fight"],
  "Burglary": ["climbing","vandalism","snatching"],
  "Arrest":   ["fight","person_down"],
}
```

Frames with `category == "Normal"` are **trusted negatives** — write
`{k: 0 for k in QUESTIONS}` with `"source": "trusted"` and never query the
model. That is ~4,000 of the frames.

**Answer parsing matters.** If the model returns prose rather than a bare
"Yes"/"No", detect that and handle it — either append "Answer with only yes
or no." to each question, or match yes/no anywhere in the response. A silent
misparse would label every frame `0` and quietly ruin the downstream
training, so please verify on ~20 real frames and report what the raw
responses look like.

---

## Constraints

- Do not modify anything under `/root/aniket/`, `/root/khushi/`,
  `/root/abhijay/`. Read-only.
- Do not change the main venv
  `/root/interns/avani/git_clones/image-captioning/venv` — its
  `torch 2.6.0+cu124` was painful to get working and other work depends on it.
- New venvs under `/root/interns/avani/` are fine.
- The GPU is shared. Free it when idle; don't leave a 60 GB server running
  after the job finishes.
- `moondream3-preview` is preferred. `moondream3.1-9B-A2B` (MoE, faster) or
  `moondream2` (small) are acceptable fallbacks **if** you report clearly
  which you used and why.

---

## What to report back

When finished, please state plainly:

1. **Which backend worked** — vLLM in-process / vLLM server / transformers /
   Photon — and which model.
2. **The exact environment**: venv path, torch version, and any environment
   variables required (especially `LD_LIBRARY_PATH`).
3. **The exact command** to run the labelling job.
4. **Raw sample output** — 10–20 real question/answer pairs, unedited, so
   the answer format can be verified.
5. **Measured throughput** (frames/sec) and the projected wall-clock time for
   ~20,000 frames.
6. **Anything you changed** outside `scripts/moondream_verify.py`.
7. **What did not work**, briefly, so it isn't retried.

If none of the local paths work on sm80 with this driver, say so explicitly
rather than silently falling back to a weaker model — that is a legitimate
result and changes the plan.
