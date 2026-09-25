#!/usr/bin/env bash
# JANA2 — environment bootstrap. Run once from /root/avani/code_files.
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/4] venv"
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip wheel

echo "[2/4] deps"
.venv/bin/pip install -r requirements.txt

echo "[3/4] symlinks to read-only source assets"
SRC=/root/aniket/git_clones/image-captioning
[ -e data ]            || ln -s "$SRC/data"          data
[ -e checkpoints_rpn ] || ln -s "$SRC/runs/rpn_v2"   checkpoints_rpn
mkdir -p assets outputs

echo "[4/4] sanity"
.venv/bin/python -c "
import torch, open_clip, ultralytics
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('open_clip', open_clip.__version__ if hasattr(open_clip,'__version__') else 'ok')
names=[n for n,_ in open_clip.list_pretrained()]
print('SigLIP2 available:', any('SigLIP2' in n for n in names))
"
echo "done. activate with: source .venv/bin/activate"
