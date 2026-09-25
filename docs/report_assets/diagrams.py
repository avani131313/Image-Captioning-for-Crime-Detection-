import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

INK="#1a1a1a"; MUT="#6b6b6b"; ACC="#2563eb"; GOOD="#059669"; WARN="#d97706"; BAD="#dc2626"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9})

def box(ax,x,y,w,h,text,fc="#eff6ff",ec=ACC,fs=8.5,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.012",
                 fc=fc,ec=ec,lw=1.2))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs,
            fontweight="bold" if bold else "normal",color=INK)

def arrow(ax,x1,y1,x2,y2,label=None):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",
                 mutation_scale=11,color=MUT,lw=1.1))
    if label:
        ax.text((x1+x2)/2,(y1+y2)/2+0.012,label,ha="center",fontsize=7,color=MUT)

# ---------- pipeline ----------
fig,ax=plt.subplots(figsize=(8.2,4.4)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
box(ax,0.02,0.78,0.18,0.13,"IMAGE",fc="#f8fafc",ec=MUT,bold=True)
box(ax,0.26,0.86,0.30,0.10,"yolo26n  (COCO, 3M)",fc="#f1f5f9",ec=MUT)
box(ax,0.26,0.73,0.30,0.10,"YOLO-World-s  (open vocab, 13M)",fc="#f1f5f9",ec=MUT)
arrow(ax,0.20,0.86,0.26,0.91); arrow(ax,0.20,0.83,0.26,0.78)
box(ax,0.62,0.78,0.34,0.13,"boxes + labels\nmerge, NMS, part-suppression",fc="#f1f5f9",ec=MUT)
arrow(ax,0.56,0.91,0.62,0.88); arrow(ax,0.56,0.78,0.62,0.83)

box(ax,0.30,0.55,0.40,0.12,"CLIP encoder\nSigLIP2 93M  ->  distilled 13.4M",fc="#eff6ff",ec=ACC,bold=True)
arrow(ax,0.79,0.78,0.62,0.67)

box(ax,0.02,0.33,0.21,0.13,"naming\n725-way softmax",fc="#eff6ff",ec=ACC,fs=8)
box(ax,0.26,0.33,0.21,0.13,"attributes\nsoftmax in group",fc="#eff6ff",ec=ACC,fs=8)
box(ax,0.50,0.33,0.22,0.13,"frame checks\nLEARNED PROBES",fc="#ecfdf5",ec=GOOD,fs=8,bold=True)
box(ax,0.75,0.33,0.21,0.13,"person checks\nphrase pairs",fc="#fef3c7",ec=WARN,fs=8)
for x in (0.125,0.365,0.61,0.855): arrow(ax,0.50,0.55,x,0.46)

box(ax,0.20,0.10,0.28,0.13,"importance ranking\nexplicit weights",fc="#f1f5f9",ec=MUT,fs=8)
box(ax,0.52,0.10,0.28,0.13,"composer\ndeterministic, no LM",fc="#f1f5f9",ec=MUT,fs=8)
arrow(ax,0.34,0.33,0.34,0.23); arrow(ax,0.48,0.16,0.52,0.16)
ax.text(0.5,0.015,"caption + auditable operator lines + alerts",ha="center",
        fontsize=9,fontweight="bold",color=INK)
ax.set_title("JANA2 pipeline — every stage inspectable, no language-model decoder",
             loc="left",fontweight="bold",fontsize=10)
plt.tight_layout(); plt.savefig("pipeline.png",dpi=200); plt.close()

# ---------- distillation ----------
fig,ax=plt.subplots(figsize=(8.2,3.0)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
box(ax,0.03,0.60,0.24,0.16,"154k crops\nfrom real frames",fc="#f8fafc",ec=MUT,fs=8)
box(ax,0.36,0.76,0.28,0.16,"TEACHER\nSigLIP2 image tower 93M",fc="#f1f5f9",ec=MUT,fs=8)
box(ax,0.36,0.50,0.28,0.16,"STUDENT\nMobileCLIP2-S0 11.4M (frozen)",fc="#eff6ff",ec=ACC,fs=8)
box(ax,0.70,0.50,0.26,0.16,"projector 2M\nTRAINED",fc="#ecfdf5",ec=GOOD,fs=8,bold=True)
arrow(ax,0.27,0.70,0.36,0.84); arrow(ax,0.27,0.66,0.36,0.58); arrow(ax,0.64,0.58,0.70,0.58)
ax.text(0.83,0.40,"same embedding space",ha="center",fontsize=7.5,color=MUT)
ax.add_patch(FancyArrowPatch((0.83,0.50),(0.60,0.76),arrowstyle="<|-|>",
             mutation_scale=10,color=GOOD,lw=1.2,ls="--"))
ax.text(0.5,0.30,"loss = cosine  +  batch geometry  +  KL over the 1,046 phrases the system reads",
        ha="center",fontsize=8.5,color=INK)
ax.text(0.5,0.20,"the third term is the one that matters: it protects the margins the checks depend on",
        ha="center",fontsize=7.5,color=MUT,style="italic")
ax.text(0.5,0.06,"result:  agreement 0.881   |   probe AP within ~3% of teacher   |   7x fewer parameters",
        ha="center",fontsize=8.5,fontweight="bold",color=GOOD)
ax.set_title("Knowledge distillation — teacher embeddings ARE the labels (no human annotation)",
             loc="left",fontweight="bold",fontsize=10)
plt.tight_layout(); plt.savefig("distill.png",dpi=200); plt.close()

# ---------- labelling loop ----------
fig,ax=plt.subplots(figsize=(8.2,2.6)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
box(ax,0.01,0.55,0.17,0.30,"UCF-Crime\n1,150 videos\nVIDEO-level labels",fc="#f8fafc",ec=MUT,fs=7.5)
box(ax,0.21,0.55,0.17,0.30,"56k frames\n1 per second",fc="#f1f5f9",ec=MUT,fs=7.5)
box(ax,0.41,0.55,0.19,0.30,"Moondream3\nyes/no per frame\n43 q/s",fc="#eff6ff",ec=ACC,fs=7.5)
box(ax,0.63,0.55,0.17,0.30,"FRAME-level\nlabels",fc="#ecfdf5",ec=GOOD,fs=7.5)
box(ax,0.83,0.55,0.16,0.30,"linear probe\n768 numbers",fc="#ecfdf5",ec=GOOD,fs=7.5,bold=True)
for x1,x2 in ((0.18,0.21),(0.38,0.41),(0.60,0.63),(0.80,0.83)):
    arrow(ax,x1,0.70,x2,0.70)
ax.text(0.5,0.36,'"this video contains a fight" does not say WHICH FRAMES.',
        ha="center",fontsize=8.5,color=INK)
ax.text(0.5,0.24,"Frames from a fight video with no fight visible become HARD NEGATIVES —",
        ha="center",fontsize=8,color=MUT)
ax.text(0.5,0.14,"same camera, same place, no fight. Nothing is discarded.",
        ha="center",fontsize=8,color=MUT)
ax.set_title("Weak supervision: a VLM converts video-level labels into frame-level ones",
             loc="left",fontweight="bold",fontsize=10)
plt.tight_layout(); plt.savefig("labelling.png",dpi=200); plt.close()

# ---------- timeline ----------
fig,ax=plt.subplots(figsize=(8.2,3.6))
phases=[("Environment & CUDA",0,1,"#94a3b8"),
        ("Architecture: drop RPN, pick detectors",1,1,"#94a3b8"),
        ("Bug hunt: duplicates, parts, threat tiers",2,2,"#64748b"),
        ("Readout calibration (softmax/sigmoid/pairs)",4,1,"#64748b"),
        ("Distillation: harvest, projector, eval",5,2,ACC),
        ("UCF-Crime: 96GB, frames, Moondream",7,2,ACC),
        ("Probe training + wiring",9,1,GOOD),
        ("Threshold calibration on real CCTV",10,1,GOOD)]
for i,(n,s,d,c) in enumerate(phases):
    ax.barh(i,d,left=s,color=c,height=0.55)
    ax.text(s+d+0.12,i,n,va="center",fontsize=8.5)
ax.set_yticks([]); ax.invert_yaxis()
ax.set_xlim(0,19); ax.set_xlabel("relative effort")
ax.set_xticks([])
for sp in ("top","right","left","bottom"): ax.spines[sp].set_visible(False)
ax.set_title("Project phases",loc="left",fontweight="bold",fontsize=10)
plt.tight_layout(); plt.savefig("timeline.png",dpi=200); plt.close()
print("diagrams done")
