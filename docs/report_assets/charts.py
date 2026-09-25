import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

INK="#1a1a1a"; MUT="#6b6b6b"; ACC="#2563eb"; GOOD="#059669"; BAD="#dc2626"; WARN="#d97706"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,
    "axes.edgecolor":"#cccccc","axes.labelcolor":INK,"text.color":INK,
    "xtick.color":MUT,"ytick.color":MUT,"axes.spines.top":False,"axes.spines.right":False})

# ---------- 1. probe AP ----------
checks=["accident","climbing","person_down","snatching","smoke","fight",
        "vandalism","fire","weapon_scene"]
teach=[0.945,0.943,0.941,0.937,0.921,0.899,0.823,0.801,0.340]
stud={"accident":0.930,"climbing":0.941,"fight":0.872,"fire":0.751}
fig,ax=plt.subplots(figsize=(8,3.6))
y=np.arange(len(checks))
cols=[GOOD if v>=0.7 else BAD for v in teach]
ax.barh(y,teach,color=cols,height=0.55,label="teacher (SigLIP2 93M)")
sy=[i for i,c in enumerate(checks) if c in stud]
sv=[stud[checks[i]] for i in sy]
ax.scatter(sv,sy,color=INK,zorder=5,s=28,marker="D",label="student (13.4M)")
ax.axvline(0.7,color=WARN,ls="--",lw=1)
ax.text(0.705,-0.7,"adoption bar 0.70",color=WARN,fontsize=7.5)
ax.set_yticks(y); ax.set_yticklabels(checks); ax.invert_yaxis()
ax.set_xlim(0,1.02); ax.set_xlabel("Average Precision (held-out)")
for i,v in enumerate(teach): ax.text(v+0.012,i,f"{v:.3f}",va="center",fontsize=8)
ax.legend(loc="lower right",frameon=False,fontsize=8)
ax.set_title("Learned probes: 8 of 9 checks cleared the bar",loc="left",fontweight="bold")
plt.tight_layout(); plt.savefig("ap.png",dpi=200); plt.close()

# ---------- 2. margins before/after ----------
names=["accident","weapon_scene","crowd_surge","vandalism","wrong_side","person_down",
       "fight","darkness","climbing","unattended_bag","littering","flooding"]
before=[0.0043,-0.0043,0.0087,0.0104,0.0109,0.0136,0.0130,0.0172,0.0118,-0.0239,-0.0011,-0.0006]
after=[0.0367,0.0383,0.0388,0.0316,0.0426,0.0342,0.0272,0.0549,0.0217,0.0286,0.0196,0.0173]
fig,ax=plt.subplots(figsize=(8,3.4))
x=np.arange(len(names)); w=0.38
ax.bar(x-w/2,before,w,color="#cbd5e1",label="before (negation in negatives)")
ax.bar(x+w/2,after,w,color=ACC,label="after rewrite")
ax.axhline(0,color=MUT,lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(names,rotation=40,ha="right",fontsize=8)
ax.set_ylabel("cosine margin")
ax.legend(frameon=False,fontsize=8)
ax.set_title("Removing negation from negative phrases: margins doubled to tripled",
             loc="left",fontweight="bold")
plt.tight_layout(); plt.savefig("margins.png",dpi=200); plt.close()

# ---------- 3. parameter budget ----------
fig,ax=plt.subplots(figsize=(7.4,2.5))
labels=["Before\n(SigLIP2)","After\n(distilled student)"]
det=[3,3]; wor=[13,13]; clip=[93,13.4]
ax.barh(labels,det,color="#94a3b8",label="yolo26n")
ax.barh(labels,wor,left=det,color="#64748b",label="YOLO-World-s")
ax.barh(labels,clip,left=np.array(det)+np.array(wor),color=ACC,label="CLIP encoder")
ax.axvline(100,color=BAD,ls="--",lw=1.2)
ax.text(101,1.35,"100M budget",color=BAD,fontsize=8)
for i,t in enumerate([109,29.4]):
    ax.text(t+2,i,f"{t}M",va="center",fontweight="bold")
ax.set_xlabel("million parameters"); ax.set_xlim(0,125)
ax.legend(frameon=False,fontsize=8,ncol=3,loc="lower right")
ax.set_title("Encoder distillation: 3.7x smaller, back inside budget",
             loc="left",fontweight="bold")
plt.tight_layout(); plt.savefig("params.png",dpi=200); plt.close()

# ---------- 4. data funnel ----------
fig,ax=plt.subplots(figsize=(7.4,3.0))
stages=["UCF-Crime\ndownload","videos\nextracted","frames\nsampled","frames\nlabelled",
        "Moondream\nqueries","usable\nlabels"]
vals=[96,1150,56000,23500,48782,45891]
disp=["96 GB","1,150","56,000","23,500","48,782","45,891"]
xs=np.arange(len(stages))
ax.bar(xs,np.log10(np.array(vals,dtype=float)+1),color=[ACC]*4+[GOOD]*2,width=0.62)
for i,d in enumerate(disp):
    ax.text(i,np.log10(vals[i]+1)+0.08,d,ha="center",fontweight="bold",fontsize=8.5)
ax.set_xticks(xs); ax.set_xticklabels(stages,fontsize=8)
ax.set_ylabel("log scale"); ax.set_yticks([])
ax.set_title("Labelling pipeline volumes (5.9% of answers unparseable, dropped not guessed)",
             loc="left",fontweight="bold")
plt.tight_layout(); plt.savefig("funnel.png",dpi=200); plt.close()
print("charts done")
