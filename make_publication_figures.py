from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
DATA = ROOT / "results" / "architecture_comparison"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
metrics = pd.read_csv(DATA / "architecture_trial_metrics.csv")
qc = pd.read_csv(DATA / "architecture_fold_qc.csv")

labels = {"persistence":"Persistence", "linear_ssm":"Linear SSM", "gru_ssm":"GRU SSM", "transformer":"Transformer"}
order = list(labels)
colors = {"persistence":"#7A7A7A", "linear_ssm":"#167D9A", "gru_ssm":"#D28B26", "transformer":"#8B5FBF"}
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9, "axes.spines.top":False, "axes.spines.right":False, "figure.dpi":150})

def save(fig, stem):
    fig.savefig(OUT/f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT/f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

# Figure 1: within-subject aggregate and paired trials.
w = metrics[metrics.split == "within_subject"]
means = w.groupby("model").latent_rmse.mean().reindex(order)
sem = w.groupby("model").latent_rmse.sem().reindex(order)
fig, axes = plt.subplots(1,2,figsize=(10,3.7),gridspec_kw={"width_ratios":[1,1.65]})
axes[0].bar(range(4),means,color=[colors[x] for x in order],yerr=sem,capsize=3,width=.7)
axes[0].set_xticks(range(4),[labels[x] for x in order],rotation=22,ha="right")
axes[0].set_ylabel("Latent RMSE")
axes[0].set_title("A  Mean held out trial error",loc="left",fontweight="bold")
for i,v in enumerate(means): axes[0].text(i,v+.012,f"{v:.3f}",ha="center",va="bottom",fontsize=8)
piv=w.pivot(index="trial",columns="model",values="latent_rmse")
x=np.arange(len(piv)); width=.19
for j,m in enumerate(order): axes[1].bar(x+(j-1.5)*width,piv[m],width,label=labels[m],color=colors[m])
axes[1].set_xticks(x,[t.replace(".mat","").replace("Subj_", "S") for t in piv.index],rotation=45,ha="right")
axes[1].set_ylabel("Latent RMSE"); axes[1].set_title("B  Performance by held out trial",loc="left",fontweight="bold")
axes[1].legend(frameon=False,ncol=2,fontsize=8)
fig.tight_layout(); save(fig,"figure1_within_subject_performance")

# Figure 2: cross-subject transfer.
c=metrics[metrics.split=="cross_subject"].groupby(["fold","model"]).latent_rmse.mean().unstack()
fig,ax=plt.subplots(figsize=(7,4)); x=np.arange(len(c)); width=.19
for j,m in enumerate(order): ax.bar(x+(j-1.5)*width,c[m],width,label=labels[m],color=colors[m])
ax.set_xticks(x,[s.replace("Subj_1","Subject 1").replace("Subj_2","Subject 2") for s in c.index])
ax.set_ylabel("Latent RMSE"); ax.set_title("Cross Subject one step prediction",loc="left",fontweight="bold")
ax.legend(frameon=False,ncol=2)
for i,fold in enumerate(c.index):
    best=c.loc[fold].idxmin(); j=order.index(best); y=c.loc[fold,best]
    ax.text(i+(j-1.5)*width,y+.009,"best",ha="center",fontsize=8,fontweight="bold")
fig.tight_layout(); save(fig,"figure2_cross_subject_transfer")

# Figure 3: improvement and PCA QC.
fig,axes=plt.subplots(1,2,figsize=(9,3.6))
baseline=means["persistence"]
improvement=(1-means[["linear_ssm","gru_ssm","transformer"]]/baseline)*100
axes[0].bar(range(3),improvement,color=[colors[x] for x in improvement.index])
axes[0].axhline(0,color="#333333",lw=.8); axes[0].set_xticks(range(3),[labels[x] for x in improvement.index],rotation=18,ha="right")
axes[0].set_ylabel("RMSE improvement vs persistence (%)"); axes[0].set_title("A  Relative model improvement",loc="left",fontweight="bold")
axes[0].set_ylim(min(-8, improvement.min()-2), max(10, improvement.max()+2))
for i,v in enumerate(improvement): axes[0].text(i,v+(0.6 if v>=0 else -0.6),f"{v:.1f}%",ha="center",va="bottom" if v>=0 else "top")
q=qc.copy(); q["type"]=np.where(q.split=="within_subject","Held out trial","Held out subject")
vals=[q[q.type==t].pc_variance_total*100 for t in ["Held out trial","Held out subject"]]
axes[1].boxplot(vals,tick_labels=["Held out\ntrial","Held out\nsubject"],patch_artist=True,boxprops={"facecolor":"#DCECF1"},medianprops={"color":"#167D9A","linewidth":2})
axes[1].set_ylabel("Variance explained by 3 training PCs (%)"); axes[1].set_title("B  Fold-specific PCA coverage",loc="left",fontweight="bold")
fig.tight_layout(); save(fig,"figure3_improvement_and_pca_qc")

print(f"Saved figures to {OUT}")
