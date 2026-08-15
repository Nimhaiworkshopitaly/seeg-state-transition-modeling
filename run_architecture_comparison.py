"""Compare linear SSM, compact GRU SSM, and tiny causal Transformer.

Uses previously extracted shared-parcel edge-change trajectories. PCA is fit
inside each training fold. Evaluation is one-step-ahead teacher-forced
prediction on wholly held-out trials/subjects, never random window splits.
"""

from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn


SEED = 20260813
CONTEXT = 8
N_PC = 3


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


class PCA:
    def fit(self, x):
        self.mean_=x.mean(0); _,s,vt=np.linalg.svd(x-self.mean_,full_matrices=False)
        self.components_=vt[:N_PC]; self.explained_=(s[:N_PC]**2)/(s**2).sum(); return self
    def transform(self,x): return (x-self.mean_)@self.components_.T


def sequences(meta, z, indices):
    xs, ys, rows = [], [], []
    idx_set=set(indices.tolist())
    for trial in meta.iloc[indices].trial.unique():
        idx=np.array([i for i in indices if meta.iloc[i].trial==trial])
        idx=idx[np.argsort(meta.iloc[idx].window_index.to_numpy())]
        target=float(meta.iloc[idx[0]].stim_target=='PULVINAR')
        for j in range(CONTEXT,len(idx)):
            history=z[idx[j-CONTEXT:j]]
            context=np.column_stack([history,np.full(CONTEXT,target)])
            xs.append(context); ys.append(z[idx[j]]); rows.append(idx[j])
    return np.asarray(xs,np.float32),np.asarray(ys,np.float32),np.asarray(rows)


def ridge_fit(x,y,alpha=1.0):
    flat=x.reshape(len(x),-1); mean=flat.mean(0); scale=flat.std(0); scale[scale==0]=1
    a=np.column_stack([np.ones(len(flat)),(flat-mean)/scale]); pen=np.eye(a.shape[1])*alpha; pen[0,0]=0
    coef=np.linalg.solve(a.T@a+pen,a.T@y); return mean,scale,coef
def ridge_predict(model,x):
    mean,scale,coef=model; flat=x.reshape(len(x),-1); return np.column_stack([np.ones(len(flat)),(flat-mean)/scale])@coef


class GRUSSM(nn.Module):
    def __init__(self):
        super().__init__(); self.gru=nn.GRU(4,12,batch_first=True); self.head=nn.Linear(12,3)
    def forward(self,x): return self.head(self.gru(x)[0][:,-1])


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__(); self.embed=nn.Linear(4,16); self.pos=nn.Parameter(torch.zeros(1,CONTEXT,16))
        layer=nn.TransformerEncoderLayer(16,2,32,dropout=.1,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,1); self.head=nn.Linear(16,3)
    def forward(self,x):
        h=self.embed(x)+self.pos; mask=torch.triu(torch.ones(CONTEXT,CONTEXT,device=x.device,dtype=torch.bool),1)
        return self.head(self.encoder(h,mask=mask)[:,-1])


def train_nn(cls,x,y,seed):
    seed_all(seed); model=cls(); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-2)
    xt=torch.tensor(x); yt=torch.tensor(y); model.train()
    for _ in range(120):
        order=torch.randperm(len(xt))
        for start in range(0,len(xt),32):
            ii=order[start:start+32]; opt.zero_grad(); loss=((model(xt[ii])-yt[ii])**2).mean(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    model.eval(); return model


def evaluate_fold(meta,edges,fold,split,tr,te):
    pca=PCA().fit(edges[tr]); z=pca.transform(edges); xtr,ytr,_=sequences(meta,z,tr); xte,yte,row_idx=sequences(meta,z,te)
    models={"persistence":xte[:,-1,:3],"linear_ssm":ridge_predict(ridge_fit(xtr,ytr),xte)}
    for name,cls in [("gru_ssm",GRUSSM),("transformer",TinyTransformer)]:
        ensemble=[]
        for seed in [SEED,SEED+1,SEED+2]:
            net=train_nn(cls,xtr,ytr,seed); ensemble.append(net(torch.tensor(xte)).detach().numpy())
        models[name]=np.mean(ensemble,axis=0)
    pred_rows=[]; metric_rows=[]
    for name,pred in models.items():
        for k,gi in enumerate(row_idx):
            r=meta.iloc[gi]; pred_rows.append({"split":split,"fold":fold,"model":name,"subject":r.subject,"trial":r.trial,
                "stim_target":r.stim_target,"window_index":int(r.window_index),
                **{f"true_pc{i+1}":float(yte[k,i]) for i in range(N_PC)},**{f"pred_pc{i+1}":float(pred[k,i]) for i in range(N_PC)}})
        test_trials=meta.iloc[row_idx].trial.to_numpy()
        for trial in np.unique(test_trials):
            mask=test_trials==trial; a=yte[mask]; b=pred[mask]
            corr=[np.corrcoef(a[:,i],b[:,i])[0,1] for i in range(N_PC) if np.std(a[:,i])>0 and np.std(b[:,i])>1e-12]
            metric_rows.append({"split":split,"fold":fold,"test_subject":meta.iloc[row_idx][mask].subject.iloc[0],"trial":trial,
                "stim_target":meta.iloc[row_idx][mask].stim_target.iloc[0],"model":name,"latent_mae":float(np.mean(np.abs(a-b))),
                "latent_rmse":float(np.sqrt(np.mean((a-b)**2))),"mean_pc_correlation":float(np.mean(corr)) if corr else np.nan})
    variance={"split":split,"fold":fold,"pc_variance_total":float(pca.explained_.sum()),"n_train_sequences":len(xtr),"n_test_sequences":len(xte)}
    return pred_rows,metric_rows,variance


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--edges',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.edges); edge_cols=[c for c in df if '--' in c]; meta=df[['subject','trial','stim_target','window_index','time_s']].copy(); edges=df[edge_cols].to_numpy()
    folds=[]
    for subject in sorted(meta.subject.unique()):
        sm=meta[meta.subject==subject]
        for trial in sorted(sm.trial.unique()):
            tr=np.flatnonzero((meta.subject==subject)&(meta.trial!=trial)); te=np.flatnonzero((meta.subject==subject)&(meta.trial==trial)); folds.append((f'{subject}:{trial}','within_subject',tr,te))
    for a,b in [('Subj_1','Subj_2'),('Subj_2','Subj_1')]: folds.append((f'{a}->{b}','cross_subject',np.flatnonzero(meta.subject==a),np.flatnonzero(meta.subject==b)))
    pr,mr,vr=[],[],[]
    for i,(name,split,tr,te) in enumerate(folds):
        print(f'fold {i+1}/{len(folds)} {name}',flush=True); p,m,v=evaluate_fold(meta,edges,name,split,tr,te); pr+=p; mr+=m; vr.append(v)
    preds=pd.DataFrame(pr); metrics=pd.DataFrame(mr); variance=pd.DataFrame(vr)
    preds.to_csv(args.output/'architecture_predictions.csv',index=False); metrics.to_csv(args.output/'architecture_trial_metrics.csv',index=False); variance.to_csv(args.output/'architecture_fold_qc.csv',index=False)
    summary={"endpoint":"one-step-ahead prediction of 3 training-fold PCA parcel-connectivity states","context_windows":CONTEXT,"ensemble_seeds":3,
      "within":metrics[metrics.split=='within_subject'].groupby('model')[["latent_mae","latent_rmse","mean_pc_correlation"]].mean().to_dict('index'),
      "cross":metrics[metrics.split=='cross_subject'].groupby(['fold','model'])[["latent_mae","latent_rmse","mean_pc_correlation"]].mean().reset_index().to_dict('records')}
    (args.output/'architecture_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
