"""Leakage-safe shared-parcel connectivity PCA experiment."""

from __future__ import annotations

import argparse
import gc
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from run_analysis import load_localizations, matlab_inclusive_slice, normalize_channel
from run_preliminary_models import Ridge, preprocess


def window_edges(x: np.ndarray, groups: list[np.ndarray], size: int = 500, step: int = 250) -> np.ndarray:
    """Fisher-z parcel connectivity edges in 2-s windows at 250 Hz."""
    tri = np.triu_indices(len(groups), 1)
    rows = []
    for start in range(0, len(x) - size + 1, step):
        block = x[start : start + size]
        parcel = np.column_stack([block[:, idx].mean(axis=1) for idx in groups])
        corr = np.corrcoef(parcel, rowvar=False)
        corr = np.clip(corr, -0.999999, 0.999999)
        rows.append(np.arctanh(corr[tri]))
    return np.asarray(rows)


def extract_trial(archive, member, row, loc, parcels):
    with archive.open(member) as stream:
        mat = loadmat(stream, variable_names=["ch_names", "dat"], simplify_cells=True)
    dat = np.asarray(mat["dat"])
    names = np.array([normalize_channel(x) for x in np.ravel(mat["ch_names"]).astype(str)])
    lookup = loc.set_index("channel_norm")
    channel_indices, labels = [], []
    for i, name in enumerate(names):
        if name in lookup.index and str(lookup.at[name, "aBNlab"]) in parcels:
            channel_indices.append(i)
            labels.append(str(lookup.at[name, "aBNlab"]))
    groups = [np.flatnonzero(np.asarray(labels) == parcel) for parcel in parcels]
    if any(len(g) == 0 for g in groups):
        raise ValueError(f"Missing shared parcel in {member}")
    fs = int(row.SamplingFreq)
    pre_slc = matlab_inclusive_slice(int(row.BeforeStimStartI), int(row.BeforeStimEndI), len(dat))
    post_slc = matlab_inclusive_slice(int(row.AfterStimStartI), int(row.AfterStimEndI), len(dat))
    pre = preprocess(dat[pre_slc, :][:, channel_indices], fs)
    post = preprocess(dat[post_slc, :][:, channel_indices], fs)
    pre_edges, post_edges = window_edges(pre, groups), window_edges(post, groups)
    baseline = pre_edges.mean(axis=0)
    changes = post_edges - baseline
    target = "ANT" if "ANT" in str(row.StimLocation).upper() else "PULVINAR"
    meta = pd.DataFrame({
        "subject": row.Subject, "trial": row.FileName, "stim_target": target,
        "window_index": np.arange(len(changes)), "time_s": np.arange(len(changes)) + 1.0,
    })
    qc = {"subject": row.Subject, "trial": row.FileName, "stim_target": target,
          "contacts_retained": len(channel_indices), "pre_windows": len(pre_edges),
          "post_windows": len(post_edges), "baseline_edge_sd": float(baseline.std(ddof=1))}
    del mat, dat, pre, post
    gc.collect()
    return meta, changes, qc


class PCA:
    def __init__(self, n_components=3): self.n_components = n_components
    def fit(self, x):
        self.mean_ = x.mean(axis=0)
        _, s, vt = np.linalg.svd(x - self.mean_, full_matrices=False)
        self.components_ = vt[: self.n_components]
        variance = s * s
        self.explained_ratio_ = variance[: self.n_components] / variance.sum()
        return self
    def transform(self, x): return (x - self.mean_) @ self.components_.T
    def inverse_transform(self, z): return z @ self.components_ + self.mean_


def design(frame):
    t = frame.window_index.to_numpy(float) / max(1.0, float(frame.window_index.max()))
    pul = (frame.stim_target == "PULVINAR").to_numpy(float)
    return np.column_stack([t, t*t, pul, t*pul, t*t*pul])


def target_mean(train_meta, train_z, test_meta):
    table = pd.DataFrame(train_z)
    table["target"] = train_meta.stim_target.to_numpy()
    table["window"] = train_meta.window_index.to_numpy()
    component_cols = list(range(train_z.shape[1]))
    by_target = table.groupby(["target", "window"])[component_cols].mean()
    overall = table.groupby("window")[component_cols].mean()
    rows = []
    for r in test_meta.itertuples():
        key = (r.stim_target, r.window_index)
        rows.append(by_target.loc[key].to_numpy() if key in by_target.index else overall.loc[r.window_index].to_numpy())
    return np.vstack(rows)


def evaluate(meta, edges, split, direction="", n_components=3):
    if split == "within_subject":
        folds=[]
        for subject in sorted(meta.subject.unique()):
            sm = meta[meta.subject == subject]
            for trial in sorted(sm.trial.unique()):
                test=np.flatnonzero((meta.subject == subject) & (meta.trial == trial))
                train=np.flatnonzero((meta.subject == subject) & (meta.trial != trial))
                folds.append((f"{subject}:{trial}",train,test))
    else:
        a,b=direction.split("->")
        folds=[(direction,np.flatnonzero(meta.subject==a),np.flatnonzero(meta.subject==b))]
    pred_rows, metric_rows, pca_rows = [], [], []
    for fold, tr, te in folds:
        pca=PCA(n_components).fit(edges[tr]); ztr=pca.transform(edges[tr]); zte=pca.transform(edges[te])
        model=Ridge(1.0).fit(design(meta.iloc[tr]),ztr,meta.iloc[tr].trial.to_numpy())
        preds={"ridge":model.predict(design(meta.iloc[te])),"persistence":pca.transform(np.zeros_like(edges[te])),
               "target_mean":target_mean(meta.iloc[tr],ztr,meta.iloc[te])}
        pca_rows.append({"split":split,"fold":fold,**{f"pc{i+1}_variance":float(v) for i,v in enumerate(pca.explained_ratio_)},
                         "variance_3pc":float(pca.explained_ratio_.sum())})
        for name,zpred in preds.items():
            epred=pca.inverse_transform(zpred)
            for local_i, global_i in enumerate(te):
                r=meta.iloc[global_i]
                pred_rows.append({"split":split,"fold":fold,"model":name,"subject":r.subject,"trial":r.trial,
                                  "stim_target":r.stim_target,"window_index":int(r.window_index),"time_s":float(r.time_s),
                                  **{f"true_pc{k+1}":float(zte[local_i,k]) for k in range(n_components)},
                                  **{f"pred_pc{k+1}":float(zpred[local_i,k]) for k in range(n_components)}})
            test_trials=meta.iloc[te].trial.to_numpy()
            for trial in np.unique(test_trials):
                mask=test_trials==trial; actual=edges[te][mask]; predicted=epred[mask]
                latent_actual=zte[mask]; latent_pred=zpred[mask]
                metric_rows.append({"split":split,"fold":fold,"test_subject":meta.iloc[te][mask].subject.iloc[0],
                    "trial":trial,"stim_target":meta.iloc[te][mask].stim_target.iloc[0],"model":name,
                    "edge_mae":float(np.mean(np.abs(actual-predicted))),
                    "edge_rmse":float(np.sqrt(np.mean((actual-predicted)**2))),
                    "latent_rmse":float(np.sqrt(np.mean((latent_actual-latent_pred)**2))),
                    "edge_pattern_correlation":float(np.mean([np.corrcoef(a,b)[0,1] for a,b in zip(actual,predicted) if np.std(b)>1e-12])) if np.any(np.std(predicted,axis=1)>1e-12) else np.nan})
    return pd.DataFrame(pred_rows),pd.DataFrame(metric_rows),pd.DataFrame(pca_rows)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--subj1",type=Path,required=True); ap.add_argument("--subj2",type=Path,required=True)
    ap.add_argument("--stim-info",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    info=pd.read_excel(args.stim_info); locs={p.stem:load_localizations(p) for p in [args.subj1,args.subj2]}
    parcels=sorted((set(locs[args.subj1.stem].aBNlab.dropna().astype(str)) & set(locs[args.subj2.stem].aBNlab.dropna().astype(str)))-{"no_label_found"})
    metas=[]; edge_sets=[]; qcs=[]
    for zp in [args.subj1,args.subj2]:
        with zipfile.ZipFile(zp) as ar:
            members={Path(n).name:n for n in ar.namelist() if n.lower().endswith('.mat')}
            for _,row in info[info.Subject==zp.stem].iterrows():
                m,e,q=extract_trial(ar,members[row.FileName],row,locs[zp.stem],parcels); metas.append(m); edge_sets.append(e); qcs.append(q)
    meta=pd.concat(metas,ignore_index=True); edges=np.vstack(edge_sets)
    tri=np.triu_indices(len(parcels),1); edge_names=[f"{parcels[i]}--{parcels[j]}" for i,j in zip(*tri)]
    pd.DataFrame(edges,columns=edge_names).join(meta).to_csv(args.output/'parcel_edge_trajectories.csv',index=False)
    pd.DataFrame(qcs).to_csv(args.output/'parcel_qc.csv',index=False)
    outputs=[evaluate(meta,edges,'within_subject')]+[evaluate(meta,edges,'cross_subject',d) for d in ['Subj_1->Subj_2','Subj_2->Subj_1']]
    preds=pd.concat([x[0] for x in outputs],ignore_index=True); metrics=pd.concat([x[1] for x in outputs],ignore_index=True); pca=pd.concat([x[2] for x in outputs],ignore_index=True)
    preds.to_csv(args.output/'parcel_pca_predictions.csv',index=False); metrics.to_csv(args.output/'parcel_pca_trial_metrics.csv',index=False); pca.to_csv(args.output/'parcel_pca_fold_variance.csv',index=False)
    summary={"shared_parcels":parcels,"n_parcels":len(parcels),"n_edges":len(edge_names),"n_windows":len(meta),
      "mean_3pc_variance":float(pca.variance_3pc.mean()),
      "within":metrics[metrics.split=='within_subject'].groupby('model')[["edge_mae","edge_rmse","latent_rmse","edge_pattern_correlation"]].mean().to_dict('index'),
      "cross":metrics[metrics.split=='cross_subject'].groupby(['fold','model'])[["edge_mae","edge_rmse","latent_rmse","edge_pattern_correlation"]].mean().reset_index().to_dict('records')}
    (args.output/'parcel_pca_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
