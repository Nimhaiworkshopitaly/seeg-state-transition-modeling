"""Preliminary held-out-trial and cross-subject prediction experiment.

Primary state: percent change from pre-stimulation total functional-network
strength. Connectivity is estimated in 2-second, 50%-overlapping windows after
common-average referencing, 60-Hz notch filtering, 1-80-Hz band-pass filtering,
and resampling to 250 Hz. Active-stimulation windows are deliberately excluded.
"""

from __future__ import annotations

import argparse
import gc
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, sosfiltfilt

from run_analysis import load_localizations, matlab_inclusive_slice, normalize_channel


def preprocess(x: np.ndarray, fs: int, target_fs: int = 250) -> np.ndarray:
    """CAR, notch, band-pass, and resample a localized sEEG segment."""
    x = np.asarray(x, dtype=np.float64)
    x = x - np.median(x, axis=1, keepdims=True)  # robust common-average reference
    b, a = iirnotch(60.0, 30.0, fs=fs)
    x = filtfilt(b, a, x, axis=0)
    sos = butter(4, [1.0, 80.0], btype="bandpass", fs=fs, output="sos")
    x = sosfiltfilt(sos, x, axis=0)
    if fs != target_fs:
        x = resample_poly(x, target_fs, fs, axis=0)
    return x


def network_strength(x: np.ndarray, fs: int, window_s: float = 2.0, step_s: float = 1.0) -> np.ndarray:
    """Mean absolute off-diagonal Pearson correlation per sliding window."""
    size, step = int(round(window_s * fs)), int(round(step_s * fs))
    tri = np.triu_indices(x.shape[1], 1)
    values = []
    for start in range(0, len(x) - size + 1, step):
        block = x[start : start + size]
        block = block - block.mean(axis=0, keepdims=True)
        scale = np.sqrt(np.sum(block * block, axis=0, keepdims=True))
        valid = scale.ravel() > 0
        z = np.divide(block, scale, out=np.zeros_like(block), where=scale > 0)
        corr = z.T @ z
        edges = np.abs(corr[tri])
        edge_valid = valid[tri[0]] & valid[tri[1]]
        values.append(float(edges[edge_valid].mean()))
    return np.asarray(values)


def extract_trial(archive: zipfile.ZipFile, member: str, row: pd.Series, loc: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    with archive.open(member) as stream:
        mat = loadmat(stream, variable_names=["ch_names", "dat"], simplify_cells=True)
    dat = np.asarray(mat["dat"])
    names = np.array([normalize_channel(x) for x in np.ravel(mat["ch_names"]).astype(str)])
    localized = set(loc["channel_norm"])
    keep = np.array([i for i, name in enumerate(names) if name in localized], dtype=int)
    if len(keep) != len(loc):
        raise ValueError(f"Localized channel mismatch in {member}: {len(keep)} vs {len(loc)}")
    pre_slice = matlab_inclusive_slice(int(row.BeforeStimStartI), int(row.BeforeStimEndI), len(dat))
    post_slice = matlab_inclusive_slice(int(row.AfterStimStartI), int(row.AfterStimEndI), len(dat))
    fs = int(row.SamplingFreq)
    pre = preprocess(dat[pre_slice, :][:, keep], fs)
    post = preprocess(dat[post_slice, :][:, keep], fs)
    pre_strength = network_strength(pre, 250)
    post_strength = network_strength(post, 250)
    baseline = float(pre_strength.mean())
    if not np.isfinite(baseline) or baseline <= 0:
        raise ValueError(f"Invalid baseline strength in {member}")
    pre_time = np.arange(len(pre_strength), dtype=float)
    pre_slope = float(np.polyfit(pre_time, pre_strength / baseline - 1.0, 1)[0]) if len(pre_strength) > 1 else 0.0
    target = "ANT" if "ANT" in str(row.StimLocation).upper() else "PULVINAR"
    frame = pd.DataFrame(
        {
            "subject": row.Subject,
            "trial": row.FileName,
            "stim_target": target,
            "window_index": np.arange(len(post_strength)),
            "time_s": np.arange(len(post_strength), dtype=float) + 1.0,
            "network_strength": post_strength,
            "baseline_strength": baseline,
            "baseline_cv": float(pre_strength.std(ddof=1) / baseline),
            "baseline_slope_per_window": pre_slope,
            "state_change": post_strength / baseline - 1.0,
        }
    )
    qc = {
        "subject": row.Subject,
        "trial": row.FileName,
        "stim_target": target,
        "n_channels": int(len(keep)),
        "pre_windows": int(len(pre_strength)),
        "post_windows": int(len(post_strength)),
        "baseline_strength": baseline,
        "baseline_cv": float(pre_strength.std(ddof=1) / baseline),
        "baseline_slope_per_window": pre_slope,
    }
    del mat, dat, pre, post
    gc.collect()
    return frame, qc


def design_matrix(frame: pd.DataFrame) -> np.ndarray:
    t = frame["window_index"].to_numpy(dtype=float)
    denom = max(1.0, float(frame["window_index"].max()))
    t = t / denom
    pul = (frame["stim_target"] == "PULVINAR").to_numpy(dtype=float)
    return np.column_stack(
        [
            t,
            t * t,
            pul,
            t * pul,
            t * t * pul,
        ]
    )


class Ridge:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> "Ridge":
        # Equalize trial influence even if a future dataset has unequal trajectory lengths.
        _, counts = np.unique(groups, return_counts=True)
        count_map = dict(zip(*np.unique(groups, return_counts=True)))
        weights = np.array([1.0 / count_map[g] for g in groups])
        weights *= len(weights) / weights.sum()
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        z = (x - self.mean_) / self.scale_
        z = np.column_stack([np.ones(len(z)), z])
        sw = np.sqrt(weights)[:, None]
        penalty = np.eye(z.shape[1]) * self.alpha
        penalty[0, 0] = 0.0
        weighted_y = y * (sw if y.ndim == 2 else sw[:, 0])
        self.coef_ = np.linalg.solve((z * sw).T @ (z * sw) + penalty, (z * sw).T @ weighted_y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean_) / self.scale_
        return np.column_stack([np.ones(len(z)), z]) @ self.coef_


def target_mean_prediction(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    curves = train.groupby(["stim_target", "window_index"])["state_change"].mean()
    overall = train.groupby("window_index")["state_change"].mean()
    pred = []
    for row in test.itertuples():
        key = (row.stim_target, row.window_index)
        pred.append(curves[key] if key in curves.index else overall.get(row.window_index, 0.0))
    return np.asarray(pred)


def score_predictions(frame: pd.DataFrame, split: str, direction: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, metrics = [], []
    if split == "within_subject":
        folds = []
        for subject in sorted(frame.subject.unique()):
            subject_frame = frame[frame.subject == subject]
            for trial in sorted(subject_frame.trial.unique()):
                folds.append((f"{subject}:{trial}", subject_frame[subject_frame.trial != trial], subject_frame[subject_frame.trial == trial]))
    else:
        train_subject, test_subject = direction.split("->")
        folds = [(direction, frame[frame.subject == train_subject], frame[frame.subject == test_subject])]

    for fold, train, test in folds:
        model = Ridge(alpha=1.0).fit(design_matrix(train), train.state_change.to_numpy(), train.trial.to_numpy())
        predictions = {
            "ridge": model.predict(design_matrix(test)),
            "persistence": np.zeros(len(test)),
            "target_mean": target_mean_prediction(train, test),
        }
        for model_name, pred in predictions.items():
            out = test[["subject", "trial", "stim_target", "window_index", "time_s", "state_change"]].copy()
            out["split"] = split
            out["fold"] = fold
            out["model"] = model_name
            out["prediction"] = pred
            rows.append(out)
            for trial, trial_frame in out.groupby("trial"):
                y, yhat = trial_frame.state_change.to_numpy(), trial_frame.prediction.to_numpy()
                metrics.append(
                    {
                        "split": split,
                        "fold": fold,
                        "test_subject": trial_frame.subject.iloc[0],
                        "trial": trial,
                        "stim_target": trial_frame.stim_target.iloc[0],
                        "model": model_name,
                        "mae": float(np.mean(np.abs(y - yhat))),
                        "rmse": float(np.sqrt(np.mean((y - yhat) ** 2))),
                        "correlation": float(np.corrcoef(y, yhat)[0, 1]) if np.std(y) > 0 and np.std(yhat) > 0 else np.nan,
                    }
                )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj1", type=Path, required=True)
    parser.add_argument("--subj2", type=Path, required=True)
    parser.add_argument("--stim-info", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    info = pd.read_excel(args.stim_info)
    trajectories, qcs = [], []
    for zip_path in [args.subj1, args.subj2]:
        loc = load_localizations(zip_path)
        subject = zip_path.stem
        with zipfile.ZipFile(zip_path) as archive:
            members = {Path(n).name: n for n in archive.namelist() if n.lower().endswith(".mat")}
            for _, row in info[info.Subject == subject].iterrows():
                trajectory, qc = extract_trial(archive, members[row.FileName], row, loc)
                trajectories.append(trajectory)
                qcs.append(qc)
    states = pd.concat(trajectories, ignore_index=True)
    state_qc = pd.DataFrame(qcs)
    states.to_csv(args.output / "network_strength_trajectories.csv", index=False)
    state_qc.to_csv(args.output / "network_strength_qc.csv", index=False)

    within_pred, within_metrics = score_predictions(states, "within_subject", "")
    cross_outputs = [score_predictions(states, "cross_subject", d) for d in ["Subj_1->Subj_2", "Subj_2->Subj_1"]]
    predictions = pd.concat([within_pred] + [x[0] for x in cross_outputs], ignore_index=True)
    metrics = pd.concat([within_metrics] + [x[1] for x in cross_outputs], ignore_index=True)
    predictions.to_csv(args.output / "model_predictions.csv", index=False)
    metrics.to_csv(args.output / "trial_metrics.csv", index=False)
    aggregate = metrics.groupby(["split", "fold", "model"], as_index=False).agg(mae=("mae", "mean"), rmse=("rmse", "mean"), correlation=("correlation", "mean"), n_trials=("trial", "nunique"))
    aggregate.to_csv(args.output / "aggregate_metrics.csv", index=False)
    summary = {
        "state": "percent change from pre-stimulation mean absolute functional-connectivity strength",
        "n_trials": int(state_qc.trial.nunique()),
        "n_post_windows": int(len(states)),
        "within_subject_mean_metrics": metrics[metrics.split == "within_subject"].groupby("model")[["mae", "rmse", "correlation"]].mean().to_dict(orient="index"),
        "cross_subject_mean_metrics": metrics[metrics.split == "cross_subject"].groupby(["fold", "model"])[["mae", "rmse", "correlation"]].mean().reset_index().to_dict(orient="records"),
    }
    (args.output / "preliminary_model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
