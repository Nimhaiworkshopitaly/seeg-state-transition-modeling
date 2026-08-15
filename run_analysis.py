"""Initial QC and feature extraction for the LBCM pilot dataset.

The script reads the two subject ZIP archives without changing or unpacking
the source data. It validates Michelle's stimulation indices against DC01,
extracts pre/stimulation/post features, joins electrode localization, and
summarizes the available CCEP context.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import find_peaks, welch


BANDS_HZ = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 80.0),
    "high_gamma": (80.0, 150.0),
}


def normalize_channel(value: object) -> str:
    """Map EDF/MAT/CCEP/localization labels to a common uppercase contact form."""
    s = str(value).upper().strip()
    s = re.sub(r"^POL[ _-]*", "", s)
    s = re.sub(r"[ _-]*REF$", "", s)
    return re.sub(r"[^A-Z0-9]", "", s)


def is_non_seeg(channel: str) -> bool:
    return channel.startswith(("DC", "CCEP", "EKG", "ECG"))


def resolve_stim_contacts(stim_location: str, available: set[str]) -> tuple[str, str]:
    """Resolve folder labels, including the Subject 2 PLV/PUL naming variant."""
    raw = [normalize_channel(x) for x in str(stim_location).split("-")]
    if len(raw) != 2:
        raise ValueError(f"Expected a bipolar stimulation location, got {stim_location}")
    resolved = []
    for contact in raw:
        base_candidates = [contact, contact.replace("PLV", "PUL"), contact.replace("PUL", "PLV")]
        candidates = list(dict.fromkeys(base_candidates + [f"{x}1" for x in base_candidates]))
        matches = [x for x in candidates if x in available]
        if len(matches) != 1:
            raise ValueError(f"Cannot uniquely resolve {contact} from {stim_location}: {matches}")
        resolved.append(matches[0])
    return resolved[0], resolved[1]


def matlab_inclusive_slice(start_i: int, end_i: int, n: int) -> slice:
    """Convert validated MATLAB 1-based inclusive endpoints to a Python slice."""
    if not (1 <= start_i <= end_i <= n):
        raise ValueError(f"Invalid MATLAB interval [{start_i}, {end_i}] for n={n}")
    return slice(start_i - 1, end_i)


def feature_table(x: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    """Return channel-wise time and frequency-domain features."""
    x = np.asarray(x, dtype=np.float64)
    centered = x - np.nanmean(x, axis=0, keepdims=True)
    std = np.nanstd(x, axis=0, ddof=1)
    rms = np.sqrt(np.nanmean(centered * centered, axis=0))
    line_length = np.nanmean(np.abs(np.diff(x, axis=0)), axis=0)
    f, pxx = welch(centered, fs=fs, axis=0, nperseg=min(2000, len(x)))
    total_mask = (f >= 1.0) & (f <= min(200.0, fs / 2.0))
    total = np.trapezoid(pxx[total_mask], f[total_mask], axis=0)
    out = {"std": std, "rms": rms, "line_length": line_length, "power_1_200": total}
    for name, (lo, hi) in BANDS_HZ.items():
        mask = (f >= lo) & (f < hi)
        bp = np.trapezoid(pxx[mask], f[mask], axis=0)
        out[f"power_{name}"] = bp
        out[f"relative_{name}"] = np.divide(bp, total, out=np.full_like(bp, np.nan), where=total > 0)
    return out


def load_localizations(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".xlsx")]
        if len(names) != 1:
            raise ValueError(f"Expected one localization workbook in {zip_path}, found {names}")
        with archive.open(names[0]) as stream:
            loc = pd.read_excel(stream)
    loc = loc.copy()
    loc["channel_norm"] = loc["Label"].map(normalize_channel)
    if loc["channel_norm"].duplicated().any():
        raise ValueError(f"Duplicate normalized localization labels in {zip_path}")
    return loc


def analyze_trial(
    archive: zipfile.ZipFile,
    member: str,
    info: pd.Series,
    localization: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    with archive.open(member) as stream:
        mat = loadmat(stream, variable_names=["ch_names", "dat", "samp_times"], simplify_cells=True)
    dat = np.asarray(mat["dat"])
    names_raw = np.ravel(mat["ch_names"]).astype(str)
    names = np.array([normalize_channel(x) for x in names_raw])
    fs = float(info["SamplingFreq"])
    if dat.ndim != 2 or dat.shape[1] != len(names):
        raise ValueError(f"Shape/channel mismatch in {member}: {dat.shape} vs {len(names)}")

    dc_matches = np.flatnonzero(names == "DC01")
    if len(dc_matches) != 1:
        raise ValueError(f"Expected one DC01 in {member}, found {len(dc_matches)}")
    ttl = dat[:, dc_matches[0]].astype(np.float64)
    ttl_z = (ttl - ttl.mean()) / ttl.std(ddof=1)
    peaks, _ = find_peaks(ttl_z, height=2.0)
    expected_start = int(info["StimStartI"]) - 1
    expected_end = int(info["StimEndI"]) - 1
    ttl_valid = bool(len(peaks) and peaks[0] == expected_start and peaks[-1] == expected_end)

    intervals = {
        "pre": matlab_inclusive_slice(int(info["BeforeStimStartI"]), int(info["BeforeStimEndI"]), len(dat)),
        "stim": matlab_inclusive_slice(int(info["StimStartI"]), int(info["StimEndI"]), len(dat)),
        "post": matlab_inclusive_slice(int(info["AfterStimStartI"]), int(info["AfterStimEndI"]), len(dat)),
    }
    loc_index = localization.set_index("channel_norm")
    localized_contacts = set(loc_index.index)
    candidate_idx = np.array([i for i, n in enumerate(names) if not is_non_seeg(n)], dtype=int)
    # The observed brain state is restricted to anatomically localized contacts.
    # This excludes Subject 2 scalp/auxiliary channels (F7/F8/T3-T6/O1/O2/RINP*).
    neural_idx = np.array([i for i in candidate_idx if names[i] in localized_contacts], dtype=int)
    stim_left, stim_right = resolve_stim_contacts(str(info["StimLocation"]), set(names))
    feature_frames = []
    for window, slc in intervals.items():
        feats = feature_table(dat[slc, :][:, neural_idx], fs)
        frame = pd.DataFrame(feats)
        frame.insert(0, "channel", names[neural_idx])
        frame.insert(0, "window", window)
        frame.insert(0, "stim_location", info["StimLocation"])
        frame.insert(0, "stim_contact_right", stim_right)
        frame.insert(0, "stim_contact_left", stim_left)
        frame.insert(0, "trial", info["FileName"])
        frame.insert(0, "subject", info["Subject"])
        feature_frames.append(frame)
    features = pd.concat(feature_frames, ignore_index=True)
    features = features.join(loc_index, on="channel", rsuffix="_localization")
    features["localization_matched"] = features["Label"].notna()

    samp_times = np.ravel(mat.get("samp_times", np.array([])))
    summary = {
        "subject": info["Subject"],
        "trial": info["FileName"],
        "stim_location": info["StimLocation"],
        "sampling_hz": fs,
        "n_samples_dat": int(dat.shape[0]),
        "n_samples_samp_times": int(len(samp_times)),
        "samp_times_matches_dat": bool(len(samp_times) == dat.shape[0]),
        "n_channels_total": int(dat.shape[1]),
        "n_channels_candidate_seeg": int(len(candidate_idx)),
        "n_channels_neural": int(len(neural_idx)),
        "n_channels_excluded_unlocalized": int(len(candidate_idx) - len(neural_idx)),
        "n_channels_localization_matched": int(features.loc[features.window == "pre", "localization_matched"].sum()),
        "stim_contact_left_resolved": stim_left,
        "stim_contact_right_resolved": stim_right,
        "stim_start_matlab": int(info["StimStartI"]),
        "stim_end_matlab": int(info["StimEndI"]),
        "stim_span_seconds": (expected_end - expected_start) / fs,
        "ttl_peak_count": int(len(peaks)),
        "ttl_rate_from_span_hz": (len(peaks) - 1) * fs / (peaks[-1] - peaks[0]),
        "ttl_indices_valid": ttl_valid,
        "pre_samples_inclusive": intervals["pre"].stop - intervals["pre"].start,
        "stim_samples_inclusive": intervals["stim"].stop - intervals["stim"].start,
        "post_samples_inclusive": intervals["post"].stop - intervals["post"].start,
    }
    del mat, dat, ttl, ttl_z
    gc.collect()
    return summary, features


def analyze_ccep(path: Path, output_dir: Path) -> dict:
    ccep = pd.read_csv(path, skiprows=1)
    for col in ["stim_chan", "record_chan", "sc1", "sc2", "rc1", "rc2"]:
        ccep[f"{col}_norm"] = ccep[col].map(normalize_channel)
    ccep["subject"] = ccep["subject"].astype(int)
    summary = (
        ccep.groupby(["subject", "stim_chan_norm"], as_index=False)
        .agg(
            n_rows=("record_chan_norm", "size"),
            n_record_pairs=("record_chan_norm", "nunique"),
            n_blocks=("block_name", "nunique"),
            activated_n=("activated", "sum"),
            activation_rate=("activated", "mean"),
            median_cecs=("CECS", "median"),
        )
    )
    summary.to_csv(output_dir / "ccep_stim_pair_summary.csv", index=False)
    edge = (
        ccep.groupby(["subject", "stim_chan_norm", "record_chan_norm"], as_index=False)
        .agg(n_blocks=("block_name", "nunique"), activation_rate=("activated", "mean"), mean_cecs=("CECS", "mean"))
    )
    edge.to_csv(output_dir / "ccep_edge_context.csv", index=False)
    return {
        "rows": int(len(ccep)),
        "subjects": int(ccep.subject.nunique()),
        "stim_pairs_per_subject": ccep.groupby("subject")["stim_chan_norm"].nunique().astype(int).to_dict(),
        "record_pairs_per_subject": ccep.groupby("subject")["record_chan_norm"].nunique().astype(int).to_dict(),
        "full_square_connectome_available": False,
        "reason": "Only two stimulated pairs per subject are present, so the CCEP table is a rectangular thalamic-input context rather than a full state-by-state A matrix.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj1", type=Path, required=True)
    parser.add_argument("--subj2", type=Path, required=True)
    parser.add_argument("--stim-info", type=Path, required=True)
    parser.add_argument("--ccep", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    stim_info = pd.read_excel(args.stim_info)
    if len(stim_info) != 10 or stim_info["FileName"].duplicated().any():
        raise ValueError("The stimulation manifest must contain 10 unique files")

    summaries, feature_sets = [], []
    for zip_path in (args.subj1, args.subj2):
        subject = zip_path.stem
        loc = load_localizations(zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            members = {Path(n).name: n for n in archive.namelist() if n.lower().endswith(".mat")}
            subject_rows = stim_info.loc[stim_info["Subject"] == subject]
            if set(subject_rows["FileName"]) != set(members):
                raise ValueError(f"Manifest/archive mismatch for {subject}")
            for _, row in subject_rows.iterrows():
                summary, features = analyze_trial(archive, members[row["FileName"]], row, loc)
                summaries.append(summary)
                feature_sets.append(features)

    trial_qc = pd.DataFrame(summaries).sort_values(["subject", "stim_location", "trial"])
    channel_features = pd.concat(feature_sets, ignore_index=True)
    trial_qc.to_csv(args.output / "trial_qc.csv", index=False)
    channel_features.to_csv(args.output / "channel_window_features.csv", index=False)

    paired = channel_features.pivot_table(
        index=["subject", "trial", "stim_location", "channel"],
        columns="window",
        values=["power_1_200", "relative_beta", "relative_gamma", "relative_high_gamma", "rms"],
    )
    paired.columns = [f"{metric}_{window}" for metric, window in paired.columns]
    paired = paired.reset_index()
    for metric in ["power_1_200", "relative_beta", "relative_gamma", "relative_high_gamma", "rms"]:
        paired[f"{metric}_stim_over_pre"] = paired[f"{metric}_stim"] / paired[f"{metric}_pre"]
        paired[f"{metric}_post_over_pre"] = paired[f"{metric}_post"] / paired[f"{metric}_pre"]
    paired.to_csv(args.output / "paired_window_changes.csv", index=False)

    ccep_summary = analyze_ccep(args.ccep, args.output)
    run_summary = {
        "trial_count": int(len(trial_qc)),
        "all_ttl_indices_valid": bool(trial_qc["ttl_indices_valid"].all()),
        "samp_times_mismatch_trials": trial_qc.loc[~trial_qc["samp_times_matches_dat"], "trial"].tolist(),
        "median_ttl_rate_from_span_hz": float(trial_qc["ttl_rate_from_span_hz"].median()),
        "feature_rows": int(len(channel_features)),
        "ccep": ccep_summary,
    }
    (args.output / "analysis_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
