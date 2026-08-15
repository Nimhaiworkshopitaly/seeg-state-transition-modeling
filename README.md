# Shared-parcel state-transition analysis

This package contains the code used for the two-subject pilot analysis.

## Analysis sequence

1. `run_analysis.py` validates the raw MATLAB recordings, timing manifest,
   localization files, and CCEP table and creates quality-control outputs.
2. `run_parcel_pca_models.py` finds the intersection of Brainnetome
   macro-parcels represented in both subjects, averages contacts within each
   parcel, computes two-second parcel-connectivity windows with one-second
   steps, and subtracts each trial's prestimulation connectivity baseline.
3. `run_architecture_comparison.py` fits PCA inside each training fold and
   compares persistence, a regularized linear state-space model, a compact GRU,
   and a small causal Transformer. It performs complete-trial within-subject
   holdout and bidirectional complete-subject holdout.
4. `make_publication_figures.py` creates the three summary figures from the
   saved architecture-comparison metrics.

## Run in PowerShell

```powershell
python -m pip install -r requirements.txt

python run_analysis.py `
  --subj1 "C:\Users\deys4\Downloads\Subj_1.zip" `
  --subj2 "C:\Users\deys4\Downloads\Subj_2.zip" `
  --stim-info "C:\Users\deys4\Downloads\stimFileInfo_Subham.xlsx" `
  --ccep "C:\Users\deys4\Downloads\CCEP_results.csv" `
  --output "results\qc"

python run_parcel_pca_models.py `
  --subj1 "C:\Users\deys4\Downloads\Subj_1.zip" `
  --subj2 "C:\Users\deys4\Downloads\Subj_2.zip" `
  --stim-info "C:\Users\deys4\Downloads\stimFileInfo_Subham.xlsx" `
  --output "results\parcel_pca_models"

python run_architecture_comparison.py `
  --edges "results\parcel_pca_models\parcel_edge_trajectories.csv" `
  --output "results\architecture_comparison"

python make_publication_figures.py
```

## Interpretation boundary

The cross-subject experiment uses only parcel labels represented in both
subjects; missing parcels are not zero-filled. PCA and linear-model scaling are
estimated from the training fold only. Nevertheless, different subjects may
sample different locations and numbers of contacts within the same macro-parcel.
The transfer results are therefore exploratory and do not establish global
network-state generalizability.

