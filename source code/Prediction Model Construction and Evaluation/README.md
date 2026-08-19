# DeepSurv Proteomic Risk Prediction for Rheumatoid Arthritis
This repository contains the analysis code for a UK Biobank proteomics study of
incident rheumatoid arthritis (RA) risk prediction. A DeepSurv-style Cox neural
network (scalar log-risk output trained with the Cox partial-likelihood objective)
is fitted with a three-stage, hold-out-based workflow:

1. **Stage 0 — hold-out split**: all samples are split 80/20, stratified by
   event status (`train_test_split`). The 20% test set is never touched
   during selection or tuning.
2. **Stage 1 — feature selection**: a protein-only DeepSurv importance model
   is fitted on an inner split of the 80%
   training data. Proteins are ranked by permutation importance (drop in c-index,
   5 repeats), and the smallest set of proteins holding ≥ 65% of the total
   (clipped) importance is kept.
3. **Stage 2 — 5-fold cross-validation**: stratified 5-fold CV on the 80% training
   split with the fixed selected protein set, comparing three models — `Clinical`
   (covariates only), `Protein` (selected proteins only) and `Integrated`
   (proteins + covariates).
4. **Stage 3 — independent validation**: each model is refit on the full 80%
   training split and evaluated once on the untouched 20% test set.

## Dependencies
- Python 3 (with `pandas`, `numpy`)
- `torch`
- `pycox`, `torchtuples`
- `scikit-survival`
- `lifelines`
- `scikit-learn`

## Usage
```bash
python ra_deepsurv_holdout_sel.py
```
All settings (data paths, random seed, protein-selection share, training epochs,
output directory, GPU usage) are hard-coded in the configuration block at the top
of `ra_deepsurv_holdout_sel.py` — edit them there if needed.

