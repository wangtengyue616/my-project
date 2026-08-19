#!/usr/bin/env python
"""DeepSurv survival modeling for RA risk: 80/20 hold-out, permutation-importance protein
selection, 5-fold CV and independent test evaluation."""

from __future__ import annotations

import copy
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

try:
    from sksurv.metrics import brier_score as sksurv_brier_score
    from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc, integrated_brier_score
    from sksurv.util import Surv
except ImportError as exc:
    raise ImportError("Please install scikit-survival: pip install scikit-survival") from exc

try:
    from pycox.models import CoxPH
    import torchtuples as tt
except ImportError as exc:
    raise ImportError("Please install pycox+torchtuples: pip install pycox torchtuples") from exc

from torch.utils.data import DataLoader, TensorDataset

ID_COL = "eid"
EVENT_COL = "later_illness_label"
TIME_COL = "follow_time"

# --- Data paths ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_SCRIPT_DIR, "data")
LABEL_CSV = "RA_label.csv"
PROTEIN_CSV = "RA_proteins.csv"
COV_CSV = "RA_cov.csv"


RESULT_DIR = "results_outputs"

FEATURE_SELECTION_SUBDIR = "feature_selection"
CROSS_VALIDATION_SUBDIR = "cross_validation"
INDEPENDENT_VALIDATION_SUBDIR = "independent_validation"
OUTPUT_CSV = "survival_deepsurv_permimp_results.csv"
EXPERIMENT_SUMMARY_CSV = "protein_count_experiment_average_metrics.csv"
CINDEX_AUC_SUMMARY_CSV = "cindex_auc_summary.csv"
FEATURE_IMPORTANCE_CSV = "feature_importance.csv"
SELECTED_PROTEINS_TXT = "selected_proteins.txt"
SELECTION_METADATA_CSV = "selection_metadata.csv"
HOLDOUT_SPLIT_CSV = "holdout_split_eids.csv"
N_SPLITS = 5
RANDOM_SEED = 31
USE_GPU = True

# --- Hold-out split (stage 0) ---
HOLDOUT_TEST_FRAC = 0.2
INNER_VAL_FRAC = 0.15

# --- Stage-1 feature selection (permutation importance on the DeepSurv model) ---
PFI_REPEATS = 5
PFI_INNER_VAL_FRAC = 0.2
STAGE1_FULL_BATCH_EPOCHS = 800
STAGE1_FULL_BATCH_EARLY_STOP_PATIENCE = 80
MIN_SELECTED = 1
CUMULATIVE_IMPORTANCE_SHARE = 0.65
TOP_N_IMPORTANCE_REFERENCE = 34
SELECTION_TOP_N = None

STAGE1_INCLUDE_COVARIATES = False

IMPORTANCE_METHOD = "permutation"


BATCH_SIZE_GPU = 1024
BATCH_SIZE_CPU = 512
USE_FULL_BATCH_COX_FOR_V2 = True
FULL_BATCH_MAX_N = 65536
FULL_BATCH_EPOCHS = 1200
FULL_BATCH_EARLY_STOP_PATIENCE = 120
FIXED_TRAIN_EPOCHS = 60
MINIBATCH_EPOCHS = 160
MINIBATCH_EARLY_STOP_PATIENCE = 20

BALANCED_MIN_EVENTS_PER_BATCH = 18


WEIGHT_DECAY = 1e-4
LEARNING_RATE = 5e-4
DROPOUT = 0.3

GAUSSIAN_NOISE_STD = 0.0

FEATURE_DROPOUT_P = 0.05


USE_XTILE_FOR_V2 = True
XTILE_MIN_GROUP_FRAC = 0.2
XTILE_MAX_CANDIDATES = 1024
RISK_GROUP_SPLIT: str = "tertile"


SURV_METRIC_TIME_QUANTILES: Tuple[float, ...] = (0.1, 0.25, 0.4, 0.55, 0.7, 0.85)
FIXED_EVAL_YEARS: Tuple[float, ...] = (3.0, 5.0, 10.0)
ENABLE_PROTEIN_LOG1P = False


def _select_batch_size(device: torch.device) -> int:
    return BATCH_SIZE_GPU if device.type == "cuda" else BATCH_SIZE_CPU


def _select_v2_batching(n_train: int, device: torch.device) -> Tuple[int, bool, str]:
    """Choose v2 batching policy: full-batch exact Cox, else large mini-batch fallback."""
    if USE_FULL_BATCH_COX_FOR_V2 and int(n_train) <= int(FULL_BATCH_MAX_N):
        return max(1, int(n_train)), False, "full_batch_exact_cox"
    return _select_batch_size(device), True, "large_minibatch_balanced_cox_approx"


def _select_v2_training_schedule(batching_mode: str) -> Tuple[int, int]:
    if batching_mode == "full_batch_exact_cox":
        return int(FULL_BATCH_EPOCHS), int(FULL_BATCH_EARLY_STOP_PATIENCE)
    return int(MINIBATCH_EPOCHS), int(MINIBATCH_EARLY_STOP_PATIENCE)


# -------------------------
def _unpack_xy_te(data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unpack tuplefy survival training data into numpy X, time, event."""
    data_np = tt.tuplefy(data).to_numpy()
    left, right = data_np[0], data_np[1]

    if isinstance(left, (tuple, list)) or (
        hasattr(left, "__len__") and not isinstance(left, np.ndarray) and hasattr(left, "__getitem__")
    ):
        try:
            x = left[0] if len(left) == 1 else np.asarray(left)
        except Exception:
            x = np.asarray(left)
    else:
        x = np.asarray(left)
    t = np.asarray(right[0]).reshape(-1)
    e = np.asarray(right[1]).reshape(-1)
    return np.asarray(x, dtype=np.float32), t.astype(np.float32), e.astype(np.float32)


class StratifiedSurvivalBatchSampler:
    """Yields batch index lists with at least ``min_events`` event rows per batch (when available)."""

    def __init__(
        self,
        events: np.ndarray,
        batch_size: int,
        min_events: int,
        base_seed: int = 0,
    ) -> None:
        self.events = events.astype(bool)
        self.event_idx = np.where(self.events)[0]
        self.cens_idx = np.where(~self.events)[0]
        self.batch_size = int(batch_size)
        self.min_events = int(min_events)
        self.base_seed = int(base_seed)
        self._epoch = 0

    def __len__(self) -> int:
        n = len(self.events)
        return max(1, (n + self.batch_size - 1) // self.batch_size)

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.RandomState(self.base_seed + self._epoch)
        self._epoch += 1
        n_ev = len(self.event_idx)
        n_ce = len(self.cens_idx)
        B = self.batch_size

        if n_ev == 0:
            idx = np.arange(len(self.events))
            rng.shuffle(idx)
            for i in range(0, len(idx), B):
                yield idx[i : i + B].tolist()
            return

        k = min(self.min_events, n_ev)
        for _ in range(len(self)):
            ev_pick = rng.choice(self.event_idx, size=k, replace=n_ev < k)
            need_cens = B - k
            if need_cens <= 0:
                batch = ev_pick
            elif n_ce == 0:
                batch = rng.choice(self.event_idx, size=B, replace=n_ev < B)
            else:
                cens_pick = rng.choice(self.cens_idx, size=need_cens, replace=n_ce < need_cens)
                batch = np.concatenate([ev_pick, cens_pick])
            rng.shuffle(batch)
            yield batch.tolist()


def _survival_collate(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Batch as pycox expects: net(*input) with input=(x,), target=(duration, event)."""
    xs, ts, es = zip(*batch)
    xb = torch.stack(xs, 0)
    tb = torch.stack(ts, 0)
    eb = torch.stack(es, 0)
    return (xb,), (tb, eb)


class CoxPHBalanced(CoxPH):
    """CoxPH with optional stratified mini-batches on training (shuffle=True) only."""

    @staticmethod
    def make_dataloader(data, batch_size, shuffle, num_workers=0, **kwargs):
        kwargs = dict(kwargs)
        sampler_seed = int(kwargs.pop("sampler_seed", 0))
        balanced_min_events = int(kwargs.pop("balanced_min_events", BALANCED_MIN_EVENTS_PER_BATCH))
        use_balanced_batches = bool(kwargs.pop("use_balanced_batches", True))


        tt_only = {k: kwargs[k] for k in ("to_tensor", "make_dataset", "torch_ds_dl") if k in kwargs}

        if shuffle and use_balanced_batches:
            x, t, e = _unpack_xy_te(data)
            events_np = e.astype(bool)
            n_ev = int(events_np.sum())
            ds = TensorDataset(
                torch.as_tensor(x, dtype=torch.float32),
                torch.as_tensor(t, dtype=torch.float32),
                torch.as_tensor(e, dtype=torch.float32),
            )
            k_eff = min(balanced_min_events, n_ev) if n_ev else 0
            if n_ev == 0:
                return tt.make_dataloader(data, batch_size, shuffle, num_workers, **tt_only)
            batch_sampler = StratifiedSurvivalBatchSampler(
                events_np,
                batch_size=batch_size,
                min_events=max(1, k_eff),
                base_seed=sampler_seed,
            )
            return DataLoader(
                ds,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                collate_fn=_survival_collate,
            )

        return tt.make_dataloader(data, batch_size, shuffle, num_workers, **tt_only)


def set_seed(seed: int) -> None:
    """Fix NumPy / PyTorch RNGs for reproducible deep-model runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------

# -------------------------
def c_index(event: np.ndarray, time: np.ndarray, risk: np.ndarray) -> float:
    """Harrell concordance index using scikit-survival (higher risk = higher score)."""
    return float(concordance_index_censored(event.astype(bool), time.astype(float), risk.astype(float))[0])


def tertile_cuts_from_train_risk(risk_train: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Return (q1,q2,q3, low_group_upper, high_group_lower) for tertile splits of training risk."""
    q1, q2, q3 = np.quantile(risk_train, [1 / 3, 2 / 3, 1.0]).astype(float)
    return q1, q2, q3, q1, q2


def median_cut_from_train_risk(risk_train: np.ndarray) -> float:
    return float(np.median(risk_train))


def xtile_threshold_from_train(
    t_train: np.ndarray,
    e_train: np.ndarray,
    risk_train: np.ndarray,
    min_group_frac: float = 0.2,
) -> float:
    """Approximate X-tile threshold: maximize train log-rank statistic over valid cutpoints."""
    r = np.asarray(risk_train, dtype=float).reshape(-1)
    t = np.asarray(t_train, dtype=float).reshape(-1)
    e = np.asarray(e_train, dtype=int).reshape(-1)
    n = r.shape[0]
    if n < 20:
        return float(np.median(r))
    order = np.argsort(r)
    r_sorted = r[order]
    unique_vals = np.unique(r_sorted)
    if unique_vals.size < 5:
        return float(np.median(r))
    lo = int(np.ceil(min_group_frac * n))
    hi = int(np.floor((1.0 - min_group_frac) * n))
    best_thr = float(np.median(r))
    best_score = -np.inf

    mids = (unique_vals[:-1] + unique_vals[1:]) / 2.0
    if mids.size > XTILE_MAX_CANDIDATES:
        idx = np.linspace(0, mids.size - 1, XTILE_MAX_CANDIDATES).astype(int)
        mids = mids[idx]
    for thr in mids:
        high = r >= thr
        n_high = int(high.sum())
        if n_high < lo or n_high > hi:
            continue
        low = ~high
        if low.sum() == 0 or high.sum() == 0:
            continue
        lr = logrank_test(t[high], t[low], e[high], e[low])
        score = float(getattr(lr, "test_statistic", np.nan))
        if np.isnan(score):
            continue
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    return float(best_thr)


def _hr_from_binary_group(t: np.ndarray, e: np.ndarray, high_mask: np.ndarray) -> Dict[str, float]:
    """Hazard ratio (High vs Low) with 95% CI from CoxPH on a binary group indicator."""
    df_hr = pd.DataFrame(
        {
            "T": np.asarray(t, dtype=float),
            "E": np.asarray(e, dtype=int),
            "group_high": np.asarray(high_mask, dtype=int),
        }
    )
    out = {
        "hazard_ratio_high_vs_low": float("nan"),
        "hr_ci95_lower": float("nan"),
        "hr_ci95_upper": float("nan"),
        "hr_p_value": float("nan"),
    }
    if df_hr["group_high"].nunique() < 2 or df_hr["E"].sum() < 5:
        return out
    try:
        cph = CoxPHFitter()
        cph.fit(df_hr, duration_col="T", event_col="E", robust=True)
        out["hazard_ratio_high_vs_low"] = float(np.exp(cph.params_["group_high"]))
        ci = cph.confidence_intervals_.loc["group_high"].values.astype(float)
        out["hr_ci95_lower"] = float(np.exp(ci[0]))
        out["hr_ci95_upper"] = float(np.exp(ci[1]))
        out["hr_p_value"] = float(cph.summary.loc["group_high", "p"])
    except Exception:
        pass
    return out


def _hr_per_sd_from_continuous_risk(t: np.ndarray, e: np.ndarray, risk: np.ndarray) -> Dict[str, float]:
    """Cox HR per 1-SD increase in risk score (continuous)."""
    r = np.asarray(risk, dtype=float).reshape(-1)
    sd = float(np.std(r, ddof=0))
    out = {
        "hr_per_sd": float("nan"),
        "hr_per_sd_ci95_lower": float("nan"),
        "hr_per_sd_ci95_upper": float("nan"),
        "hr_per_sd_p_value": float("nan"),
        "risk_sd": sd,
    }
    if sd <= 1e-12:
        return out
    z = (r - float(np.mean(r))) / sd
    df_hr = pd.DataFrame({"T": np.asarray(t, dtype=float), "E": np.asarray(e, dtype=int), "risk_z": z})
    if df_hr["E"].sum() < 5:
        return out
    try:
        cph = CoxPHFitter()
        cph.fit(df_hr, duration_col="T", event_col="E", robust=True)
        out["hr_per_sd"] = float(np.exp(cph.params_["risk_z"]))
        ci = cph.confidence_intervals_.loc["risk_z"].values.astype(float)
        out["hr_per_sd_ci95_lower"] = float(np.exp(ci[0]))
        out["hr_per_sd_ci95_upper"] = float(np.exp(ci[1]))
        out["hr_per_sd_p_value"] = float(cph.summary.loc["risk_z", "p"])
    except Exception:
        pass
    return out


def _append_val_c_index_to_monitor(model: "tt.Model", x_val: np.ndarray, t_val: np.ndarray, e_val: np.ndarray) -> None:
    """Compute val c-index and append to ``model.val_metrics`` for torchtuples EarlyStopping."""
    with torch.no_grad():
        r = model.predict(x_val, batch_size=16384, eval_=True, numpy=True)
    r = np.asarray(r, dtype=np.float32).reshape(-1)
    score = c_index(e_val, t_val, r)
    if np.isnan(score):
        score = 0.5

    model.val_metrics.append_score("c_index", float(score))


class ValConcordanceIndexCallback(tt.callbacks.Callback):
    """Append validation Harrell c-index into ``val_metrics`` as ``c_index`` (for EarlyStopping)."""

    def __init__(self, x_val: np.ndarray, t_val: np.ndarray, e_val: np.ndarray) -> None:
        super().__init__()
        self.x_val = np.asarray(x_val, dtype=np.float32)
        self.t_val = np.asarray(t_val, dtype=np.float32)
        self.e_val = np.asarray(e_val, dtype=np.float32)

    def on_epoch_end(self) -> bool:
        _append_val_c_index_to_monitor(self.model, self.x_val, self.t_val, self.e_val)
        return False


# -------------------------

# -------------------------
def apply_protein_log1p(df: pd.DataFrame, protein_cols: Sequence[str]) -> pd.DataFrame:
    if not ENABLE_PROTEIN_LOG1P:
        return df.copy()
    out = df.copy()
    for c in protein_cols:
        if c in out.columns:
            out[c] = np.log1p(out[c].astype(float).clip(lower=0.0))
    return out


def scale_train_val(
    x_train: np.ndarray,
    x_val: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Any]:
    scaler: Any = RobustScaler()
    xs_tr = scaler.fit_transform(x_train).astype(np.float32)
    xs_va = scaler.transform(x_val).astype(np.float32)
    return xs_tr, xs_va, scaler


# -------------------------

# -------------------------
def _stratified_inner_split(
    train_df: pd.DataFrame, val_frac: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified (by event) inner split of a training fold; first fold is the inner validation set."""
    e = train_df[EVENT_COL].to_numpy(dtype=int)
    n = len(train_df)
    n_splits = max(2, int(round(1.0 / max(1e-3, min(0.5, float(val_frac))))))
    n_splits = min(n_splits, n)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    tr_idx, va_idx = next(skf.split(np.zeros(n), e))
    inner_train = train_df.iloc[tr_idx].reset_index(drop=True)
    pfi_val = train_df.iloc[va_idx].reset_index(drop=True)
    return inner_train, pfi_val


def top_n_importance_share(
    sorted_items: List[Tuple[str, float]], top_n: int
) -> Tuple[float, float, float]:
    """Share of total importance held by the top-``top_n`` features: (share, top_n_sum, total_sum)."""
    vals = np.asarray([float(v) for _, v in sorted_items], dtype=float)
    if vals.size == 0 or int(top_n) <= 0:
        return float("nan"), 0.0, 0.0
    total = float(vals.sum())
    k = min(int(top_n), vals.size)
    top_sum = float(vals[:k].sum())
    share = (top_sum / total) if total > 0 else float("nan")
    return share, top_sum, total


def select_by_cumulative_importance(
    importances: Dict[str, float],
    share_threshold: float = CUMULATIVE_IMPORTANCE_SHARE,
    min_selected: int = MIN_SELECTED,
) -> Tuple[List[str], int, List[Tuple[str, float]], np.ndarray]:
    """Select the smallest top-k whose clipped (>=0) importance sums to >= ``share_threshold`` of the total."""
    items = sorted(
        ((str(k), float(max(0.0, v))) for k, v in importances.items()),
        key=lambda kv: kv[1],
        reverse=True,
    )
    names = [k for k, _ in items]
    vals = np.asarray([v for _, v in items], dtype=float)
    n = len(vals)
    if n == 0:
        return [], 0, items, vals
    total = float(vals.sum())
    if total <= 0:
        k = max(1, min(int(min_selected), n))
        return names[:k], k, items, vals
    cum_share = np.cumsum(vals) / total
    ge = cum_share >= float(share_threshold)
    k = int(np.argmax(ge)) + 1 if bool(ge.any()) else n
    k = max(int(min_selected), min(k, n))
    return names[:k], k, items, vals


def select_top_n(
    importances: Dict[str, float],
    top_n: int,
    min_selected: int = MIN_SELECTED,
) -> Tuple[List[str], int, List[Tuple[str, float]], np.ndarray]:
    """Select the top-``top_n`` proteins by clipped importance (fixed count)."""
    items = sorted(
        ((str(k), float(max(0.0, v))) for k, v in importances.items()),
        key=lambda kv: kv[1],
        reverse=True,
    )
    names = [k for k, _ in items]
    vals = np.asarray([v for _, v in items], dtype=float)
    n = len(vals)
    if n == 0:
        return [], 0, items, vals
    k = max(int(min_selected), min(int(top_n), n))
    return names[:k], k, items, vals


# -------------------------

# -------------------------
def _mlp_res_block(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
        nn.BatchNorm1d(out_dim),
    )


class _ResStack(nn.Module):
    """Two-layer bottleneck residual block: x + W2 act(W1 x)."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.l1 = nn.Linear(dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.l2 = nn.Linear(hidden_dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.l1(x)
        h = self.bn1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.l2(h)
        h = self.bn2(h)
        return self.act(x + h)


class DualBranchResidualDeepSurvNet(nn.Module):
    """Protein and phenotype encoders, fused, then residual MLP with skip connections to scalar log-risk."""

    def __init__(
        self,
        n_prot: int,
        n_phen: int,
        branch_emb: int = 64,
        branch_hidden: int = 256,
        fuse_width: int = 256,
        n_res_blocks: int = 2,
        dropout: float = DROPOUT,
        noise_std: float = GAUSSIAN_NOISE_STD,
        feat_drop_p: float = FEATURE_DROPOUT_P,
    ) -> None:
        super().__init__()
        if n_prot < 0 or n_phen < 0:
            raise ValueError("Invalid branch sizes")
        self.n_prot = int(n_prot)
        self.n_phen = int(n_phen)
        self.noise_std = float(noise_std)
        self.feat_drop_p = float(feat_drop_p)

        in_dim = self.n_prot + self.n_phen
        if in_dim <= 0:
            raise ValueError("Need at least one input feature")

        if self.n_prot == 0:
            self.prot = None
        else:
            self.prot = nn.Sequential(
                _mlp_res_block(self.n_prot, max(branch_hidden, 2 * self.n_prot), branch_emb, dropout),
                nn.ReLU(),
            )
        if self.n_phen == 0:
            self.phen = None
        else:
            self.phen = nn.Sequential(
                _mlp_res_block(self.n_phen, max(branch_hidden, 2 * self.n_phen), branch_emb, dropout),
                nn.ReLU(),
            )

        if self.prot is not None and self.phen is not None:
            fuse_in = 2 * branch_emb
        else:
            fuse_in = branch_emb

        self.fuse_in = _mlp_res_block(fuse_in, max(fuse_width, fuse_in), fuse_width, dropout)
        self.res = nn.ModuleList(
            [_ResStack(fuse_width, max(128, fuse_width // 2), dropout) for _ in range(int(n_res_blocks))]
        )
        self.head = nn.Linear(fuse_width, 1)

    def _feat_aug(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.noise_std > 0.0:
            x = x + self.noise_std * torch.randn_like(x)
        if self.training and self.feat_drop_p > 0.0:
            p = self.feat_drop_p

            m = (torch.rand_like(x) > p).to(dtype=x.dtype) / (1.0 - p)
            x = x * m
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            x = x.view(x.size(0), -1)
        x = self._feat_aug(x)
        p = int(self.n_prot)
        if p == 0:
            z = self.phen(x)
        elif p >= x.size(1):
            z = self.prot(x)
        else:
            xp, xb = x[:, :p], x[:, p:]
            if self.phen is None:
                z = self.prot(xp)
            elif self.prot is None:
                z = self.phen(xb)
            else:
                z = torch.cat([self.prot(xp), self.phen(xb)], dim=1)
        h = self.fuse_in(z)
        h = torch.relu(h)
        for blk in self.res:
            h = blk(h)
        return self.head(h)


# -------------------------

# -------------------------
def _sksurv_y(event: np.ndarray, time: np.ndarray) -> np.ndarray:
    return Surv.from_arrays(event.astype(bool), time.astype(float))


def _extract_time_event_from_y(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    names = list(y.dtype.names or [])
    if "event" in names:
        e = y["event"].astype(bool)
    else:
        e = y[names[0]].astype(bool)
    if "time" in names:
        t = y["time"].astype(float)
    elif "t" in names:
        t = y["t"].astype(float)
    else:
        t = y[names[1]].astype(float)
    return t, e


def eval_times_from_y(y: np.ndarray, quantiles: Sequence[float], eps: float = 1e-6) -> np.ndarray:


    names = list(y.dtype.names or [])
    if "time" in names:
        t = y["time"].astype(float)
    elif "t" in names:
        t = y["t"].astype(float)
    elif len(names) >= 2:
        t = y[names[1]].astype(float)
    else:
        raise ValueError(f"Unexpected survival dtype layout: names={names}")
    tmin, tmax = float(np.min(t)), float(np.max(t))
    qs = [float(q) for q in quantiles]
    ts = np.quantile(t, qs).astype(float)
    ts = np.unique(np.clip(ts, tmin + eps, tmax - eps))
    if ts.size < 3:
        ts = np.linspace(tmin + eps, tmax - eps, num=5)
    return ts


def eval_times_train_test_overlap(
    y_train: np.ndarray,
    y_test: np.ndarray,
    quantiles: Sequence[float],
    eps: float = 1e-6,
) -> np.ndarray:
    train_names = list(y_train.dtype.names or [])
    test_names = list(y_test.dtype.names or [])
    t_train = y_train["time"].astype(float) if "time" in train_names else y_train[train_names[1]].astype(float)
    t_test = y_test["time"].astype(float) if "time" in test_names else y_test[test_names[1]].astype(float)
    lo = max(float(np.min(t_train)), float(np.min(t_test))) + eps
    hi = min(float(np.max(t_train)), float(np.max(t_test))) - eps
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:

        return eval_times_from_y(y_train, quantiles, eps=eps)
    q = np.asarray([float(v) for v in quantiles], dtype=float)
    ts = np.quantile(t_train, q).astype(float)
    ts = np.unique(np.clip(ts, lo, hi))
    if ts.size < 3:
        ts = np.linspace(lo, hi, num=5)
    return ts


def time_dependent_auc_mean(
    y_train: np.ndarray, y_test: np.ndarray, risk_test: np.ndarray, times: np.ndarray
) -> Tuple[float, np.ndarray, float]:
    auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_test, times)
    auc = np.asarray(auc, dtype=np.float32)
    return float(mean_auc), auc, float(np.max(times) - np.min(times))


def time_dependent_auc_at_years(
    y_train: np.ndarray,
    y_test: np.ndarray,
    risk_test: np.ndarray,
    years: Sequence[float] = FIXED_EVAL_YEARS,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Compute IPCW dynamic AUC at fixed follow-up years."""
    train_names = list(y_train.dtype.names or [])
    test_names = list(y_test.dtype.names or [])
    t_train = y_train["time"].astype(float) if "time" in train_names else y_train[train_names[1]].astype(float)
    t_test = y_test["time"].astype(float) if "time" in test_names else y_test[test_names[1]].astype(float)
    lo = max(float(np.min(t_train)), float(np.min(t_test))) + eps
    hi = min(float(np.max(t_train)), float(np.max(t_test))) - eps
    out: Dict[str, float] = {}
    for y in years:
        key = f"td_auc_{int(y)}y"
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            out[key] = float("nan")
            continue
        t_eval = float(np.clip(float(y), lo, hi))
        try:
            auc_arr, _ = cumulative_dynamic_auc(y_train, y_test, risk_test, np.asarray([t_eval], dtype=float))
            out[key] = float(np.asarray(auc_arr, dtype=float).reshape(-1)[0])
        except Exception:
            out[key] = float("nan")
    return out


def _clip_eval_years(y_train: np.ndarray, y_test: np.ndarray, years: Sequence[float], eps: float = 1e-6) -> np.ndarray:
    t_train, _ = _extract_time_event_from_y(y_train)
    t_test, _ = _extract_time_event_from_y(y_test)
    lo = max(float(np.min(t_train)), float(np.min(t_test))) + eps
    hi = min(float(np.max(t_train)), float(np.max(t_test))) - eps
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return np.asarray([], dtype=float)
    return np.asarray([float(np.clip(float(y), lo, hi)) for y in years], dtype=float)


def predict_survival_probs_at_times(cox: CoxPH, x: np.ndarray, eval_times: np.ndarray) -> np.ndarray:
    """Return S(t|x) with shape [n_samples, n_eval_times]."""
    if eval_times.size == 0:
        return np.zeros((x.shape[0], 0), dtype=np.float32)
    surv_df = cox.predict_surv_df(x, max_duration=None, batch_size=16384, eval_=True, num_workers=0)
    grid = surv_df.index.to_numpy(dtype=float)
    vals = surv_df.to_numpy(dtype=np.float32)
    out = np.empty((vals.shape[1], eval_times.size), dtype=np.float32)
    for j in range(vals.shape[1]):
        out[j, :] = np.interp(eval_times, grid, vals[:, j], left=vals[0, j], right=vals[-1, j]).astype(np.float32)
    return out


def time_dependent_auprc_at_years(
    y_train: np.ndarray,
    y_test: np.ndarray,
    risk_test: np.ndarray,
    years: Sequence[float] = FIXED_EVAL_YEARS,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """IPCW-style time-dependent AUPRC at fixed horizons."""
    t_train, e_train = _extract_time_event_from_y(y_train)
    t_test, e_test = _extract_time_event_from_y(y_test)
    km_c = KaplanMeierFitter()

    km_c.fit(t_train, event_observed=(~e_train).astype(int))
    eval_times = _clip_eval_years(y_train, y_test, years, eps=eps)
    out: Dict[str, float] = {f"td_auprc_{int(y)}y": float("nan") for y in years}
    if eval_times.size == 0:
        return out
    for y_raw, t_h in zip(years, eval_times):
        cases = e_test & (t_test <= t_h)
        controls = t_test > t_h
        valid = cases | controls
        y_bin = cases[valid].astype(int)
        if np.unique(y_bin).size < 2:
            out[f"td_auprc_{int(y_raw)}y"] = float("nan")
            continue
        w = np.zeros_like(t_test, dtype=float)
        g_t = float(np.asarray(km_c.survival_function_at_times([float(t_h)])).reshape(-1)[0])
        g_t = max(g_t, eps)
        if np.any(cases):
            g_case = np.asarray(km_c.survival_function_at_times(t_test[cases])).reshape(-1).astype(float)
            g_case = np.maximum(g_case, eps)
            w[cases] = 1.0 / g_case
        w[controls] = 1.0 / g_t
        out[f"td_auprc_{int(y_raw)}y"] = float(
            average_precision_score(y_bin, risk_test[valid].astype(float), sample_weight=w[valid].astype(float))
        )
    return out


def brier_at_years(
    y_train: np.ndarray,
    y_test: np.ndarray,
    surv_probs: np.ndarray,
    eval_times: np.ndarray,
    years: Sequence[float] = FIXED_EVAL_YEARS,
) -> Dict[str, float]:
    """Time-dependent Brier scores at fixed horizons using IPCW."""
    out: Dict[str, float] = {f"brier_{int(y)}y": float("nan") for y in years}
    if eval_times.size == 0 or surv_probs.size == 0:
        return out
    _, bs = sksurv_brier_score(y_train, y_test, surv_probs, eval_times)
    bs = np.asarray(bs, dtype=float).reshape(-1)
    for i, y in enumerate(years):
        if i < bs.size:
            out[f"brier_{int(y)}y"] = float(bs[i])
    return out


def integrated_brier(
    cox: CoxPH,
    y_train: np.ndarray,
    y_test: np.ndarray,
    x_test: np.ndarray,
    times: np.ndarray,
) -> float:
    if not hasattr(cox, "baseline_hazards_"):
        raise RuntimeError("Need model.compute_baseline_hazards() before IBS.")

    surv_df = cox.predict_surv_df(x_test, max_duration=None, batch_size=16384, eval_=True, num_workers=0)
    model_times = surv_df.index.to_numpy(dtype=float)
    if model_times.size < 3:
        raise ValueError("Too few model time grid points for IBS.")
    t_train = np.asarray(y_train[y_train.dtype.names[1]], dtype=float)
    t_test = np.asarray(y_test[y_test.dtype.names[1]], dtype=float)
    lo = max(float(np.min(t_train)), float(np.min(t_test))) + 1e-6
    hi = min(float(np.max(t_train)), float(np.max(t_test))) - 1e-6
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        raise ValueError("No valid overlap in train/test follow-up range for IBS.")
    use_times = model_times[(model_times >= lo) & (model_times <= hi)]
    if use_times.size < 3:

        q = np.quantile(model_times[(model_times >= lo) & (model_times <= hi)], [0.2, 0.4, 0.6, 0.8])
        use_times = np.unique(q.astype(float))
    if use_times.size < 3:
        raise ValueError("Insufficient valid times for IBS after clipping.")

    surv = surv_df.loc[use_times].to_numpy(dtype=np.float32).T
    return float(integrated_brier_score(y_train, y_test, surv, use_times))


# -------------------------

# -------------------------
def compute_km_hr_metrics_v2(
    val_df: pd.DataFrame,
    risk_train: np.ndarray,
    risk_val: np.ndarray,
    split: str,
    threshold: float,
) -> Dict[str, Any]:
    """Risk-group metrics on ``val_df``: HR per SD, binary high/low Cox HR, log-rank p."""
    risk_train = np.asarray(risk_train, dtype=np.float32).reshape(-1)
    risk_val = np.asarray(risk_val, dtype=np.float32).reshape(-1)
    t_va = val_df[TIME_COL].to_numpy(dtype=float)
    e_va = val_df[EVENT_COL].to_numpy(dtype=int)

    km_meta: Dict[str, Any] = {"split": split}
    km_meta.update(_hr_per_sd_from_continuous_risk(t_va, e_va, risk_val))

    if split == "youden":
        cut = float(threshold)
        high_mask = risk_val >= cut
    elif split == "xtile":

        cut = float(threshold)
        high_mask = risk_val >= cut
        km_meta.update({"grouping": "xtile_train_logrank_max", "xtile_threshold": float(cut)})
    elif split == "median":
        cut = median_cut_from_train_risk(risk_train)
        high_mask = risk_val >= cut
        km_meta.update({"cut": float(cut), "grouping": "median_split"})
    else:

        q1, q2, q3, low_cut, high_cut = tertile_cuts_from_train_risk(risk_train)
        km_meta.update(
            {
                "q1": float(q1),
                "q2": float(q2),
                "q3": float(q3),
                "low_risk_le_q1": float(low_cut),
                "high_risk_gt_q2": float(high_cut),
                "tertile_excludes_middle": True,
            }
        )
        low_mask = risk_val <= low_cut
        high_mask = risk_val > high_cut
        km_meta.update(_hr_from_binary_group(t_va, e_va, high_mask))
        p_value = float("nan")
        if int(high_mask.sum()) > 0 and int(low_mask.sum()) > 0:
            lr = logrank_test(t_va[high_mask], t_va[low_mask], e_va[high_mask], e_va[low_mask])
            p_value = float(lr.p_value)
        km_meta.update(
            {
                "threshold": float(high_cut),
                "logrank_p_value": float(p_value),
                "test_high_n": int(high_mask.sum()),
                "test_low_n": int(low_mask.sum()),
                "test_mid_excluded_n": int(np.sum(~(high_mask | low_mask))),
            }
        )
        return km_meta


    high_mask = np.asarray(high_mask, dtype=bool)
    low_mask = ~high_mask
    km_meta.update(_hr_from_binary_group(t_va, e_va, high_mask))
    p_value = float("nan")
    if int(high_mask.sum()) > 0 and int(low_mask.sum()) > 0:
        lr = logrank_test(t_va[high_mask], t_va[low_mask], e_va[high_mask], e_va[low_mask])
        p_value = float(lr.p_value)
    km_meta.update(
        {
            "threshold": float(cut) if split != "youden" else float(threshold),
            "logrank_p_value": float(p_value),
            "train_high_n": int((risk_train >= float(cut if split != "youden" else threshold)).sum()),
            "train_low_n": int((risk_train < float(cut if split != "youden" else threshold)).sum()),
            "test_high_n": int(high_mask.sum()),
            "test_low_n": int(low_mask.sum()),
        }
    )
    return km_meta


# -------------------------

# -------------------------
def read_dataset(
    label_path: str,
    protein_path: str,
    cov_path: str,
    selected_protein_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Load label, protein, and covariate CSVs; merge on ``ID_COL``; median-impute features."""
    label_df = pd.read_csv(label_path)
    protein_df = pd.read_csv(protein_path)
    cov_df = pd.read_csv(cov_path)

    if not {ID_COL, EVENT_COL, TIME_COL}.issubset(label_df.columns):
        missing = {ID_COL, EVENT_COL, TIME_COL} - set(label_df.columns)
        raise ValueError(f"Label file missing required columns: {missing}")
    if ID_COL not in protein_df.columns:
        raise ValueError(f"Protein file must contain '{ID_COL}' column.")
    if ID_COL not in cov_df.columns:
        raise ValueError(f"Covariate file must contain '{ID_COL}' column.")

    all_protein_cols = [c for c in protein_df.columns if c != ID_COL]
    if selected_protein_cols is None:
        protein_cols = all_protein_cols
    else:
        protein_cols = [str(c).strip() for c in selected_protein_cols]
        missing_selected = [c for c in protein_cols if c not in protein_df.columns]
        if missing_selected:
            preview = ", ".join(missing_selected[:10])
            raise ValueError(
                f"{len(missing_selected)} selected proteins are missing from protein file: {preview}"
            )
        protein_df = protein_df[[ID_COL] + protein_cols].copy()
    if not protein_cols:
        raise ValueError("Protein file has no feature columns (only id).")

    phenotype_cols = [c for c in cov_df.columns if c != ID_COL]
    if not phenotype_cols:
        raise ValueError("Covariate file has no feature columns (only id).")

    overlap = set(protein_cols) & set(phenotype_cols)
    if overlap:
        raise ValueError(f"Protein and covariate files have overlapping feature columns: {sorted(overlap)}")

    label_df = label_df.drop_duplicates(subset=[ID_COL], keep="first")
    protein_df = protein_df.drop_duplicates(subset=[ID_COL], keep="first")
    cov_df = cov_df.drop_duplicates(subset=[ID_COL], keep="first")

    label_keep = label_df[[ID_COL, EVENT_COL, TIME_COL]].copy()
    df = label_keep.merge(protein_df, on=ID_COL, how="inner").merge(cov_df, on=ID_COL, how="inner")
    if df.empty:
        raise ValueError(
            "Inner merge produced no rows; check ``eid`` overlap between label, protein, and covariate files."
        )

    feature_cols = protein_cols + phenotype_cols

    work_df = df[[ID_COL, EVENT_COL, TIME_COL] + feature_cols].copy()
    work_df = work_df.dropna(subset=[EVENT_COL, TIME_COL])
    work_df[EVENT_COL] = work_df[EVENT_COL].astype(int)
    work_df[TIME_COL] = pd.to_numeric(work_df[TIME_COL], errors="coerce")
    work_df = work_df.dropna(subset=[TIME_COL])

    x_df = work_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy="median")
    x_imp = imputer.fit_transform(x_df)
    x_imp_df = pd.DataFrame(x_imp, columns=feature_cols, index=work_df.index)
    clean_df = pd.concat([work_df[[ID_COL, EVENT_COL, TIME_COL]], x_imp_df], axis=1)
    return clean_df, protein_cols, phenotype_cols


def make_cv_splits(y_event: np.ndarray, n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros_like(y_event), y_event))


def make_holdout_split(df: pd.DataFrame, test_frac: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified (by ``EVENT_COL``) train/test split of the full cohort."""
    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(
        idx,
        test_size=float(test_frac),
        random_state=int(seed),
        shuffle=True,
        stratify=df[EVENT_COL].to_numpy(dtype=int),
    )
    train_df = df.iloc[np.sort(tr_idx)].reset_index(drop=True)
    test_df = df.iloc[np.sort(te_idx)].reset_index(drop=True)
    return train_df, test_df


def _torchtuples_adam_weight_decay(params=None) -> Any:
    """Factory for torchtuples ``Adam`` with fixed L2 penalty (``weight_decay``)."""
    return tt.optim.Adam(lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY, amsgrad=False, params=params)


# -------------------------

# -------------------------
def run_deepsurv_v2_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    protein_cols: List[str],
    phenotype_cols: List[str],
    all_feature_cols: List[str],
    device: torch.device,
    seed: int,
    return_model: bool = False,
    stage1: bool = False,
    eval_df: Optional[pd.DataFrame] = None,
    disable_early_stop: bool = False,
) -> Dict[str, Any]:
    """DeepSurv_v2 Cox neural net for one fold (training + metric evaluation)."""
    set_seed(seed)

    tr = apply_protein_log1p(train_df, protein_cols)
    va = apply_protein_log1p(val_df, protein_cols)

    selected: List[str] = list(all_feature_cols)

    n_prot = len([c for c in selected if c in set(protein_cols)])
    n_phen = len(selected) - n_prot
    if n_phen < 0:
        raise RuntimeError("Internal: invalid protein/phenotype selection")

    x_train, x_val, _scaler = scale_train_val(
        tr[selected].to_numpy(dtype=np.float32), va[selected].to_numpy(dtype=np.float32)
    )
    t_train = tr[TIME_COL].to_numpy(dtype=np.float32)
    e_train = tr[EVENT_COL].to_numpy(dtype=np.float32)
    t_val = va[TIME_COL].to_numpy(dtype=np.float32)
    e_val = va[EVENT_COL].to_numpy(dtype=np.float32)


    use_eval = eval_df is not None and not stage1
    x_eval: Optional[np.ndarray] = None
    t_eval: Optional[np.ndarray] = None
    e_eval: Optional[np.ndarray] = None
    if use_eval:
        ev = apply_protein_log1p(eval_df, protein_cols)
        x_eval = _scaler.transform(ev[selected].to_numpy(dtype=np.float32)).astype(np.float32)
        t_eval = ev[TIME_COL].to_numpy(dtype=np.float32)
        e_eval = ev[EVENT_COL].to_numpy(dtype=np.float32)

    batch_size, use_balanced_batches, batching_mode = _select_v2_batching(len(t_train), device)
    no_es = bool(disable_early_stop) and not stage1
    if stage1:
        max_epochs = int(STAGE1_FULL_BATCH_EPOCHS)
        early_stop_patience = int(STAGE1_FULL_BATCH_EARLY_STOP_PATIENCE)
    elif no_es:
        max_epochs = int(FIXED_TRAIN_EPOCHS)
        early_stop_patience = 0
    else:
        max_epochs, early_stop_patience = _select_v2_training_schedule(batching_mode)

    y_tr = _sksurv_y(e_train, t_train)
    y_va = _sksurv_y(e_val, t_val)

    y_met = _sksurv_y(e_eval, t_eval) if use_eval else y_va
    times = eval_times_train_test_overlap(y_tr, y_met, SURV_METRIC_TIME_QUANTILES)
    eval_years = _clip_eval_years(y_tr, y_met, years=FIXED_EVAL_YEARS)

    set_seed(seed)
    net = DualBranchResidualDeepSurvNet(
        n_prot=n_prot,
        n_phen=n_phen,
        branch_emb=64,
        branch_hidden=256,
        fuse_width=256,
        n_res_blocks=2,
        dropout=DROPOUT,
        noise_std=GAUSSIAN_NOISE_STD,
        feat_drop_p=FEATURE_DROPOUT_P,
    )
    model = CoxPHBalanced(net, _torchtuples_adam_weight_decay)
    model.optimizer.set_lr(LEARNING_RATE)
    model.net.to(device)

    cidx_cb = ValConcordanceIndexCallback(x_val, t_val, e_val)
    if no_es:
        callbacks: List[Any] = [cidx_cb]
    else:
        es = tt.callbacks.EarlyStopping(
            metric="c_index",
            dataset="val",
            get_score=None,
            minimize=False,
            patience=early_stop_patience,
            file_path=None,
            load_best=True,
            rm_file=True,
        )
        callbacks = [cidx_cb, es]
    model.fit(
        x_train,
        (t_train, e_train),
        batch_size=batch_size,
        epochs=max_epochs,
        callbacks=callbacks,
        verbose=False,
        val_data=(x_val, (t_val, e_val)),
        balanced_min_events=BALANCED_MIN_EVENTS_PER_BATCH,
        use_balanced_batches=use_balanced_batches,
        sampler_seed=seed,
    )

    with torch.no_grad():
        _ = model.compute_baseline_hazards()

    risk_train = model.predict(x_train).reshape(-1).astype(np.float32)
    risk_inner_val = model.predict(x_val).reshape(-1).astype(np.float32)
    if use_eval:
        risk_val = model.predict(x_eval).reshape(-1).astype(np.float32)
        metrics_df = eval_df
    else:
        risk_val = risk_inner_val
        metrics_df = val_df
    c_val = c_index(metrics_df[EVENT_COL].values, metrics_df[TIME_COL].values, risk_val)

    model_state: Optional[Dict[str, Any]] = None
    if stage1 or return_model:
        model_state = {
            "state_dict": copy.deepcopy(model.net.state_dict()),
            "batch_size": int(batch_size),
            "n_prot": int(n_prot),
            "n_phen": int(n_phen),
            "selected_features": list(selected),
            "x_train": x_train.copy(),
            "x_val": x_val.copy(),
            "t_train": t_train.copy(),
            "e_train": e_train.copy(),
            "t_val": t_val.copy(),
            "e_val": e_val.copy(),
            "train_df": train_df.copy(),
            "val_df": val_df.copy(),
            "method_selected": list(selected),
        }

    if stage1:
        return {
            "c_index": float(c_val),
            "n_features": int(len(selected)),
            "n_prot_features": int(n_prot),
            "n_phen_features": int(n_phen),
            "selected_features": list(selected),
            "risk_train": risk_train,
            "risk_val": risk_val,
            "model_state": model_state,
        }

    x_met = x_eval if use_eval else x_val
    td_curve = np.asarray([], dtype=float)
    try:
        td_mean, td_curve, _ = time_dependent_auc_mean(y_tr, y_met, risk_val, times)
    except Exception:
        td_mean = float("nan")
    td_years = time_dependent_auc_at_years(y_tr, y_met, risk_val, years=FIXED_EVAL_YEARS)
    td_auprc_years = time_dependent_auprc_at_years(y_tr, y_met, risk_val, years=FIXED_EVAL_YEARS)
    surv_val_years = predict_survival_probs_at_times(model, x_met, eval_years)
    brier_years = brier_at_years(y_tr, y_met, surv_val_years, eval_years, years=FIXED_EVAL_YEARS)
    ibs = float("nan")
    try:
        ibs = float(integrated_brier(model, y_tr, y_met, x_met, times))
    except Exception:
        pass

    return {
        "c_index": float(c_val),
        "ibs": float(ibs),
        "td_auc_mean": float(td_mean),
        "td_auc_times": [float(t) for t in np.asarray(times).reshape(-1).tolist()],
        "td_auc_curve": [float(v) for v in np.asarray(td_curve).reshape(-1).tolist()],
        **td_years,
        **td_auprc_years,
        **brier_years,
        "n_features": int(len(selected)),
        "n_prot_features": int(n_prot),
        "n_phen_features": int(n_phen),
        "selected_features": list(selected),
        "risk_train": risk_train,
        "risk_val": risk_val,
        "eval_df_used": bool(use_eval),
        "model_state": model_state,
        "train_config": {
            "model": "deepsurv_v2",
            "loss_name": "pycox_default_cox_partial_likelihood",
            "is_standard_deepsurv_loss": True,
            "batch_size": int(batch_size),
            "batching_mode": batching_mode,
            "use_full_batch_cox": bool(batching_mode == "full_batch_exact_cox"),
            "full_batch_max_n": int(FULL_BATCH_MAX_N),
            "max_epochs": int(max_epochs),
            "early_stop_patience": int(early_stop_patience),
            "disable_early_stop": bool(no_es),
            "balanced_min_events_per_batch": BALANCED_MIN_EVENTS_PER_BATCH,
            "use_balanced_batches": bool(use_balanced_batches),
            "learning_rate": float(LEARNING_RATE),
            "weight_decay": float(WEIGHT_DECAY),
            "scaler": "RobustScaler",
            "protein_transform": "log1p" if ENABLE_PROTEIN_LOG1P else "none",
            "optimizer": f"Adam(weight_decay={float(WEIGHT_DECAY):g})",
            "dropout": float(DROPOUT),
            "early_stop_on": (
                "none_fixed_epochs"
                if no_es
                else ("inner_val_c_index" if use_eval else "val_c_index")
            ),
            "metrics_on": "eval_df" if use_eval else "val_df",
            "feature_selection": "fixed_stage1_selected_proteins_plus_covariates",
            "data_aug": {"gaussian_noise_std": float(GAUSSIAN_NOISE_STD), "feature_dropout_p": float(FEATURE_DROPOUT_P)},
        },
    }


@dataclass
class Config:
    label_csv_path: str
    protein_csv_path: str
    cov_csv_path: str
    n_splits: int
    seed: int
    use_gpu: bool
    holdout_test_frac: float = HOLDOUT_TEST_FRAC
    inner_val_frac: float = INNER_VAL_FRAC
    pfi_repeats: int = PFI_REPEATS
    pfi_inner_val_frac: float = PFI_INNER_VAL_FRAC
    cumulative_importance_share: float = CUMULATIVE_IMPORTANCE_SHARE
    importance_method: str = IMPORTANCE_METHOD
    selection_top_n: Optional[int] = SELECTION_TOP_N
    stage1_include_covariates: bool = STAGE1_INCLUDE_COVARIATES


def _summarize_metric(fold_list: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    vals = [x.get(key) for x in fold_list]
    vals_f = [float(v) for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if len(vals_f) == 0:
        return {
            f"mean_{key}": float("nan"),
            f"std_{key}": float("nan"),
            f"fold_{key}": vals,
        }
    return {f"mean_{key}": float(np.mean(vals_f)), f"std_{key}": float(np.std(vals_f)), f"fold_{key}": vals}


# -------------------------

# -------------------------
METHOD_TEMPLATE: List[Dict[str, Any]] = [
    {"key": "covariates_only", "display_name": "Clinical"},
    {"key": "protein_only", "display_name": "Protein"},
    {"key": "integrated_cov_plus_protein", "display_name": "Integrated"},
]


def _method_feature_specs(
    selected_proteins: Sequence[str], phenotype_cols: Sequence[str]
) -> Dict[str, Tuple[List[str], List[str]]]:
    """Feature lists per method with the FIXED stage-1 selected protein set."""
    return {
        "covariates_only": ([], list(phenotype_cols)),
        "protein_only": (list(selected_proteins), []),
        "integrated_cov_plus_protein": (list(selected_proteins), list(phenotype_cols)),
    }


def _train_method_fold(
    fit_df: pd.DataFrame,
    stop_df: pd.DataFrame,
    eval_df: Optional[pd.DataFrame],
    method_key: str,
    method_display_name: str,
    method_protein_cols: List[str],
    method_phenotype_cols: List[str],
    device: torch.device,
    seed: int,
    fold: int,
    return_model: bool = False,
    disable_early_stop: bool = False,
) -> Dict[str, Any]:
    """Fit one DeepSurv method and evaluate it (metrics + KM/HR) for a single fold."""
    method_feature_cols = list(method_protein_cols) + list(method_phenotype_cols)
    v2 = run_deepsurv_v2_fold(
        fit_df,
        stop_df,
        method_protein_cols,
        method_phenotype_cols,
        method_feature_cols,
        device=device,
        seed=seed,
        return_model=return_model,
        eval_df=eval_df,
        disable_early_stop=disable_early_stop,
    )
    metric_df = eval_df if eval_df is not None else stop_df
    if USE_XTILE_FOR_V2:

        rtr = np.asarray(v2["risk_train"], dtype=float)
        rva = np.asarray(v2["risk_val"], dtype=float)
        xt_thr = xtile_threshold_from_train(
            fit_df[TIME_COL].values,
            fit_df[EVENT_COL].values,
            rtr,
            min_group_frac=XTILE_MIN_GROUP_FRAC,
        )
        km2 = compute_km_hr_metrics_v2(
            metric_df,
            rtr,
            rva,
            "xtile",
            xt_thr,
        )
        high_mask_val = rva >= xt_thr
        hr_info = _hr_from_binary_group(
            metric_df[TIME_COL].values,
            metric_df[EVENT_COL].values,
            high_mask_val,
        )
        km2.update(
            {
                "risk_split_method": "xtile_train_logrank_max",
                "xtile_threshold": float(xt_thr),
                **hr_info,
            }
        )
    else:
        km2 = compute_km_hr_metrics_v2(
            metric_df,
            v2["risk_train"],
            v2["risk_val"],
            RISK_GROUP_SPLIT,
            0.0,
        )
    v2.update(km2)
    print(
        f"{method_display_name} c-index={v2['c_index']:.4f}, IBS={v2.get('ibs', float('nan')):.4f}, "
        f"td-AUC mean={v2.get('td_auc_mean', float('nan')):.4f}, "
        f"3y={v2.get('td_auc_3y', float('nan')):.4f}, "
        f"5y={v2.get('td_auc_5y', float('nan')):.4f}, "
        f"10y={v2.get('td_auc_10y', float('nan')):.4f}, "
        f"AUPRC(3/5/10y)=({v2.get('td_auprc_3y', float('nan')):.4f}/"
        f"{v2.get('td_auprc_5y', float('nan')):.4f}/{v2.get('td_auprc_10y', float('nan')):.4f}), "
        f"Brier(3/5/10y)=({v2.get('brier_3y', float('nan')):.4f}/"
        f"{v2.get('brier_5y', float('nan')):.4f}/{v2.get('brier_10y', float('nan')):.4f})",
        flush=True,
    )
    return v2


def _incremental_gain_rows_for_fold(
    t_val: np.ndarray,
    e_val: np.ndarray,
    fold_method_outputs: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Delta-c-index and category-NRI rows comparing the three methods on one eval set."""
    rows: List[Dict[str, Any]] = []
    if not all(k in fold_method_outputs for k in ["covariates_only", "protein_only", "integrated_cov_plus_protein"]):
        return rows
    integrated_risk_val = np.asarray(fold_method_outputs["integrated_cov_plus_protein"]["risk_val"], dtype=float)
    protein_risk_val = np.asarray(fold_method_outputs["protein_only"]["risk_val"], dtype=float)
    clinin_risk_val = np.asarray(fold_method_outputs["covariates_only"]["risk_val"], dtype=float)
    if integrated_risk_val.size == 0 or protein_risk_val.size == 0 or clinin_risk_val.size == 0:
        return rows
    rows.extend(
        [
            {
                "metric": "delta_c_index",
                "comparison": "integrated_vs_protein",
                "value": float(c_index(e_val, t_val, integrated_risk_val) - c_index(e_val, t_val, protein_risk_val)),
            },
            {
                "metric": "delta_c_index",
                "comparison": "integrated_vs_clininflam",
                "value": float(c_index(e_val, t_val, integrated_risk_val) - c_index(e_val, t_val, clinin_risk_val)),
            },
            {
                "metric": "delta_c_index",
                "comparison": "protein_vs_clininflam",
                "value": float(c_index(e_val, t_val, protein_risk_val) - c_index(e_val, t_val, clinin_risk_val)),
            },
        ]
    )
    for horizon, metric_key in [(3.0, "nri_3y"), (5.0, "nri_5y"), (10.0, "nri_10y")]:
        rows.extend(
            [
                {
                    "metric": metric_key,
                    "comparison": "integrated_vs_protein",
                    "value": _category_nri(t_val, e_val, integrated_risk_val, protein_risk_val, horizon),
                },
                {
                    "metric": metric_key,
                    "comparison": "integrated_vs_clininflam",
                    "value": _category_nri(t_val, e_val, integrated_risk_val, clinin_risk_val, horizon),
                },
                {
                    "metric": metric_key,
                    "comparison": "protein_vs_clininflam",
                    "value": _category_nri(t_val, e_val, protein_risk_val, clinin_risk_val, horizon),
                },
            ]
        )
    return rows


def _summarize_results(results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Mean/std metric summary per method (JSON-compatible); also valid for a single 'fold'."""
    summary: Dict[str, Any] = {}
    for method, fold_list in results.items():
        c_block = {
            "mean_c_index": float(np.mean([x["c_index"] for x in fold_list])),
            "std_c_index": float(np.std([x["c_index"] for x in fold_list])),
            "fold_c_index": [float(x["c_index"]) for x in fold_list],
        }
        if "ibs" in fold_list[0]:
            c_block.update(
                {
                    **_summarize_metric(fold_list, "ibs"),
                    **_summarize_metric(fold_list, "td_auc_mean"),
                    **_summarize_metric(fold_list, "td_auc_3y"),
                    **_summarize_metric(fold_list, "td_auc_5y"),
                    **_summarize_metric(fold_list, "td_auc_10y"),
                    **_summarize_metric(fold_list, "td_auprc_3y"),
                    **_summarize_metric(fold_list, "td_auprc_5y"),
                    **_summarize_metric(fold_list, "td_auprc_10y"),
                    **_summarize_metric(fold_list, "brier_3y"),
                    **_summarize_metric(fold_list, "brier_5y"),
                    **_summarize_metric(fold_list, "brier_10y"),
                }
            )
        if "td_auc_mean" in fold_list[0]:
            c_block.update(
                {
                    **_summarize_metric(fold_list, "td_auc_mean"),
                    **_summarize_metric(fold_list, "td_auc_3y"),
                    **_summarize_metric(fold_list, "td_auc_5y"),
                    **_summarize_metric(fold_list, "td_auc_10y"),
                }
            )
        c_block["metric_notes"] = {
            "c_index": "primary (discrimination, IPCW-robust in td-AUC uses train censoring dist.)",
            "td_auc_mean": "primary (IPCW; mean over time grid)",
            "td_auc_3y/5y/10y": "primary (IPCW dynamic AUC at fixed follow-up years)",
            "td_auprc_3y/5y/10y": "IPCW time-dependent precision-recall AUC at fixed horizons",
            "ibs": "primary (calibration+discrimination via survival probs; lower is better)",
            "brier_3y/5y/10y": "time-dependent Brier score at fixed horizons; lower is better",
        }
        summary[method] = c_block
    return summary


def _config_output_block(config: Config, device: torch.device, stage_note: str) -> Dict[str, Any]:
    """JSON-compatible config block written into each stage's result dict."""
    selection_desc = (
        f"holdout_train_{config.importance_method}_importance + top_{int(config.selection_top_n)}"
        if config.selection_top_n is not None
        else f"holdout_train_{config.importance_method}_importance + cumulative_{int(round(config.cumulative_importance_share * 100))}%_share"
    )
    return {
        "stage": stage_note,
        "label_csv_path": config.label_csv_path,
        "protein_csv_path": config.protein_csv_path,
        "cov_csv_path": config.cov_csv_path,
        "holdout_test_frac": float(config.holdout_test_frac),
        "inner_val_frac": float(config.inner_val_frac),
        "pfi_repeats": int(config.pfi_repeats),
        "pfi_inner_val_frac": float(config.pfi_inner_val_frac),
        "n_splits": config.n_splits,
        "seed": config.seed,
        "device": str(device),
        "use_xtile_for_v2": bool(USE_XTILE_FOR_V2),
        "xtile_min_group_frac": float(XTILE_MIN_GROUP_FRAC),
        "xtile_max_candidates": int(XTILE_MAX_CANDIDATES),
        "risk_group_split": RISK_GROUP_SPLIT,
        "deepsurv_v2": {
            "model_family": "DeepSurv / Cox neural network",
            "batch_size_policy": f"{BATCH_SIZE_GPU} (GPU) / {BATCH_SIZE_CPU} (CPU)",
            "use_full_batch_cox": bool(USE_FULL_BATCH_COX_FOR_V2),
            "full_batch_max_n": int(FULL_BATCH_MAX_N),
            "full_batch_note": "v2 uses the full training fold as one Cox risk set when train n is below the cap; mini-batch balanced sampler is only a fallback.",
            "full_batch_epochs": int(FULL_BATCH_EPOCHS),
            "full_batch_early_stop_patience": int(FULL_BATCH_EARLY_STOP_PATIENCE),
            "minibatch_epochs": int(MINIBATCH_EPOCHS),
            "minibatch_early_stop_patience": int(MINIBATCH_EARLY_STOP_PATIENCE),
            "balanced_min_events_per_batch": BALANCED_MIN_EVENTS_PER_BATCH,
            "loss_name": "pycox_default_cox_partial_likelihood",
            "is_standard_deepsurv_loss": True,
            "learning_rate": float(LEARNING_RATE),
            "weight_decay": float(WEIGHT_DECAY),
            "dropout": float(DROPOUT),
            "feature_selection": selection_desc,
            "covariates": "RA_cox_cov1222.csv columns are used as phenotype/covariate branch inputs",
            "enable_protein_log1p": bool(ENABLE_PROTEIN_LOG1P),
            "augmentation": {
                "gaussian_noise_std": float(GAUSSIAN_NOISE_STD),
                "feature_dropout_p": float(FEATURE_DROPOUT_P),
            },
            "td_auc_time_quantiles": [float(x) for x in SURV_METRIC_TIME_QUANTILES],
            "fixed_eval_years": [float(x) for x in FIXED_EVAL_YEARS],
        },
        "method_definitions": [
            {
                "method_key": str(m["key"]),
                "method_name": str(m["display_name"]),
                "note": "Protein set is fixed from the stage-1 selection on the holdout-train split; covariates are always all included.",
            }
            for m in METHOD_TEMPLATE
        ],
    }


def run_feature_selection(
    config: Config,
    train_full_df: pd.DataFrame,
    protein_cols: List[str],
    phenotype_cols: List[str],
    device: torch.device,
    out_dir: str,
) -> Dict[str, Any]:
    """Stage 1: importance + fixed protein selection on the 80% holdout-train split only."""
    os.makedirs(out_dir, exist_ok=True)
    imp_method = str(config.importance_method).lower()
    print(
        f"\n[Stage 1] Computing {imp_method} importance on a stage-1 DeepSurv "
        f"(inner split of the {len(train_full_df)} holdout-train samples)...",
        flush=True,
    )
    importances, pfi_meta = compute_deepsurv_permutation_importance(
        train_full_df,
        protein_cols,
        phenotype_cols,
        device=device,
        seed=int(config.seed),
        val_frac=float(config.pfi_inner_val_frac),
        n_repeats=int(config.pfi_repeats),
        include_covariates=bool(config.stage1_include_covariates),
    )

    if config.selection_top_n is not None:
        selected_proteins, n_sel, sorted_items, sorted_vals = select_top_n(
            importances, int(config.selection_top_n), min_selected=MIN_SELECTED
        )
        sel_share, _, _ = top_n_importance_share(sorted_items, n_sel)
        selection_label = f"top-{int(config.selection_top_n)}"
    else:
        selected_proteins, n_sel, sorted_items, sorted_vals = select_by_cumulative_importance(
            importances,
            share_threshold=float(config.cumulative_importance_share),
            min_selected=MIN_SELECTED,
        )
        sel_share, _, _ = top_n_importance_share(sorted_items, n_sel)
        selection_label = f"cumul. >= {int(config.cumulative_importance_share * 100)}% = {sel_share * 100:.1f}%"
    print(
        f"[Stage 1] Selected {n_sel}/{len(protein_cols)} proteins ({selection_label}; "
        f"stage-1 inner-val c-index={pfi_meta['stage1_cidx']:.4f}, "
        f"importance={pfi_meta.get('importance_method', '?')} on {pfi_meta.get('stage1_features', '?')} model)",
        flush=True,
    )
    if selected_proteins:
        print(f"[Stage 1] Selected proteins: {', '.join(selected_proteins)}", flush=True)


    top_share, top_sum, total_imp = top_n_importance_share(sorted_items, TOP_N_IMPORTANCE_REFERENCE)
    print(
        f"[Stage 1] Top-{TOP_N_IMPORTANCE_REFERENCE} importance share = "
        f"{top_share * 100:.1f}% (sum={top_sum:.4f} / total={total_imp:.4f})",
        flush=True,
    )


    imp_rows: List[Dict[str, Any]] = []
    for rank, (pname, pval) in enumerate(sorted_items, start=1):
        imp_rows.append(
            {
                "protein": pname,
                "importance": float(pval),
                "rank": int(rank),
                "selected": bool(rank <= n_sel),
                "stage1_inner_val_c_index": float(pfi_meta["stage1_cidx"]),
                "pfi_baseline_c_index": float(pfi_meta.get("baseline_cidx", float("nan"))),
                "importance_method": str(pfi_meta.get("importance_method", config.importance_method)),
                "stage1_features": str(pfi_meta.get("stage1_features", "")),
            }
        )
    imp_csv = os.path.join(out_dir, FEATURE_IMPORTANCE_CSV)
    pd.DataFrame(imp_rows).to_csv(imp_csv, index=False, encoding="utf-8-sig")
    print(f"[Stage 1] Saved feature-importance table CSV to: {imp_csv}")


    sel_txt = os.path.join(out_dir, SELECTED_PROTEINS_TXT)
    with open(sel_txt, "w", encoding="utf-8") as fh:
        fh.write("\n".join(selected_proteins) + "\n")
    print(f"[Stage 1] Saved selected protein list to: {sel_txt}")


    meta_csv = os.path.join(out_dir, SELECTION_METADATA_CSV)
    pd.DataFrame(
        [
            {
                "importance_method": str(config.importance_method),
                "selection_rule": selection_label,
                "selection_top_n": int(config.selection_top_n) if config.selection_top_n is not None else "",
                "cumulative_importance_share": float(config.cumulative_importance_share),
                "n_selected": int(n_sel),
                "n_protein_pool": int(len(protein_cols)),
                "n_train_full": int(len(train_full_df)),
                f"top_{TOP_N_IMPORTANCE_REFERENCE}_importance_share": float(top_share),
                "stage1_inner_val_c_index": float(pfi_meta["stage1_cidx"]),
                "pfi_baseline_c_index": float(pfi_meta.get("baseline_cidx", float("nan"))),
                "pfi_repeats": int(config.pfi_repeats),
                "pfi_inner_val_frac": float(config.pfi_inner_val_frac),
                "stage1_include_covariates": bool(config.stage1_include_covariates),
                "seed": int(config.seed),
            }
        ]
    ).to_csv(meta_csv, index=False, encoding="utf-8-sig")
    print(f"[Stage 1] Saved selection metadata CSV to: {meta_csv}")

    return {
        "selected_proteins": list(selected_proteins),
        "n_selected": int(n_sel),
        "selection_label": selection_label,
        "sorted_items": sorted_items,
        "sorted_vals": sorted_vals,
        "top_n_share": float(top_share),
        "pfi_meta": pfi_meta,
    }


def run_cross_validation(
    config: Config,
    train_full_df: pd.DataFrame,
    selected_proteins: List[str],
    phenotype_cols: List[str],
    device: torch.device,
    out_dir: str,
) -> Dict[str, Any]:
    """Stage 2: stratified K-fold CV on the 80% train split with the fixed selected proteins."""
    os.makedirs(out_dir, exist_ok=True)
    splits = make_cv_splits(train_full_df[EVENT_COL].values, config.n_splits, config.seed)
    method_specs = _method_feature_specs(selected_proteins, phenotype_cols)
    results: Dict[str, List[Dict[str, Any]]] = {str(m["key"]): [] for m in METHOD_TEMPLATE}
    incremental_gain_rows: List[Dict[str, Any]] = []

    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        print(f"\n===== Fold {fold}/{config.n_splits} (holdout-train split only) =====", flush=True)
        train_df = train_full_df.iloc[tr_idx].reset_index(drop=True)
        val_df = train_full_df.iloc[va_idx].reset_index(drop=True)

        fold_method_outputs: Dict[str, Dict[str, Any]] = {}
        for method_idx, method in enumerate(METHOD_TEMPLATE, start=1):
            method_key = str(method["key"])
            method_display_name = str(method["display_name"])
            method_protein_cols, method_phenotype_cols = method_specs[method_key]
            method_feature_cols = list(method_protein_cols) + list(method_phenotype_cols)
            method_seed = int(config.seed + fold + 1000 * method_idx)
            if method_key == "integrated_cov_plus_protein":
                method_seed = int(config.seed + fold)

            print(
                f"[Fold {fold}] Running method {method_idx}/{len(METHOD_TEMPLATE)}: {method_display_name} "
                f"(n_features={len(method_feature_cols)})...",
                flush=True,
            )
            v2 = _train_method_fold(
                train_df,
                val_df,
                None,
                method_key,
                method_display_name,
                list(method_protein_cols),
                list(method_phenotype_cols),
                device=device,
                seed=method_seed,
                fold=fold,
                return_model=False,
                disable_early_stop=True,
            )
            v2["method_name"] = method_display_name
            v2["method_key"] = method_key
            fold_method_outputs[method_key] = {
                "c_index": float(v2["c_index"]),
                "risk_val": np.asarray(v2.get("risk_val", np.asarray([])), dtype=float),
            }
            for risk_key in ("risk_train", "risk_val"):
                v2.pop(risk_key, None)
            results[method_key].append(v2)
        incremental_gain_rows.extend(
            _incremental_gain_rows_for_fold(val_df[TIME_COL].values, val_df[EVENT_COL].values, fold_method_outputs)
        )

    summary = _summarize_results(results)
    output: Dict[str, Any] = {
        "config": _config_output_block(config, device, stage_note="cross_validation_on_holdout_train"),
        "fold_results": results,
        "incremental_gain_rows": incremental_gain_rows,
        "summary": summary,
    }
    return output


def run_independent_validation(
    config: Config,
    train_full_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected_proteins: List[str],
    phenotype_cols: List[str],
    device: torch.device,
    out_dir: str,
) -> Dict[str, Any]:
    """Stage 3: train each method on the 80% train split, evaluate once on the 20% test set."""
    os.makedirs(out_dir, exist_ok=True)
    inner_train_df, inner_val_df = _stratified_inner_split(train_full_df, config.inner_val_frac, config.seed)
    print(
        f"\n[Stage 3] Independent validation: fit on inner train (n={len(inner_train_df)}) for a fixed "
        f"{FIXED_TRAIN_EPOCHS} epochs (NO early stopping; inner val n={len(inner_val_df)} only monitored), "
        f"evaluate once on held-out test (n={len(test_df)}).",
        flush=True,
    )
    method_specs = _method_feature_specs(selected_proteins, phenotype_cols)
    results: Dict[str, List[Dict[str, Any]]] = {str(m["key"]): [] for m in METHOD_TEMPLATE}
    fold_method_outputs: Dict[str, Dict[str, Any]] = {}

    for method_idx, method in enumerate(METHOD_TEMPLATE, start=1):
        method_key = str(method["key"])
        method_display_name = str(method["display_name"])
        method_protein_cols, method_phenotype_cols = method_specs[method_key]
        method_feature_cols = list(method_protein_cols) + list(method_phenotype_cols)
        method_seed = int(config.seed + 1 + 1000 * method_idx)
        if method_key == "integrated_cov_plus_protein":
            method_seed = int(config.seed + 1)

        print(
            f"[Stage 3] Running method {method_idx}/{len(METHOD_TEMPLATE)}: {method_display_name} "
            f"(n_features={len(method_feature_cols)})...",
            flush=True,
        )
        v2 = _train_method_fold(
            inner_train_df,
            inner_val_df,
            test_df,
            method_key,
            method_display_name,
            list(method_protein_cols),
            list(method_phenotype_cols),
            device=device,
            seed=method_seed,
            fold=1,
            return_model=False,
            disable_early_stop=True,
        )
        v2["method_name"] = method_display_name
        v2["method_key"] = method_key
        fold_method_outputs[method_key] = {
            "c_index": float(v2["c_index"]),
            "risk_val": np.asarray(v2.get("risk_val", np.asarray([])), dtype=float),
        }
        for risk_key in ("risk_train", "risk_val"):
            v2.pop(risk_key, None)
        results[method_key].append(v2)

    incremental_gain_rows = _incremental_gain_rows_for_fold(
        test_df[TIME_COL].values, test_df[EVENT_COL].values, fold_method_outputs
    )
    summary = _summarize_results(results)
    output: Dict[str, Any] = {
        "config": _config_output_block(config, device, stage_note="independent_test_evaluation"),
        "fold_results": results,
        "incremental_gain_rows": incremental_gain_rows,
        "summary": summary,
    }
    return output


def _experiment_rows_from_summary(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten a per-method summary dict into experiment-summary rows (and print them)."""
    exp_rows: List[Dict[str, Any]] = []
    for model_name, block in summary.items():
        exp_rows.append(
            {
                "model": str(model_name),
                "mean_c_index": float(block.get("mean_c_index", np.nan)),
                "std_c_index": float(block.get("std_c_index", np.nan)),
                "n_c_index": int(len(block.get("fold_c_index", []))),
                "mean_ibs": float(block.get("mean_ibs", np.nan)),
                "std_ibs": float(block.get("std_ibs", np.nan)),
                "mean_td_auc_mean": float(block.get("mean_td_auc_mean", np.nan)),
                "std_td_auc_mean": float(block.get("std_td_auc_mean", np.nan)),
                "mean_td_auc_3y": float(block.get("mean_td_auc_3y", np.nan)),
                "mean_td_auc_5y": float(block.get("mean_td_auc_5y", np.nan)),
                "mean_td_auc_10y": float(block.get("mean_td_auc_10y", np.nan)),
                "mean_td_auprc_3y": float(block.get("mean_td_auprc_3y", np.nan)),
                "mean_td_auprc_5y": float(block.get("mean_td_auprc_5y", np.nan)),
                "mean_td_auprc_10y": float(block.get("mean_td_auprc_10y", np.nan)),
                "mean_brier_3y": float(block.get("mean_brier_3y", np.nan)),
                "mean_brier_5y": float(block.get("mean_brier_5y", np.nan)),
                "mean_brier_10y": float(block.get("mean_brier_10y", np.nan)),
            }
        )
        print(
            f"[{model_name}] mean c-index={exp_rows[-1]['mean_c_index']:.4f}, "
            f"mean IBS={exp_rows[-1]['mean_ibs']:.4f}, "
            f"mean td-AUC={exp_rows[-1]['mean_td_auc_mean']:.4f}",
            flush=True,
        )
    return exp_rows


def _save_cindex_auc_summary(result: Dict[str, Any], out_dir: str) -> str:
    """Save a compact per-model c-index / fixed-year td-AUC summary CSV and print it."""
    summary = result.get("summary", {})
    n_folds = max((len(block.get("fold_c_index", [])) for block in summary.values()), default=0)
    years = [int(y) for y in FIXED_EVAL_YEARS]
    rows: List[Dict[str, Any]] = []
    for model_name, block in summary.items():
        row: Dict[str, Any] = {"model": str(model_name)}
        if n_folds > 1:
            row["c_index_mean"] = float(block.get("mean_c_index", np.nan))
            row["c_index_std"] = float(block.get("std_c_index", np.nan))
            for y in years:
                row[f"td_auc_{y}y_mean"] = float(block.get(f"mean_td_auc_{y}y", np.nan))
                row[f"td_auc_{y}y_std"] = float(block.get(f"std_td_auc_{y}y", np.nan))
        else:
            row["c_index"] = float(block.get("mean_c_index", np.nan))
            for y in years:
                row[f"td_auc_{y}y"] = float(block.get(f"mean_td_auc_{y}y", np.nan))
        rows.append(row)
    df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, CINDEX_AUC_SUMMARY_CSV)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved c-index/AUC summary CSV to: {out_csv}")
    print(f"\nC-index / td-AUC summary ({os.path.basename(out_dir)}):", flush=True)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)
    return out_csv


def _save_prediction_outputs(
    result: Dict[str, Any],
    out_dir: str,
) -> None:
    """Save the per-fold metrics CSV and summary CSVs into ``out_dir``."""
    result_csv = save_result_csv(result, os.path.join(out_dir, OUTPUT_CSV))
    exp_rows = _experiment_rows_from_summary(result.get("summary", {}))
    summary_csv_path = save_experiment_summary_csv(exp_rows, os.path.join(out_dir, EXPERIMENT_SUMMARY_CSV))
    print(f"Saved per-fold metrics CSV to: {result_csv}")
    print(f"Saved experiment summary CSV to: {summary_csv_path}")
    _save_cindex_auc_summary(result, out_dir)


def save_result_csv(result: Dict[str, Any], output_csv_path: str) -> str:
    """Save fold metrics + summary mean/std to a single CSV file."""
    fold_results_by_model = result.get("fold_results", {})
    metric_cols = [
        "c_index",
        "ibs",
        "td_auc_mean",
        "td_auc_3y",
        "td_auc_5y",
        "td_auc_10y",
        "td_auprc_3y",
        "td_auprc_5y",
        "td_auprc_10y",
        "brier_3y",
        "brier_5y",
        "brier_10y",
        "logrank_p_value",
        "hazard_ratio_high_vs_low",
        "hr_ci95_lower",
        "hr_ci95_upper",
        "hr_p_value",
        "hr_per_sd",
        "hr_per_sd_ci95_lower",
        "hr_per_sd_ci95_upper",
        "hr_per_sd_p_value",
    ]

    rows: List[Dict[str, Any]] = []
    for model_name, fold_list in fold_results_by_model.items():
        for i, fold in enumerate(fold_list, start=1):
            row: Dict[str, Any] = {
                "row_type": "fold",
                "model": str(model_name),
                "fold": int(i),
                "split": str(fold.get("split", "")),
            }
            for key in metric_cols:
                v = fold.get(key, np.nan)
                row[key] = float(v) if isinstance(v, (int, float, np.number)) else np.nan
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        summary_rows: List[Dict[str, Any]] = []
        for model_name, g in df[df["row_type"] == "fold"].groupby("model", sort=False):
            mean_row: Dict[str, Any] = {
                "row_type": "mean",
                "model": str(model_name),
                "fold": "all",
                "split": "",
            }
            std_row: Dict[str, Any] = {
                "row_type": "std",
                "model": str(model_name),
                "fold": "all",
                "split": "",
            }
            for key in metric_cols:
                mean_row[key] = float(pd.to_numeric(g[key], errors="coerce").mean())
                std_row[key] = float(pd.to_numeric(g[key], errors="coerce").std())
            summary_rows.extend([mean_row, std_row])
        df = pd.concat([df, pd.DataFrame(summary_rows)], ignore_index=True)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    return output_csv_path


def save_experiment_summary_csv(rows: List[Dict[str, Any]], output_csv_path: str) -> str:
    """Save averaged metrics by protein-count configuration."""
    df = pd.DataFrame(rows)
    if not df.empty:
        if "protein_count" in df.columns:
            df = df.sort_values("protein_count", kind="mergesort").reset_index(drop=True)
        elif "model" in df.columns:
            df = df.sort_values("model", kind="mergesort").reset_index(drop=True)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    return output_csv_path


def _build_net_from_state(state: Dict[str, Any]) -> DualBranchResidualDeepSurvNet:
    n_prot = int(state["n_prot"])
    n_phen = int(state["n_phen"])
    net = DualBranchResidualDeepSurvNet(
        n_prot=n_prot,
        n_phen=n_phen,
        branch_emb=64,
        branch_hidden=256,
        fuse_width=256,
        n_res_blocks=2,
        dropout=DROPOUT,
        noise_std=GAUSSIAN_NOISE_STD,
        feat_drop_p=FEATURE_DROPOUT_P,
    )
    net.load_state_dict(state["state_dict"])
    net.eval()
    return net


def _fit_stage1_deepsurv(
    train_df: pd.DataFrame,
    protein_cols: Sequence[str],
    phenotype_cols: Sequence[str],
    device: torch.device,
    seed: int,
    val_frac: float,
    include_covariates: bool = True,
) -> Dict[str, Any]:
    """Stage-1 importance model: inner train/val split + a stage-1 DeepSurv fit."""
    protein_cols = list(protein_cols)
    phenotype_cols = list(phenotype_cols)
    stage1_protein_cols = protein_cols
    if include_covariates:
        stage1_phenotype_cols = phenotype_cols
        all_feat = protein_cols + phenotype_cols
        stage1_features = "proteins_and_covariates"
    else:
        stage1_phenotype_cols = []
        all_feat = protein_cols
        stage1_features = "proteins_only"

    inner_train, inner_val = _stratified_inner_split(train_df, val_frac, seed)
    stage1 = run_deepsurv_v2_fold(
        inner_train,
        inner_val,
        stage1_protein_cols,
        stage1_phenotype_cols,
        all_feat,
        device=device,
        seed=seed,
        return_model=True,
        stage1=True,
    )
    state = stage1.get("model_state")
    if not isinstance(state, dict) or "state_dict" not in state:
        raise RuntimeError("Stage-1 DeepSurv did not return a model state for feature importance.")

    net = _build_net_from_state(state).to(device)
    net.eval()
    n_prot = int(state["n_prot"])

    protein_order = [str(c) for c in state["selected_features"]][:n_prot]
    return {
        "net": net,
        "stage1_cidx": float(stage1.get("c_index", float("nan"))),
        "stage1_features": stage1_features,
        "n_proteins": int(n_prot),
        "n_inner_train": int(len(inner_train)),
        "n_inner_val": int(len(inner_val)),
        "protein_order": protein_order,
        "x_train": np.ascontiguousarray(np.asarray(state["x_train"], dtype=np.float32)),
        "x_val": np.ascontiguousarray(np.asarray(state["x_val"], dtype=np.float32)),
        "t_val": np.asarray(state["t_val"], dtype=np.float32),
        "e_val": np.asarray(state["e_val"], dtype=np.float32),
    }


def compute_deepsurv_permutation_importance(
    train_df: pd.DataFrame,
    protein_cols: Sequence[str],
    phenotype_cols: Sequence[str],
    device: torch.device,
    seed: int,
    n_repeats: int = PFI_REPEATS,
    val_frac: float = PFI_INNER_VAL_FRAC,
    include_covariates: bool = True,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Protein importance via permutation (c-index drop) on a stage-1 DeepSurv model."""
    s = _fit_stage1_deepsurv(
        train_df, protein_cols, phenotype_cols, device, seed, val_frac, include_covariates=include_covariates
    )
    net = s["net"]
    x_val = s["x_val"]
    t_val = s["t_val"]
    e_val = s["e_val"]
    protein_order = s["protein_order"]

    def _risk(x_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            r = net(torch.as_tensor(x_np, dtype=torch.float32, device=device)).reshape(-1)
        return r.detach().cpu().numpy().astype(np.float32)

    baseline_cidx = c_index(e_val, t_val, _risk(x_val))
    if not np.isfinite(baseline_cidx):
        baseline_cidx = 0.5

    rng = np.random.RandomState(int(seed) + 7919)
    importances: Dict[str, float] = {}
    for j, prot in enumerate(protein_order):
        col = x_val[:, j].copy()
        drops: List[float] = []
        for _ in range(int(n_repeats)):
            x_perm = x_val.copy()
            x_perm[:, j] = rng.permutation(col)
            c_perm = c_index(e_val, t_val, _risk(x_perm))
            if not np.isfinite(c_perm):
                c_perm = baseline_cidx
            drops.append(float(baseline_cidx - c_perm))
        importances[prot] = float(np.mean(drops))

    meta: Dict[str, Any] = {
        "importance_method": "permutation",
        "stage1_features": s["stage1_features"],
        "baseline_cidx": float(baseline_cidx),
        "stage1_cidx": s["stage1_cidx"],
        "n_proteins": int(s["n_proteins"]),
        "n_inner_train": int(s["n_inner_train"]),
        "n_inner_val": int(s["n_inner_val"]),
        "n_repeats": int(n_repeats),
        "protein_order": protein_order,
    }
    return importances, meta


def _category_nri(
    t_true: np.ndarray,
    e_true: np.ndarray,
    risk_new: np.ndarray,
    risk_old: np.ndarray,
    horizon: float,
) -> float:
    """Category NRI using pooled tertile risk categories on the valid subset."""
    t_true = np.asarray(t_true, dtype=float).reshape(-1)
    e_true = np.asarray(e_true, dtype=int).reshape(-1)
    risk_new = np.asarray(risk_new, dtype=float).reshape(-1)
    risk_old = np.asarray(risk_old, dtype=float).reshape(-1)
    valid = (e_true.astype(bool) & (t_true <= float(horizon))) | (t_true > float(horizon))
    if int(valid.sum()) < 10:
        return float("nan")
    y_event = e_true.astype(bool) & (t_true <= float(horizon))
    y_nonevent = t_true > float(horizon)
    y_event = y_event[valid]
    y_nonevent = y_nonevent[valid]
    if np.unique(y_event).size < 2 or np.unique(y_nonevent).size < 2:
        return float("nan")

    pooled = np.concatenate([risk_new[valid], risk_old[valid]])
    q1, q2 = np.quantile(pooled, [1.0 / 3.0, 2.0 / 3.0]).astype(float)
    bins = np.asarray([q1, q2], dtype=float)
    cat_new = np.digitize(risk_new[valid], bins=bins, right=False)
    cat_old = np.digitize(risk_old[valid], bins=bins, right=False)
    if np.unique(cat_new).size < 2 and np.unique(cat_old).size < 2:
        return float("nan")

    event_up = float(np.mean(cat_new[y_event] > cat_old[y_event]))
    event_down = float(np.mean(cat_new[y_event] < cat_old[y_event]))
    nonevent_up = float(np.mean(cat_new[y_nonevent] > cat_old[y_nonevent]))
    nonevent_down = float(np.mean(cat_new[y_nonevent] < cat_old[y_nonevent]))
    return float((event_up - event_down) + (nonevent_down - nonevent_up))


def main() -> None:

    result_dir = RESULT_DIR

    feature_selection_dir = os.path.join(result_dir, FEATURE_SELECTION_SUBDIR)
    cross_validation_dir = os.path.join(result_dir, CROSS_VALIDATION_SUBDIR)
    independent_validation_dir = os.path.join(result_dir, INDEPENDENT_VALIDATION_SUBDIR)
    for d in (result_dir, feature_selection_dir, cross_validation_dir, independent_validation_dir):
        os.makedirs(d, exist_ok=True)

    print(
        f"\n########## DeepSurv holdout experiment: {IMPORTANCE_METHOD}-importance selection on an 80% "
        f"train split, {N_SPLITS}-fold CV + independent 20% test ##########",
        flush=True,
    )
    print(f"Data dir: {DATA_DIR}", flush=True)
    print(
        f"Outputs -> {result_dir}/[{FEATURE_SELECTION_SUBDIR}|{CROSS_VALIDATION_SUBDIR}|{INDEPENDENT_VALIDATION_SUBDIR}]",
        flush=True,
    )
    cfg = Config(
        label_csv_path=os.path.join(DATA_DIR, LABEL_CSV),
        protein_csv_path=os.path.join(DATA_DIR, PROTEIN_CSV),
        cov_csv_path=os.path.join(DATA_DIR, COV_CSV),
        n_splits=N_SPLITS,
        seed=RANDOM_SEED,
        use_gpu=USE_GPU,
        holdout_test_frac=HOLDOUT_TEST_FRAC,
        inner_val_frac=INNER_VAL_FRAC,
        pfi_repeats=PFI_REPEATS,
        pfi_inner_val_frac=PFI_INNER_VAL_FRAC,
        cumulative_importance_share=CUMULATIVE_IMPORTANCE_SHARE,
        importance_method=IMPORTANCE_METHOD,
        selection_top_n=SELECTION_TOP_N,
        stage1_include_covariates=STAGE1_INCLUDE_COVARIATES,
    )

    df, protein_cols, phenotype_cols = read_dataset(
        cfg.label_csv_path,
        cfg.protein_csv_path,
        cfg.cov_csv_path,
        selected_protein_cols=None,
    )
    feature_cols = protein_cols + phenotype_cols
    print(
        f"Loaded {len(protein_cols)} proteins + {len(phenotype_cols)} covariates from "
        f"{os.path.basename(cfg.protein_csv_path)}; protein selection runs ONCE on the holdout-train split.",
        flush=True,
    )
    print(f"Data shape: {df.shape}", flush=True)
    print(f"Protein features: {len(protein_cols)}, phenotype features: {len(phenotype_cols)}", flush=True)
    print(f"Total candidate features: {len(feature_cols)}", flush=True)

    device = torch.device("cuda" if cfg.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("CUDA unavailable or disabled; using CPU (batch size will auto-downshift).", flush=True)
    if USE_FULL_BATCH_COX_FOR_V2:
        print(
            f"DeepSurv_v2 batching: full-batch exact Cox when train n <= {FULL_BATCH_MAX_N}; "
            f"otherwise {_select_batch_size(device)} mini-batch fallback.",
            flush=True,
        )
    else:
        print(f"DeepSurv_v2 effective batch size: {_select_batch_size(device)}", flush=True)

    # --- Stage 0: stratified 80/20 hold-out split of ALL samples ---
    train_full_df, test_df = make_holdout_split(df, cfg.holdout_test_frac, cfg.seed)
    print(
        f"\n[Stage 0] Holdout split: train_full n={len(train_full_df)} "
        f"(events={int(train_full_df[EVENT_COL].sum())}), test n={len(test_df)} "
        f"(events={int(test_df[EVENT_COL].sum())}).",
        flush=True,
    )
    split_eids_df = pd.concat(
        [
            pd.DataFrame({ID_COL: train_full_df[ID_COL].values, "split": "train_full"}),
            pd.DataFrame({ID_COL: test_df[ID_COL].values, "split": "test"}),
        ],
        ignore_index=True,
    )
    split_csv_path = os.path.join(result_dir, HOLDOUT_SPLIT_CSV)
    split_eids_df.to_csv(split_csv_path, index=False, encoding="utf-8-sig")
    print(f"[Stage 0] Saved holdout split eid lists to: {split_csv_path}", flush=True)

    # --- Stage 1: feature selection on the 80% split ONLY (fixed protein set) ---
    fs_result = run_feature_selection(cfg, train_full_df, protein_cols, phenotype_cols, device, feature_selection_dir)
    selected_proteins = list(fs_result["selected_proteins"])

    # --- Stage 2: 5-fold CV on the 80% split ONLY (fixed selected proteins) ---
    print(
        f"\n[Stage 2] {cfg.n_splits}-fold CV on the holdout-train split with the fixed "
        f"{len(selected_proteins)}-protein set...",
        flush=True,
    )
    cv_result = run_cross_validation(cfg, train_full_df, selected_proteins, phenotype_cols, device, cross_validation_dir)
    _save_prediction_outputs(cv_result, cross_validation_dir)
    print(f"Cross-validation metric CSVs are saved in: {cross_validation_dir}", flush=True)

    # --- Stage 3: independent validation on the 20% test set ---
    iv_result = run_independent_validation(
        cfg, train_full_df, test_df, selected_proteins, phenotype_cols, device, independent_validation_dir
    )
    _save_prediction_outputs(iv_result, independent_validation_dir)
    print(f"Independent-validation metric CSVs are saved in: {independent_validation_dir}", flush=True)


if __name__ == "__main__":
    main()
