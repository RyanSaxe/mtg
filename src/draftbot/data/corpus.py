"""Multi-set pretraining corpus loader (PLAN P3.T2, in-memory variant).

Loads each manifest set's TRAIN-split drafts (capped per set), keeps arrays as
int16 (~1.5GB for 29 sets — well under the 10GB ceiling), and yields single-set
batches sampled by manifest weight. Feature scaler is fit on the corpus sets'
card tables ONLY (stats-leakage discipline; the sandbag never contributes).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from draftbot.data.cards import PROCESSED_DIR
from draftbot.data.dataset import DraftArrays, load_draft_arrays
from draftbot.data.features import assemble, fit_scaler
from draftbot.data.splits import load_splits
from draftbot.models.base import importance_weights


@dataclass
class SetBundle:
    code: str
    arrays: DraftArrays
    weights: np.ndarray          # importance weights (N, t)
    features_full: np.ndarray    # (n_cards+1, F) scaled, stats present
    features_statless: np.ndarray  # same shape, stats zeroed + masks on
    sample_weight: float
    columns: list[str] | None = None  # feature column names (+ "is_bias")


def _statless(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    stat_cols = [c for c in out.columns if c.endswith("__miss")]
    for mask_col in stat_cols:
        out[mask_col] = 1.0
        out[mask_col.removesuffix("__miss")] = 0.0
    return out


def fit_corpus_scaler(set_codes: list[str], text: str | None = None) -> dict:
    """One scaler over all corpus sets' raw feature tables (static [+ text] +
    full stats). The sandbag never contributes (quarantine covers this)."""
    from draftbot.data.features import PROCESSED_DIR as PD, load_snapshot
    frames = []
    for code in set_codes:
        parts = [pd.read_parquet(PD / code / "features.static.parquet")]
        if text:
            from draftbot.data.text_features import load_text_features
            parts.append(load_text_features(code, text))
        parts.append(load_snapshot(code, "full"))
        frames.append(pd.concat(parts, axis=1))
    return fit_scaler(pd.concat(frames, axis=0, ignore_index=True))


def add_bias_row(feats: pd.DataFrame) -> np.ndarray:
    arr = feats.to_numpy(np.float32)
    arr = np.concatenate([arr, np.zeros((1, arr.shape[1]), np.float32)], 0)
    is_bias = np.zeros((arr.shape[0], 1), np.float32)
    is_bias[-1, 0] = 1.0
    return np.concatenate([arr, is_bias], 1)


def load_corpus(manifest_path: Path, skill_filter: dict | None = None,
                scaler: dict | None = None,
                text: str | None = None) -> tuple[list[SetBundle], dict]:
    manifest = yaml.safe_load(Path(manifest_path).read_text())
    quarantined = set(manifest["quarantine"])
    codes = [c for c in manifest["sets"] if
             (PROCESSED_DIR / c / "draft.PremierDraft.parquet").exists()]
    assert not (set(codes) & quarantined), "quarantined set in corpus!"
    cap = manifest.get("max_drafts_per_set")
    if scaler is None:
        scaler = fit_corpus_scaler(codes, text)

    bundles = []
    for code in codes:
        splits = load_splits(code)
        arr = load_draft_arrays(code, draft_ids=set(splits["random"]["train"]))
        if skill_filter:
            m = arr.meta
            keep = ((m["user_win_rate"].fillna(0) >= skill_filter["min_win_rate"])
                    & (m["user_n_games"] >= skill_filter["min_games"])).to_numpy()
            if keep.sum() == 0:
                # 2021-era exports carry no skill metadata; keep the whole set
                # (weights fall back to the neutral 0.5 pathway) rather than
                # silently dropping it from the corpus.
                print(f"corpus: {code} has no skill metadata — filter skipped",
                      flush=True)
            else:
                arr = DraftArrays(arr.packs[keep], arr.picks[keep],
                                  arr.prev_picks[keep], m[keep].reset_index(drop=True))
        if cap and arr.n_drafts > cap:
            sel = np.random.default_rng(13).choice(arr.n_drafts, cap, replace=False)
            arr = DraftArrays(arr.packs[sel], arr.picks[sel], arr.prev_picks[sel],
                              arr.meta.iloc[sel].reset_index(drop=True))
        m = arr.meta
        w = importance_weights(m["rank"].to_numpy(), m["user_win_rate"].to_numpy(),
                               m["event_wins"].to_numpy(np.float64),
                               m["event_losses"].to_numpy(np.float64),
                               m["draft_time"].to_numpy(), arr.t).astype(np.float32)
        full, _ = assemble(code, "full", scaler=scaler, text=text)
        bundles.append(SetBundle(
            code=code, arrays=arr, weights=w,
            features_full=add_bias_row(full),
            features_statless=add_bias_row(_statless(full)),
            sample_weight=float(manifest["sets"][code].get("weight", 1.0)) * arr.n_drafts,
            columns=list(full.columns) + ["is_bias"],
        ))
        print(f"corpus: {code} {arr.n_drafts} drafts t={arr.t} P={arr.max_pack}",
              flush=True)
    return bundles, scaler


def batch_iterator(bundles: list[SetBundle], batch_drafts: int, rng: np.random.Generator):
    """Yield (bundle, idx) forever; sets sampled by weight, drafts uniformly."""
    probs = np.array([b.sample_weight for b in bundles], dtype=np.float64)
    probs /= probs.sum()
    while True:
        b = bundles[int(rng.choice(len(bundles), p=probs))]
        idx = rng.integers(0, b.arrays.n_drafts, size=batch_drafts)
        yield b, idx
