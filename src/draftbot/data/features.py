"""Card feature tables (PLAN P0.T6).

Two kinds of features, kept separate on disk and concatenated at model-load time:

- **static**: derived purely from Scryfall card data (available day 0 for any set).
  `data/processed/<SET>/features.static.parquet`, indexed by card id.
- **stats**: 17lands ratings-API snapshots over an explicit [start, end] date range,
  all 32 color filters, saved immutably to
  `data/snapshots/<SET>/<start>_<end>.parquet` with a tag registry (full/week1).
  Missing values stay NaN in storage; `assemble()` turns each continuous column
  into (z-scored value with NaN→0, paired `<col>__miss` mask).

Scalers are fit from a *training* feature table only (fit_scaler/apply live here;
training code persists the scaler next to the checkpoint).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from draftbot.data.cards import HEADERS, PROCESSED_DIR, norm_name

SNAPSHOT_DIR = Path("data/snapshots")

COLOR_FILTERS = [
    None, "W", "U", "B", "R", "G",
    "WU", "WB", "WR", "WG", "UB", "UR", "UG", "BR", "BG", "RG",
    "WUB", "WUR", "WUG", "WBR", "WBG", "WRG", "UBR", "UBG", "URG", "BRG",
    "WUBR", "WUBG", "WURG", "WBRG", "UBRG", "WUBRG",
]

# Fixed keyword vocabulary so static-feature dimensionality is identical across
# sets (Phase-3 requirement). Mechanics beyond this list are captured only via
# n_keywords (and later, oracle-text embeddings — PLAN P3.T5).
KEYWORDS = [
    "flying", "first strike", "double strike", "deathtouch", "defender", "flash",
    "haste", "hexproof", "indestructible", "lifelink", "menace", "reach",
    "trample", "vigilance", "ward", "scry", "surveil", "mill", "fight", "goad",
]

TYPES = ["creature", "instant", "sorcery", "enchantment", "artifact",
         "planeswalker", "land", "battle"]
RARITIES = ["common", "uncommon", "rare", "mythic", "special", "bonus"]


def set_release_date(set_code: str) -> str:
    cache = Path("data/raw/scryfall") / f"set_{set_code.lower()}.json"
    if cache.exists():
        info = json.loads(cache.read_text())
    else:
        resp = requests.get(f"https://api.scryfall.com/sets/{set_code.lower()}",
                            headers=HEADERS, timeout=30)
        resp.raise_for_status()
        info = resp.json()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(info))
    return info["released_at"]


def _pt_to_float(v) -> float:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return -1.0  # '*', '1+*', 'X' → sentinel, as the original repo did


def build_static_features(set_code: str) -> pd.DataFrame:
    cards = pd.read_parquet(PROCESSED_DIR / set_code / "cards.parquet")
    out = pd.DataFrame(index=cards["id"].values)
    out["cmc"] = cards["cmc"].fillna(0).values
    out["power"] = [_pt_to_float(v) for v in cards["power"]]
    out["toughness"] = [_pt_to_float(v) for v in cards["toughness"]]
    out[["power", "toughness"]] = out[["power", "toughness"]].fillna(0.0)
    mana = cards["mana_cost"].fillna("").values
    for c in "WUBRG":
        out[f"pips_{c}"] = [str(m).count(c) for m in mana]
        out[f"color_{c}"] = [int(c in str(ci)) for ci in cards["colors"].fillna("")]
        out[f"produces_{c}"] = [int(c in str(p)) for p in cards["produced_mana"].fillna("")]
    out["produces_C"] = [int("C" in str(p)) for p in cards["produced_mana"].fillna("")]
    out["n_pips"] = out[[f"pips_{c}" for c in "WUBRG"]].sum(axis=1)
    out["n_colors"] = out[[f"color_{c}" for c in "WUBRG"]].sum(axis=1)
    tl = cards["type_line"].fillna("").str.lower().values
    for t in TYPES:
        out[f"type_{t}"] = [int(t in s) for s in tl]
    for r in RARITIES:
        out[f"rarity_{r}"] = (cards["rarity"] == r).astype(int).values
    kw = cards["keywords"].fillna("").str.lower().values
    for k in KEYWORDS:
        out[f"kw_{k.replace(' ', '_')}"] = [int(k in s) for s in kw]
    out["n_keywords"] = [len([x for x in s.split(",") if x]) for s in kw]
    out["is_flip"] = cards["is_flip"].values
    out["is_basic"] = cards["is_basic"].values
    out.index.name = "id"
    path = PROCESSED_DIR / set_code / "features.static.parquet"
    out.to_parquet(path)
    return out


def _registry_path(set_code: str) -> Path:
    return SNAPSHOT_DIR / set_code / "registry.json"


# NOTE (2026-07-28, journal): the anonymous 17lands ratings API stopped serving
# win rates and ignores start/end dates (verified across MSH/FIN/EOE/TDM). Stat
# snapshots are therefore COMPUTED from the raw public game_data + draft_data
# files — the backlog item "rating-free stat features", promoted to P0.T6. Date
# windows are exact: a game/pick counts iff its timestamp date ∈ [start, end].

MIN_GAMES_FOR_WR = 5  # below this, win-rate columns stay NaN (masked), not fake 0


def _draft_stage_stats(set_code: str, start: str, end: str,
                       event: str) -> pd.DataFrame:
    """seen_count / avg_seen (ALSA) / pick_count / avg_pick (ATA) from draft data."""
    from draftbot.data.dataset import PAD, load_draft_arrays

    arrays = load_draft_arrays(set_code, event)
    times = pd.to_datetime(arrays.meta["draft_time"]).dt.date
    keep = ((times >= pd.Timestamp(start).date())
            & (times <= pd.Timestamp(end).date())).to_numpy()
    packs, picks = arrays.packs[keep], arrays.picks[keep]
    t = packs.shape[1]
    picks_per_pack = t // 3
    cards = pd.read_parquet(PROCESSED_DIR / set_code / "cards.parquet")
    n_cards = len(cards)

    seen_sum = np.zeros(n_cards)
    seen_cnt = np.zeros(n_cards)
    # last-seen per (draft, pack, card): walk pick positions in reverse; the first
    # time we see a card (going backward) is its last sighting in that pack.
    for pack_no in range(3):
        recorded = np.zeros((packs.shape[0], n_cards), dtype=bool)
        for p in range(picks_per_pack - 1, -1, -1):
            step = pack_no * picks_per_pack + p
            slot_ids = packs[:, step]  # (N, P)
            for slot in range(slot_ids.shape[1]):
                ids = slot_ids[:, slot]
                valid = ids != PAD
                idx = ids[valid].astype(np.int64)
                rows = np.flatnonzero(valid)
                fresh = ~recorded[rows, idx]
                np.add.at(seen_sum, idx[fresh], p + 1)
                np.add.at(seen_cnt, idx[fresh], 1)
                recorded[rows[fresh], idx[fresh]] = True
    pick_cnt = np.bincount(picks.ravel(), minlength=n_cards).astype(float)
    pick_sum = np.zeros(n_cards)
    pick_pos_in_pack = (np.arange(t) % picks_per_pack) + 1
    np.add.at(pick_sum, picks.ravel().astype(np.int64),
              np.broadcast_to(pick_pos_in_pack, picks.shape).ravel())
    with np.errstate(invalid="ignore"):
        out = pd.DataFrame({
            "seen_count": seen_cnt,
            "avg_seen": np.where(seen_cnt > 0, seen_sum / seen_cnt, np.nan),
            "pick_count": pick_cnt,
            "avg_pick": np.where(pick_cnt > 0, pick_sum / pick_cnt, np.nan),
        }, index=[norm_name(n) for n in cards["name"]])
    return out


def _game_stage_stats(set_code: str, start: str, end: str,
                      event: str) -> pd.DataFrame:
    """GP/OH/GD/GIH/GNS win rates + IWD, overall and per exact-deck-color filter,
    streamed from game_data in chunks."""
    from draftbot.data.fetch import dest_path

    cards = pd.read_parquet(PROCESSED_DIR / set_code / "cards.parquet")
    names = list(cards["name"])
    n_cards = len(names)
    csv_path = dest_path(set_code, event, kind="game")
    blocks = {b: [f"{b}_{n}" for n in names]
              for b in ("deck", "opening_hand", "drawn")}
    header = pd.read_csv(csv_path, nrows=0).columns
    has_game_time = "game_time" in header  # absent in pre-2023 game_data exports
    has_main_colors = "main_colors" in header  # absent in 2021 exports (STX)
    time_cols = ["draft_time"] + (["game_time"] if has_game_time else [])
    usecols = (time_cols + (["main_colors"] if has_main_colors else [])
               + ["won"] + sum(blocks.values(), []))
    # float32: 2021 exports carry NAs in the card blocks (NaN > 0 is False)
    dtypes = {c: "float32" for cols in blocks.values() for c in cols}

    filt_keys = [None] + [c for c in COLOR_FILTERS if c]
    STATS = ["game", "opening_hand", "drawn", "ever_drawn", "never_drawn"]
    acc = {k: {s: np.zeros((2, n_cards)) for s in STATS} for k in filt_keys}

    start_d, end_d = pd.Timestamp(start), pd.Timestamp(end)
    for chunk in pd.read_csv(csv_path, usecols=usecols, dtype=dtypes,
                             chunksize=20_000):
        when = pd.to_datetime(chunk["game_time"].fillna(chunk["draft_time"])
                              if has_game_time else chunk["draft_time"])
        keep = (when >= start_d) & (when < end_d + pd.Timedelta(days=1))
        chunk = chunk[keep]
        if not len(chunk):
            continue
        won = chunk["won"].astype(bool).to_numpy()
        deck = chunk[blocks["deck"]].to_numpy() > 0
        oh = chunk[blocks["opening_hand"]].to_numpy() > 0
        drawn = chunk[blocks["drawn"]].to_numpy() > 0
        ever = oh | drawn
        never = deck & ~ever
        mats = {"game": deck, "opening_hand": oh, "drawn": drawn,
                "ever_drawn": ever, "never_drawn": never}
        mc = (chunk["main_colors"].fillna("").to_numpy() if has_main_colors
              else np.full(len(chunk), "", dtype=object))  # color filters stay NaN→masked
        for key in filt_keys:
            rows = slice(None) if key is None else np.flatnonzero(mc == key)
            if key is not None and len(rows) == 0:
                continue
            w = won[rows]
            for s, m in mats.items():
                sub = m[rows]
                acc[key][s][0] += sub.sum(0)
                acc[key][s][1] += (sub & w[:, None]).sum(0)

    out = {}
    for key in filt_keys:
        suffix = "" if key is None else f"_{key}"
        for s in STATS:
            games, wins = acc[key][s]
            out[f"{s}_count{suffix}"] = games
            with np.errstate(invalid="ignore", divide="ignore"):
                wr = np.where(games >= MIN_GAMES_FOR_WR, wins / np.maximum(games, 1),
                              np.nan)
            out[f"{s}_win_rate{suffix}"] = wr
        out[f"drawn_improvement_win_rate{suffix}"] = (
            out[f"ever_drawn_win_rate{suffix}"] - out[f"never_drawn_win_rate{suffix}"])
    return pd.DataFrame(out, index=[norm_name(n) for n in names])


def build_snapshot(set_code: str, start: str, end: str, tag: str | None = None,
                   event: str = "PremierDraft") -> pd.DataFrame:
    """Compute the [start, end] stat snapshot from raw data; immutable on disk."""
    out_dir = SNAPSHOT_DIR / set_code
    out_path = out_dir / f"{start}_{end}.parquet"
    if not out_path.exists():
        snap = pd.concat([
            _draft_stage_stats(set_code, start, end, event),
            _game_stage_stats(set_code, start, end, event),
        ], axis=1)
        out_dir.mkdir(parents=True, exist_ok=True)
        snap.to_parquet(out_path)
    if tag:
        reg_path = _registry_path(set_code)
        reg = json.loads(reg_path.read_text()) if reg_path.exists() else {}
        reg[tag] = out_path.name
        reg_path.write_text(json.dumps(reg, indent=2))
    return pd.read_parquet(out_path)


def load_snapshot(set_code: str, tag: str) -> pd.DataFrame:
    """Load a snapshot by tag, reindexed to card-id order (NaN for missing cards)."""
    reg = json.loads(_registry_path(set_code).read_text())
    snap = pd.read_parquet(SNAPSHOT_DIR / set_code / reg[tag])
    cards = pd.read_parquet(PROCESSED_DIR / set_code / "cards.parquet")
    snap = snap.reindex([norm_name(n) for n in cards["name"]])
    snap.index = cards["id"].values
    snap.index.name = "id"
    return snap


# ---------------------------------------------------------------- scaler ----

def fit_scaler(df: pd.DataFrame, binary_ok: bool = True) -> dict:
    """Per-column mean/std over non-NaN entries. Binary 0/1 columns pass through."""
    scaler = {}
    for col in df.columns:
        vals = df[col].astype(float)
        uniq = set(vals.dropna().unique())
        if binary_ok and uniq <= {0.0, 1.0}:
            scaler[col] = {"mean": 0.0, "std": 1.0, "binary": True}
        else:
            std = float(vals.std()) or 1.0
            scaler[col] = {"mean": float(vals.mean()), "std": std, "binary": False}
    return scaler


def save_scaler(scaler: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scaler, indent=1))


def load_scaler(path: Path) -> dict:
    return json.loads(path.read_text())


def apply_scaler(df: pd.DataFrame, scaler: dict, add_masks: bool) -> pd.DataFrame:
    """Z-score with stored stats; NaN→0 after scaling, plus paired __miss masks."""
    out = {}
    for col in df.columns:
        s = scaler[col]
        vals = df[col].astype(float)
        scaled = vals if s["binary"] else (vals - s["mean"]) / s["std"]
        if add_masks:
            out[f"{col}__miss"] = vals.isna().astype(np.float32)
        out[col] = scaled.fillna(0.0).astype(np.float32)
    return pd.DataFrame(out, index=df.index)


def assemble(set_code: str, stats: str | None, scaler: dict | None = None,
             fit: bool = False, text: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Full feature table for a set: static ∥ [text] ∥ (stat snapshot ∥ masks).

    stats: None → static only, masks all-on for the stat block is handled by the
    caller keeping dims fixed (Phase 3); for single-set work None simply means
    no stat columns. text: encoder name → the frozen oracle-text embedding
    block (P3.T5c; never masked — text never goes missing). Returns
    (features, scaler).
    """
    static_path = PROCESSED_DIR / set_code / "features.static.parquet"
    static = pd.read_parquet(static_path) if static_path.exists() \
        else build_static_features(set_code)
    parts = [("static", static, False)]
    if text:
        from draftbot.data.text_features import load_text_features
        parts.append(("text", load_text_features(set_code, text), False))
    if stats:
        parts.append(("stats", load_snapshot(set_code, stats), True))
    if fit:
        scaler = {}
        for _, df, _ in parts:
            scaler.update(fit_scaler(df))
    assert scaler is not None, "need a scaler (fit=True at train time, load at eval)"
    scaled = [apply_scaler(df, {c: scaler[c] for c in df.columns}, add_masks=masks)
              for _, df, masks in parts]
    return pd.concat(scaled, axis=1), scaler
