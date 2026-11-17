"""Oracle-text card embeddings (P3.T5c; docs/TEXT_EMBED_PLAN.md §D1).

One canonical string per card, built from fields cards.parquet already
persists, through a FROZEN sentence encoder →
data/processed/<SET>/features.text.<encoder>.parquet (index = card id,
columns text_<encoder>_000…). A sidecar .json stores the sha of the strings,
so the table is rebuilt only when a card's text changes.

The hand-built static block stays the interpretable one; this block is what
lets the model read a card nobody has played yet. Day-0 available: Scryfall
oracle text exists before Arena ships the set.

    uv run python -m draftbot.data.text_features --set HOB [--encoder bge-base]
    uv run python -m draftbot.data.text_features --corpus configs/corpus/pretrain_v2.yaml
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from draftbot.data.cards import PROCESSED_DIR

ENCODERS = {
    "bge-small": "BAAI/bge-small-en-v1.5",                 # 384-d, 33M params
    "bge-base": "BAAI/bge-base-en-v1.5",                   # 768-d, 110M — default
    "bge-large": "BAAI/bge-large-en-v1.5",                 # 1024-d, 335M
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",    # 384-d, 22M
    # Control (journal 2026-09-05): a fixed random unit vector per card NAME —
    # all the identity of a text embedding, none of the semantics. Reprints
    # share a vector, as they share text.
    "random": "random-unit-vector-control",
}
DEFAULT_ENCODER = "bge-base"
RANDOM_DIM = 768  # matches bge-base so the trunk is the same size


def random_vectors(names: list[str], dim: int | None = None) -> np.ndarray:
    dim = dim or RANDOM_DIM  # read at call time (tests shrink it)
    out = np.zeros((len(names), dim), np.float32)
    for i, n in enumerate(names):
        key = n.split("//")[0].strip().lower().encode()
        v = np.random.default_rng(int(hashlib.sha1(key).hexdigest()[:8], 16)).standard_normal(dim)
        out[i] = v / np.linalg.norm(v)
    return out
# Probe ablation (plan §3): `masked` = header without the name and the name
# replaced by "this card" inside the rules text (the CARDNAME convention);
# `named` = name kept everywhere; `oracle` = rules text only.
FORMATS = ("masked", "named", "oracle")


def _aliases(name: str) -> list[str]:
    """Strings the rules text uses to refer to the card itself: the full
    front-face name and, for legends, the part before the comma."""
    front = name.split("//")[0].strip()
    short = front.split(",")[0].strip()
    return sorted({a for a in (front, short) if a}, key=len, reverse=True)


def card_text(row: dict, fmt: str = "masked") -> str:
    name = str(row["name"]).split("//")[0].strip()
    oracle = str(row.get("oracle_text") or "").replace("\n", " ").strip()
    if fmt != "named":
        for alias in _aliases(name):
            oracle = re.sub(re.escape(alias), "this card", oracle)
    if fmt == "oracle":
        return oracle or "(no rules text)"
    power, toughness = row.get("power"), row.get("toughness")
    pt = f" {power}/{toughness}." if isinstance(power, str) and power else ""
    head = f"{name}. " if fmt == "named" else ""
    s = (f"{head}{row.get('mana_cost') or ''} {row.get('type_line') or ''}.{pt} "
         f"{row.get('rarity') or ''}. {oracle or '(no rules text)'}")
    return re.sub(r"\s+", " ", s).strip()


def text_path(set_code: str, encoder: str = DEFAULT_ENCODER) -> Path:
    return PROCESSED_DIR / set_code / f"features.text.{encoder}.parquet"


def card_strings(set_code: str, fmt: str = "masked") -> list[str]:
    cards = pd.read_parquet(PROCESSED_DIR / set_code / "cards.parquet")
    return [card_text(r, fmt) for r in cards.to_dict("records")]


_MODELS: dict[str, object] = {}


def _encoder(name: str):
    if name not in _MODELS:
        from sentence_transformers import SentenceTransformer  # optional extra [text]
        from draftbot.models.base import device_auto
        _MODELS[name] = SentenceTransformer(ENCODERS[name], device=str(device_auto()))
    return _MODELS[name]


def embed(texts: list[str], encoder: str = DEFAULT_ENCODER) -> np.ndarray:
    """(n, dim) L2-normalised float32 embeddings from the frozen encoder."""
    return _encoder(encoder).encode(
        texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False).astype(np.float32)


def build_text_features(set_code: str, encoder: str = DEFAULT_ENCODER,
                        fmt: str = "masked") -> pd.DataFrame:
    cards = pd.read_parquet(PROCESSED_DIR / set_code / "cards.parquet")
    texts = [card_text(r, fmt) for r in cards.to_dict("records")]
    sha = hashlib.sha1(("\n".join(texts) + f"|{fmt}").encode()).hexdigest()
    path, meta = text_path(set_code, encoder), text_path(set_code, encoder).with_suffix(".json")
    if path.exists() and meta.exists() and json.loads(meta.read_text()).get("sha") == sha:
        return pd.read_parquet(path)
    vecs = (random_vectors(list(cards["name"])) if encoder == "random"
            else embed(texts, encoder))
    cols = [f"text_{encoder}_{i:03d}" for i in range(vecs.shape[1])]
    df = pd.DataFrame(vecs, index=cards["id"].values, columns=cols)
    df.index.name = "id"
    df.to_parquet(path)
    meta.write_text(json.dumps({"encoder": ENCODERS[encoder], "format": fmt,
                                "sha": sha, "dim": int(vecs.shape[1])}, indent=1))
    return df


PRIOR_DIR = Path("checkpoints")


def load_strength_prior(set_code: str, exp: str) -> pd.DataFrame:
    """Text-predicted strength stats from a fine-tuned encoder (TXT-001,
    `draftbot.train.text_strength`): `prior_<stat>` columns, one row per card.
    For corpus sets these are OUT-OF-FOLD predictions — the model that scored a
    set never trained on it — so the trunk sees them exactly as accurate as
    they will be on a new set. Deliberately not named `text_*`: the trunk's
    text noise / text dropout must not touch them, and they are not stats
    (no mask, never dropped) — a prior is available on day 0 by construction.
    """
    path = PRIOR_DIR / exp / f"prior.{set_code}.parquet"
    if not path.exists():
        raise TextBlockUnavailable(f"{set_code}: no strength prior at {path} "
                                   f"(run draftbot.train.text_strength for {exp})")
    df = pd.read_parquet(path)
    n = len(pd.read_parquet(PROCESSED_DIR / set_code / "cards.parquet", columns=["id"]))
    assert len(df) == n, f"{path}: {len(df)} rows for {n} cards"
    assert all(c.startswith("prior_") for c in df.columns)
    df.index.name = "id"
    return df.astype(np.float32)


class TextBlockUnavailable(RuntimeError):
    """A text-trained checkpoint needs `features.text.<encoder>.parquet` for
    this set; it is missing and the encoder cannot be loaded here."""


def load_text_features(set_code: str, encoder: str = DEFAULT_ENCODER) -> pd.DataFrame:
    """Cached table if present (the bundle ships it — no encoder needed on the
    laptop), else build it — with a clear failure when that is impossible
    (day-0 onboarding on a machine without the optional [text] extra)."""
    if "+" in encoder:  # composite spec, e.g. "bge-base+prior:TXT-001"
        return pd.concat([load_text_features(set_code, part) for part in encoder.split("+")], axis=1)
    if encoder.startswith("prior:"):
        return load_strength_prior(set_code, encoder.removeprefix("prior:"))
    path = text_path(set_code, encoder)
    if path.exists():
        return pd.read_parquet(path)
    try:
        return build_text_features(set_code, encoder)
    except ImportError as e:
        raise TextBlockUnavailable(
            f"{set_code}: no {path.name} and sentence-transformers is not installed, "
            f"so it cannot be built here. Either `uv sync --extra dev --extra text` "
            f"and re-run, copy the file from a machine that has it, or use a "
            f"checkpoint trained without text (e.g. --models checkpoints/EXP-033)."
        ) from e


def main(argv=None):
    p = argparse.ArgumentParser(prog="draftbot.data.text_features")
    p.add_argument("--set", dest="sets", action="append", default=[])
    p.add_argument("--corpus", type=Path, default=None, help="manifest: build for all its sets")
    p.add_argument("--encoder", default=DEFAULT_ENCODER, choices=sorted(ENCODERS))
    args = p.parse_args(argv)
    sets = list(args.sets)
    if args.corpus:
        import yaml
        sets += [c for c in yaml.safe_load(args.corpus.read_text())["sets"] if c not in sets]
    for code in sets:
        if not (PROCESSED_DIR / code / "cards.parquet").exists():
            print(f"{code}: no cards.parquet — skipped", flush=True)
            continue
        df = build_text_features(code, args.encoder)
        print(f"{code}: {df.shape[0]} cards × {df.shape[1]} dims ({args.encoder})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
