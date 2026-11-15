"""Build the per-set card vocabulary (cards.parquet) from the CSV header + Scryfall.

Vocab rules (PLAN P0.T5):
- The universe of draftable cards for a set = exactly the names appearing as
  `pack_card_<name>` columns in the 17lands CSV (MSH: 339, basics included).
- Scryfall data comes from `set:<code>` UNION its companion sheets (registry below),
  with a per-name `/cards/named?exact=` fallback for anything still unmatched — this
  makes onboarding a new set work even before its companion sheet is registered.
- Card ids are assigned once, in sorted-name order, and persisted. If cards.parquet
  exists it is loaded and verified, never regenerated (id stability guardrail).
"""

import csv
import gzip
import json
import time
from pathlib import Path

import pandas as pd
import requests

from draftbot.data.fetch import dest_path

# Known companion/bonus sheets that put extra cards in a set's draft boosters.
COMPANION_SHEETS = {
    "MSH": ["MAR"],   # Marvel Universe masterpieces
    "BRO": ["BRR"],   # Retro artifacts
    "MOM": ["MUL"],   # Multiverse legends
    "WOE": ["WOT"],   # Enchanting tales
    "STX": ["STA"],   # Mystical archive
    "TDM": ["TDC"],   # (probe; harmless if empty)
    "FIN": ["FCA"],   # Through the Ages
    "EOE": ["EOS"],   # Stellar sights
    "LCI": ["SPG"], "MKM": ["SPG", "PLST"], "OTJ": ["OTP", "BIG", "SPG"], "MH3": ["SPG"],
    "BLB": ["SPG"], "DSK": ["SPG"], "FDN": ["SPG"], "DFT": ["SPG"],
    "TLA": ["SPG"], "ECL": ["SPG"],
}

# Written by draftbot.data.day0 next to a provisional (pre-17lands) vocab.
DAY0_MARKER = ".day0"
# Optional per-set override of the companion-sheet registry, written by day0
# from Scryfall's parent_set_code graph so a new set needs no code edit.
SOURCES_JSON = "sources.json"

SCRYFALL_SEARCH = "https://api.scryfall.com/cards/search"
SCRYFALL_NAMED = "https://api.scryfall.com/cards/named"
HEADERS = {"User-Agent": "draft-research-client/0.1", "Accept": "application/json"}

PROCESSED_DIR = Path("data/processed")
SCRYFALL_CACHE_DIR = Path("data/raw/scryfall")

BASICS = ["plains", "island", "swamp", "mountain", "forest"]

# Scryfall fields we persist per card (raw-ish; feature engineering happens later).
KEEP_FIELDS = [
    "name", "set", "collector_number", "rarity", "cmc", "mana_cost", "colors",
    "color_identity", "type_line", "power", "toughness", "keywords",
    "produced_mana", "layout", "oracle_text",
]


def norm_name(name: str) -> str:
    """Canonical matching key: front face, lowercase (matches original repo)."""
    return name.split("//")[0].strip().lower()


def csv_pack_names(set_code: str, event: str = "PremierDraft") -> list[str]:
    """The draftable-card universe: pack_card_ column names from the raw CSV."""
    raw = dest_path(set_code, event)
    with gzip.open(raw, "rt") as f:
        header = next(csv.reader(f))
    return [c[len("pack_card_"):] for c in header if c.startswith("pack_card_")]


def _scryfall_page(url: str, params=None) -> dict:
    for attempt in range(6):
        resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
        if resp.status_code == 429:
            time.sleep(3 * 2 ** attempt)
            continue
        if resp.status_code == 404:
            return {"data": [], "has_more": False}
        resp.raise_for_status()
        return resp.json()
    raise IOError(f"Scryfall gave repeated 429s for {url}")


def fetch_set_cards(set_code: str) -> list[dict]:
    """All Scryfall card objects for set:<code> (cached to data/raw/scryfall/)."""
    cache = SCRYFALL_CACHE_DIR / f"{set_code.lower()}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    cards, url = [], SCRYFALL_SEARCH
    params = {"q": f"set:{set_code.lower()}", "unique": "cards"}
    while url:
        page = _scryfall_page(url, params)
        cards.extend(page["data"])
        url = page.get("next_page") if page.get("has_more") else None
        params = None  # next_page URLs are self-contained
        time.sleep(0.11)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(cards))
    return cards


def fetch_named(name: str) -> dict | None:
    cache = SCRYFALL_CACHE_DIR / "named" / (norm_name(name).replace("/", "_") + ".json")
    if cache.exists():
        return json.loads(cache.read_text())
    time.sleep(0.25)  # politeness: the fallback can run for many names in a row
    page = _scryfall_page(SCRYFALL_NAMED, {"exact": name})
    if not page or "name" not in page:
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(page))
    return page


def _flatten(card: dict) -> dict:
    """Extract KEEP_FIELDS, merging front-face data for multi-faced cards."""
    faces = card.get("card_faces") or []
    front = faces[0] if faces else {}
    out = {}
    for f in KEEP_FIELDS:
        v = card.get(f)
        if v is None and front:
            v = front.get(f)
        if isinstance(v, list):
            v = ",".join(str(x) for x in v)
        out[f] = v
    out["is_flip"] = 1 if faces else 0
    return out


def source_sets(set_code: str) -> list[str]:
    """Set codes whose cards can appear in <SET>'s boosters, this set first.

    A `sources.json` written by day-0 onboarding wins over the static registry,
    so a newly-released set with a bonus sheet needs no code change here.
    """
    override = PROCESSED_DIR / set_code / SOURCES_JSON
    if override.exists():
        return json.loads(override.read_text())
    return [set_code] + COMPANION_SHEETS.get(set_code.upper(), [])


def _discard_day0_artifacts(set_code: str) -> None:
    """Drop everything keyed on the provisional day-0 card ids.

    The marker, the static feature table (`onboard()` rebuilds it only when
    ABSENT — a stale one would silently mis-index every feature row) and the
    all-NaN placeholder snapshot with its registry. `sources.json` stays: the
    Scryfall pool it names is still the right place to resolve CSV names.
    """
    from draftbot.data.features import SNAPSHOT_DIR  # lazy: features imports cards

    proc = PROCESSED_DIR / set_code
    for name in (DAY0_MARKER, "features.static.parquet"):
        (proc / name).unlink(missing_ok=True)
    reg = SNAPSHOT_DIR / set_code / "registry.json"
    if reg.exists():
        for fname in set(json.loads(reg.read_text()).values()):
            start, _, end = Path(fname).stem.partition("_")
            if start == end:  # day-0 placeholder: an empty [release, release] window
                (reg.parent / fname).unlink(missing_ok=True)
        reg.unlink()


def build_cards(set_code: str, event: str = "PremierDraft") -> pd.DataFrame:
    """Build (or load, if already built) data/processed/<SET>/cards.parquet."""
    out_path = PROCESSED_DIR / set_code / "cards.parquet"
    day0 = out_path.parent / DAY0_MARKER
    names = csv_pack_names(set_code, event)
    if out_path.exists() and not day0.exists():
        df = pd.read_parquet(out_path)
        assert list(df["name"]) == sorted(names), f"{out_path} inconsistent with CSV header"
        return df
    if day0.exists():
        # 17lands has published: the provisional day-0 vocab is superseded by
        # the CSV's authoritative universe. Ids change here — see day0's
        # docstring for why nothing may be carried across that boundary.
        _discard_day0_artifacts(set_code)

    pool: dict[str, dict] = {}
    for code in source_sets(set_code):
        for card in fetch_set_cards(code):
            pool.setdefault(norm_name(card["name"]), card)

    rows, missing = [], []
    for name in sorted(names):
        card = pool.get(norm_name(name))
        if card is None:
            card = fetch_named(name)
        if card is None and name.startswith("A-"):
            # Alchemy-rebalanced card: Scryfall has only the paper original.
            # Static features of the base card are the right approximation.
            card = pool.get(norm_name(name[2:])) or fetch_named(name[2:])
        if card is None:
            missing.append(name)
            continue
        row = _flatten(card)
        row["name"] = name  # canonical = CSV spelling (join key for all pipelines)
        rows.append(row)
    if missing:
        raise KeyError(f"{set_code}: {len(missing)} CSV names unresolved on Scryfall: {missing[:10]}")

    df = pd.DataFrame(rows)
    df.insert(0, "id", range(len(df)))
    df["is_basic"] = df["name"].str.lower().isin(BASICS).astype(int)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


def name_to_id(cards: pd.DataFrame) -> dict[str, int]:
    """Mapping that resolves CSV `pick` values (front-face, any case) to card ids."""
    return {norm_name(n): i for n, i in zip(cards["name"], cards["id"])}
