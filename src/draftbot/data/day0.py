"""Day-0 onboarding: make a brand-new set draftable on its release day.

The normal path (`draftbot.data.onboard`) derives the card vocabulary from the
`pack_card_*` columns of the 17lands CSV and the stat features from that set's
public draft/game files. On release day neither exists — and Scryfall has not
yet backfilled `arena_id`, so even the HUD's grpId bridge is missing. This is
the gap PLAN §Phase 3 always pointed at: *"On next-set release day, the day-0
claim gets tested for real — that's the ultimate eval."*

This module builds the same artifacts from the two sources that DO exist on
day 0:

- **Arena's local card database** (`draftbot.data.arena_db`) → the draftable
  grpIds and their names, straight out of the client that is about to write
  those grpIds into Player.log.
- **Scryfall** → the card objects behind those names, hence the static
  feature table (cmc/pips/types/rarity/keywords — all release-day available).

Stats come out all-NaN, which `apply_scaler(add_masks=True)` turns into zeros
with every missing-mask flipped on. That is not a degraded mode: it is exactly
the stat-dropout condition the trunk was pretrained under (PLAN P3), so run
the HUD with `--stats none` and you are on the trained-for zero-shot path.

    uv run python -m draftbot.data.day0 --set HOB

Everything written here is PROVISIONAL. A `.day0` marker lands next to
cards.parquet, and `draftbot.data.cards.build_cards` regenerates the vocab
from the CSV (dropping the marker) the moment 17lands publishes. Card ids
change at that point, so nothing trained on day-0 ids may outlive it — refresh
the set the normal way and rebuild the bundle.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from draftbot.data import arena_db
from draftbot.data.cards import (BASICS, DAY0_MARKER, HEADERS, PROCESSED_DIR,
                                 SCRYFALL_CACHE_DIR, SOURCES_JSON, _flatten,
                                 fetch_named, fetch_set_cards, norm_name)
from draftbot.data.features import (SNAPSHOT_DIR, _registry_path,
                                    build_static_features, load_snapshot,
                                    set_release_date)

SCRYFALL_SETS = "https://api.scryfall.com/sets"


def _all_sets() -> list[dict]:
    """Scryfall's set list, disk-cached (the parent_set_code graph lives here)."""
    cache = SCRYFALL_CACHE_DIR / "sets.json"
    if cache.exists():
        return json.loads(cache.read_text())
    import requests
    resp = requests.get(SCRYFALL_SETS, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    sets = resp.json()["data"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(sets))
    return sets


def discover_sources(set_code: str, db: Path | None = None,
                     extra: list[str] | None = None) -> list[str]:
    """Set codes that can appear in <SET>'s boosters, this set first.

    Scryfall's `parent_set_code` names the bonus sheets (HOB → HOC); we keep
    only children Arena has actually downloaded, so a paper-only supplement
    never widens the vocab. `extra` is the manual escape hatch.
    """
    code = set_code.upper()
    present = arena_db.expansion_codes(db)
    # Only masterpiece-type children are booster bonus sheets (BRR, EOS, FCA,
    # MUL, STA, WOT…). A set's Commander / "Eternal" companion (LTC, HOC) is
    # also a parent_set_code child and Arena downloads it for Brawl — but none
    # of its cards is ever in a draft pack (HOB's 17lands CSV: 193 names, zero
    # HOC; the day-0 vocab had carried 103 of them).
    children = [s["code"].upper() for s in _all_sets()
                if (s.get("parent_set_code") or "").upper() == code
                and s.get("set_type") == "masterpiece"]
    out = [code]
    for c in children + [e.upper() for e in (extra or [])]:
        if c not in out and c in present:
            out.append(c)
    return out


def build_day0_cards(set_code: str, names: list[str],
                     sources: list[str]) -> pd.DataFrame:
    """cards.parquet from a name list instead of a 17lands CSV header.

    Same schema, same id rule (sorted-name order) as `cards.build_cards`, so
    every downstream consumer is unchanged. Names that Scryfall cannot resolve
    are dropped and reported — a dropped card simply never gets ranked, where a
    hard failure would cost the whole set.
    """
    pool: dict[str, dict] = {}
    for code in sources:
        for card in fetch_set_cards(code):
            pool.setdefault(norm_name(card["name"]), card)

    rows, missing = [], []
    for name in sorted(set(names)):
        card = pool.get(norm_name(name)) or fetch_named(name)
        if card is None:
            missing.append(name)
            continue
        row = _flatten(card)
        row["name"] = name  # canonical = Arena's spelling (front face, as 17lands)
        rows.append(row)
    if missing:
        print(f"{set_code}: {len(missing)} names unresolved on Scryfall, dropped: "
              f"{missing[:10]}{' …' if len(missing) > 10 else ''}", file=sys.stderr)

    df = pd.DataFrame(rows)
    df.insert(0, "id", range(len(df)))
    df["is_basic"] = df["name"].str.lower().isin(BASICS).astype(int)
    out_dir = PROCESSED_DIR / set_code
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "cards.parquet", index=False)
    (out_dir / SOURCES_JSON).write_text(json.dumps(sources))
    (out_dir / DAY0_MARKER).write_text(
        "provisional vocab from draftbot.data.day0 — rebuilt from the 17lands "
        "CSV (with new card ids) as soon as that data exists\n")
    return df


def _template_columns(exclude: str) -> list[str]:
    """Stat-column layout, borrowed from a set that has a REAL snapshot.

    The columns are set-independent (4 draft-stage + the per-color-filter game
    blocks), so any normally-onboarded set defines the shape the corpus scaler
    expects. Day-0 sets are skipped — including `exclude`, which on a re-run
    would otherwise let this set seed its own layout.
    """
    for reg in sorted(SNAPSHOT_DIR.glob("*/registry.json")):
        code = reg.parent.name
        if code == exclude or (PROCESSED_DIR / code / DAY0_MARKER).exists():
            continue
        if "full" in json.loads(reg.read_text()):
            return list(load_snapshot(code, "full").columns)
    raise SystemExit(
        "no normally-onboarded set with a 'full' snapshot to take the "
        "stat-column layout from — untar laptop-bundle.tar.gz first")


def build_statless_snapshot(set_code: str) -> pd.DataFrame:
    """An all-NaN stat snapshot: 'what 17lands knows on release day'.

    Registered under both `full` and `week1` so every stats mode degrades to
    the day-0 condition instead of raising. `apply_scaler` maps NaN → 0 with
    the paired missing-mask on, which is what the model was trained to read.
    """
    cards = pd.read_parquet(PROCESSED_DIR / set_code / "cards.parquet")
    snap = pd.DataFrame(float("nan"), columns=_template_columns(set_code),
                        index=[norm_name(n) for n in cards["name"]])
    rel = set_release_date(set_code)
    out_dir = SNAPSHOT_DIR / set_code
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{rel}_{rel}.parquet"  # empty window = no data yet
    snap.to_parquet(out_path)
    _registry_path(set_code).write_text(
        json.dumps({"full": out_path.name, "week1": out_path.name}, indent=2))
    return snap


def onboard_day0(set_code: str, db: Path | None = None,
                 extra: list[str] | None = None) -> pd.DataFrame:
    code = set_code.upper()
    db = db or arena_db.find_db()
    sources = discover_sources(code, db, extra)
    grps = arena_db.grp_names(sources, db)
    if not grps:
        raise SystemExit(
            f"{code}: no cards in Arena's database at {db} — has the client "
            "finished downloading the set? (check with expansion_codes())")
    cards = build_day0_cards(code, list(grps.values()), sources)
    build_static_features(code)
    build_statless_snapshot(code)
    try:  # oracle-text block for text-trained trunks (optional extra [text])
        from draftbot.data.text_features import build_text_features
        build_text_features(code)
    except ImportError:
        print(f"{code}: sentence-transformers not installed — no text block "
              "(only text-trained checkpoints need it)", file=sys.stderr)
    print(f"{code}: {len(cards)} cards, {len(grps)} grpIds, "
          f"sources {'+'.join(sources)} — day-0 artifacts written")
    print(f"  next: uv run python -m draftbot.bundle --set {code} --set MSH")
    print(f"        uv run python -m draftbot.hud --ui window --stats none")
    return cards


def main(argv=None):
    p = argparse.ArgumentParser(prog="draftbot.data.day0")
    p.add_argument("--set", dest="set_code", required=True)
    p.add_argument("--arena-db", type=Path, default=None,
                   help=f"path to Raw_CardDatabase_*.mtga (default: newest under "
                        f"{arena_db.DEFAULT_ROOT})")
    p.add_argument("--include", action="append", default=None,
                   help="extra bonus-sheet set code (repeatable)")
    args = p.parse_args(argv)
    onboard_day0(args.set_code, args.arena_db, args.include)
    return 0


if __name__ == "__main__":
    sys.exit(main())
