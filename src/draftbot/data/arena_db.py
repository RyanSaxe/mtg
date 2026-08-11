"""Arena's local card database — the day-0 grpId → name bridge.

The Arena client ships a SQLite card database next to its downloads:

    ~/Library/Application Support/com.wizards.mtga/Downloads/Raw/
        Raw_CardDatabase_<hash>.mtga

`Cards` carries GrpId / ExpansionCode / TitleId; `Localizations_enUS` resolves
TitleId → the English display name (`Formatted = 1` is the display variant).
Together that is grpId → name for every set the client has downloaded —
available the hour a set goes live, i.e. weeks before 17lands publishes a
draft CSV and days before Scryfall backfills `arena_id` (verified 2026-08-11:
all 193 HOB + 103 HOC cards had `arena_id` null while the set was drafted).

Names are front-face only ("Bruce Banner", not "Bruce Banner // Hulk"), which
is exactly the convention `data/processed/<SET>/cards.parquet` already uses.

Read-only throughout: we open with `mode=ro` and never write to Arena's files.
"""

import re
import sqlite3
from pathlib import Path

DEFAULT_ROOT = (Path.home() / "Library/Application Support/com.wizards.mtga"
                / "Downloads/Raw")
DB_GLOB = "Raw_CardDatabase_*.mtga"

# Localization strings carry display markup: Arena wraps hyphenated names to
# stop them breaking across lines ("<nobr>Lake-town</nobr> Lookout").
_MARKUP = re.compile(r"<[^>]+>")


def find_db(root: Path | str | None = None) -> Path:
    """Newest Raw_CardDatabase_*.mtga under `root` (Arena rewrites it on patch)."""
    root = Path(root) if root is not None else DEFAULT_ROOT
    dbs = sorted(root.glob(DB_GLOB), key=lambda p: p.stat().st_mtime)
    if not dbs:
        raise FileNotFoundError(
            f"no {DB_GLOB} under {root} — Arena must be installed on this "
            "machine and have downloaded the set")
    return dbs[-1]


def _connect(db: Path | str | None) -> sqlite3.Connection:
    path = Path(db) if db is not None else find_db()
    return sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)


def expansion_codes(db: Path | str | None = None) -> dict[str, int]:
    """ExpansionCode → non-token card count. Diagnostic: has the client got the set?"""
    with _connect(db) as con:
        rows = con.execute(
            "SELECT ExpansionCode, COUNT(*) FROM Cards WHERE IsToken = 0 "
            "GROUP BY ExpansionCode").fetchall()
    return {code: n for code, n in rows if code}


def grp_names(set_codes: list[str], db: Path | str | None = None) -> dict[int, str]:
    """arena grpId → English card name, over the given expansion codes.

    Filter is `IsToken = 0 AND IsPrimaryCard = 1` — deliberately NOT
    `DraftContent = 1`. That column looked like the right one until two sets
    contradicted it: basics carry `DraftContent = 0` (never opened in a pack,
    but pool-legal and allocated by the deck builder), and so does every one
    of HOC's 104 cards. Validated against MSH on 2026-08-11: this filter
    covers 339/339 of the 17lands-derived vocab, at the cost of 36 bonus-sheet
    cards that never reach a pack. That trade is the right way round — an
    over-wide vocab costs nothing at inference, a missing card is a hole in
    the pack the model cannot rank.
    """
    codes = [c.upper() for c in set_codes]
    sql = (
        "SELECT c.GrpId, l.Loc FROM Cards c "
        "JOIN Localizations_enUS l ON l.LocId = c.TitleId AND l.Formatted = 1 "
        f"WHERE c.ExpansionCode IN ({','.join('?' * len(codes))}) "
        "AND c.IsToken = 0 AND c.IsPrimaryCard = 1"
    )
    with _connect(db) as con:
        rows = con.execute(sql, codes).fetchall()
    return {int(grp): _MARKUP.sub("", name).strip() for grp, name in rows if name}
