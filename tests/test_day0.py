"""Day-0 onboarding (draftbot.data.day0 + arena_db): make a set released today
draftable before 17lands publishes anything about it.

The Arena-DB tests are hermetic — they build a miniature Raw_CardDatabase and
assert the two filters that actually bit during the HOB rollout. The artifact
tests are gated on a day-0 set existing locally, like the rest of the suite.
"""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from draftbot.data import arena_db
from draftbot.data.cards import DAY0_MARKER, PROCESSED_DIR, SOURCES_JSON, norm_name

# --- hermetic: Arena's card database ---------------------------------------

ROWS = [
    # grp,   title, expansion, token, primary, draftContent
    (1001, "Bilbo, Thief in the Night", "HOB", 0, 1, 1),
    (1002, "<nobr>Lake-town</nobr> Lookout", "HOB", 0, 1, 1),  # display markup
    (1003, "Forest", "HOB", 0, 1, 0),                          # basic: DraftContent 0
    (1004, "Andúril, Flame of the West", "HOC", 0, 1, 0),      # bonus sheet: all 0
    (1005, "Food", "HOB", 1, 0, 0),                            # token
    (1006, "At the Door", "HOB", 0, 0, 0),                     # DFC back face
    (2001, "Lightning Strike", "MSH", 0, 1, 1),                # other set
]


@pytest.fixture
def fake_db(tmp_path) -> Path:
    db = tmp_path / "Raw_CardDatabase_deadbeef.mtga"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE Cards (GrpId INT, TitleId INT, ExpansionCode TEXT, "
                "IsToken BOOLEAN, IsPrimaryCard BOOLEAN, DraftContent BOOLEAN)")
    con.execute("CREATE TABLE Localizations_enUS (LocId INT, Formatted INT, Loc TEXT)")
    for grp, title, exp, token, primary, draft in ROWS:
        con.execute("INSERT INTO Cards VALUES (?,?,?,?,?,?)",
                    (grp, grp, exp, token, primary, draft))
        con.execute("INSERT INTO Localizations_enUS VALUES (?,?,?)", (grp, 1, title))
        con.execute("INSERT INTO Localizations_enUS VALUES (?,?,?)",
                    (grp, 0, f"raw::{title}"))  # non-display variant must be ignored
    con.commit()
    con.close()
    return db


def test_grp_names_strips_display_markup(fake_db):
    """Arena wraps hyphenated names in <nobr> to stop mid-word line breaks;
    unstripped they miss Scryfall entirely (26 HOB names dropped before this)."""
    names = arena_db.grp_names(["HOB"], fake_db)
    assert names[1002] == "Lake-town Lookout"
    assert not any("<" in n for n in names.values())


def test_grp_names_keeps_zero_draftcontent_rows(fake_db):
    """DraftContent is NOT the draftability signal: basics carry 0, and so does
    every card on HOC's bonus sheet. Filtering on it silently lost 104 cards."""
    names = arena_db.grp_names(["HOB", "HOC"], fake_db)
    assert names[1003] == "Forest"
    assert names[1004] == "Andúril, Flame of the West"


def test_grp_names_excludes_tokens_backfaces_and_other_sets(fake_db):
    names = arena_db.grp_names(["HOB", "HOC"], fake_db)
    assert set(names) == {1001, 1002, 1003, 1004}


def test_expansion_codes_counts_non_tokens(fake_db):
    assert arena_db.expansion_codes(fake_db) == {"HOB": 4, "HOC": 1, "MSH": 1}


def test_find_db_picks_newest(tmp_path):
    old, new = tmp_path / "Raw_CardDatabase_a.mtga", tmp_path / "Raw_CardDatabase_b.mtga"
    old.write_bytes(b""), new.write_bytes(b"")
    import os
    os.utime(old, (1, 1))
    assert arena_db.find_db(tmp_path) == new


def test_find_db_without_arena_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Arena must be installed"):
        arena_db.find_db(tmp_path)


# --- day-0 artifacts (gated on a local day-0 set) ---------------------------

DAY0_SETS = [p.parent.name for p in PROCESSED_DIR.glob(f"*/{DAY0_MARKER}")] \
    if PROCESSED_DIR.exists() else []
needs_day0 = pytest.mark.skipif(not DAY0_SETS, reason="no day-0 set onboarded locally")
CODE = DAY0_SETS[0] if DAY0_SETS else "HOB"


@pytest.fixture(scope="module")
def cards():
    return pd.read_parquet(PROCESSED_DIR / CODE / "cards.parquet")


@needs_day0
def test_vocab_has_stable_sorted_ids(cards):
    """Same id rule as the CSV path — downstream code cannot tell them apart."""
    assert list(cards["id"]) == list(range(len(cards)))
    assert list(cards["name"]) == sorted(cards["name"])
    assert cards["is_basic"].sum() == 5
    assert not cards["name"].str.contains("<").any()


@needs_day0
def test_static_features_match_a_normally_onboarded_set(cards):
    """The scaler is corpus-fit: a column that differs here is a silent
    mis-scaling at inference, not an error."""
    day0 = pd.read_parquet(PROCESSED_DIR / CODE / "features.static.parquet")
    msh = pd.read_parquet(PROCESSED_DIR / "MSH" / "features.static.parquet")
    assert list(day0.columns) == list(msh.columns)
    assert len(day0) == len(cards)


@needs_day0
def test_snapshot_is_all_nan_over_the_real_stat_layout(cards):
    from draftbot.data.features import load_snapshot

    snap = load_snapshot(CODE, "full")
    assert list(snap.columns) == list(load_snapshot("MSH", "full").columns)
    assert snap.isna().all().all(), "day-0 stats must be missing, never zero"
    assert len(snap) == len(cards)


@needs_day0
def test_both_stat_tags_registered():
    """`week1` must resolve too — otherwise `--stats week1` raises KeyError on
    a set whose first week has not happened yet."""
    reg = json.loads((Path("data/snapshots") / CODE / "registry.json").read_text())
    assert {"full", "week1"} <= set(reg)


@needs_day0
def test_day0_stats_present_as_masked_not_as_zero_winrate():
    """The whole claim: NaN stats become 0 with every missing-mask ON, which is
    the stat-dropout condition the trunk was pretrained under (PLAN P3)."""
    from draftbot.data.corpus import _statless
    from draftbot.data.features import assemble, load_scaler

    scaler = load_scaler(Path("checkpoints/EXP-033/scaler.json"))
    feats, _ = assemble(CODE, "full", scaler=scaler)
    masks = [c for c in feats.columns if c.endswith("__miss")]
    assert masks, "stat block missing entirely"
    assert (feats[masks] == 1.0).all().all()
    statless = _statless(feats)
    assert np.allclose(statless[[c.removesuffix("__miss") for c in masks]].to_numpy(), 0.0)


@needs_day0
def test_sources_json_overrides_the_companion_registry():
    """HOB's bonus sheet (HOC) is in no hardcoded table — it was discovered from
    Scryfall's parent_set_code graph and must survive into every consumer."""
    from draftbot.data.cards import source_sets

    sources = json.loads((PROCESSED_DIR / CODE / SOURCES_JSON).read_text())
    assert sources[0] == CODE
    assert source_sets(CODE) == sources


@needs_day0
def test_every_arena_grpid_resolves_to_the_vocab(cards):
    """The day-0 bridge itself: Scryfall has no arena_id for a set this new, so
    all of these come from the local Arena DB."""
    from draftbot.hud.state import grp_to_vocab

    mapping = grp_to_vocab(CODE)
    assert set(mapping.values()) == set(cards["id"]), "vocab id unreachable from a grpId"
    names = arena_db.grp_names(json.loads(
        (PROCESSED_DIR / CODE / SOURCES_JSON).read_text()))
    nti = {norm_name(n): i for n, i in zip(cards["name"], cards["id"])}
    unmapped = [n for g, n in names.items() if g not in mapping and norm_name(n) in nti]
    assert not unmapped, f"known cards with no grpId route: {unmapped[:5]}"
