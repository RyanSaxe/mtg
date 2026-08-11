"""Draft-state reconstruction from follower events (HUD H1/H2).

Maps Arena grpIds (== Scryfall arena_id) into our per-set vocab, accumulates
pack/pick history, and exposes the (packs, prev_picks, position) arrays the
models consume. Position comes from COUNTING packs seen per draft — Arena's
reported pack/pick numbers vary in base across event kinds, so they are used
for display only.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from draftbot.data.cards import (PROCESSED_DIR, SCRYFALL_CACHE_DIR, name_to_id,
                                 norm_name, source_sets)
from draftbot.data.dataset import PAD
from draftbot.hud.follower import (DraftCompleted, DraftJoined, PackSeen,
                                   PickMade, SealedPool, set_code_from_event)


_GLOBAL_GRP_INDEX: dict[int, str] | None = None


def global_grp_index() -> dict[int, str]:
    """arena grpId → set code, across every onboarded set. Lets the HUD infer
    the set from pack contents alone — real Arena logs don't reliably emit a
    parseable join message (seen live 2026-07-29: bare-JSON Course lines)."""
    global _GLOBAL_GRP_INDEX
    if _GLOBAL_GRP_INDEX is None:
        idx: dict[int, str] = {}
        for pq in PROCESSED_DIR.glob("*/cards.parquet"):
            set_code = pq.parent.name
            for grp in grp_to_vocab(set_code):
                idx.setdefault(grp, set_code)
        _GLOBAL_GRP_INDEX = idx
    return _GLOBAL_GRP_INDEX


def infer_set(card_ids: list[int]) -> str | None:
    """Majority-vote the set from a pack's grpIds."""
    idx = global_grp_index()
    votes: dict[str, int] = {}
    for g in card_ids:
        code = idx.get(int(g))
        if code:
            votes[code] = votes.get(code, 0) + 1
    if not votes:
        return None
    best = max(votes, key=votes.get)  # type: ignore[arg-type]
    return best if votes[best] >= max(2, len(card_ids) // 2) else None


def grp_to_vocab(set_code: str) -> dict[int, int]:
    """arena grpId → our card id, via the cached Scryfall set JSONs, with
    Arena's own card database as the backstop.

    Scryfall is authoritative where it has an `arena_id`, but it backfills
    those days after a set goes live — on release day every card in the set
    has `arena_id: null` (observed for all 193 HOB + 103 HOC cards on
    2026-08-11), which would leave the HUD unable to read a single pack. The
    Arena client's local database always knows the grpIds it is about to
    write into Player.log, so it fills the gaps here (and, incidentally, the
    alternate printings Scryfall's `unique=cards` search collapses away)."""
    cards = pd.read_parquet(PROCESSED_DIR / set_code / "cards.parquet")
    nti = name_to_id(cards)
    mapping: dict[int, int] = {}
    sources = source_sets(set_code)
    for code in sources:
        cache = SCRYFALL_CACHE_DIR / f"{code.lower()}.json"
        if not cache.exists():
            continue
        for card in json.loads(cache.read_text()):
            arena = card.get("arena_id")
            vid = nti.get(norm_name(card["name"]))
            if arena and vid is not None:
                mapping[int(arena)] = vid
    for f in (SCRYFALL_CACHE_DIR / "named").glob("*.json"):
        card = json.loads(f.read_text())
        arena = card.get("arena_id")
        vid = nti.get(norm_name(card.get("name", "")))
        if arena and vid is not None:
            mapping.setdefault(int(arena), vid)
    mapping.update(_arena_db_grps(sources, nti, mapping))
    return mapping


def _arena_db_grps(sources: list[str], nti: dict[str, int],
                   known: dict[int, int]) -> dict[int, int]:
    """grpIds Scryfall could not supply, read from the local Arena client.

    Absent/unreadable is normal (CI, a machine without Arena) — the Scryfall
    mapping stands alone there, which is every set past its first week."""
    import sqlite3

    from draftbot.data import arena_db
    try:
        names = arena_db.grp_names(sources)
    except (FileNotFoundError, sqlite3.Error, OSError):
        return {}
    extra = {}
    for grp, name in names.items():
        vid = nti.get(norm_name(name))
        if vid is not None and grp not in known:
            extra[int(grp)] = vid
    return extra


@dataclass
class DraftState:
    t_max: int = 45
    pack_max: int = 15
    set_code: str | None = None
    draft_id: str | None = None
    event_name: str | None = None
    completed: bool = False
    packs: list[list[int]] = field(default_factory=list)   # vocab ids per step
    picks: list[int] = field(default_factory=list)
    unknown_grps: set[int] = field(default_factory=set)
    _grp_map: dict[int, int] | None = None

    @property
    def position(self) -> int:
        return len(self.packs) - 1  # current (last-seen) pack's step

    def _ensure_set(self, event_name: str | None):
        code = set_code_from_event(event_name)
        if code and code != self.set_code:
            self.set_code = code
            self._grp_map = grp_to_vocab(code)

    def _resolve_unknown(self, grp: int) -> int | None:
        """Alternate printings (e.g. second basic-land arts) carry arena ids
        Scryfall's unique=cards search collapses away — resolve by id via the
        /cards/arena endpoint (disk-cached), then map by name into the vocab."""
        import pandas as pd

        from draftbot.data.cards import (HEADERS, PROCESSED_DIR,
                                         SCRYFALL_CACHE_DIR, name_to_id, norm_name)
        cache = SCRYFALL_CACHE_DIR / "arena" / f"{grp}.json"
        if cache.exists():
            card = json.loads(cache.read_text())
        else:
            import requests
            try:
                resp = requests.get(f"https://api.scryfall.com/cards/arena/{grp}",
                                    headers=HEADERS, timeout=10)
            except requests.RequestException:
                return None
            if resp.status_code != 200:
                return None
            card = resp.json()
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(card))
        if not self.set_code:
            return None
        cards = pd.read_parquet(PROCESSED_DIR / self.set_code / "cards.parquet")
        vid = name_to_id(cards).get(norm_name(card.get("name", "")))
        if vid is not None and self._grp_map is not None:
            self._grp_map[grp] = vid
        return vid

    def _map(self, grp_ids: list[int]) -> list[int]:
        out = []
        for g in grp_ids:
            vid = (self._grp_map or {}).get(int(g))
            if vid is None:
                vid = self._resolve_unknown(int(g))
            if vid is None:
                self.unknown_grps.add(int(g))
            else:
                out.append(vid)
        return out

    def apply(self, ev) -> bool:
        """Consume one event; returns True if the pick-context changed."""
        if isinstance(ev, DraftJoined):
            self._ensure_set(ev.event_name)
            return False
        if isinstance(ev, DraftCompleted):
            self.completed = ev.draft_id == self.draft_id or self.completed
            return False
        if isinstance(ev, PackSeen):
            if ev.event_name:
                self._ensure_set(ev.event_name)
            if self._grp_map is None:  # no join message parsed — infer from pack
                code = infer_set(ev.card_ids)
                if code:
                    self.set_code = code
                    self._grp_map = grp_to_vocab(code)
            if self._grp_map is None:
                return False
            if ev.draft_id != self.draft_id:  # new draft begins
                self.draft_id = ev.draft_id
                self.packs, self.picks = [], []
                self.completed = False
            # replays/dedup: identical pack at same step arrives via multiple
            # message kinds — only append if it's genuinely the next step
            ids = self._map(ev.card_ids)
            if self.packs and len(self.picks) < len(self.packs):
                self.packs[-1] = ids  # refresh current step (dedup)
            else:
                self.packs.append(ids)
            return True
        if isinstance(ev, PickMade):
            if ev.draft_id != self.draft_id or not self.packs:
                return False
            if len(self.picks) < len(self.packs):
                vids = self._map(ev.grp_ids)
                if vids:
                    self.picks.append(vids[0])
            return True
        return False

    def arrays(self) -> tuple[np.ndarray, np.ndarray, int]:
        """(packs (1,k,P), prev_picks (1,k), current position index)."""
        k = len(self.packs)
        assert k > 0, "no pack seen yet"
        packs = np.full((1, k, self.pack_max), PAD, dtype=np.int64)
        for i, ids in enumerate(self.packs):
            packs[0, i, : len(ids)] = ids[: self.pack_max]
        prev = np.full((1, k), PAD, dtype=np.int64)
        for i, p in enumerate(self.picks[: k - 1]):
            prev[0, i + 1] = p
        return packs, prev, k - 1

    def pool_ids(self) -> list[int]:
        return list(self.picks)


_BASIC_IDS: dict[str, set[int]] = {}


def _basic_ids(set_code: str) -> set[int]:
    if set_code not in _BASIC_IDS:
        cards = pd.read_parquet(PROCESSED_DIR / set_code / "cards.parquet")
        _BASIC_IDS[set_code] = set(cards.loc[cards["is_basic"] == 1, "id"])
    return _BASIC_IDS[set_code]


@dataclass
class SealedState:
    """Sealed pool from the course messages (P5.S4). No picks to track — the
    pool arrives whole; basics are dropped per the pool-identity rule (the
    basics head owns the manabase). grp→vocab mapping reuses DraftState's
    resolver, including the /cards/arena variant-printing fallback."""

    set_code: str | None = None
    event_name: str | None = None
    course_id: str | None = None
    pool: list[int] = field(default_factory=list)   # vocab ids, nonbasics
    unknown_grps: set[int] = field(default_factory=set)
    _mapper: DraftState | None = None

    def apply(self, ev) -> bool:
        """Consume one event; True if the pool changed."""
        if not isinstance(ev, SealedPool):
            return False
        if ev.course_id and ev.course_id == self.course_id and self.pool:
            return False   # course re-announced (relaunch, module change)
        code = set_code_from_event(ev.event_name) or infer_set(ev.card_ids)
        if code is None:
            return False
        if self._mapper is None or code != self.set_code:
            self.set_code = code
            self._mapper = DraftState(set_code=code, _grp_map=grp_to_vocab(code))
        vids = self._mapper._map(ev.card_ids)
        self.unknown_grps |= self._mapper.unknown_grps
        basics = _basic_ids(code)
        pool = [v for v in vids if v not in basics]
        if not pool:
            return False
        self.pool = pool
        self.event_name, self.course_id = ev.event_name, ev.course_id
        return True
