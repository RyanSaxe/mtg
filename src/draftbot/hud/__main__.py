"""Terminal HUD (H2): live model rankings alongside an Arena draft.

Live:    uv run python -m draftbot.hud --ui window
Replay:  uv run python -m draftbot.hud --replay path/to/Player.log

Production checkpoints (owner sign-off 2026-07-31, see DECKBUILDER_HANDOFF.md
"Final model selection"): the corpus-pretrained trunks, not the best in-set
scorers. EXP-013 (draft) and EXP-121 (deck) measure marginally higher ON MSH,
but the corpus trunks carry day-1 zero-shot on an unseen set and the
onboard-set fine-tune path — generalization is what the extra machinery buys,
and a HUD that only works on one set is not a HUD.
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from draftbot.hud.advisor import Advisor
from draftbot.hud.follower import DEFAULT_LOG, LogFollower, PackSeen, SealedPool
from draftbot.hud.state import DraftState, SealedState

RARITY_STYLE = {"mythic": "bold orange1", "rare": "gold3",
                "uncommon": "grey70", "common": "white"}
PIPS = "WUBRG"

# production defaults — see the module docstring for why these and not the
# best MSH scorers. EXP-116 stays the rebuild slot: it is the only builder
# with diffusion=True, i.e. the only one that conditions on locks natively.
# 2026-09-06: EXP-056 — the oracle-text trunk (bge-base block + embedding
# noise σ2, P3.T5c). HOB zero-shot expert top-1 .5675/.6869/.6930 vs
# EXP-033 .5387/.6801/.6855; passes all 17 fixtures at day 0. Needs
# features.text.bge-base.parquet for the set (bundle ships it; day-0 builds
# it with the [text] extra). EXP-033 remains the no-text fallback.
# 2026-11-15: EXP-061 — the same recipe on pretrain_v3 (HOB + FRA promoted,
# TRK held out); gate passed (journal 2026-11-15).
PROD_DRAFT_MODELS = "checkpoints/EXP-061"
PROD_DECK_MODELS = "checkpoints/EXP-126,checkpoints/EXP-116"
# sealed mode (P5.S4): the joint trunk with the sealed format token. No
# diffusion slot — locks go through the probability-pinning fallback, which
# keeps the sealed-aware model in charge rather than the draft-trained EXP-116.
PROD_SEALED_DECK_MODELS = "checkpoints/EXP-135"


def render(state: DraftState, ranked: dict, pips: list[int]) -> Table:
    pos = state.position
    pack_no, pick_no = pos // 14 + 1, pos % 14 + 1
    primary_name = next(iter(ranked))
    primary = ranked[primary_name]
    others = {n: r for n, r in ranked.items() if n != primary_name}

    title = f"P{pack_no}P{pick_no} · {state.set_code} · {len(state.picks)} picks"
    if others:
        for name, r in others.items():
            if r[0].card_id != primary[0].card_id:
                title += f"  ⚠ {name.split('@')[0]} prefers {r[0].name}"
    table = Table(title=title, expand=True, header_style="dim")
    table.add_column("#", width=3)
    table.add_column("card")
    table.add_column("conf", width=26)
    table.add_column("GIH", width=5)
    table.add_column("ALSA", width=5)
    for s in primary:
        bar = "▓" * max(1, int(s.prob * 24)) if s.prob >= 0.005 else "·"
        style = RARITY_STYLE.get(s.rarity, "white")
        emphasis = "bold " if s.rank == 1 else ""
        table.add_row(
            str(s.rank), Text(s.name, style=emphasis + style),
            f"{bar} {s.prob * 100:4.1f}%",
            f"{s.gih:.3f}" if s.gih == s.gih and s.gih is not None else "—",
            f"{s.alsa:.1f}" if s.alsa == s.alsa and s.alsa is not None else "—",
        )
    pip_str = "  ".join(f"{c}:{n}" for c, n in zip(PIPS, pips) if n)
    table.caption = f"pool pips  {pip_str or '—'}"
    return table


def pool_pips(state: DraftState, set_code: str) -> list[int]:
    import pandas as pd
    static = pd.read_parquet(f"data/processed/{set_code}/features.static.parquet")
    pip_cols = static[[f"pips_{c}" for c in PIPS]].to_numpy()
    total = [0] * 5
    for cid in state.pool_ids():
        for i in range(5):
            total[i] += int(pip_cols[cid, i])
    return total


def snapshot(state: DraftState, ranked: dict, pips: list[int], stats: str,
             curve: list[int]) -> dict:
    """JSON state for the window panel (shape consumed by panel.html)."""
    pos = state.position
    names = list(ranked)
    primary = ranked[names[0]]
    alt = ranked[names[1]] if len(names) > 1 else None
    return {
        "status": "live",
        "pos_label": f"P{pos // 14 + 1}P{pos % 14 + 1}",
        "picks_count": len(state.picks),
        "pool_pips": pips,
        "curve": curve,
        "stats_mode": stats,
        "status_line": f"{state.set_code} · {stats} stats",
        "version": "v0.1",
        "suggestions": [
            {"card_id": s.card_id, "name": s.name, "rarity": s.rarity,
             "prob": s.prob,
             "gih": None if s.gih is None or s.gih != s.gih else s.gih,
             "alsa": None if s.alsa is None or s.alsa != s.alsa else s.alsa,
             "art": f"/art/{s.card_id}"}
            for s in primary],
        "alt_top1": None if alt is None else {
            "card_id": alt[0].card_id, "name": alt[0].name,
            "disagrees": alt[0].card_id != primary[0].card_id},
    }


def draft_finished(state: DraftState) -> bool:
    """Arena does not reliably emit Draft_CompleteDraft (absent from the
    captured 2026-07-29 fixture), so the pack shape is the dependable signal:
    a one-card pack ends EVERY pack, so it only means "draft over" once all
    three boosters have been seen. Pack size comes from the draft itself —
    MSH play boosters are 14, other sets differ."""
    if state.completed:
        return True
    if not state.packs:
        return False
    size = max(len(p) for p in state.packs) or 1
    return (len(state.packs) >= 3 * size
            and len(state.packs[-1]) <= 1
            and len(state.picks) >= len(state.packs))


def pool_curve(state: DraftState, set_code: str) -> list[int]:
    import pandas as pd
    static = pd.read_parquet(f"data/processed/{set_code}/features.static.parquet")
    cmc = static["cmc"].to_numpy()
    land = static["type_land"].to_numpy()
    buckets = [0] * 7
    for cid in state.pool_ids():
        if land[cid]:
            continue
        buckets[min(6, int(cmc[cid]))] += 1
    return buckets


def make_deck_advisor(args, set_code: str):
    """DeckAdvisor for --deck-models, or None if the checkpoints are absent."""
    from draftbot.hud.deck import DeckAdvisor

    paths = [Path(p.strip()) for p in args.deck_models.split(",") if p.strip()]
    paths = [p for p in paths if p.exists()]
    if not paths:
        return None
    return DeckAdvisor(set_code=set_code, stats=args.stats,
                       primary_ckpt=paths[0],
                       rebuild_ckpt=paths[1] if len(paths) > 1 else None)


def make_sealed_advisor(args, set_code: str):
    """Sealed-mode DeckAdvisor (format token = sealed), or None if absent."""
    from draftbot.hud.deck import DeckAdvisor

    paths = [Path(p.strip()) for p in args.sealed_deck_models.split(",")
             if p.strip()]
    paths = [p for p in paths if p.exists()]
    if not paths:
        return None
    return DeckAdvisor(set_code=set_code, stats=args.stats,
                       primary_ckpt=paths[0],
                       rebuild_ckpt=paths[1] if len(paths) > 1 else None,
                       format_id=1)


def sealed_snapshot(sealed: SealedState, stats: str) -> dict:
    """Minimal /state payload for sealed mode — the deck section carries the
    content; this keeps the header/status/footer coherent."""
    return {
        "status": "live", "pos_label": "SEALED",
        "picks_count": len(sealed.pool),
        "pool_pips": [0] * 5, "curve": [0] * 7,
        "stats_mode": stats,
        "status_line": f"{sealed.set_code} sealed · {stats} stats",
        "version": "v0.1", "suggestions": [], "alt_top1": None,
    }


def sealed_action(action: str, payload: dict, adv, sealed: SealedState):
    """Panel POSTs in sealed mode — same lock verbs, whole-pool rebuilds."""
    if action == "lock":
        adv.set_lock(int(payload["card_id"]), payload.get("mode"))
    elif action == "clear_locks":
        adv.clear_locks()
    return adv.build(sealed.pool, provisional=False, keep_diff=True)


def refresh_deck(server, adv, state: DraftState, args, built: int,
                 final: bool = False) -> int:
    """Rebuild the deck section when the pool moved on. Returns the pick count
    the published build reflects. `final` forces the non-provisional view (a
    replay's log ending is itself proof the draft is over)."""
    n = len(state.picks)
    if adv is None or not adv.loaded or (n == built and not final):
        return built
    finished = final or draft_finished(state)
    if not finished and n < args.provisional_from:
        return built
    server.update_deck(adv.build(state.pool_ids(), provisional=not finished))
    return n


def deck_action(action: str, payload: dict, adv, state: DraftState):
    """Panel POSTs — lock toggles and rebuild — served on the HTTP thread."""
    if action == "lock":
        adv.set_lock(int(payload["card_id"]), payload.get("mode"))
    elif action == "clear_locks":
        adv.clear_locks()
    return adv.build(state.pool_ids(), provisional=not draft_finished(state),
                     keep_diff=True)


def render_deck(d: dict) -> Table:
    """Terminal v0 rendering of the proposed build."""
    head = f"{d['n_cards']} cards · {d['n_lands']} lands"
    if d.get("provisional"):
        head = "provisional · " + head
    table = Table(title=f"{head} · {d['model_id']}", expand=True,
                  header_style="dim")
    table.add_column("", width=3)
    table.add_column("card")
    table.add_column("conf", width=16)
    for g in d["groups"]:
        table.add_row("", Text(g["label"], style="dim"), "")
        for c in g["cards"]:
            table.add_row(f"{c['count']}×", Text(c["name"], style=RARITY_STYLE.get(
                c["rarity"], "white")),
                f"{'▓' * max(1, int(c['conf'] * 14))} {c['conf']:.2f}")
    for side, rows in (("in", d["boundary"]["in"]), ("out", d["boundary"]["out"])):
        for c in rows:
            table.add_row(f"{c['count']}×" if side == "in" else "—",
                          Text(c["name"], style="magenta"),
                          f"{c['conf']:.2f} {side}")
    lands = "  ".join(f"{n} {nm}" for nm, n in
                      zip(d["basic_names"], d["basics"]) if n)
    extra = ", ".join(c["name"] for c in d["nonbasic_lands"])
    table.caption = f"mana  {lands}" + (f" + {extra}" if extra else "")
    return table


def main(argv=None):
    ap = argparse.ArgumentParser(prog="draftbot.hud")
    ap.add_argument("--models", default=PROD_DRAFT_MODELS,
                    help=f"draft scorer(s), comma-separated (default {PROD_DRAFT_MODELS})")
    ap.add_argument("--stats", default="full", choices=["none", "week1", "full"])
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--replay", type=Path, default=None)
    ap.add_argument("--ui", default="terminal", choices=["terminal", "window"])
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--replay-delay", type=float, default=0.0,
                    help="seconds per pick when replaying into the window UI")
    ap.add_argument("--deck-models", default=PROD_DECK_MODELS,
                    help=f"one-shot builder[,diffusion builder for lock+rebuild] "
                         f"(default {PROD_DECK_MODELS})")
    ap.add_argument("--sealed-deck-models", default=PROD_SEALED_DECK_MODELS,
                    help=f"builder for sealed pools found in the log "
                         f"(default {PROD_SEALED_DECK_MODELS})")
    ap.add_argument("--no-deck", action="store_true",
                    help="skip the deck-builder section (HUD_PLAN H5)")
    ap.add_argument("--provisional-from", type=int, default=30,
                    help="pick number from which a provisional build is offered")
    args = ap.parse_args(argv)
    if args.ui == "window":
        return window_main(args)

    console = Console()
    state = DraftState()
    sealed = SealedState()
    advisor = None
    scorers = None
    follower = LogFollower(args.replay or args.log, replay=args.replay is not None)
    console.print(f"[dim]following {follower.path}…[/dim]")

    with Live(console=console, auto_refresh=False) as live:
        for ev in follower.events():
            if isinstance(ev, SealedPool):
                if sealed.apply(ev) and not args.no_deck:
                    adv = make_sealed_advisor(args, sealed.set_code)
                    if adv is not None:
                        console.print(render_deck(adv.build(sealed.pool)))
                continue
            changed = state.apply(ev)
            if not changed or not isinstance(ev, PackSeen):
                continue
            if scorers is None and state.set_code:
                from draftbot.eval.__main__ import resolve_scorer
                scorers = [resolve_scorer(m.strip(), state.set_code, args.stats)
                           for m in args.models.split(",")]
                advisor = Advisor(scorers, state.set_code)
            if advisor is None:
                continue
            ranked = advisor.rank_pack(state)
            live.update(render(state, ranked, pool_pips(state, state.set_code)),
                        refresh=True)
    if not args.no_deck and state.set_code and draft_finished(state):
        deck_adv = make_deck_advisor(args, state.set_code)
        if deck_adv is not None:
            console.print(render_deck(deck_adv.build(state.pool_ids())))
    return 0


def window_main(args):
    import threading
    import time
    import webbrowser

    from draftbot.hud.server import HudServer, art_url_map

    server = HudServer(args.port)
    url = server.start()
    print(f"HUD panel: {url}")
    opened = False
    try:  # always-on-top native window if pywebview is installed
        import webview  # noqa: F401
        threading.Thread(target=lambda: (webview.create_window(
            "Draft Buddy", url, width=400, height=680, on_top=True),
            webview.start()), daemon=True).start()
        opened = True
    except ImportError:
        pass
    if not opened:
        webbrowser.open(url)

    state = DraftState()
    sealed = SealedState()
    advisor = None
    scorers = None
    deck_adv = None
    sealed_adv = None
    built = -1
    follower = LogFollower(args.replay or args.log, replay=args.replay is not None)
    for ev in follower.events():
        if isinstance(ev, SealedPool):
            if sealed.apply(ev) and not args.no_deck:
                if sealed_adv is None or sealed_adv.set_code != sealed.set_code:
                    sealed_adv = make_sealed_advisor(args, sealed.set_code)
                    if sealed_adv is not None:
                        server.set_action_handler(
                            lambda a, p: sealed_action(a, p, sealed_adv, sealed))
                        server.set_art_urls(art_url_map(sealed.set_code))
                if sealed_adv is not None:
                    sealed_adv.preload()
                    server.update(sealed_snapshot(sealed, args.stats))
                    server.update_deck(
                        sealed_adv.build(sealed.pool, provisional=False))
            continue
        changed = state.apply(ev)
        if state.set_code and deck_adv is None and not args.no_deck:
            deck_adv = make_deck_advisor(args, state.set_code)
            if deck_adv is not None:
                # off-thread so a live pick never waits on checkpoint loading
                threading.Thread(target=deck_adv.preload, daemon=True).start()
                server.set_action_handler(
                    lambda a, p: deck_action(a, p, deck_adv, state))
        built = refresh_deck(server, deck_adv, state, args, built)
        if not changed or not isinstance(ev, PackSeen):
            continue
        if scorers is None and state.set_code:
            from draftbot.eval.__main__ import resolve_scorer
            scorers = [resolve_scorer(m.strip(), state.set_code, args.stats)
                       for m in args.models.split(",")]
            advisor = Advisor(scorers, state.set_code)
            server.set_art_urls(art_url_map(state.set_code))
        if advisor is None:
            continue
        ranked = advisor.rank_pack(state)
        server.update(snapshot(state, ranked, pool_pips(state, state.set_code),
                               args.stats, pool_curve(state, state.set_code)))
        if args.replay and args.replay_delay:
            time.sleep(args.replay_delay)
    if deck_adv is not None and (args.replay or draft_finished(state)):
        # a fast replay can outrun the background load — settle synchronously.
        # A replay reaching EOF is itself the end of the draft.
        deck_adv.preload()
        refresh_deck(server, deck_adv, state, args, built, final=True)
    if args.replay:
        print("replay complete — panel stays up (ctrl-c to quit)")
        try:
            while True:
                __import__("time").sleep(3600)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
