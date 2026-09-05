"""Build the laptop deployment bundle (owner workflow, 2026-07-31).

`python -m draftbot.bundle --set MSH [--set EOE ...] [--checkpoints EXP-033 ...]`

Packs everything a machine needs to RUN the models — checkpoints (without
last.pt resume state), per-set card tables / static features / deck tables /
splits / stat snapshots, and the Scryfall cache — into laptop-bundle.tar.gz
at the repo root. Training data (draft parquets, raw CSVs) is excluded.

When the next set drops: onboard it (`draftbot onboard-set <CODE>`), retrain /
refresh per docs/PLAN.md Appendix D, then
`python -m draftbot.bundle --set <CODE> --set MSH` and copy the tarball over.
"""

import argparse
import sys
import tarfile
from pathlib import Path

from draftbot.data.cards import DAY0_MARKER, SOURCES_JSON

# Production set as of 2026-08-01 (journal: joint program closed):
# draft advisor, deck lock-and-rebuild, deck trunk, deck MSH fine-tune,
# joint draft+sealed trunk (P5.S joint; supersedes the sealed-only EXP-134).
DEFAULT_CHECKPOINTS = ["EXP-033", "EXP-116", "EXP-126", "EXP-127", "EXP-135"]
BUNDLE = Path("laptop-bundle.tar.gz")


def bundle_paths(set_codes: list[str], checkpoints: list[str]) -> list[Path]:
    paths = []
    for exp in checkpoints:
        ckpt = Path("checkpoints") / exp
        if not (ckpt / "best.pt").exists():
            raise SystemExit(f"missing checkpoint: {ckpt}/best.pt")
        paths += [p for p in sorted(ckpt.iterdir())
                  if p.name != "last.pt" and p.is_file()]
    for code in set_codes:
        proc = Path("data/processed") / code
        for name in ("cards.parquet", "features.static.parquet"):
            if not (proc / name).exists():
                raise SystemExit(f"missing {proc / name} — onboard {code} first")
            paths.append(proc / name)
        if (proc / "decks.parquet").exists():  # absent pre-extraction is fine
            paths.append(proc / "decks.parquet")
        # oracle-text blocks (+ sha sidecars): the laptop needs no encoder
        paths += sorted(p for p in proc.glob("features.text.*") if p.is_file())
        # Day-0 sets carry two more files: sources.json names the bonus sheets
        # (the target machine's grpId bridge and art map both read it), and the
        # .day0 marker keeps the vocab flagged provisional over there too.
        for name in (SOURCES_JSON, DAY0_MARKER):
            if (proc / name).exists():
                paths.append(proc / name)
        splits = Path("data/splits") / f"{code}.json"
        if splits.exists():
            paths.append(splits)
        snaps = Path("data/snapshots") / code
        if snaps.exists():
            paths += [p for p in sorted(snaps.iterdir()) if p.is_file()]
    cache = Path("data/raw/scryfall")
    if cache.exists():
        paths += [p for p in sorted(cache.rglob("*")) if p.is_file()]
    return paths


def build(set_codes: list[str], checkpoints: list[str],
          out: Path = BUNDLE) -> Path:
    paths = bundle_paths(set_codes, checkpoints)
    with tarfile.open(out, "w:gz") as tf:
        for p in paths:
            tf.add(p)
    size_mb = out.stat().st_size / 1e6
    print(f"{out}: {len(paths)} files, {size_mb:.0f} MB "
          f"(sets: {', '.join(set_codes)}; ckpts: {', '.join(checkpoints)})")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="draftbot.bundle")
    p.add_argument("--set", dest="set_codes", action="append", required=True,
                   help="set code to include (repeatable)")
    p.add_argument("--checkpoints", nargs="+", default=DEFAULT_CHECKPOINTS)
    p.add_argument("--out", type=Path, default=BUNDLE)
    args = p.parse_args(argv)
    build(args.set_codes, args.checkpoints, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
