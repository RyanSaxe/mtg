"""Laptop-bundle builder smoke test (skips without local data/checkpoints)."""

import tarfile
from pathlib import Path

import pytest

needs_artifacts = pytest.mark.skipif(
    not (Path("checkpoints/EXP-126/best.pt").exists()
         and Path("data/processed/MSH/cards.parquet").exists()),
    reason="production checkpoints/data not on this machine")


@needs_artifacts
def test_bundle_builds_and_contains_essentials(tmp_path):
    from draftbot.bundle import build

    out = build(["MSH"], ["EXP-126"], out=tmp_path / "b.tar.gz")
    names = set(tarfile.open(out).getnames())
    assert "checkpoints/EXP-126/best.pt" in names
    assert "checkpoints/EXP-126/scaler.json" in names
    assert "data/processed/MSH/cards.parquet" in names
    assert "data/processed/MSH/features.static.parquet" in names
    assert not any(n.endswith("last.pt") for n in names)
    assert not any("draft.PremierDraft.parquet" in n for n in names)


@pytest.mark.skipif(not list(Path("data/processed").glob("*/.day0")),
                    reason="no day-0 set onboarded locally")
def test_bundle_carries_day0_sidecars(tmp_path):
    """sources.json names the bonus sheets; without it the target machine's
    grpId bridge and art map silently drop the whole sheet."""
    from draftbot.bundle import build

    code = next(Path("data/processed").glob("*/.day0")).parent.name
    names = set(tarfile.open(build([code], ["EXP-033"], out=tmp_path / "b.tar.gz")).getnames())
    assert f"data/processed/{code}/sources.json" in names
    assert f"data/processed/{code}/.day0" in names
    assert f"data/snapshots/{code}/registry.json" in names
