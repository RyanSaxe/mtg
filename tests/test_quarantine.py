"""Stop-the-world guardrail (PLAN §0): the current sandbagged set must never
enter any pretraining corpus, warm-start source, or scaler-fit source.

History: MSH was the sandbag from project start until 2026-07-29, when the
Phase-3 gate passed (owner ruling, see journal) and it joined the corpus in
pretrain_v2. pretrain_v1 is frozen as EXP-030's training manifest and must keep
its MSH quarantine for provenance. The next Arena release becomes the new
sandbag: add its code to the LATEST manifest's `quarantine` on announcement day
(HOB, 2026-09-05) — test_latest_manifest_has_sandbag fails while it is empty.
"""

from pathlib import Path

import pytest
import yaml

CORPUS_DIR = Path("configs/corpus")
CONFIG_DIR = Path("configs")


def manifests() -> dict[str, dict]:
    return {p.name: yaml.safe_load(p.read_text())
            for p in sorted(CORPUS_DIR.glob("*.yaml"))}


def test_no_manifest_trains_on_its_own_quarantine():
    ms = manifests()
    assert ms, "no corpus manifests found"
    for name, m in ms.items():
        overlap = set(m.get("quarantine", [])) & set(m.get("sets", {}))
        assert not overlap, f"{name}: quarantined sets {overlap} in corpus!"


def test_v1_keeps_historical_msh_quarantine():
    """EXP-030 trained under v1's MSH quarantine — that record is immutable."""
    m = manifests().get("pretrain_v1.yaml")
    assert m is not None, "pretrain_v1.yaml must not be deleted (provenance)"
    assert "MSH" in m.get("quarantine", []), "v1's MSH quarantine is frozen"
    assert "MSH" not in m.get("sets", {})


def test_latest_manifest_has_sandbag():
    """HOB became the sandbag on 2026-09-05 (its 17lands drop day); the xfail
    that waited for it is gone. A future empty quarantine is a real failure."""
    ms = manifests()
    latest = sorted(ms)[-1]
    assert ms[latest].get("quarantine"), f"{latest}: no sandbag declared"


def test_no_pretrain_config_trains_on_a_quarantined_set():
    """Any config with a `corpus` key must not have its manifest's quarantine
    overlap its sets (single-set configs with `set:` are fine-tunes, exempt)."""
    for path in CONFIG_DIR.glob("*.yaml"):
        cfg = yaml.safe_load(path.read_text())
        if not isinstance(cfg, dict) or "corpus" not in cfg:
            continue
        manifest = yaml.safe_load(Path(cfg["corpus"]).read_text())
        overlap = set(manifest.get("quarantine", [])) & set(manifest.get("sets", {}))
        assert not overlap, f"{path}: pretrains on quarantined {overlap}"
