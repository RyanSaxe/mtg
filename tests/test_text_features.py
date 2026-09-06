"""Oracle-text feature block (P3.T5c): hermetic — a fake encoder stands in for
sentence-transformers, so these run on any machine."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from draftbot.data import features as feats_mod
from draftbot.data import text_features as tf

CARDS = pd.DataFrame([
    dict(id=0, name="Aardvark Ranger", cmc=2.0, mana_cost="{1}{G}", colors="G",
         produced_mana=None, type_line="Creature — Aardvark", rarity="common",
         power="2", toughness="2", keywords="Reach",
         oracle_text="Reach\nWhen Aardvark Ranger enters, you gain 2 life.",
         is_flip=0, is_basic=0),
    dict(id=1, name="Bilbo, Thief in the Night", cmc=2.0, mana_cost="{1}{U}", colors="U",
         produced_mana=None, type_line="Legendary Creature — Halfling Rogue",
         rarity="mythic", power="2", toughness="2", keywords="",
         oracle_text="Whenever Bilbo attacks, draw a card.", is_flip=0, is_basic=0),
    dict(id=2, name="Forest", cmc=0.0, mana_cost="", colors="", produced_mana="G",
         type_line="Basic Land — Forest", rarity="common", power=None, toughness=None,
         keywords="", oracle_text="({T}: Add {G}.)", is_flip=0, is_basic=1),
])


def fake_embed(texts, encoder="fake"):
    """Deterministic 8-d vectors from the text hash: a stand-in encoder."""
    out = np.zeros((len(texts), 8), np.float32)
    for i, t in enumerate(texts):
        h = hashlib.sha1(t.encode()).digest()
        out[i] = np.frombuffer(h[:8], dtype=np.uint8) / 255.0
    return out


@pytest.fixture
def two_sets(tmp_path, monkeypatch):
    monkeypatch.setattr(tf, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(feats_mod, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(tf, "embed", fake_embed)
    tf.ENCODERS["fake"] = "fake/encoder"
    for code in ("AAA", "BBB"):
        (tmp_path / code).mkdir()
        CARDS.to_parquet(tmp_path / code / "cards.parquet", index=False)
        feats_mod.build_static_features(code)
    return tmp_path


def test_card_text_masks_the_name_and_the_legend_short_name():
    t0 = tf.card_text(CARDS.iloc[0].to_dict())
    t1 = tf.card_text(CARDS.iloc[1].to_dict())
    assert "Aardvark Ranger" not in t0 and "this card enters" in t0
    assert "Bilbo" not in t1 and "Whenever this card attacks" in t1
    assert t0.startswith("{1}{G} Creature — Aardvark. 2/2. common.")
    assert "Bilbo, Thief in the Night. " in tf.card_text(CARDS.iloc[1].to_dict(), "named")
    assert tf.card_text(CARDS.iloc[1].to_dict(), "oracle") == "Whenever this card attacks, draw a card."
    assert "2/2" not in tf.card_text(CARDS.iloc[2].to_dict())  # no P/T on a land


def test_build_caches_by_text_sha(two_sets, monkeypatch):
    df = tf.build_text_features("AAA", "fake")
    assert df.shape == (3, 8) and list(df.index) == [0, 1, 2]
    assert list(df.columns)[:2] == ["text_fake_000", "text_fake_001"]
    meta = json.loads(tf.text_path("AAA", "fake").with_suffix(".json").read_text())
    assert meta["dim"] == 8 and meta["format"] == "masked"
    calls = []
    monkeypatch.setattr(tf, "embed", lambda texts, encoder: (calls.append(1), fake_embed(texts))[1])
    tf.build_text_features("AAA", "fake")
    assert not calls, "unchanged text must not re-run the encoder"


def test_assemble_inserts_text_between_static_and_stats_with_stable_dims(two_sets):
    a, scaler = feats_mod.assemble("AAA", None, fit=True, text="fake")
    b, _ = feats_mod.assemble("BBB", None, scaler=scaler, text="fake")
    assert list(a.columns) == list(b.columns), "dims must match across sets (corpus scaler)"
    static_cols = list(pd.read_parquet(two_sets / "AAA" / "features.static.parquet").columns)
    assert list(a.columns[:len(static_cols)]) == static_cols
    assert list(a.columns[len(static_cols):]) == [f"text_fake_{i:03d}" for i in range(8)]
    assert not any(c.endswith("__miss") for c in a.columns), "text is never masked"
    plain, _ = feats_mod.assemble("AAA", None, fit=True)
    assert not any(c.startswith("text_") for c in plain.columns), "text=None → unchanged"


def test_statless_leaves_the_text_block_alone(two_sets):
    from draftbot.data.corpus import _statless
    a, _ = feats_mod.assemble("AAA", None, fit=True, text="fake")
    a["ever_drawn_win_rate"], a["ever_drawn_win_rate__miss"] = 0.7, 0.0  # fake stat pair
    s = _statless(a)
    assert (s[[c for c in s.columns if c.startswith("text_")]].to_numpy()
            == a[[c for c in a.columns if c.startswith("text_")]].to_numpy()).all()
    assert s["ever_drawn_win_rate__miss"].iloc[0] == 1.0 and s["ever_drawn_win_rate"].iloc[0] == 0.0


def test_random_control_encoder_is_identity_without_semantics(two_sets, monkeypatch):
    """`text_features: random` — a fixed unit vector per card NAME: same name
    in two sets → same vector (like shared text); different names → unrelated."""
    monkeypatch.setattr(tf, "RANDOM_DIM", 16)
    a = tf.build_text_features("AAA", "random")
    b = tf.build_text_features("BBB", "random")
    assert a.shape == (3, 16) and np.allclose(np.linalg.norm(a.to_numpy(), axis=1), 1.0)
    assert np.allclose(a.to_numpy(), b.to_numpy())          # same names, same vectors
    assert abs(float(a.iloc[0] @ a.iloc[1])) < 0.6           # different names: unrelated


def test_missing_block_without_encoder_fails_clearly(two_sets, monkeypatch):
    """Day-0 on a laptop without the [text] extra: a text-trained trunk must
    fail with instructions, not an ImportError from inside sentence-transformers."""
    def no_encoder(texts, encoder="fake"):
        raise ImportError("No module named 'sentence_transformers'")
    monkeypatch.setattr(tf, "embed", no_encoder)
    with pytest.raises(tf.TextBlockUnavailable, match="uv sync --extra dev --extra text"):
        tf.load_text_features("AAA", "fake")
    monkeypatch.setattr(tf, "embed", fake_embed)  # with the encoder it builds fine
    assert tf.load_text_features("AAA", "fake").shape == (3, 8)
    assert tf.load_text_features("AAA", "fake").shape == (3, 8)  # and now loads from disk
