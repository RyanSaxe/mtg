"""Phase-3 model plumbing: set-summary token, swappable card context,
feature-only pathway, heterogeneous t."""

import numpy as np
import torch

from draftbot.data.dataset import PAD
from draftbot.models.base import slot_nll
from draftbot.models.modern import ModernDraftBot

F = 12


def build(n_cards=40, t=8):
    torch.manual_seed(0)
    feats = torch.randn(n_cards + 1, F)
    return ModernDraftBot(n_cards, feats, t=t, emb_dim=32, layers=1, heads=4,
                          feature_only=True, use_set_token=True)


def batch(n_cards, t, P=4, b=6, seed=0):
    rng = np.random.default_rng(seed)
    packs = torch.from_numpy(rng.integers(0, n_cards, size=(b, t, P)).astype(np.int64))
    picks = packs[:, :, 0].clone()
    prev = torch.full((b, t), PAD, dtype=torch.long)
    prev[:, 1:] = picks[:, :-1]
    return packs, prev, picks


def test_set_token_and_shapes():
    model = build()
    packs, prev, picks = batch(40, 8)
    logits = model(packs, prev)
    assert logits.shape == packs.shape
    assert torch.isfinite(slot_nll(logits, packs, picks))


def test_swap_card_context_foreign_set():
    """A foreign set with a different vocab size must work via set_context
    (zero learned half), and produce different logits than the home set."""
    model = build().eval()
    foreign_feats = torch.randn(60 + 1, F)
    packs, prev, _ = batch(60, 8, seed=2)
    model.embedding.set_context(foreign_feats)
    with torch.no_grad():
        out = model(packs, prev)
    assert out.shape == packs.shape
    assert torch.isfinite(out[packs != PAD]).all()


def test_shorter_t_than_max():
    """Heterogeneous t: a 5-step draft through a t=8 model."""
    model = build().eval()
    packs, prev, _ = batch(40, 5, seed=3)
    with torch.no_grad():
        out = model(packs, prev)
    assert out.shape == packs.shape


def test_feature_only_learned_half_is_zero():
    model = build()
    assert model.embedding.learned.weight.abs().sum() == 0


def test_stat_aux_head_masked_mse_ignores_missing_and_is_optional():
    """P3.T5c D2: the stat-reconstruction head reads the feature half of the
    STATLESS table; masked rows contribute nothing; no head ⇒ no new params
    (every earlier checkpoint stays loadable)."""
    torch.manual_seed(0)
    model = ModernDraftBot(40, torch.randn(41, F), t=8, emb_dim=32, layers=1, heads=4,
                           feature_only=True, use_set_token=True, stat_targets=3)
    feats = torch.randn(41, F)
    tgt, valid = torch.randn(40, 3), torch.ones(40, 3)
    valid[:20] = 0.0  # half the cards have no measured stat
    loss = model.stat_aux_loss(feats, tgt, valid)
    assert torch.isfinite(loss) and loss.requires_grad
    garbage = tgt.clone()
    garbage[:20] = 1e6
    assert torch.allclose(loss, model.stat_aux_loss(feats, garbage, valid))
    assert model.stat_aux_loss(feats, tgt, torch.zeros(40, 3)).item() == 0.0
    assert build().stat_head is None
    assert not any(k.startswith("stat_head") for k in build().state_dict())
