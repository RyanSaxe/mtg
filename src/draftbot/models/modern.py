"""ModernDraftBot — the v2 architecture (PLAN Phase 2).

Decoder-only causal stack over the t picks. Per-step input token =
proj(concat(pack summary, previous-pick embedding, position embedding)).

- Pack summary: permutation-invariant set attention (2 blocks) + PMA pooling
  over the ≤P card embeddings in the pack (config-fallback: masked mean).
- Pointer head: logit(card) = <q(decoder state), k(card embedding)> / sqrt(d)
  over in-pack cards only (config-fallback: dense vocab head, for ablation).
- Pre-LN RMSNorm, SwiGLU FFN, no behavioral priors (skill curation + the
  rare-take eval watchdog replace them).
"""

import torch
from torch import nn

from draftbot.data.dataset import PAD
from draftbot.models.base import NEG_INF


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.in_proj = nn.Linear(dim, hidden * 2)
        self.out_proj = nn.Linear(hidden, dim)

    def forward(self, x):
        a, b = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(nn.functional.silu(a) * b)


class SetAttentionBlock(nn.Module):
    """Pre-LN self-attention over pack slots (no causality — sets)."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True,
                                          dropout=dropout)
        self.norm2 = nn.RMSNorm(dim)
        self.ffn = SwiGLU(dim, dim * 2)

    def forward(self, x, key_padding_mask):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask,
                         need_weights=False)
        x = x + a
        return x + self.ffn(self.norm2(x))


class PMA(nn.Module):
    """Pooling by multihead attention: one learned seed queries the set."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.norm = nn.RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x, key_padding_mask):
        h = self.norm(x)
        q = self.seed.expand(x.shape[0], 1, -1)
        out, _ = self.attn(q, h, h, key_padding_mask=key_padding_mask,
                           need_weights=False)
        return out[:, 0]


class DecoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True,
                                          dropout=dropout)
        self.norm2 = nn.RMSNorm(dim)
        self.ffn = SwiGLU(dim, dim * 2)

    def forward(self, x, causal_mask):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + a
        return x + self.ffn(self.norm2(x))


class CardEncoder(nn.Module):
    """Learned-id half ∥ feature-MLP half (ConcatEmbedding, modern internals).

    feature_only=True zero-inits the id half (Phase-3 pretraining: zero-shot =
    pure feature pathway; a fresh per-set table is trained only at fine-tune).
    For multi-set training, `set_context(features, table)` swaps the card
    universe; the feature MLP is shared across sets."""

    def __init__(self, n_tokens: int, emb_dim: int, card_features: torch.Tensor,
                 feature_only: bool = False):
        super().__init__()
        half = emb_dim // 2
        self.register_buffer("card_features", card_features)
        f = card_features.shape[1]
        self.mlp = nn.Sequential(
            nn.Linear(f, emb_dim), nn.SiLU(), nn.Linear(emb_dim, half),
        )
        self.learned = nn.Embedding(n_tokens, half)
        if feature_only:
            nn.init.zeros_(self.learned.weight)
        self.feature_only = feature_only
        self._ext_features: torch.Tensor | None = None
        self._ext_table: nn.Embedding | None = None

    def set_context(self, card_features: torch.Tensor,
                    table: nn.Embedding | None = None):
        """Swap the card universe (multi-set pretraining / new-set inference)."""
        self._ext_features = card_features
        self._ext_table = table

    def all_embeddings(self) -> torch.Tensor:
        feats = self._ext_features if self._ext_features is not None \
            else self.card_features
        feat_half = self.mlp(feats)
        if self._ext_features is not None:
            if self._ext_table is not None:
                learned_half = self._ext_table.weight
            else:  # pure feature pathway for a foreign set
                learned_half = torch.zeros(
                    feats.shape[0], self.learned.embedding_dim,
                    device=feats.device, dtype=feat_half.dtype)
        else:
            learned_half = self.learned.weight
        return torch.cat([learned_half, feat_half], -1)


class ModernDraftBot(nn.Module):
    name = "modern"

    def __init__(self, n_cards: int, card_features: torch.Tensor, t: int = 42,
                 emb_dim: int = 128, layers: int = 4, heads: int = 8,
                 dropout: float = 0.0, use_set_encoder: bool = True,
                 pointer_head: bool = True, feature_only: bool = False,
                 use_set_token: bool = False, stat_targets: int = 0):
        super().__init__()
        self.n_cards, self.t, self.emb_dim = n_cards, t, emb_dim
        assert card_features.shape[0] == n_cards + 1  # + start-of-draft bias token
        self.embedding = CardEncoder(n_cards + 1, emb_dim, card_features,
                                     feature_only)
        # P3.T5c D2: stat-reconstruction head on the FEATURE half of the card
        # embedding — predicts a card's measured strength from its stat-less
        # features (text + static), giving the text pathway direct supervision
        # (docs/TEXT_EMBED_PLAN.md). Discarded at inference; None keeps every
        # earlier checkpoint loadable unchanged.
        self.stat_head = nn.Linear(emb_dim // 2, stat_targets) if stat_targets else None
        self.use_set_token = use_set_token
        if use_set_token:  # summary token prepended at position 0
            self.set_token_mlp = nn.Sequential(
                nn.Linear(card_features.shape[1], emb_dim), nn.SiLU(),
                nn.Linear(emb_dim, emb_dim))
        self.pos_embedding = nn.Embedding(t + (1 if use_set_token else 0), emb_dim)
        self.use_set_encoder = use_set_encoder
        if use_set_encoder:
            self.pack_blocks = nn.ModuleList(
                [SetAttentionBlock(emb_dim, heads, dropout) for _ in range(2)])
            self.pma = PMA(emb_dim, heads)
        self.fusion = nn.Linear(emb_dim * 3, emb_dim)
        self.blocks = nn.ModuleList(
            [DecoderBlock(emb_dim, heads, dropout) for _ in range(layers)])
        self.final_norm = nn.RMSNorm(emb_dim)
        self.pointer = pointer_head
        if pointer_head:
            self.q_proj = nn.Linear(emb_dim, emb_dim)
            self.k_proj = nn.Linear(emb_dim, emb_dim)
        else:
            self.out_mlp = nn.Sequential(
                nn.Linear(emb_dim, emb_dim * 2), nn.SiLU(),
                nn.Linear(emb_dim * 2, n_cards))
        self.drop = nn.Dropout(dropout)

    def stat_aux_loss(self, statless_features: torch.Tensor, targets: torch.Tensor,
                      valid: torch.Tensor) -> torch.Tensor:
        """Masked MSE of predicted vs measured (corpus-z-scored) stats over one
        set's card table. statless_features: (n+1, F) incl. the bias row;
        targets / valid: (n, k), valid = 1 − missing-mask."""
        pred = self.stat_head(self.embedding.mlp(statless_features[:-1]))
        return ((pred - targets) ** 2 * valid).sum() / valid.sum().clamp(min=1.0)

    def forward(self, packs: torch.Tensor, prev_picks: torch.Tensor):
        B, t, P = packs.shape
        device = packs.device
        all_emb = self.embedding.all_embeddings()

        pack_ids = packs.clamp(min=0)
        pack_emb = all_emb[pack_ids]                      # (B,t,P,E)
        pad_mask = packs == PAD                           # (B,t,P)
        if self.use_set_encoder:
            flat = pack_emb.view(B * t, P, -1)
            flat_mask = pad_mask.view(B * t, P)
            # fully-padded rows would NaN in attention; none exist (packs ≥1 card)
            for blk in self.pack_blocks:
                flat = blk(flat, flat_mask)
            summary = self.pma(flat, flat_mask).view(B, t, -1)
        else:
            valid = (~pad_mask).unsqueeze(-1)
            summary = (pack_emb * valid).sum(2) / valid.sum(2).clamp(min=1)

        bias_id = all_emb.shape[0] - 1  # bias token is always the last row
        prev_ids = torch.where(prev_picks == PAD,
                               torch.full_like(prev_picks, bias_id), prev_picks)
        prev_emb = all_emb[prev_ids]
        offset = 1 if self.use_set_token else 0
        pos = self.pos_embedding(
            torch.arange(offset, t + offset, device=device))[None].expand(B, -1, -1)
        x = self.fusion(torch.cat([summary, prev_emb, pos], -1))
        x = self.drop(x)

        if self.use_set_token:
            feats = self.embedding._ext_features if self.embedding._ext_features \
                is not None else self.embedding.card_features
            set_vec = self.set_token_mlp(feats[:-1].mean(0))  # exclude bias row
            tok = (set_vec + self.pos_embedding.weight[0])[None, None].expand(B, 1, -1)
            x = torch.cat([tok, x], dim=1)

        L = x.shape[1]
        causal = torch.triu(torch.full((L, L), float("-inf"), device=device), 1)
        for blk in self.blocks:
            x = blk(x, causal)
        x = self.final_norm(x)
        if self.use_set_token:
            x = x[:, 1:]

        if self.pointer:
            q = self.q_proj(x)                            # (B,t,E)
            k = self.k_proj(pack_emb)                     # (B,t,P,E)
            logits = torch.einsum("bte,btpe->btp", q, k) / (self.emb_dim ** 0.5)
        else:
            vocab_logits = self.out_mlp(x)
            logits = torch.gather(vocab_logits, 2, pack_ids)
        return logits.masked_fill(pad_mask, NEG_INF)
