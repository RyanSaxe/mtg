"""Rebuild a trained model + feature pathway from a checkpoint directory."""

from pathlib import Path

import pandas as pd
import torch

from draftbot.data.cards import PROCESSED_DIR
from draftbot.data.features import load_scaler
from draftbot.models.base import TorchScorer, device_auto


def scorer_from_checkpoint(ckpt: Path, set_code: str, stats: str) -> TorchScorer:
    """ckpt may be .../best.pt, .../last.pt or the EXP directory itself.

    `stats` selects the snapshot used to build features at EVAL time (none|week1|
    full) — the day-0/day-7 evaluation lever. The scaler always comes from
    training (scaler.json next to the checkpoint)."""
    ckpt = Path(ckpt)
    ckpt_dir = ckpt if ckpt.is_dir() else ckpt.parent
    ckpt_file = ckpt if ckpt.is_file() else ckpt_dir / "best.pt"
    state = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    cfg = state["cfg"]
    text = cfg.get("text_features")  # oracle-text block the checkpoint trained with

    if "corpus" in cfg:  # pretrained trunk: zero-shot via swapped card context
        from draftbot.data.corpus import _statless, add_bias_row
        from draftbot.data.features import assemble
        from draftbot.train.pretrain import build_trunk
        scaler = load_scaler(ckpt_dir / "scaler.json")
        feats_df, _ = assemble(set_code, "full" if stats == "none" else stats,
                               scaler=scaler, text=text)
        if stats == "none":
            feats_df = _statless(feats_df)
        elif text and cfg.get("text_when") == "statless":
            from draftbot.data.features import zero_text_block
            feats_df = zero_text_block(feats_df)  # text is a day-0 fallback only
        table = torch.tensor(add_bias_row(feats_df))
        model = build_trunk(cfg, table.shape[1])
        model.load_state_dict(state["model"])
        scorer = TorchScorer(model, f"{cfg['exp']}-zeroshot@{ckpt_file.stem}",
                             device_auto())
        model.embedding.set_context(table.to(scorer.device))
        return scorer

    from draftbot.train.loop import build_model, feature_tensor
    scaler = load_scaler(ckpt_dir / "scaler.json")
    has_stat_cols = "avg_seen" in scaler  # draft-stage stat present ⇔ stats-trained
    if stats == "none" and has_stat_cols:
        # stat-trained model in day-0 mode: full-width table, stats zeroed, masks on
        from draftbot.data.corpus import _statless, add_bias_row
        from draftbot.data.features import assemble
        feats_df, _ = assemble(set_code, "full", scaler=scaler, text=text)
        feats = torch.tensor(add_bias_row(_statless(feats_df)))
    elif stats != "none" and not has_stat_cols:
        raise SystemExit("model was trained without stats; use --stats-snapshot none")
    else:
        feats, _ = feature_tensor(set_code, stats, ckpt_dir=None, scaler=scaler,
                                  text=text, text_when=cfg.get("text_when"))
    cards = pd.read_parquet(PROCESSED_DIR / set_code / "cards.parquet")
    # t comes from the checkpoint itself: pos_embedding rows (minus the set-token
    # slot if present) — sets differ (MSH t=42, EOE t=39, draft boosters 45)
    t = state["model"]["pos_embedding.weight"].shape[0] - (
        1 if cfg.get("use_set_token") else 0)
    model = build_model(cfg, len(cards), feats, t)
    if any("parametrizations" in k for k in state["model"]):
        from draftbot.models.lora import apply_finetune_mode
        apply_finetune_mode(model, "lora", rank=cfg.get("lora_rank", 4),
                            alpha=cfg.get("lora_alpha", 16.0))
    model.load_state_dict(state["model"])
    name = f"{cfg['exp']}@{ckpt_file.stem}"
    return TorchScorer(model, name, device_auto())


def deck_builder_from_checkpoint(ckpt: Path, set_code: str, stats: str):
    """Deck-builder analogue of scorer_from_checkpoint (PLAN P5.T1).

    The scaler comes from training; `stats` picks the eval-time snapshot
    (`none` = zeroed stats + masks on, the day-0 condition)."""
    from draftbot.data.deck_dataset import land_flags
    from draftbot.models.builder import TorchDeckBuilder
    from draftbot.train.decks import build_deck_model

    ckpt = Path(ckpt)
    ckpt_dir = ckpt if ckpt.is_dir() else ckpt.parent
    ckpt_file = ckpt if ckpt.is_file() else ckpt_dir / "best.pt"
    state = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    cfg = state["cfg"]
    text = cfg.get("text_features")
    scaler = load_scaler(ckpt_dir / "scaler.json")
    has_stat_cols = "avg_seen" in scaler
    if stats == "none" and has_stat_cols:
        from draftbot.data.corpus import _statless, add_bias_row
        from draftbot.data.features import assemble
        feats_df, _ = assemble(set_code, "full", scaler=scaler, text=text)
        feats = torch.tensor(add_bias_row(_statless(feats_df)))
    elif stats != "none" and not has_stat_cols:
        raise SystemExit("model was trained without stats; use --stats-snapshot none")
    else:
        from draftbot.train.loop import feature_tensor
        feats, _ = feature_tensor(set_code, stats, ckpt_dir=None, scaler=scaler,
                                  text=text, text_when=cfg.get("text_when"))
    model = build_deck_model(cfg, feats)
    sd = state["model"]
    # the checkpoint buffer holds TRAIN-time features (possibly another set's,
    # for corpus checkpoints); the freshly built eval-set table must win
    sd.pop("card_features", None)
    model.load_state_dict(sd, strict=False)
    model.card_features = feats
    cards = pd.read_parquet(PROCESSED_DIR / set_code / "cards.parquet")
    return TorchDeckBuilder(model, f"{cfg['exp']}@{ckpt_file.stem}",
                            device_auto(), land_flags(cards),
                            decode=cfg.get("decode", "greedy"),
                            steps=cfg.get("steps", 8))
