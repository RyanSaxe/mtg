"""Config-driven training (PLAN P1.T5).

`python -m draftbot.train --config configs/EXP-001.yaml [--resume]`

Checkpoints: checkpoints/<EXP>/last.pt (model+optimizer+sched+epoch+rng, written
every epoch) and best.pt (best val top-1). Scaler + feature-column list persisted
next to the checkpoints so eval reconstructs the exact feature pathway.
"""

import argparse
import copy
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from draftbot.data.cards import PROCESSED_DIR
from draftbot.data.dataset import PAD, load_draft_arrays
from draftbot.data.features import assemble, save_scaler
from draftbot.data.splits import load_splits
from draftbot.eval.harness import topk_hits
from draftbot.models.base import TorchScorer, device_auto, importance_weights, slot_nll

CKPT_ROOT = Path("checkpoints")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def feature_tensor(set_code: str, stats: str | None, ckpt_dir: Path | None = None,
                   scaler: dict | None = None,
                   text: str | None = None) -> tuple[torch.Tensor, dict]:
    feats, scaler = assemble(set_code, stats if stats not in (None, "none") else None,
                             scaler=scaler, fit=scaler is None, text=text)
    tensor = torch.tensor(feats.to_numpy(np.float32))
    # bias token row: zeros + is_bias flag column
    tensor = torch.cat([tensor, torch.zeros(1, tensor.shape[1])], 0)
    is_bias = torch.zeros(tensor.shape[0], 1)
    is_bias[-1, 0] = 1.0
    tensor = torch.cat([tensor, is_bias], 1)
    if ckpt_dir is not None:
        save_scaler(scaler, ckpt_dir / "scaler.json")
        (ckpt_dir / "feature_columns.json").write_text(json.dumps(list(feats.columns)))
    return tensor, scaler


def build_model(cfg: dict, n_cards: int, card_features: torch.Tensor, t: int):
    kind = cfg.get("model", "legacy")
    if kind == "legacy":
        from draftbot.models.legacy import LegacyDraftBot
        return LegacyDraftBot(
            n_cards, card_features, t=t, emb_dim=cfg.get("emb_dim", 128),
            enc_layers=cfg.get("enc_layers", 2), dec_layers=cfg.get("dec_layers", 2),
            enc_heads=cfg.get("enc_heads", 8), dec_heads=cfg.get("dec_heads", 8),
            dropout=cfg.get("dropout", 0.0))
    if kind == "modern":
        from draftbot.models.modern import ModernDraftBot
        return ModernDraftBot(
            n_cards, card_features, t=cfg.get("t_max", t),
            emb_dim=cfg.get("emb_dim", 128),
            layers=cfg.get("layers", 4), heads=cfg.get("heads", 8),
            dropout=cfg.get("dropout", 0.0),
            use_set_encoder=cfg.get("use_set_encoder", True),
            pointer_head=cfg.get("pointer_head", True),
            feature_only=cfg.get("feature_only", False),
            use_set_token=cfg.get("use_set_token", False))
    raise ValueError(f"unknown model kind {kind}")


def noam_lr(step: int, d_model: int, warmup: int) -> float:
    step = max(step, 1)
    return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


def cosine_lr(step: int, total: int, warmup: int, peak: float) -> float:
    if step < warmup:
        return peak * step / max(warmup, 1)
    frac = (step - warmup) / max(total - warmup, 1)
    return peak * 0.5 * (1 + np.cos(np.pi * min(frac, 1.0)))


class Trainer:
    def __init__(self, cfg: dict, resume: bool = False):
        self.cfg = cfg
        self.exp = cfg["exp"]
        self.dir = CKPT_ROOT / self.exp
        self.dir.mkdir(parents=True, exist_ok=True)
        set_seed(cfg.get("seed", 17))
        self.device = device_auto()
        set_code = cfg["set"]
        splits = load_splits(set_code)
        scheme = cfg.get("scheme", "random")

        self.train_arr = load_draft_arrays(set_code, draft_ids=set(splits[scheme]["train"]))
        self.val_arr = load_draft_arrays(set_code, draft_ids=set(splits[scheme]["val"]))
        self.t = self.train_arr.t
        # Owner directive (journal 2026-07-29): select models on agreement with
        # GOOD drafters — early stop tracks expert-subset val top-1.
        es = splits["expert_subset"]
        vm = self.val_arr.meta
        self.val_expert_mask = ((vm["user_win_rate"].fillna(0) >= es["min_win_rate"])
                                & (vm["user_n_games"] >= es["min_games"])).to_numpy()

        md = cfg.get("max_drafts")
        if md and md < self.train_arr.n_drafts:
            keep = np.zeros(self.train_arr.n_drafts, dtype=bool)
            keep[np.random.default_rng(cfg.get("seed", 17)).choice(
                self.train_arr.n_drafts, md, replace=False)] = True
            m = self.train_arr.meta
            self.train_arr = type(self.train_arr)(
                packs=self.train_arr.packs[keep], picks=self.train_arr.picks[keep],
                prev_picks=self.train_arr.prev_picks[keep],
                meta=m[keep].reset_index(drop=True))

        sf = cfg.get("skill_filter")
        if sf:
            m = self.train_arr.meta
            keep = ((m["user_win_rate"].fillna(0) >= sf["min_win_rate"])
                    & (m["user_n_games"] >= sf["min_games"])).to_numpy()
            self.train_arr = type(self.train_arr)(
                packs=self.train_arr.packs[keep], picks=self.train_arr.picks[keep],
                prev_picks=self.train_arr.prev_picks[keep],
                meta=m[keep].reset_index(drop=True))
            print(f"skill filter kept {keep.sum()}/{len(keep)} drafts")

        cards = pd.read_parquet(PROCESSED_DIR / set_code / "cards.parquet")
        self.n_cards = len(cards)
        pre_scaler = None
        if cfg.get("scaler_from"):  # fine-tune: reuse the pretrain-corpus scaler
            from draftbot.data.features import load_scaler
            pre_scaler = load_scaler(Path(cfg["scaler_from"]) / "scaler.json")
        feats, _ = feature_tensor(set_code, cfg.get("stats", "full"),
                                  ckpt_dir=self.dir, scaler=pre_scaler,
                                  text=cfg.get("text_features"))
        self.model = build_model(cfg, self.n_cards, feats, self.t).to(self.device)
        if cfg.get("init_from"):  # warm-start from a pretrained trunk
            src = Path(cfg["init_from"])
            src_file = src if src.is_file() else src / "best.pt"
            state = torch.load(src_file, map_location=self.device,
                               weights_only=False)["model"]
            own = self.model.state_dict()
            loadable = {k: v for k, v in state.items()
                        if k in own and own[k].shape == v.shape}
            skipped = sorted(set(state) - set(loadable))
            self.model.load_state_dict(loadable, strict=False)
            print(f"init_from {src_file}: {len(loadable)} tensors loaded, "
                  f"skipped {skipped}")

        self.ft_mode = cfg.get("finetune_mode", "full")
        if self.ft_mode != "full":
            from draftbot.models.lora import apply_finetune_mode
            stats = apply_finetune_mode(self.model, self.ft_mode,
                                        rank=cfg.get("lora_rank", 4),
                                        alpha=cfg.get("lora_alpha", 16.0))
            print(f"finetune_mode={self.ft_mode}: {stats['trainable']:,} of "
                  f"{stats['total']:,} params trainable ({stats['frac']:.2%})")
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"{self.exp}: {n_params / 1e6:.2f}M params on {self.device}")

        rare_like = cards["rarity"].isin(["rare", "mythic", "special", "bonus"])
        self.rare_flag = torch.tensor(rare_like.to_numpy(np.float32)).to(self.device)
        self.cmc = torch.tensor(cards["cmc"].fillna(0).to_numpy(np.float32)).to(self.device)

        if cfg.get("weighting", "importance") == "importance":
            m = self.train_arr.meta
            w = importance_weights(m["rank"].to_numpy(), m["user_win_rate"].to_numpy(),
                                   m["event_wins"].to_numpy(np.float64),
                                   m["event_losses"].to_numpy(np.float64),
                                   m["draft_time"].to_numpy(), self.t)
        else:
            w = np.ones((self.train_arr.n_drafts, self.t))
        self.weights = w.astype(np.float32)

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.opt_kind = "adamw" if cfg.get("model") == "modern" else "adam"
        if self.opt_kind == "adamw":
            self.optim = torch.optim.AdamW(trainable, lr=1.0,
                                           betas=(0.9, 0.98), eps=1e-9,
                                           weight_decay=cfg.get("weight_decay", 0.01))
        else:
            self.optim = torch.optim.Adam(trainable, lr=1.0,
                                          betas=(0.9, 0.98), eps=1e-9)
        self.step = 0
        self.epoch = 0
        self.best_val = -1.0
        self.bad_epochs = 0
        self.writer = SummaryWriter(log_dir=str(Path("runs") / self.exp))
        if resume:
            self._load(self.dir / "last.pt")

    # ------------------------------------------------------------------ io ---
    def _save(self, path: Path):
        torch.save({
            "model": self.model.state_dict(), "optim": self.optim.state_dict(),
            "step": self.step, "epoch": self.epoch, "best_val": self.best_val,
            "bad_epochs": self.bad_epochs, "cfg": self.cfg,
            "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state(),
        }, path)

    def _load(self, path: Path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optim.load_state_dict(state["optim"])
        self.step, self.epoch = state["step"], state["epoch"]
        self.best_val, self.bad_epochs = state["best_val"], state["bad_epochs"]
        torch.set_rng_state(state["torch_rng"].cpu())
        np.random.set_state(state["np_rng"])
        print(f"resumed at epoch {self.epoch}, step {self.step}, best {self.best_val:.4f}")

    # --------------------------------------------------------------- batches ---
    def _batches(self, arr, weights, batch_drafts, shuffle=True):
        idx = np.arange(arr.n_drafts)
        if shuffle:
            np.random.shuffle(idx)
        for i in range(0, len(idx), batch_drafts):
            b = idx[i:i + batch_drafts]
            yield (
                torch.from_numpy(arr.packs[b].astype(np.int64)).to(self.device),
                torch.from_numpy(arr.prev_picks[b].astype(np.int64)).to(self.device),
                torch.from_numpy(arr.picks[b].astype(np.int64)).to(self.device),
                torch.from_numpy(weights[b]).to(self.device) if weights is not None else None,
            )

    def _lr(self, total_steps):
        cfg = self.cfg
        if self.opt_kind == "adam":
            return noam_lr(self.step, cfg.get("emb_dim", 128), cfg.get("lr_warmup", 1000))
        return cosine_lr(self.step, total_steps, cfg.get("lr_warmup", 500),
                         cfg.get("peak_lr", 3e-4))

    # ----------------------------------------------------------------- train ---
    def val_top1(self) -> tuple[float, float]:
        """(overall, expert-subset) val top-1; selection uses the expert number."""
        scorer = TorchScorer(self.model, self.exp, self.device)
        scores = scorer.score_drafts(self.val_arr.packs, self.val_arr.prev_picks)
        hits = topk_hits(self.val_arr.packs, self.val_arr.picks, scores, kmax=1)
        self.model.train()
        return float(hits[0].mean()), float(hits[0][self.val_expert_mask].mean())

    def train(self):
        cfg = self.cfg
        bs = cfg.get("batch_drafts", 64)
        epochs = cfg.get("epochs", 20)
        steps_per_epoch = int(np.ceil(self.train_arr.n_drafts / bs))
        total_steps = steps_per_epoch * epochs
        use_priors = cfg.get("priors", False)
        smoothing = cfg.get("label_smoothing", 0.0)
        clip = cfg.get("grad_clip", 1.0)
        self.model.train()
        start = time.time()
        while self.epoch < epochs:
            epoch_loss, n_batches = 0.0, 0
            for packs, prev, picks, w in self._batches(self.train_arr, self.weights, bs):
                lr = self._lr(total_steps)
                for g in self.optim.param_groups:
                    g["lr"] = lr
                self.optim.zero_grad(set_to_none=True)
                if use_priors:
                    logits, y, pack_emb = self.model(packs, prev, return_states=True)
                    loss = slot_nll(logits, packs, picks, w, smoothing)
                    loss = loss + self.model.auxiliary_loss(
                        packs, picks, logits, y, pack_emb,
                        w if w is not None else torch.ones_like(picks, dtype=torch.float32),
                        self.rare_flag, self.cmc,
                        emb_lambda=cfg.get("emb_lambda", 1.0),
                        rare_lambda=cfg.get("rare_lambda", 10.0),
                        cmc_lambda=cfg.get("cmc_lambda", 1.0))
                else:
                    logits = self.model(packs, prev)
                    loss = slot_nll(logits, packs, picks, w, smoothing)
                loss.backward()
                if clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                self.optim.step()
                self.step += 1
                epoch_loss += float(loss.detach())
                n_batches += 1
                if self.step % 100 == 0:
                    self.writer.add_scalar("train/loss", float(loss), self.step)
                    self.writer.add_scalar("train/lr", lr, self.step)
            self.epoch += 1
            val_overall, val = self.val_top1()  # `val` (expert) drives selection
            self.writer.add_scalar("val/top1", val_overall, self.step)
            self.writer.add_scalar("val/expert_top1", val, self.step)
            mins = (time.time() - start) / 60
            print(f"epoch {self.epoch}/{epochs} loss {epoch_loss / max(n_batches,1):.4f} "
                  f"val_top1 {val_overall:.4f} expert {val:.4f} ({mins:.1f} min)",
                  flush=True)
            improved = val > self.best_val
            if improved:
                self.best_val = val
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1
            self._save(self.dir / "last.pt")
            if improved:
                self._save(self.dir / "best.pt")
                if self.ft_mode != "full":  # adapter checkpoint (PLAN P4.T1)
                    from draftbot.models.lora import adapter_state
                    torch.save({"adapter": adapter_state(self.model),
                                "cfg": self.cfg}, self.dir / "adapter.pt")
            elif self.bad_epochs >= cfg.get("patience", 3):
                print(f"early stop at epoch {self.epoch} (best {self.best_val:.4f})")
                break
        self.writer.close()
        return self.best_val


def main(argv=None):
    p = argparse.ArgumentParser(prog="draftbot.train")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    ckpt_dir = CKPT_ROOT / cfg["exp"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, ckpt_dir / "config.yaml")
    trainer = Trainer(cfg, resume=args.resume)
    best = trainer.train()
    print(f"best val top-1: {best:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
