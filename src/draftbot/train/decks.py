"""Deck-builder training (PLAN P5.T1d).

`python -m draftbot.train.decks --config configs/EXP-101.yaml [--resume]`

Loss = per-copy membership BCE (weighted by pool copies × example weight)
     + land_lambda · CE over total-land count (14..20)
     + basics_lambda · soft-CE over the W/U/B/R/G basic split.
Example weight per DECK_DATA_PLAN curation: (1 + n_wins) · soft_skill(wr),
config-switchable (`weighting: win_skill|uniform`). Trains on ALL builds;
selection = val trophy-F1 (agreement with ≥5-win builds), computed through the
same greedy decode used at eval time.
"""

import argparse
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
from draftbot.data.deck_dataset import (example_weights, land_flags,
                                        load_deck_arrays)
from draftbot.data.splits import load_splits
from draftbot.models.base import device_auto
from draftbot.models.builder import (LAND_MAX, LAND_MIN, DeckBuilder,
                                     TorchDeckBuilder)
from draftbot.train.loop import CKPT_ROOT, cosine_lr, feature_tensor


def build_deck_model(cfg: dict, card_features: torch.Tensor) -> DeckBuilder:
    return DeckBuilder(card_features, emb_dim=cfg.get("emb_dim", 128),
                       heads=cfg.get("heads", 8), blocks=cfg.get("blocks", 3),
                       dropout=cfg.get("dropout", 0.1),
                       diffusion=cfg.get("model", "builder") == "diffusion",
                       quality_basics=cfg.get("quality_basics", False),
                       formats=cfg.get("formats", False))


def deck_targets(arr, is_land: np.ndarray):
    """(member_frac (N,P), slot_weight (N,P), land_class (N,), basics_frac (N,5))"""
    pool = arr.pool_counts.astype(np.float32)
    frac = np.divide(arr.deck_counts, pool, out=np.zeros_like(pool),
                     where=pool > 0)
    ids = np.clip(arr.pool_ids, 0, None)
    nb_lands = (arr.deck_counts * is_land[ids] * (arr.pool_ids >= 0)).sum(1)
    total_lands = arr.basics.sum(1) + nb_lands
    land_class = np.clip(total_lands, LAND_MIN, LAND_MAX) - LAND_MIN
    bsum = arr.basics.sum(1, keepdims=True).astype(np.float32)
    basics_frac = np.divide(arr.basics, bsum,
                            out=np.full(arr.basics.shape, 0.2, np.float32),
                            where=bsum > 0)
    return frac.astype(np.float32), pool, land_class.astype(np.int64), basics_frac


def deck_loss(model, arr, targets, weights: np.ndarray, b: np.ndarray,
              device, cfg: dict, format_id: int = 0) -> torch.Tensor:
    """The full training objective for a batch of build indices `b`.
    Diffusion models: reveal a random fraction of true deck counts as
    conditioning and take the membership loss on the MASKED slots only."""
    from draftbot.models.builder import MASK_STATE

    frac, pool_w, land_cls, basics_frac = targets
    ids = torch.from_numpy(arr.pool_ids[b].astype(np.int64)).to(device)
    cnt = torch.from_numpy(arr.pool_counts[b].astype(np.int64)).to(device)
    ex_w = torch.from_numpy(weights[b]).to(device)
    slot_w = torch.from_numpy(pool_w[b]).to(device) * ex_w[:, None]

    if getattr(model, "diffusion", False):
        import math
        deck_cnt = torch.from_numpy(arr.deck_counts[b].astype(np.int64)).to(device)
        rho = torch.rand(ids.shape[0], device=device)
        if cfg.get("mask_dist") == "cosine":  # bias toward the all-masked
            rho = torch.cos(math.pi / 2 * (1 - rho))  # inference condition
        reveal = torch.rand_like(slot_w) >= rho[:, None]
        state = torch.where(reveal & (ids >= 0),
                            deck_cnt.clamp(0, MASK_STATE - 1),
                            torch.full_like(ids, MASK_STATE))
        member, land, basics = model(ids, cnt, ids == -1, state,
                                     format_id=format_id)
        # hidden slots carry the objective; revealed ones optionally kept at a
        # low weight so masking doesn't halve the supervision per step
        rw = cfg.get("reveal_weight", 0.0)
        slot_w = slot_w * torch.where(reveal, torch.full_like(slot_w, rw),
                                      torch.ones_like(slot_w))
    else:
        member, land, basics = model(ids, cnt, ids == -1, format_id=format_id)
    target = torch.from_numpy(frac[b]).to(device)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        member.clamp(min=-30), target, reduction="none")  # pads sit at -1e9
    member_loss = (bce * slot_w).sum() / slot_w.sum().clamp(min=1)

    land_t = torch.from_numpy(land_cls[b]).to(device)
    land_ce = torch.nn.functional.cross_entropy(land, land_t, reduction="none")
    land_loss = (land_ce * ex_w).sum() / ex_w.sum()

    basics_t = torch.from_numpy(basics_frac[b]).to(device)
    basics_ce = -(basics_t * torch.log_softmax(basics, -1)).sum(-1)
    basics_loss = (basics_ce * ex_w).sum() / ex_w.sum()

    return (member_loss + cfg.get("land_lambda", 0.2) * land_loss
            + cfg.get("basics_lambda", 0.2) * basics_loss)


class DeckTrainer:
    def __init__(self, cfg: dict, resume: bool = False):
        self.cfg = cfg
        self.exp = cfg["exp"]
        self.dir = CKPT_ROOT / self.exp
        self.dir.mkdir(parents=True, exist_ok=True)
        random.seed(cfg.get("seed", 17))
        np.random.seed(cfg.get("seed", 17))
        torch.manual_seed(cfg.get("seed", 17))
        self.device = device_auto()
        set_code = cfg["set"]
        source = cfg.get("source", "draft")  # 'sealed' → sealed decks + splits
        self.format_id = 1 if source == "sealed" else 0  # formats models only
        splits = load_splits(set_code, source)

        self.train_arr = load_deck_arrays(set_code,
                                          set(splits["random"]["train"]),
                                          source=source)
        val_ids = set(splits["random"]["val"])
        self.val_arr = load_deck_arrays(set_code, val_ids, view="most_played",
                                        source=source)
        win = load_deck_arrays(set_code, val_ids, view="winningest",
                               source=source)
        self.val_trophy = win.take(win.meta["n_wins"].to_numpy() >= 5)
        cards = pd.read_parquet(PROCESSED_DIR / set_code / "cards.parquet")
        self.cards = cards
        self.is_land = land_flags(cards)

        if cfg.get("min_wins"):  # hard curation lever (hill-climb contrast)
            keep = self.train_arr.meta["n_wins"].to_numpy() >= cfg["min_wins"]
            self.train_arr = type(self.train_arr)(
                pool_ids=self.train_arr.pool_ids[keep],
                pool_counts=self.train_arr.pool_counts[keep],
                deck_counts=self.train_arr.deck_counts[keep],
                basics=self.train_arr.basics[keep],
                meta=self.train_arr.meta[keep].reset_index(drop=True))
            print(f"min_wins={cfg['min_wins']} kept {keep.sum()}/{len(keep)}")

        if cfg.get("weighting", "win_skill") == "win_skill":
            self.weights = example_weights(self.train_arr.meta)
        else:
            self.weights = np.ones(self.train_arr.n_builds, dtype=np.float32)

        pre_scaler = None
        if cfg.get("scaler_from"):  # fine-tune: reuse the pretrain-corpus scaler
            from draftbot.data.features import load_scaler
            pre_scaler = load_scaler(Path(cfg["scaler_from"]) / "scaler.json")
        feats, _ = feature_tensor(set_code, cfg.get("stats", "full"),
                                  ckpt_dir=self.dir, scaler=pre_scaler,
                                  text=cfg.get("text_features"))
        self.model = build_deck_model(cfg, feats).to(self.device)
        if cfg.get("init_from"):  # warm-start (stage 2 / corpus-pretrained trunk)
            src = Path(cfg["init_from"])
            src_file = src if src.is_file() else src / "best.pt"
            state = torch.load(src_file, map_location=self.device,
                               weights_only=False)["model"]
            state.pop("card_features", None)  # this set's table, not the source's
            own = self.model.state_dict()
            loadable = {k: v for k, v in state.items()
                        if k in own and own[k].shape == v.shape}
            self.model.load_state_dict(loadable, strict=False)
            print(f"init_from {src_file}: {len(loadable)} tensors loaded")

        self.pairs = None
        if cfg.get("quality_lambda"):  # within-pool winner/loser contrast
            m = self.train_arr.meta
            multi = m[m["draft_id"].duplicated(keep=False)]
            scope = cfg.get("pair_scope", "trophy")  # trophy | all
            min_gap = cfg.get("pair_min_gap", 1 if scope == "trophy" else 2)
            pairs = []
            for _, g in multi.groupby("draft_id", sort=False):
                order = g.sort_values(["n_wins", "n_games", "build_index"],
                                      ascending=[False, False, True])
                w, l = order.index[0], order.index[-1]
                gap_ok = m.loc[w, "n_wins"] - m.loc[l, "n_wins"] >= min_gap
                if gap_ok and (scope == "all" or m.loc[w, "n_wins"] >= 5):
                    pairs.append((w, l))
            self.pairs = np.array(pairs) if pairs else None
            print(f"quality contrast pairs: {0 if self.pairs is None else len(self.pairs)}")
        n_params = sum(p.numel() for p in self.model.parameters()
                       if p.requires_grad)
        print(f"{self.exp}: {n_params / 1e6:.2f}M params on {self.device}, "
              f"{self.train_arr.n_builds} train builds")

        self.targets = deck_targets(self.train_arr, self.is_land)
        self.optim = torch.optim.AdamW(
            self.model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9,
            weight_decay=cfg.get("weight_decay", 0.01))
        self.step = 0
        self.epoch = 0
        self.best_val = -1.0
        self.bad_epochs = 0
        self.writer = SummaryWriter(log_dir=str(Path("runs") / self.exp))
        if resume:
            self._load(self.dir / "last.pt")

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
        print(f"resumed at epoch {self.epoch}, best {self.best_val:.4f}")

    def _loss(self, b: np.ndarray) -> torch.Tensor:
        loss = deck_loss(self.model, self.train_arr, self.targets,
                         self.weights, b, self.device, self.cfg,
                         format_id=self.format_id)
        if self.pairs is not None:
            loss = loss + self.cfg["quality_lambda"] * self._quality_loss()
        return loss

    def _quality_loss(self) -> torch.Tensor:
        """Margin ranking: the build that won ≥5 must outscore the same pool's
        strictly-worse build (the winner-pref construction, as a loss)."""
        from draftbot.models.builder import MASK_STATE
        arr = self.train_arr
        k = min(64, len(self.pairs))
        sel = self.pairs[np.random.choice(len(self.pairs), k, replace=False)]
        rows = np.concatenate([sel[:, 0], sel[:, 1]])
        ids = torch.from_numpy(arr.pool_ids[rows].astype(np.int64)).to(self.device)
        cnt = torch.from_numpy(arr.pool_counts[rows].astype(np.int64)).to(self.device)
        state = torch.from_numpy(arr.deck_counts[rows].astype(np.int64)) \
            .to(self.device).clamp(0, MASK_STATE - 1) \
            .masked_fill(ids == -1, MASK_STATE)
        bas = torch.from_numpy(arr.basics[rows].astype(np.float32) / 20.0) \
            .to(self.device)
        q = self.model.quality(ids, cnt, ids == -1, state, bas,
                               format_id=self.format_id)
        margin = self.cfg.get("quality_margin", 0.2)
        return torch.relu(margin - (q[:k] - q[k:])).mean()

    def val_f1(self) -> tuple[float, float]:
        """(overall deck-F1 on most-played, trophy-F1 v2 on winningest ≥5)
        via the real decode. Selection uses the trophy number."""
        from draftbot.eval.decks import _f1, _true_build, _with_basics
        builder = TorchDeckBuilder(self.model, self.exp, self.device,
                                   self.is_land,
                                   decode=self.cfg.get("decode", "greedy"),
                                   steps=self.cfg.get("steps", 8),
                                   format_id=self.format_id)

        def mean_f1(arr):
            builds = builder.build_all(arr)
            return float(np.mean([
                _f1(_with_basics(p["deck"], p["basics"]),
                    _with_basics(*_true_build(arr, i)))
                for i, p in enumerate(builds)]))

        overall, trophy = mean_f1(self.val_arr), mean_f1(self.val_trophy)
        self.model.train()
        return overall, trophy

    def train(self) -> float:
        cfg = self.cfg
        bs = cfg.get("batch_builds", 256)
        epochs = cfg.get("epochs", 30)
        total_steps = int(np.ceil(self.train_arr.n_builds / bs)) * epochs
        clip = cfg.get("grad_clip", 1.0)
        self.model.train()
        start = time.time()
        while self.epoch < epochs:
            idx = np.random.permutation(self.train_arr.n_builds)
            epoch_loss, n_batches = 0.0, 0
            for s in range(0, len(idx), bs):
                lr = cosine_lr(self.step, total_steps,
                               cfg.get("lr_warmup", 300),
                               cfg.get("peak_lr", 3e-4))
                for g in self.optim.param_groups:
                    g["lr"] = lr
                self.optim.zero_grad(set_to_none=True)
                loss = self._loss(idx[s:s + bs])
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
            val_f1, trophy_f1 = self.val_f1()  # trophy drives selection
            self.writer.add_scalar("val/deck_f1", val_f1, self.step)
            self.writer.add_scalar("val/trophy_f1", trophy_f1, self.step)
            mins = (time.time() - start) / 60
            print(f"epoch {self.epoch}/{epochs} "
                  f"loss {epoch_loss / max(n_batches, 1):.4f} "
                  f"val_f1 {val_f1:.4f} trophy {trophy_f1:.4f} "
                  f"({mins:.1f} min)", flush=True)
            improved = trophy_f1 > self.best_val
            if improved:
                self.best_val = trophy_f1
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1
            self._save(self.dir / "last.pt")
            if improved:
                self._save(self.dir / "best.pt")
            elif self.bad_epochs >= cfg.get("patience", 4):
                print(f"early stop at epoch {self.epoch} "
                      f"(best {self.best_val:.4f})")
                break
        self.writer.close()
        return self.best_val


def _deck_feature_pair(code: str, scaler: dict, text: str | None = None):
    """(full, statless) feature tensors for one set, corpus-scaled, in the
    exact layout feature_tensor produces (bias row + is_bias column) so
    pretrained encoder weights transfer to the fine-tune/eval paths."""
    from draftbot.data.corpus import _statless
    from draftbot.data.features import assemble

    full_df, _ = assemble(code, "full", scaler=scaler, fit=False, text=text)

    def tensorize(df):
        t = torch.tensor(df.to_numpy(np.float32))
        t = torch.cat([t, torch.zeros(1, t.shape[1])], 0)
        b = torch.zeros(t.shape[0], 1)
        b[-1, 0] = 1.0
        return torch.cat([t, b], 1)

    return tensorize(full_df), tensorize(_statless(full_df))


class DeckCorpusTrainer:
    """Multi-set pool→deck pretraining (DECK_DATA_PLAN scale-out).

    Single-set batches sampled ∝ manifest weight × n_builds, per-set feature
    tables swapped into the model buffer, stat-dropout p per batch (day-0 as a
    trained condition). Scaler fit on corpus sets only; holdout sets touch
    nothing. Selection: mean val trophy-F1 over fixed CORPUS proxy sets (the
    holdout never drives selection)."""

    PROXY_SETS = ["EOE", "BLB", "KTK", "SNC"]

    def __init__(self, cfg: dict, resume: bool = False):
        from draftbot.data.corpus import fit_corpus_scaler
        from draftbot.data.features import save_scaler

        self.cfg = cfg
        self.exp = cfg["exp"]
        self.dir = CKPT_ROOT / self.exp
        self.dir.mkdir(parents=True, exist_ok=True)
        random.seed(cfg.get("seed", 17))
        np.random.seed(cfg.get("seed", 17))
        torch.manual_seed(cfg.get("seed", 17))
        self.device = device_auto()

        # Single manifest (cfg `corpus`) or joint multi-format training
        # (cfg `corpora: [{manifest: ..., boost: 4.0}, ...]`, P5.S joint
        # program): each manifest carries its own `source`; `boost` scales the
        # sampling weight of that corpus (sealed is 1/11 of draft naturally).
        specs = cfg.get("corpora") or [{"manifest": cfg["corpus"]}]
        manifests, all_codes = [], set()
        for spec in specs:
            manifest = yaml.safe_load(Path(spec["manifest"]).read_text())
            codes = list(manifest["sets"])
            banned = set(manifest.get("quarantine", [])) \
                | set(manifest.get("holdout", []))
            assert not (banned & set(codes)), \
                f"banned sets in corpus: {banned & set(codes)}"
            manifests.append((manifest, float(spec.get("boost", 1.0))))
            all_codes |= set(codes)
        if cfg.get("scaler_from"):  # warm-start: features scaled as the source saw them
            from draftbot.data.features import load_scaler
            scaler = load_scaler(Path(cfg["scaler_from"]) / "scaler.json")
        else:
            scaler = fit_corpus_scaler(sorted(all_codes), cfg.get("text_features"))
        save_scaler(scaler, self.dir / "scaler.json")

        self.bundles = []
        feat_cache, land_cache = {}, {}
        for manifest, boost in manifests:
            source = cfg.get("source") or manifest.get("source", "draft")
            fmt = 1 if source == "sealed" else 0
            for code in manifest["sets"]:
                splits = load_splits(code, source)
                arr = load_deck_arrays(code, set(splits["random"]["train"]),
                                       source=source)
                if code not in feat_cache:
                    cards = pd.read_parquet(PROCESSED_DIR / code / "cards.parquet")
                    land_cache[code] = land_flags(cards)
                    ff, fn = _deck_feature_pair(code, scaler, cfg.get("text_features"))
                    feat_cache[code] = (ff.to(self.device), fn.to(self.device))
                f_full, f_none = feat_cache[code]
                is_land = land_cache[code]
                if cfg.get("weighting", "uniform") == "win_skill":
                    w = example_weights(arr.meta)
                else:
                    w = np.ones(arr.n_builds, dtype=np.float32)
                self.bundles.append({
                    "code": code, "source": source, "format_id": fmt,
                    "arr": arr, "targets": deck_targets(arr, is_land),
                    "weights": w, "is_land": is_land,
                    "f_full": f_full, "f_none": f_none,
                    "sample_w": (manifest["sets"][code].get("weight", 1.0)
                                 * arr.n_builds * boost),
                })
                print(f"loaded {code}/{source}: {arr.n_builds} builds",
                      flush=True)

        self.val_proxies = []
        for manifest, _ in manifests:
            source = cfg.get("source") or manifest.get("source", "draft")
            proxy_sets = [c for c in self.PROXY_SETS if c in manifest["sets"]]
            assert len(proxy_sets) >= 2, "too few proxy sets left in corpus"
            for code in proxy_sets:
                splits = load_splits(code, source)
                win = load_deck_arrays(code, set(splits["random"]["val"]),
                                       view="winningest", source=source)
                trophy = win.take(win.meta["n_wins"].to_numpy() >= 5)
                b = next(x for x in self.bundles
                         if x["code"] == code and x["source"] == source)
                self.val_proxies.append((f"{code}/{source}", trophy,
                                         b["is_land"], b["f_full"],
                                         b["format_id"]))
        self.val_detail = ""

        self.model = build_deck_model(cfg, self.bundles[0]["f_full"]).to(self.device)
        if cfg.get("init_from"):  # warm-start (e.g. sealed corpus from the draft trunk)
            src = Path(cfg["init_from"])
            src_file = src if src.is_file() else src / "best.pt"
            state = torch.load(src_file, map_location=self.device,
                               weights_only=False)["model"]
            state.pop("card_features", None)
            own = self.model.state_dict()
            loadable = {k: v for k, v in state.items()
                        if k in own and own[k].shape == v.shape}
            self.model.load_state_dict(loadable, strict=False)
            print(f"init_from {src_file}: {len(loadable)} tensors loaded")
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(b["arr"].n_builds for b in self.bundles)
        print(f"{self.exp}: {n_params/1e6:.2f}M params, {len(self.bundles)} sets, "
              f"{total:,} builds on {self.device}")

        self.optim = torch.optim.AdamW(self.model.parameters(), lr=1.0,
                                       betas=(0.9, 0.98), eps=1e-9,
                                       weight_decay=cfg.get("weight_decay", 0.01))
        self.step = 0
        self.epoch = 0
        self.best_val = -1.0
        self.bad_epochs = 0
        self.writer = SummaryWriter(log_dir=str(Path("runs") / self.exp))
        if resume:
            self._load(self.dir / "last.pt")

    _save = DeckTrainer._save
    _load = DeckTrainer._load

    def val_f1(self) -> float:
        from draftbot.eval.decks import _f1, _true_build, _with_basics
        scores, by_fmt = [], {}
        for label, arr, is_land, feats, fmt in self.val_proxies:
            self.model.card_features = feats
            builder = TorchDeckBuilder(self.model, self.exp, self.device, is_land,
                                       decode=self.cfg.get("decode", "greedy"),
                                       format_id=fmt)
            builds = builder.build_all(arr)
            s = np.mean([
                _f1(_with_basics(p["deck"], p["basics"]),
                    _with_basics(*_true_build(arr, i)))
                for i, p in enumerate(builds)])
            scores.append(s)
            by_fmt.setdefault(label.split("/")[1], []).append(s)
        self.model.train()
        if len(by_fmt) > 1:
            self.val_detail = " ".join(f"{k} {np.mean(v):.4f}"
                                       for k, v in sorted(by_fmt.items()))
        return float(np.mean(scores))

    def train(self) -> float:
        cfg = self.cfg
        bs = cfg.get("batch_builds", 256)
        epochs = cfg.get("epochs", 10)
        total_builds = sum(b["arr"].n_builds for b in self.bundles)
        steps_per_epoch = int(np.ceil(total_builds / bs))
        total_steps = steps_per_epoch * epochs
        p_sample = np.array([b["sample_w"] for b in self.bundles], np.float64)
        p_sample /= p_sample.sum()
        p_statless = cfg.get("stat_dropout", 0.25)
        clip = cfg.get("grad_clip", 1.0)
        self.model.train()
        start = time.time()
        while self.epoch < epochs:
            epoch_loss, n_batches = 0.0, 0
            for _ in range(steps_per_epoch):
                lr = cosine_lr(self.step, total_steps, cfg.get("lr_warmup", 1000),
                               cfg.get("peak_lr", 3e-4))
                for g in self.optim.param_groups:
                    g["lr"] = lr
                b = self.bundles[np.random.choice(len(self.bundles), p=p_sample)]
                idx = np.random.choice(b["arr"].n_builds,
                                       min(bs, b["arr"].n_builds), replace=False)
                self.model.card_features = (
                    b["f_none"] if np.random.random() < p_statless else b["f_full"])
                self.optim.zero_grad(set_to_none=True)
                loss = deck_loss(self.model, b["arr"], b["targets"], b["weights"],
                                 idx, self.device, cfg,
                                 format_id=b.get("format_id", 0))
                loss.backward()
                if clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                self.optim.step()
                self.step += 1
                epoch_loss += float(loss.detach())
                n_batches += 1
                if self.step % 200 == 0:
                    self.writer.add_scalar("train/loss", float(loss), self.step)
            self.epoch += 1
            val = self.val_f1()
            self.writer.add_scalar("val/proxy_trophy_f1", val, self.step)
            mins = (time.time() - start) / 60
            detail = f" [{self.val_detail}]" if self.val_detail else ""
            print(f"epoch {self.epoch}/{epochs} loss {epoch_loss/max(n_batches,1):.4f} "
                  f"proxy_trophy {val:.4f}{detail} ({mins:.1f} min)", flush=True)
            improved = val > self.best_val
            if improved:
                self.best_val = val
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1
            self._save(self.dir / "last.pt")
            if improved:
                self._save(self.dir / "best.pt")
            elif self.bad_epochs >= cfg.get("patience", 3):
                print(f"early stop at epoch {self.epoch} (best {self.best_val:.4f})")
                break
        self.writer.close()
        return self.best_val


def main(argv=None):
    p = argparse.ArgumentParser(prog="draftbot.train.decks")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    ckpt_dir = CKPT_ROOT / cfg["exp"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, ckpt_dir / "config.yaml")
    trainer_cls = (DeckCorpusTrainer if cfg.get("corpus") or cfg.get("corpora")
                   else DeckTrainer)
    best = trainer_cls(cfg, resume=args.resume).train()
    print(f"best val trophy-F1: {best:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
