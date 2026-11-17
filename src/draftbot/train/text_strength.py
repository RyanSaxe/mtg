"""TXT probe: fine-tune the sentence encoder to predict card strength from text.

The frozen-encoder read (eval/text_probe.py, TEXT_EMBED_PLAN §6) tops out at
GIH ρ ≈ .40 on an unseen set. This asks whether letting the encoder itself
learn "what a strong card reads like" does better. Cross-fit BY SET over the
corpus: every corpus card gets an out-of-fold prediction, the holdouts get the
mean of the fold models. Arm A (ridge on static + frozen embedding) runs on
the same folds and targets as the baseline.

    uv run python -m draftbot.train.text_strength --config configs/TXT-001.yaml

Resumable per fold (predictions cached under checkpoints/<exp>/).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from draftbot.data.cards import PROCESSED_DIR
from draftbot.data.features import load_snapshot
from draftbot.data.text_features import ENCODERS, card_strings, load_text_features
from draftbot.eval.text_probe import HEADLINE, LAMBDAS, MIN_GAMES, TARGETS, ridge_fit, ridge_predict, spearman

OUT_DIR = Path("docs/scorecards/text_probe")
GIH = TARGETS.index(HEADLINE)


def load_set(code: str, encoder: str) -> dict:
    """Everything one set contributes; y is z-scored within the set over its
    reliable rows (set-level offsets are format speed, not card strength)."""
    static = pd.read_parquet(PROCESSED_DIR / code / "features.static.parquet")
    snap = load_snapshot(code, "full")
    y = snap[TARGETS].to_numpy(np.float64)
    ok = ((snap["ever_drawn_count"].to_numpy() >= MIN_GAMES)
          & (static["is_basic"].to_numpy() == 0) & ~np.isnan(y).any(1))
    z = np.zeros_like(y)
    if ok.sum() >= 20:
        z = (y - y[ok].mean(0)) / (y[ok].std(0) + 1e-6)
    return {"code": code, "static": static.to_numpy(np.float32), "y": y, "z": z.astype(np.float32),
            "ok": ok, "texts": card_strings(code, "masked"),
            "frozen": load_text_features(code, encoder).to_numpy(np.float32)}


class StrengthModel(torch.nn.Module):
    def __init__(self, hf_name: str, n_static: int, hidden: int, dropout: float, trainable: str):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(hf_name)
        dim = self.encoder.config.hidden_size
        if trainable != "all":
            for p in self.encoder.parameters():
                p.requires_grad = False
            if trainable == "top2":
                for layer in self.encoder.encoder.layer[-2:]:
                    for p in layer.parameters():
                        p.requires_grad = True
        self.static_norm = torch.nn.BatchNorm1d(n_static)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(dim + n_static, hidden), torch.nn.SiLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, len(TARGETS)))

    def embed(self, tok) -> torch.Tensor:
        cls = self.encoder(**tok).last_hidden_state[:, 0]  # bge pools on [CLS]
        return torch.nn.functional.normalize(cls, dim=-1)

    def forward(self, tok, static):
        return self.head(torch.cat([self.embed(tok), self.static_norm(static)], -1))


def _device():
    from draftbot.models.base import device_auto
    return device_auto()


def fit_fold(cfg: dict, train: list[dict], test: list[dict], seed: int):
    """Train on `train` sets' reliable rows; return ({code: pred[n,5]}, {code: emb[n,d]}) for `test`."""
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = _device()
    hf = ENCODERS[cfg["encoder"]]
    tok = AutoTokenizer.from_pretrained(hf)
    model = StrengthModel(hf, train[0]["static"].shape[1], cfg["head_hidden"],
                          cfg["head_dropout"], cfg["trainable"]).to(dev)
    texts = [t for s in train for t, k in zip(s["texts"], s["ok"]) if k]
    static = torch.tensor(np.concatenate([s["static"][s["ok"]] for s in train]))
    z = torch.tensor(np.concatenate([s["z"][s["ok"]] for s in train]))
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    groups = [{"params": list(model.head.parameters()) + list(model.static_norm.parameters()),
               "lr": cfg["lr_head"]}]
    if enc_params:
        groups.append({"params": enc_params, "lr": cfg["lr_encoder"]})
    opt = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    bs = cfg["batch_size"]
    steps = cfg["epochs"] * int(np.ceil(len(texts) / bs))
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup_frac"] * steps), steps)

    def batch_tok(strs):
        return {k: v.to(dev) for k, v in tok(strs, padding=True, truncation=True,
                                             max_length=cfg["max_tokens"], return_tensors="pt").items()}

    for ep in range(cfg["epochs"]):
        model.train()
        perm, tot = rng.permutation(len(texts)), 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            out = model(batch_tok([texts[j] for j in idx]), static[idx].to(dev))
            loss = torch.nn.functional.huber_loss(out, z[idx].to(dev))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        print(f"    epoch {ep + 1}/{cfg['epochs']} loss {tot / len(texts):.4f}", flush=True)

    model.eval()
    preds, embs = {}, {}
    with torch.no_grad():
        for s in test:
            p, e = [], []
            for i in range(0, len(s["texts"]), 64):
                t = batch_tok(s["texts"][i:i + 64])
                st = torch.tensor(s["static"][i:i + 64]).to(dev)
                emb = model.embed(t)
                p.append(model.head(torch.cat([emb, model.static_norm(st)], -1)).cpu().numpy())
                e.append(emb.cpu().numpy())
            preds[s["code"]], embs[s["code"]] = np.concatenate(p), np.concatenate(e)
    del model, opt
    if dev.type == "mps":
        torch.mps.empty_cache()
    return preds, embs


def ridge_arm(train: list[dict], test: list[dict], inner: list[list[str]]) -> dict:
    """Arm A: ridge on static + frozen embedding; λ by inner set-grouped CV on train."""
    X = lambda s: np.concatenate([s["static"], s["frozen"]], 1).astype(np.float64)
    by = {s["code"]: s for s in train}

    def score(lam):
        rs = []
        for held in inner:
            tr = [s for s in train if s["code"] not in held]
            m = ridge_fit(np.concatenate([X(s)[s["ok"]] for s in tr]),
                          np.concatenate([s["z"][s["ok"]] for s in tr]), lam)
            rs += [spearman(ridge_predict(m, X(by[c])[by[c]["ok"]])[:, GIH], by[c]["z"][by[c]["ok"], GIH])
                   for c in held if c in by]
        return float(np.mean(rs))

    lam = max(LAMBDAS, key=score)
    m = ridge_fit(np.concatenate([X(s)[s["ok"]] for s in train]),
                  np.concatenate([s["z"][s["ok"]] for s in train]), lam)
    return {s["code"]: ridge_predict(m, X(s)) for s in test}


def rho(s: dict, pred: np.ndarray, t: int = GIH) -> float:
    return spearman(pred[s["ok"], t], s["y"][s["ok"], t])


def main(argv=None):
    p = argparse.ArgumentParser(prog="draftbot.train.text_strength")
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    manifest = yaml.safe_load(Path(cfg["corpus"]).read_text())
    for h in cfg["holdouts"]:
        assert h not in manifest["sets"], f"{h} is in the corpus — not unseen"
    labelled_holdouts = list(cfg["holdouts"])
    assert "TRK" not in cfg["holdouts"] and "TRK" not in manifest["sets"]
    sets = {c: load_set(c, cfg["encoder"]) for c in list(manifest["sets"]) + cfg["holdouts"]}
    corpus = [c for c in manifest["sets"] if sets[c]["ok"].sum() >= 20]
    # corpus sets with no reliable labels (AFR) cannot train or be scored, but a
    # trunk still needs their prior: no fold trains on them, so every fold
    # model is out-of-fold for them — treat as extra prediction-only holdouts.
    unlabelled = [c for c in manifest["sets"] if c not in corpus]
    cfg["holdouts"] = labelled_holdouts + unlabelled
    rng = np.random.default_rng(cfg["seed"])
    order = list(rng.permutation(corpus))
    folds = [order[i::cfg["folds"]] for i in range(cfg["folds"])]
    ck = Path("checkpoints") / cfg["exp"]
    ck.mkdir(parents=True, exist_ok=True)
    (ck / "folds.json").write_text(json.dumps([[str(c) for c in f] for f in folds]))
    print(f"{cfg['exp']}: {len(corpus)} corpus sets, {sum(int(sets[c]['ok'].sum()) for c in corpus)} "
          f"labelled cards, {cfg['folds']} folds, holdouts {cfg['holdouts']}", flush=True)

    ft, rd, emb = {}, {}, {}
    hold_ft = {h: [] for h in cfg["holdouts"]}
    hold_rd = {h: [] for h in cfg["holdouts"]}
    hold_emb = {h: [] for h in cfg["holdouts"]}
    t0 = time.time()
    for k, held in enumerate(folds):
        train = [sets[c] for c in corpus if c not in held]
        test = [sets[c] for c in held] + [sets[h] for h in cfg["holdouts"]]
        cache = ck / f"fold{k}.npz"
        if cache.exists() and all(f"p_{s['code']}" in np.load(cache) for s in test):
            z = np.load(cache)
            preds = {c: z[f"p_{c}"] for c in [s["code"] for s in test]}
            embs = {c: z[f"e_{c}"] for c in [s["code"] for s in test]}
            print(f"fold {k}: cached", flush=True)
        else:
            print(f"fold {k}: held out {[str(c) for c in held]}", flush=True)
            preds, embs = fit_fold(cfg, train, test, cfg["seed"] + k)
            np.savez(cache, **{f"p_{c}": v for c, v in preds.items()},
                     **{f"e_{c}": v for c, v in embs.items()})
        inner = [[c for c in f if c not in held] for j, f in enumerate(folds) if j != k]
        rpred = ridge_arm(train, test, inner)
        for c in held:
            ft[c], rd[c], emb[c] = preds[c], rpred[c], embs[c]
        for h in cfg["holdouts"]:
            hold_ft[h].append(preds[h]); hold_rd[h].append(rpred[h]); hold_emb[h].append(embs[h])
        done = [c for f in folds[:k + 1] for c in f]
        print(f"  fold {k} OOF GIH ρ: fine-tuned {np.mean([rho(sets[c], ft[c]) for c in held]):.3f} "
              f"ridge {np.mean([rho(sets[c], rd[c]) for c in held]):.3f}   running "
              f"{np.mean([rho(sets[c], ft[c]) for c in done]):.3f} vs "
              f"{np.mean([rho(sets[c], rd[c]) for c in done]):.3f}  ({(time.time() - t0) / 60:.1f} min)",
              flush=True)
    for h in cfg["holdouts"]:
        ft[h], rd[h], emb[h] = np.mean(hold_ft[h], 0), np.mean(hold_rd[h], 0), np.mean(hold_emb[h], 0)

    # per-set predictions (OOF for the corpus, fold-mean for holdouts) — what a trunk would consume
    for c in corpus + cfg["holdouts"]:
        pd.DataFrame(ft[c], columns=[f"prior_{t}" for t in TARGETS]).to_parquet(ck / f"prior.{c}.parquet")

    rows = []
    for c in corpus + labelled_holdouts:
        rows.append({"set": c, "role": "holdout" if c in labelled_holdouts else "corpus-OOF",
                     "n": int(sets[c]["ok"].sum()),
                     **{f"ridge_{t}": rho(sets[c], rd[c], i) for i, t in enumerate(TARGETS)},
                     **{f"ft_{t}": rho(sets[c], ft[c], i) for i, t in enumerate(TARGETS)}})
    df = pd.DataFrame(rows)
    summ = {}
    for role in ("corpus-OOF", "holdout"):
        d = df[df.role == role]
        summ[role] = {"ridge_gih": float(d[f"ridge_{HEADLINE}"].mean()),
                      "ft_gih": float(d[f"ft_{HEADLINE}"].mean()),
                      "delta": float((d[f"ft_{HEADLINE}"] - d[f"ridge_{HEADLINE}"]).mean()),
                      "sets_ft_better": int((d[f"ft_{HEADLINE}"] > d[f"ridge_{HEADLINE}"]).sum()),
                      "n_sets": int(len(d))}
    import subprocess
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    card = {"exp": cfg["exp"], "cfg": cfg, "git_sha": sha or "unknown", "summary": summ,
            "rows": rows, "wall_min": (time.time() - t0) / 60}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{cfg['exp']}.json").write_text(json.dumps(card, indent=1))
    md = [f"# {cfg['exp']} — fine-tuned text encoder vs frozen ridge (GIH Spearman ρ)", "",
          "| | ridge (static + frozen bge) | fine-tuned | Δ | sets better |", "|---|---|---|---|---|"]
    for role, s in summ.items():
        md.append(f"| {role} (mean of {s['n_sets']} sets) | {s['ridge_gih']:.3f} | {s['ft_gih']:.3f} "
                  f"| {s['delta']:+.3f} | {s['sets_ft_better']}/{s['n_sets']} |")
    md += ["", "| set | role | n | ridge GIH | ft GIH | ridge ALSA | ft ALSA |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['set']} | {r['role']} | {r['n']} | {r[f'ridge_{HEADLINE}']:.3f} "
                  f"| {r[f'ft_{HEADLINE}']:.3f} | {r['ridge_avg_seen']:.3f} | {r['ft_avg_seen']:.3f} |")
    (OUT_DIR / f"{cfg['exp']}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:6]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
