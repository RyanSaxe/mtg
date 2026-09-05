"""Text-lever calibration probe (docs/TEXT_EMBED_PLAN.md §3).

How much card strength is readable from the rules text BEFORE anyone has
played the card? Ridge regression card → full-season stats (GIH WR is the
headline), fit on the pretraining corpus, read out (a) leave-one-set-out over
the corpus and (b) on the held-out sandbag, for feature sets static / text /
static+text, per encoder × text format. Then two harness readouts on the
holdout's val split in the day-0 condition:
  - `textgih-*`: a greedy bot ranking packs by the text-predicted GIH — the
    pick-level signal text carries on its own (compare gih-greedy with REAL
    stats, and the trunk at `none`);
  - `<trunk>-imputed`: the production trunk with the predicted overall stats
    imputed into its stat block (masks off for those columns only) — a
    zero-training day-0 candidate.
Minutes, not hours: this is what calibrates the EXP-050 series.

    uv run python -m draftbot.eval.text_probe --holdout HOB [--encoders bge-base bge-small]
        [--formats masked named oracle] [--trunk checkpoints/EXP-033]
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from draftbot.data.cards import PROCESSED_DIR
from draftbot.data.features import load_snapshot
from draftbot.data.text_features import DEFAULT_ENCODER, card_strings, embed, load_text_features

TARGETS = ["ever_drawn_win_rate", "game_win_rate", "drawn_improvement_win_rate",
           "avg_seen", "avg_pick"]
HEADLINE = "ever_drawn_win_rate"
MIN_GAMES = 200          # y reliability: cards with fewer GIH games are excluded
LAMBDAS = (1.0, 10.0, 100.0, 1000.0)
OUT_DIR = Path("docs/scorecards/text_probe")


def set_xy(code: str, encoder: str, fmt: str):
    """(static, text, y[n,5], ok) for one set; ok = reliable, non-basic rows."""
    static = pd.read_parquet(PROCESSED_DIR / code / "features.static.parquet")
    text = (load_text_features(code, encoder).to_numpy() if fmt == "masked"
            else embed(card_strings(code, fmt), encoder))
    snap = load_snapshot(code, "full")
    y = snap[TARGETS].to_numpy(np.float64)
    ok = ((snap["ever_drawn_count"].to_numpy() >= MIN_GAMES)
          & (static["is_basic"].to_numpy() == 0) & ~np.isnan(y).any(1))
    return static.to_numpy(np.float64), text.astype(np.float64), y, ok


def ridge_fit(X, y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    ym = y.mean(0)
    W = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ (y - ym))
    return mu, sd, ym, W


def ridge_predict(model, X):
    mu, sd, ym, W = model
    return ((X - mu) / sd) @ W + ym


def _knn(train_x, train_y, test_x, k: int) -> np.ndarray:
    sims = test_x @ train_x.T  # cosine: embeddings are L2-normalised
    top = np.argpartition(-sims, k, axis=1)[:, :k]
    return train_y[top].mean(1)


def _mlp_predict(X, Y, Xtest, hidden: int = 256, epochs: int = 150, seed: int = 17) -> np.ndarray:
    """2-layer MLP regressor (CPU, seconds); inputs standardised, AdamW + wd."""
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    ym, ys = Y.mean(0), Y.std(0) + 1e-6
    x = torch.tensor((X - mu) / sd, dtype=torch.float32)
    y = torch.tensor((Y - ym) / ys, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], hidden), torch.nn.SiLU(),
                              torch.nn.Dropout(0.2), torch.nn.Linear(hidden, y.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
    for _ in range(epochs):
        for i in torch.randperm(len(x)).split(256):
            opt.zero_grad()
            torch.nn.functional.mse_loss(net(x[i]), y[i]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        out = net(torch.tensor((Xtest - mu) / sd, dtype=torch.float32)).numpy()
    return out * ys + ym


def spearman(a, b) -> float:
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def feature_sets(static, text):
    return {"static": static, "text": text, "static+text": np.concatenate([static, text], 1)}


def probe(corpus: list[str], holdout: str, encoder: str, fmt: str) -> tuple[list[dict], dict]:
    data = {c: set_xy(c, encoder, fmt) for c in corpus + [holdout]}
    gih = TARGETS.index(HEADLINE)
    rows, best_models = [], {}
    for fs in ("static", "text", "static+text"):
        Xy = {c: (feature_sets(s, t)[fs][ok], y[ok]) for c, (s, t, y, ok) in data.items()}
        # λ by leave-one-set-out Spearman on GIH over the corpus
        loso, folds = {}, {}
        for lam in LAMBDAS:
            rs = {}
            for held in corpus:
                if Xy[held][1].shape[0] < 20:  # too few reliable rows to rank
                    continue
                X = np.concatenate([Xy[c][0] for c in corpus if c != held])
                Y = np.concatenate([Xy[c][1] for c in corpus if c != held])
                m = ridge_fit(X, Y, lam)
                rs[held] = spearman(ridge_predict(m, Xy[held][0])[:, gih], Xy[held][1][:, gih])
            vals = np.array(list(rs.values()))
            loso[lam] = (float(np.nanmean(vals)), float(np.nanstd(vals)))
            folds[lam] = rs
        lam = max(LAMBDAS, key=lambda l: loso[l][0])
        if fs == "static+text":
            worst = sorted(folds[lam].items(), key=lambda kv: kv[1])[:3]
            print(f"    folds: {len(folds[lam])} sets; worst " +
                  ", ".join(f"{c} {r:.2f}" for c, r in worst), flush=True)
        X = np.concatenate([Xy[c][0] for c in corpus])
        Y = np.concatenate([Xy[c][1] for c in corpus])
        model = ridge_fit(X, Y, lam)
        pred_h = ridge_predict(model, Xy[holdout][0])
        rho_h = spearman(pred_h[:, gih], Xy[holdout][1][:, gih])
        rows.append({"encoder": encoder, "format": fmt, "features": fs, "lambda": lam,
                     "loso_gih_mean": loso[lam][0], "loso_gih_sd": loso[lam][1],
                     f"{holdout}_gih": rho_h,
                     **{f"{holdout}_{t}": spearman(pred_h[:, i], Xy[holdout][1][:, i])
                        for i, t in enumerate(TARGETS) if t != HEADLINE}})
        best_models[fs] = (model, lam)
        print(f"  {encoder:9} {fmt:7} {fs:12} λ={lam:<6g} LOSO GIH ρ {loso[lam][0]:.3f}±{loso[lam][1]:.3f}"
              f"  {holdout} GIH ρ {rho_h:.3f}", flush=True)
    # kNN on the (L2-normalised) text embedding: mean GIH of the k most similar
    # corpus cards — the signal "similar text ⇒ similar strength" carries alone
    Xt = {c: (t[ok], y[ok][:, gih]) for c, (_, t, y, ok) in data.items()}
    for k in (10, 30):
        rs = []
        for held in corpus:
            if Xt[held][1].shape[0] < 20:
                continue
            T = np.concatenate([Xt[c][0] for c in corpus if c != held])
            Y = np.concatenate([Xt[c][1] for c in corpus if c != held])
            rs.append(spearman(_knn(T, Y, Xt[held][0], k), Xt[held][1]))
        T = np.concatenate([Xt[c][0] for c in corpus]); Y = np.concatenate([Xt[c][1] for c in corpus])
        rho_h = spearman(_knn(T, Y, Xt[holdout][0], k), Xt[holdout][1])
        rows.append({"encoder": encoder, "format": fmt, "features": f"text-knn{k}", "lambda": float("nan"),
                     "loso_gih_mean": float(np.nanmean(rs)), "loso_gih_sd": float(np.nanstd(rs)),
                     f"{holdout}_gih": rho_h})
        print(f"  {encoder:9} {fmt:7} {'text-knn'+str(k):12} {'':9}LOSO GIH ρ {np.nanmean(rs):.3f}±{np.nanstd(rs):.3f}"
              f"  {holdout} GIH ρ {rho_h:.3f}", flush=True)
    # nonlinear read: a small MLP on static+text (CPU, seconds) — if it beats
    # ridge, the trunk's card MLP has headroom beyond the linear signal
    Xy = {c: (feature_sets(s, t)["static+text"][ok], y[ok]) for c, (s, t, y, ok) in data.items()}
    rs = []
    for held in corpus:
        if Xy[held][1].shape[0] < 20:
            continue
        X = np.concatenate([Xy[c][0] for c in corpus if c != held])
        Y = np.concatenate([Xy[c][1] for c in corpus if c != held])
        rs.append(spearman(_mlp_predict(X, Y, Xy[held][0])[:, gih], Xy[held][1][:, gih]))
    X = np.concatenate([Xy[c][0] for c in corpus]); Y = np.concatenate([Xy[c][1] for c in corpus])
    pred_h = _mlp_predict(X, Y, Xy[holdout][0])
    rho_h = spearman(pred_h[:, gih], Xy[holdout][1][:, gih])
    rows.append({"encoder": encoder, "format": fmt, "features": "static+text (mlp)", "lambda": float("nan"),
                 "loso_gih_mean": float(np.nanmean(rs)), "loso_gih_sd": float(np.nanstd(rs)),
                 f"{holdout}_gih": rho_h,
                 **{f"{holdout}_{t}": spearman(pred_h[:, i], Xy[holdout][1][:, i])
                    for i, t in enumerate(TARGETS) if t != HEADLINE}})
    print(f"  {encoder:9} {fmt:7} {'s+t (mlp)':12} {'':9}LOSO GIH ρ {np.nanmean(rs):.3f}±{np.nanstd(rs):.3f}"
          f"  {holdout} GIH ρ {rho_h:.3f}", flush=True)
    # full-vocab predictions on the holdout (basics included) for the harness readouts
    s, t, _, _ = data[holdout]
    preds = {fs: ridge_predict(best_models[fs][0], feature_sets(s, t)[fs]) for fs in best_models}
    return rows, preds


def harness_readouts(holdout: str, preds: dict, encoder: str, trunk: Path | None,
                     out_dir: Path) -> list[dict]:
    from draftbot.eval.bots import CardValueBot
    from draftbot.eval.harness import evaluate, render_markdown
    cards = []
    gih = TARGETS.index(HEADLINE)
    for fs in ("text", "static+text"):
        bot = CardValueBot(f"textgih-{encoder}-{fs.replace('+', '_')}", preds[fs][:, gih])
        card = evaluate(bot, holdout, "val", "text-pred", out_dir=out_dir)
        print(render_markdown(card), flush=True)
        cards.append(card)
    if trunk is not None:
        from draftbot.data.corpus import _statless, add_bias_row
        from draftbot.data.features import assemble, load_scaler
        from draftbot.models.base import TorchScorer, device_auto
        from draftbot.train.pretrain import build_trunk
        state = torch.load(trunk / "best.pt", map_location="cpu", weights_only=False)
        scaler = load_scaler(trunk / "scaler.json")
        full, _ = assemble(holdout, "full", scaler=scaler, text=state["cfg"].get("text_features"))
        tab = _statless(full)
        for i, col in enumerate(TARGETS):  # predicted overall stats in, masks off
            sc = scaler[col]
            tab[col] = ((preds["static+text"][:, i] - sc["mean"]) / sc["std"]).astype(np.float32)
            tab[f"{col}__miss"] = np.float32(0.0)
        table = torch.tensor(add_bias_row(tab))
        model = build_trunk(state["cfg"], table.shape[1])
        model.load_state_dict(state["model"])
        scorer = TorchScorer(model, f"{state['cfg']['exp']}-imputed-{encoder}@best", device_auto())
        model.embedding.set_context(table.to(scorer.device))
        card = evaluate(scorer, holdout, "val", "text-imputed", out_dir=out_dir)
        print(render_markdown(card), flush=True)
        cards.append(card)
    return cards


def _md_table(df: pd.DataFrame) -> str:
    """Markdown table without the optional `tabulate` dependency."""
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(prog="draftbot.eval.text_probe")
    p.add_argument("--holdout", default="HOB")
    p.add_argument("--corpus", type=Path, default=Path("configs/corpus/pretrain_v2.yaml"))
    p.add_argument("--encoders", nargs="+", default=[DEFAULT_ENCODER])
    p.add_argument("--formats", nargs="+", default=["masked"])
    p.add_argument("--trunk", type=Path, default=Path("checkpoints/EXP-033"))
    p.add_argument("--out", type=Path, default=OUT_DIR)
    args = p.parse_args(argv)
    manifest = yaml.safe_load(args.corpus.read_text())
    assert args.holdout in manifest.get("quarantine", []) or args.holdout not in manifest["sets"], \
        f"{args.holdout} is in the corpus — not a holdout"
    corpus = [c for c in manifest["sets"]
              if (PROCESSED_DIR / c / "features.static.parquet").exists()]
    args.out.mkdir(parents=True, exist_ok=True)
    rows, harness = [], []
    for enc in args.encoders:
        for fmt in args.formats:
            print(f"== {enc} / {fmt}: {len(corpus)} corpus sets → {args.holdout}", flush=True)
            r, preds = probe(corpus, args.holdout, enc, fmt)
            rows += r
            if fmt == "masked":
                harness += harness_readouts(args.holdout, preds, enc, args.trunk, args.out)
    df = pd.DataFrame(rows)
    stamp = date.today().isoformat()
    md = [f"# Text probe — {args.holdout} holdout, {len(corpus)}-set corpus ({stamp})", "",
          "Spearman ρ between ridge-predicted and true full-season stat. LOSO = "
          "leave-one-set-out over the corpus (mean ± sd).", "",
          _md_table(df), ""]
    if harness:
        md += ["## Harness readouts (holdout val, day-0 condition)", ""]
        md += ["| model | top-1 | expert top-1 | skill-gap | rare-take × | colors |", "|---|---|---|---|---|---|"]
        for c in harness:
            m = c["metrics"]
            md.append(f"| {c['model']} | {m['top1']:.4f} | {m['expert_top1']:.4f} | "
                      f"{m['skill_gap']:+.4f} | {m['rare_take_ratio_vs_expert']:.2f} | "
                      f"{m['pool_coherence_colors']:.2f} |")
    (args.out / f"{args.holdout}.{stamp}.md").write_text("\n".join(md) + "\n")
    (args.out / f"{args.holdout}.{stamp}.json").write_text(json.dumps(
        {"rows": rows, "harness": [{"model": c["model"], "metrics": c["metrics"]} for c in harness]},
        indent=1))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
