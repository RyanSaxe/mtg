"""One-command new-set onboarding (PLAN P4.T3).

`python -m draftbot.onboard_set --set <CODE> [--base checkpoints/EXP-064]
 [--finetune-mode lora] [--budget-drafts N] [--skip-finetune]`

Pipeline: probe/download draft+game data → vocab/parquet/features/snapshots
(full + week1)/splits/data card → zero-shot scorecard from the pretrained base
(week1 + full stat modes) → LoRA fine-tune when draft data exists → fine-tuned
scorecard. Every artifact lands in the standard locations; scorecards under
docs/scorecards/onboard/<SET>/.
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from draftbot.data import onboard as onboard_mod
from draftbot.data.features import build_snapshot, set_release_date
from draftbot.data.fetch import dest_path, download, probe


def main(argv=None):
    p = argparse.ArgumentParser(prog="draftbot.onboard_set")
    p.add_argument("--set", dest="set_code", required=True)
    p.add_argument("--base", default="checkpoints/EXP-064",
                   help="pretrained trunk (2026-11-17: EXP-064, text + strength prior on pretrain_v3)")
    p.add_argument("--finetune-mode", default="lora",
                   choices=["lora", "head", "full"])
    p.add_argument("--budget-drafts", type=int, default=None,
                   help="cap fine-tune drafts (dry-runs / early-season)")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--skip-finetune", action="store_true")
    p.add_argument("--end-date", default=None,
                   help="stat snapshot end date (default: today)")
    args = p.parse_args(argv)
    code = args.set_code.upper()
    out_dir = Path("docs/scorecards/onboard") / code

    # 1. data acquisition -----------------------------------------------------
    if probe(code, "PremierDraft") is None:
        print(f"{code}: no 17lands draft data on S3 yet — nothing to onboard")
        return 1
    for kind in ("draft", "game"):
        if not dest_path(code, "PremierDraft", kind=kind).exists():
            download(code, "PremierDraft", kind=kind)

    # 2. processing pipeline (idempotent) -------------------------------------
    end = args.end_date or datetime.now().date().isoformat()
    onboard_mod.onboard(code, end_date=end)
    rel = set_release_date(code)
    week1_end = (datetime.strptime(rel, "%Y-%m-%d") + timedelta(days=7)).date()
    build_snapshot(code, rel, str(week1_end), tag="week1")

    # the card ids are final now: rebuild the text-predicted strength prior
    # (the day-0 one was discarded with the provisional vocab)
    from draftbot.data.text_features import ensure_strength_prior
    ensure_strength_prior(code)

    # 3. zero-shot scorecards from the pretrained base ------------------------
    from draftbot.eval.harness import evaluate, render_markdown
    from draftbot.models.loading import scorer_from_checkpoint
    for stats in ("week1", "full"):
        scorer = scorer_from_checkpoint(Path(args.base), code, stats)
        card = evaluate(scorer, code, "val", stats, out_dir=out_dir)
        print(render_markdown(card))

    # 4. adapter fine-tune ----------------------------------------------------
    if args.skip_finetune:
        return 0
    exp = f"ONB-{code}-{args.finetune_mode}"
    cfg = {
        "exp": exp, "model": "modern", "set": code, "stats": "full",
        "seed": 17, "batch_drafts": 64, "epochs": args.epochs, "patience": 3,
        "emb_dim": 256, "layers": 8, "heads": 8, "t_max": 45, "dropout": 0.1,
        "use_set_encoder": True, "pointer_head": True, "feature_only": True,
        "use_set_token": True, "init_from": args.base,
        "scaler_from": args.base, "finetune_mode": args.finetune_mode,
        "weighting": "importance", "priors": False, "label_smoothing": 0.05,
        "peak_lr": 7.0e-5, "lr_warmup": 200, "weight_decay": 0.01,
        "grad_clip": 1.0,
    }
    # A text trunk (EXP-056+) needs the same feature layout at fine-tune time:
    # inherit the text keys from the base checkpoint's config.
    import torch
    base_cfg = torch.load(Path(args.base) / "best.pt", map_location="cpu",
                          weights_only=False)["cfg"]
    for key in ("text_features", "text_when"):
        if base_cfg.get(key):
            cfg[key] = base_cfg[key]
    if args.budget_drafts:
        cfg["max_drafts"] = args.budget_drafts
    cfg_path = Path("configs") / f"{exp}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    from draftbot.train.loop import Trainer
    best = Trainer(cfg).train()
    print(f"{exp}: best expert val top-1 {best:.4f}")
    scorer = scorer_from_checkpoint(Path("checkpoints") / exp, code, "full")
    card = evaluate(scorer, code, "val", "full", out_dir=out_dir)
    print(render_markdown(card))
    return 0


if __name__ == "__main__":
    sys.exit(main())
