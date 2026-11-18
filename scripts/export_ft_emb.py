"""Export TXT-001's fine-tuned embeddings as a feature block (`ft:TXT-001`).

    uv run python scripts/export_ft_emb.py            # from the fold caches
    uv run python scripts/export_ft_emb.py --afr      # + train the AFR model first

Corpus set  -> the embedding from the one fold model that held it out.
HOB / FRA   -> fold 0's embedding (ONE model, like every corpus card; not the
               6-model mean, whose misaligned components cancel and shrink).
AFR         -> no labels, so no fold holds it out and TXT-001 kept no models:
               --afr retrains the fold-0 recipe (train = folds 1-5) and embeds it.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CK = Path("checkpoints/TXT-001")


def write(code: str, e: np.ndarray) -> None:
    pd.DataFrame(e.astype(np.float32), columns=[f"text_ft_{i:03d}" for i in range(e.shape[1])]
                 ).to_parquet(CK / f"emb.{code}.parquet")


def main() -> None:
    folds = json.loads((CK / "folds.json").read_text())
    if "--afr" in sys.argv and not (CK / "emb.AFR.parquet").exists():
        from draftbot.train.text_strength import fit_fold, load_set
        cfg = yaml.safe_load(Path("configs/TXT-001.yaml").read_text())
        train = [load_set(c, cfg["encoder"]) for f in folds[1:] for c in f]
        _, embs = fit_fold(cfg, train, [load_set("AFR", cfg["encoder"])], cfg["seed"])
        write("AFR", embs["AFR"])
    for k, held in enumerate(folds):
        z = np.load(CK / f"fold{k}.npz")
        for c in held:
            write(c, z[f"e_{c}"])
        if k == 0:
            for h in ("HOB", "FRA"):
                write(h, z[f"e_{h}"])
    print(len(list(CK.glob("emb.*.parquet"))), "embedding files")


if __name__ == "__main__":
    main()
