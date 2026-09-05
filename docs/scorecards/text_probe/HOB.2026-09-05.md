# Text probe — HOB holdout, 30-set corpus (2026-09-05)

Spearman ρ between ridge-predicted and true full-season stat. LOSO = leave-one-set-out over the corpus (mean ± sd).

| encoder | format | features | lambda | loso_gih_mean | loso_gih_sd | HOB_gih | HOB_game_win_rate | HOB_drawn_improvement_win_rate | HOB_avg_seen | HOB_avg_pick |
|---|---|---|---|---|---|---|---|---|---|---|
| bge-base | masked | static | 100.000 | 0.327 | 0.094 | 0.180 | 0.245 | 0.252 | 0.768 | 0.714 |
| bge-base | masked | text | 100.000 | 0.425 | 0.076 | 0.370 | 0.377 | 0.345 | 0.762 | 0.714 |
| bge-base | masked | static+text | 100.000 | 0.446 | 0.077 | 0.399 | 0.411 | 0.400 | 0.796 | 0.748 |
| bge-base | masked | text-knn10 | nan | 0.361 | 0.075 | 0.161 | nan | nan | nan | nan |
| bge-base | masked | text-knn30 | nan | 0.383 | 0.079 | 0.206 | nan | nan | nan | nan |
| bge-small | masked | static | 100.000 | 0.327 | 0.094 | 0.180 | 0.245 | 0.252 | 0.768 | 0.714 |
| bge-small | masked | text | 1000.000 | 0.398 | 0.073 | 0.298 | 0.328 | 0.343 | 0.761 | 0.714 |
| bge-small | masked | static+text | 100.000 | 0.429 | 0.069 | 0.353 | 0.395 | 0.387 | 0.797 | 0.750 |
| bge-small | masked | text-knn10 | nan | 0.353 | 0.077 | 0.280 | nan | nan | nan | nan |
| bge-small | masked | text-knn30 | nan | 0.368 | 0.081 | 0.230 | nan | nan | nan | nan |

## Harness readouts (holdout val, day-0 condition)

| model | top-1 | expert top-1 | skill-gap | rare-take × | colors |
|---|---|---|---|---|---|
| textgih-bge-base-text | 0.2872 | 0.2912 | +0.0078 | 0.90 | 3.20 |
| textgih-bge-base-static_text | 0.2816 | 0.2869 | +0.0093 | 0.87 | 3.08 |
| EXP-033-imputed-bge-base@best | 0.5315 | 0.5268 | -0.0042 | 2.17 | 2.62 |
| textgih-bge-small-text | 0.2825 | 0.2841 | +0.0083 | 0.92 | 3.04 |
| textgih-bge-small-static_text | 0.2835 | 0.2878 | +0.0076 | 1.12 | 3.15 |
| EXP-033-imputed-bge-small@best | 0.5346 | 0.5314 | -0.0025 | 2.13 | 2.66 |
