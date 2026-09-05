# Text probe — HOB holdout, 30-set corpus (2026-09-05)

Spearman ρ between ridge-predicted and true full-season stat. LOSO = leave-one-set-out over the corpus (mean ± sd).

| encoder | format | features | lambda | loso_gih_mean | loso_gih_sd | HOB_gih | HOB_game_win_rate | HOB_drawn_improvement_win_rate | HOB_avg_seen | HOB_avg_pick |
|---|---|---|---|---|---|---|---|---|---|---|
| bge-large | masked | static | 100.000 | 0.327 | 0.094 | 0.180 | 0.245 | 0.252 | 0.768 | 0.714 |
| bge-large | masked | text | 1000.000 | 0.417 | 0.081 | 0.299 | 0.308 | 0.341 | 0.775 | 0.722 |
| bge-large | masked | static+text | 1000.000 | 0.434 | 0.081 | 0.329 | 0.357 | 0.355 | 0.796 | 0.747 |
| bge-large | masked | text-knn10 | nan | 0.334 | 0.072 | 0.179 | nan | nan | nan | nan |
| bge-large | masked | text-knn30 | nan | 0.350 | 0.079 | 0.232 | nan | nan | nan | nan |
| bge-large | masked | static+text (mlp) | nan | 0.428 | 0.073 | 0.330 | 0.366 | 0.321 | 0.797 | 0.743 |
| bge-base | masked | static | 100.000 | 0.327 | 0.094 | 0.180 | 0.245 | 0.252 | 0.768 | 0.714 |
| bge-base | masked | text | 100.000 | 0.425 | 0.076 | 0.370 | 0.377 | 0.345 | 0.762 | 0.714 |
| bge-base | masked | static+text | 100.000 | 0.446 | 0.077 | 0.399 | 0.411 | 0.400 | 0.796 | 0.748 |
| bge-base | masked | text-knn10 | nan | 0.361 | 0.075 | 0.161 | nan | nan | nan | nan |
| bge-base | masked | text-knn30 | nan | 0.383 | 0.079 | 0.206 | nan | nan | nan | nan |
| bge-base | masked | static+text (mlp) | nan | 0.439 | 0.068 | 0.328 | 0.305 | 0.329 | 0.770 | 0.728 |

## Harness readouts (holdout val, day-0 condition)

| model | top-1 | expert top-1 | skill-gap | rare-take × | colors |
|---|---|---|---|---|---|
| textgih-bge-large-text | 0.2764 | 0.2756 | +0.0013 | 0.88 | 3.13 |
| textgih-bge-large-static_text | 0.2856 | 0.2865 | +0.0018 | 0.90 | 3.30 |
| EXP-033-imputed-bge-large@best | 0.5293 | 0.5229 | -0.0073 | 2.11 | 2.53 |
| textgih-bge-base-text | 0.2872 | 0.2912 | +0.0078 | 0.90 | 3.20 |
| textgih-bge-base-static_text | 0.2816 | 0.2869 | +0.0093 | 0.87 | 3.08 |
| EXP-033-imputed-bge-base@best | 0.5315 | 0.5268 | -0.0042 | 2.17 | 2.62 |
