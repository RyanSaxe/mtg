# Text probe — HOB holdout, 30-set corpus (2026-09-05)

Spearman ρ between ridge-predicted and true full-season stat. LOSO = leave-one-set-out over the corpus (mean ± sd).

| encoder | format | features | lambda | loso_gih_mean | loso_gih_sd | HOB_gih | HOB_game_win_rate | HOB_drawn_improvement_win_rate | HOB_avg_seen | HOB_avg_pick |
|---|---|---|---|---|---|---|---|---|---|---|
| bge-base | named | static | 100.000 | 0.327 | 0.094 | 0.180 | 0.245 | 0.252 | 0.768 | 0.714 |
| bge-base | named | text | 1000.000 | 0.376 | 0.072 | 0.334 | 0.327 | 0.278 | 0.750 | 0.691 |
| bge-base | named | static+text | 1000.000 | 0.415 | 0.073 | 0.400 | 0.385 | 0.373 | 0.794 | 0.738 |
| bge-base | named | text-knn10 | nan | 0.314 | 0.083 | 0.218 | nan | nan | nan | nan |
| bge-base | named | text-knn30 | nan | 0.337 | 0.088 | 0.223 | nan | nan | nan | nan |
| bge-base | oracle | static | 100.000 | 0.327 | 0.094 | 0.180 | 0.245 | 0.252 | 0.768 | 0.714 |
| bge-base | oracle | text | 1000.000 | 0.340 | 0.070 | 0.285 | 0.288 | 0.322 | 0.473 | 0.444 |
| bge-base | oracle | static+text | 1000.000 | 0.438 | 0.075 | 0.378 | 0.378 | 0.383 | 0.784 | 0.734 |
| bge-base | oracle | text-knn10 | nan | 0.278 | 0.063 | 0.240 | nan | nan | nan | nan |
| bge-base | oracle | text-knn30 | nan | 0.304 | 0.059 | 0.222 | nan | nan | nan | nan |

