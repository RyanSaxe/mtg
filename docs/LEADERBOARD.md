# Leaderboard

All rows produced by `python -m draftbot.eval` (no hand-computed numbers).
Primary metric: top-1 on MSH test; headline: expert-subset top-1.
`skill-gap` = expert-subset top-1 − low-skill-subset top-1 (validates we mimic good drafters).

| EXP | Model | Set(s) | Split | Stats | Top-1 | Top-3 | Expert Top-1 | Skill-gap | NLL | Config | SHA | Wall-clock | Date |
|-----|-------|--------|-------|-------|-------|-------|--------------|-----------|-----|--------|-----|------------|------|
| — | random | MSH | test | full | 0.2322 | 0.5174 | 0.2347 | +0.0041 | 1.8302 | baseline | 7c25359 | <1min | 2026-07-29 |
| — | rarity-first | MSH | test | full | 0.3270 | 0.6334 | 0.3213 | -0.0072 | 13.5253 | baseline | 7c25359 | <1min | 2026-07-29 |
| — | alsa-greedy | MSH | test | full | 0.4672 | 0.8009 | 0.4837 | +0.0249 | 1.4449 | baseline | 7c25359 | <1min | 2026-07-29 |
| — | gih-greedy | MSH | test | full | 0.3785 | 0.7171 | 0.4168 | +0.0626 | 1.7862 | baseline | 7c25359 | <1min | 2026-07-29 |
| — | gih-in-lane | MSH | test | full | 0.4236 | 0.7780 | 0.4608 | +0.0617 | 2.7696 | baseline | 7c25359 | <1min | 2026-07-29 |
| EXP-001 | EXP-001@best | MSH | val | full | 0.5758 | 0.8672 | 0.5994 | +0.0383 | 1.1683 | configs/EXP-001.yaml | ef5a849 | 2.7min | 2026-07-29 |
| EXP-001 | EXP-001@best | MSH | test | full | 0.5736 | 0.8648 | 0.5980 | +0.0390 | 1.1742 | configs/EXP-001.yaml | ef5a849 | 2.7min | 2026-07-29 |
| EXP-011 | EXP-011@best | MSH | val | full | 0.6731 | 0.9406 | 0.6987 | +0.0422 | 0.8602 | configs/EXP-011.yaml | 687af40 | 14min | 2026-07-29 |
| EXP-012 | EXP-012@best | MSH | val | full | 0.6719 | 0.9400 | 0.6976 | +0.0410 | 0.8639 | configs/EXP-012.yaml | 687af40 | 11min | 2026-07-29 |
| EXP-013 | EXP-013@best | MSH | val | full | 0.6778 | 0.9427 | 0.7020 | +0.0376 | 0.8474 | configs/EXP-013.yaml | 687af40 | 28min | 2026-07-29 |
| EXP-014 | EXP-014@best | MSH | val | full | 0.6718 | 0.9398 | 0.6994 | +0.0438 | 0.8569 | configs/EXP-014.yaml | 687af40 | 12min | 2026-07-29 |
| EXP-023 | EXP-023@best | MSH | val | full | 0.6765 | 0.9424 | 0.7011 | +0.0379 | 0.8519 | configs/EXP-023.yaml | 0f5159f | 38min | 2026-07-29 |
| EXP-013 | EXP-013@best | MSH | test | full | 0.6756 | 0.9410 | 0.6985 | +0.0386 | 0.8560 | configs/EXP-013.yaml | 0f5159f | 28min | 2026-07-29 |
| EXP-020 | EXP-020@best | MSH | test | full | 0.6671 | 0.9364 | 0.6902 | +0.0400 | 0.8789 | configs/EXP-020.yaml | 0f5159f | 17min | 2026-07-29 |
| EXP-023 | EXP-023@best | MSH | test | full | 0.6742 | 0.9409 | 0.6962 | +0.0366 | 0.8595 | configs/EXP-023.yaml | 0f5159f | 38min | 2026-07-29 |
| EXP-022 | EXP-022@best | MSH | val | full | 0.6741 | 0.9411 | 0.6995 | +0.0402 | 0.8554 | configs/EXP-022.yaml | 8abd93f | 10min | 2026-07-29 |
| — | gih-greedy | MSH | test | week1 | 0.3793 | 0.7177 | 0.4174 | +0.0637 | 1.7844 | baseline | 634c503 | <1min | 2026-07-29 |
| EXP-030 | EXP-030-zeroshot@best | MSH | test | week1 | 0.6486 | 0.9265 | 0.6696 | +0.0358 | 0.9324 | configs/EXP-030.yaml | 634c503 | 98min | 2026-07-29 |
| EXP-030 | EXP-030-zeroshot@best | MSH | test | none | 0.5239 | 0.8545 | 0.5167 | -0.0112 | 1.2040 | configs/EXP-030.yaml | 634c503 | 98min | 2026-07-29 |
| EXP-031 | EXP-031@best | MSH | test | full | 0.6752 | 0.9421 | 0.6973 | +0.0383 | 0.8533 | configs/EXP-031.yaml | 634c503 | 33min | 2026-07-29 |
| EXP-032 | EXP-032@best | MSH | test | full | 0.6752 | 0.9420 | 0.6972 | +0.0379 | 0.8531 | configs/EXP-032.yaml | ed4c23a | 42min | 2026-07-29 |
| EXP-040-1k | EXP-040-1k@best | MSH | val | full | 0.6666 | 0.9377 | 0.6935 | +0.0416 | 0.8757 | configs/EXP-040-1k.yaml | 6959f47 | grid | 2026-07-29 |
| EXP-040-5k | EXP-040-5k@best | MSH | val | full | 0.6716 | 0.9404 | 0.6966 | +0.0391 | 0.8651 | configs/EXP-040-5k.yaml | 6959f47 | grid | 2026-07-29 |
| EXP-041-1k | EXP-041-1k@best | MSH | val | full | 0.6654 | 0.9370 | 0.6919 | +0.0416 | 0.8802 | configs/EXP-041-1k.yaml | 6959f47 | grid | 2026-07-29 |
| EXP-041-5k | EXP-041-5k@best | MSH | val | full | 0.6698 | 0.9392 | 0.6946 | +0.0390 | 0.8672 | configs/EXP-041-5k.yaml | 6959f47 | grid | 2026-07-29 |
| EXP-041-all | EXP-041-all@best | MSH | val | full | 0.6770 | 0.9424 | 0.7021 | +0.0391 | 0.8498 | configs/EXP-041-all.yaml | 6959f47 | grid | 2026-07-29 |
| EXP-042-1k | EXP-042-1k@best | MSH | val | full | 0.6634 | 0.9360 | 0.6899 | +0.0414 | 0.8851 | configs/EXP-042-1k.yaml | 6959f47 | grid | 2026-07-29 |
| EXP-042-5k | EXP-042-5k@best | MSH | val | full | 0.6672 | 0.9386 | 0.6924 | +0.0395 | 0.8732 | configs/EXP-042-5k.yaml | 6959f47 | grid | 2026-07-29 |
| EXP-042-all | EXP-042-all@best | MSH | val | full | 0.6733 | 0.9405 | 0.6987 | +0.0399 | 0.8611 | configs/EXP-042-all.yaml | 6959f47 | grid | 2026-07-29 |

## Deck builder (Phase 5, EXP-1NN)

Deck rows from `python -m draftbot.eval --deck`. Primary: deck-F1 (full 40 incl.
basics) on MSH val; headline: trophy-F1 (eval builds with ≥5 wins — agree with
winning builds). Target = rebuild-ceiling (human self-agreement).

| EXP | Model | Set | Split | Stats | F1 | NB-F1 | Trophy-F1 | Win-wtd F1 | Lands-MAE | Basics-L1 | Config | SHA | Wall-clock | Date |
|-----|-------|-----|-------|-------|----|-------|-----------|------------|-----------|-----------|--------|-----|------------|------|
| — | random-legal | MSH | val | full | 0.6821 | 0.6076 | 0.6880 | 0.6829 | 1.826 | 6.539 | baseline | a980284 | <1min | 2026-07-30 |
| — | gih-top23 | MSH | val | full | 0.7601 | 0.7362 | 0.7736 | 0.7699 | 2.221 | 6.499 | baseline | a980284 | <1min | 2026-07-30 |
| — | gih-in-lane-build | MSH | val | full | 0.7844 | 0.7765 | 0.7915 | 0.7933 | 2.658 | 6.474 | baseline | a980284 | <1min | 2026-07-30 |
| — | rebuild-ceiling | MSH | val | full | 0.9545 | 0.9469 | 0.9627 | 0.9608 | 0.162 | 0.973 | baseline | a980284 | <1min | 2026-07-30 |
| EXP-101 | EXP-101@best | MSH | val | full | 0.8829 | 0.8625 | 0.8920 | 0.8894 | 0.247 | 2.492 | configs/EXP-101.yaml | 8433d49 | 2.3min | 2026-07-30 |
| EXP-102 | EXP-102@best | MSH | val | full | 0.8879 | 0.8672 | 0.8960 | 0.8942 | 0.248 | 2.320 | configs/EXP-102.yaml | 2fedfd8 | 2.5min | 2026-07-30 |
| EXP-103 | EXP-103@best | MSH | val | none | 0.8802 | 0.8567 | 0.8885 | 0.8864 | 0.244 | 2.412 | configs/EXP-103.yaml | 2fedfd8 | 2.4min | 2026-07-30 |
| EXP-104 | EXP-104@best | MSH | val | full | 0.8852 | 0.8659 | 0.8946 | 0.8920 | 0.273 | 2.467 | configs/EXP-104.yaml | 2fedfd8 | 3.6min | 2026-07-30 |
| EXP-105 | EXP-105@best | MSH | val | full | 0.8814 | 0.8621 | 0.8906 | 0.8881 | 0.248 | 2.589 | configs/EXP-105.yaml | 2fedfd8 | 1.4min | 2026-07-30 |

### Deck builder — trophy-F1 v2 + winner-pref (owner refinement 2026-07-30; supersedes rows above)

Trophy-F1 v2: pools that DID trophy, target = the build that won. Winner-pref:
model's deck strictly closer to the winning than the losing build of the same
pool (91 val pairs — underpowered at val scale; treat ±0.05 as noise).

| EXP | Model | Set | Split | Stats | F1 | NB-F1 | Trophy-F1v2 | Win-pref | Win-wtd F1 | Lands-MAE | Basics-L1 | Config | SHA | Wall-clock | Date |
|-----|-------|-----|-------|-------|----|-------|-------------|----------|------------|-----------|-----------|--------|-----|------------|------|
| — | random-legal | MSH | val | full | 0.6821 | 0.6076 | 0.6880 | 0.527 | 0.6829 | 1.826 | 6.539 | baseline | 66dd1e6 | <1min | 2026-07-30 |
| — | gih-top23 | MSH | val | full | 0.7601 | 0.7362 | 0.7736 | 0.495 | 0.7699 | 2.221 | 6.499 | baseline | 66dd1e6 | <1min | 2026-07-30 |
| — | gih-in-lane-build | MSH | val | full | 0.7844 | 0.7765 | 0.7915 | 0.473 | 0.7933 | 2.658 | 6.474 | baseline | 66dd1e6 | <1min | 2026-07-30 |
| — | rebuild-ceiling | MSH | val | full | 0.9545 | 0.9469 | 0.9625 | nan | 0.9608 | 0.162 | 0.973 | baseline | 66dd1e6 | <1min | 2026-07-30 |
| EXP-101 | EXP-101@best | MSH | val | full | 0.8829 | 0.8625 | 0.8920 | 0.467 | 0.8894 | 0.247 | 2.492 | configs/EXP-101.yaml | 66dd1e6 | 2.3min | 2026-07-30 |
| EXP-102 | EXP-102@best | MSH | val | full | 0.8879 | 0.8672 | 0.8960 | 0.478 | 0.8942 | 0.248 | 2.320 | configs/EXP-102.yaml | 66dd1e6 | 2.5min | 2026-07-30 |
| EXP-103 | EXP-103@best | MSH | val | none | 0.8802 | 0.8567 | 0.8885 | 0.522 | 0.8864 | 0.244 | 2.412 | configs/EXP-103.yaml | 66dd1e6 | 2.4min | 2026-07-30 |
| EXP-104 | EXP-104@best | MSH | val | full | 0.8852 | 0.8659 | 0.8946 | 0.462 | 0.8920 | 0.273 | 2.467 | configs/EXP-104.yaml | 66dd1e6 | 3.6min | 2026-07-30 |
| EXP-105 | EXP-105@best | MSH | val | full | 0.8814 | 0.8621 | 0.8906 | 0.484 | 0.8881 | 0.248 | 2.589 | configs/EXP-105.yaml | 66dd1e6 | 1.4min | 2026-07-30 |
| EXP-106 | EXP-106@best | MSH | val | full | 0.8856 | 0.8673 | 0.8942 | 0.462 | 0.8919 | 0.256 | 2.504 | configs/EXP-106.yaml | 49ff68e | 5.5min | 2026-07-30 |
| EXP-107 | EXP-107@best | MSH | val | full | 0.8893 | 0.8675 | 0.8977 | 0.473 | 0.8956 | 0.248 | 2.217 | configs/EXP-107.yaml | 49ff68e | 2.6min | 2026-07-30 |
| EXP-108 | EXP-108@best | MSH | val | full | 0.8870 | 0.8676 | 0.8951 | 0.505 | 0.8933 | 0.678 | 2.404 | configs/EXP-108.yaml | 4c81490 | 2.7min | 2026-07-30 |
| EXP-109 | EXP-109@best | MSH | val | full | 0.8884 | 0.8676 | 0.8967 | 0.451 | 0.8945 | 0.252 | 2.299 | configs/EXP-109.yaml | 4c81490 | 2.9min | 2026-07-30 |
| EXP-110 | EXP-110@best | MSH | val | full | 0.8873 | 0.8668 | 0.8959 | 0.478 | 0.8937 | 0.252 | 2.345 | configs/EXP-110.yaml | 3a8c483 | 7.1min | 2026-07-30 |
| EXP-107 | EXP-107@best | MSH | test | full | 0.8888 | 0.8669 | 0.8968 | 0.386 | 0.8941 | 0.254 | 2.228 | configs/EXP-107.yaml | 3a8c483 | 2.6min | 2026-07-30 |
| — | rebuild-ceiling | MSH | test | full | 0.9535 | 0.9465 | 0.9654 | nan | 0.9583 | 0.180 | 1.044 | baseline | 3a8c483 | <1min | 2026-07-30 |
| EXP-111 | EXP-111@best | MSH | val | full | 0.8856 | 0.8635 | 0.8936 | 0.473 | 0.8917 | 0.243 | 2.324 | configs/EXP-111.yaml | 97d376b | 3.9min | 2026-07-30 |
| EXP-112 | EXP-112@best | MSH | val | full | 0.8861 | 0.8637 | 0.8946 | 0.467 | 0.8923 | 0.247 | 2.294 | configs/EXP-112.yaml | 97d376b | 4.1min | 2026-07-30 |
| EXP-113 | EXP-113@best | MSH | val | full | 0.8862 | 0.8642 | 0.8948 | 0.462 | 0.8924 | 0.246 | 2.307 | configs/EXP-113.yaml | 97d376b | 16min | 2026-07-30 |
| EXP-114 | EXP-114@best | MSH | val | full | 0.8861 | 0.8637 | 0.8946 | 0.467 | 0.8923 | 0.246 | 2.294 | configs/EXP-114.yaml | 81df966 | 6.8min | 2026-07-30 |
| EXP-115 | EXP-115@best | MSH | val | full | 0.8858 | 0.8636 | 0.8941 | 0.462 | 0.8920 | 0.248 | 2.311 | configs/EXP-115.yaml | 81df966 | 4.2min | 2026-07-30 |
| EXP-116 | EXP-116@best | MSH | val | full | 0.8862 | 0.8639 | 0.8951 | 0.489 | 0.8928 | 0.247 | 2.294 | configs/EXP-116.yaml | 81df966 | 4.0min | 2026-07-30 |
| EXP-117 | EXP-117@best | MSH | val | full | 0.8859 | 0.8641 | 0.8931 | 0.495 | 0.8919 | 0.249 | 2.328 | configs/EXP-117.yaml | 81df966 | 4.5min | 2026-07-30 |
| EXP-117 | EXP-117@best+rescore | MSH | val | full | 0.8860 | 0.8644 | 0.8931 | 0.511 | 0.8920 | 0.249 | 2.335 | configs/EXP-117.yaml | 81df966 | 4.5min | 2026-07-30 |
| EXP-120 | EXP-120@best | MSH | val | full | 0.8782 | 0.8525 | 0.8871 | 0.478 | 0.8843 | 0.241 | 2.360 | configs/EXP-120.yaml | da5e4d5 | 157min | 2026-07-30 |
| EXP-120 | EXP-120@best | MSH | val | none | 0.8253 | 0.7769 | 0.8298 | 0.544 | 0.8285 | 0.259 | 2.819 | configs/EXP-120.yaml | da5e4d5 | 157min | 2026-07-30 |
| EXP-121 | EXP-121@best | MSH | val | full | 0.8920 | 0.8697 | 0.9005 | 0.478 | 0.8985 | 0.240 | 2.118 | configs/EXP-121.yaml | da5e4d5 | 3.1min | 2026-07-30 |
| EXP-122 | EXP-122@best | MSH | val | full | 0.8792 | 0.8532 | 0.8878 | 0.489 | 0.8855 | 0.247 | 2.312 | configs/EXP-122.yaml | 2fee29e | 8.1h | 2026-07-30 |
| EXP-122 | EXP-122@best | MSH | val | none | 0.8241 | 0.7744 | 0.8283 | 0.533 | 0.8273 | 0.254 | 2.814 | configs/EXP-122.yaml | 2fee29e | 8.1h | 2026-07-30 |
| EXP-123 | EXP-123@best | MSH | val | full | 0.8920 | 0.8709 | 0.8998 | 0.484 | 0.8984 | 0.242 | 2.170 | configs/EXP-123.yaml | 2fee29e | 5.8min | 2026-07-30 |
| EXP-124 | EXP-124@best | MSH | val | none | 0.8269 | 0.7783 | 0.8330 | 0.544 | 0.8313 | 0.268 | 2.776 | configs/EXP-124.yaml | b038c76 | 5.1h | 2026-07-31 |
| EXP-124 | EXP-124@best | MSH | val | full | 0.8770 | 0.8513 | 0.8859 | 0.473 | 0.8832 | 0.241 | 2.396 | configs/EXP-124.yaml | b038c76 | 5.1h | 2026-07-31 |
| EXP-125 | EXP-125@best | MSH | val | full | 0.8923 | 0.8702 | 0.9002 | 0.489 | 0.8988 | 0.239 | 2.127 | configs/EXP-125.yaml | b038c76 | 8.5min | 2026-07-31 |
| EXP-127 | EXP-127@best | MSH | val | full | 0.8927 | 0.8704 | 0.9011 | 0.467 | 0.8992 | 0.242 | 2.099 | configs/EXP-127.yaml | 09846d9 | 8min | 2026-07-31 |
| EXP-126 | EXP-126@best | EOE | val | none | 0.8330 | 0.7669 | 0.8347 | 0.528 | 0.8346 | 0.180 | 2.252 | configs/EXP-126.yaml | 09846d9 | 4.9h | 2026-07-31 |
| EXP-126 | EXP-126@best | EOE | val | full | 0.8933 | 0.8611 | 0.8981 | 0.509 | 0.8969 | 0.176 | 1.938 | configs/EXP-126.yaml | 09846d9 | 4.9h | 2026-07-31 |
| EXP-124 | EXP-124@best | EOE | val | none | 0.8944 | 0.8631 | 0.8997 | 0.495 | 0.8981 | 0.173 | 1.937 | configs/EXP-124.yaml | 09846d9 | 5.1h | 2026-07-31 |
| EXP-124 | EXP-124@best | EOE | val | full | 0.9034 | 0.8744 | 0.9101 | 0.532 | 0.9081 | 0.173 | 1.755 | configs/EXP-124.yaml | 09846d9 | 5.1h | 2026-07-31 |
| EXP-128 | EXP-128@best | EOE | val | none | 0.8321 | 0.7653 | 0.8336 | 0.523 | 0.8330 | 0.179 | 2.273 | configs/EXP-128.yaml | 477a36a | 4.2h | 2026-07-31 |
| EXP-128 | EXP-128@best | EOE | val | full | 0.8937 | 0.8618 | 0.8991 | 0.546 | 0.8976 | 0.173 | 1.942 | configs/EXP-128.yaml | 477a36a | 4.2h | 2026-07-31 |
| EXP-129 | EXP-129@best | MSH | val | full | 0.8921 | 0.8701 | 0.9002 | 0.467 | 0.8988 | 0.242 | 2.123 | configs/EXP-129.yaml | 477a36a | 7min | 2026-07-31 |
| EXP-126 | EXP-126@best | MSH | val | full | 0.8921 | 0.8694 | 0.9004 | 0.456 | 0.8984 | 0.242 | 2.094 | configs/EXP-126.yaml | f1d03f9 | 4.9h | 2026-07-31 |
| EXP-135 | EXP-135@best | MSH | val | full | 0.8904 | 0.8683 | 0.8989 | 0.505 | 0.8968 | 0.241 | 2.169 | configs/EXP-135.yaml | b79aa9d | 2.4h | 2026-08-01 |
| EXP-135 | EXP-135@best | EOE | val | full | 0.8926 | 0.8624 | 0.8987 | 0.537 | 0.8967 | 0.180 | 2.051 | configs/EXP-135.yaml | b79aa9d | 2.4h | 2026-08-01 |
| EXP-135 | EXP-135@best | EOE | val | none | 0.8311 | 0.7660 | 0.8327 | 0.555 | 0.8326 | 0.195 | 2.348 | configs/EXP-135.yaml | b79aa9d | 2.4h | 2026-08-01 |

### Sealed decks (P5.S — additive; draft rows above unaffected)

Sealed pools ≈ 84 unfiltered nonbasics — color choice dominates the task, and
the human ceiling sits lower than draft's (more legitimate builds per pool).
Winner-pref has ~14 val pairs on TLA: reported, but statistically meaningless.
Rows are on the curated extraction (deck 40-43, pool-consistent rebuilds;
~1% dropped, logged) — pre-curation rows differed by ≤0.001.

| EXP | Model | Set | Split | Stats | F1 | NB-F1 | Trophy-F1v2 | Win-pref | Win-wtd F1 | Lands-MAE | Basics-L1 | Config | SHA | Wall-clock | Date |
|-----|-------|-----|-------|-------|----|-------|-------------|----------|------------|-----------|-----------|--------|-----|------------|------|
| — | rebuild-ceiling | TLA/sealed | val | full | 0.9066 | 0.9032 | 0.9224 | nan | 0.9369 | 0.154 | 2.684 | baseline | 2744973 | <1min | 2026-08-01 |
| — | random-legal | TLA/sealed | val | full | 0.4187 | 0.3068 | 0.4218 | 0.643 | 0.4208 | 1.602 | 13.265 | baseline | 2744973 | <1min | 2026-08-01 |
| — | gih-top23 | TLA/sealed | val | full | 0.5205 | 0.4556 | 0.5421 | 0.536 | 0.5381 | 1.437 | 12.269 | baseline | 2744973 | <1min | 2026-08-01 |
| — | gih-in-lane-build | TLA/sealed | val | full | 0.6024 | 0.5527 | 0.6354 | 0.464 | 0.6215 | 2.706 | 10.359 | baseline | 2744973 | <1min | 2026-08-01 |
| EXP-126 | EXP-126@best | TLA/sealed | val | full | 0.6923 | 0.6964 | 0.7195 | 0.321 | 0.7108 | 0.294 | 9.207 | configs/EXP-126.yaml | 2744973 | 0 (zero-shot) | 2026-08-01 |
| EXP-130 | EXP-130@best | TLA/sealed | val | full | 0.7269 | 0.7159 | 0.7570 | 0.286 | 0.7473 | 0.282 | 7.828 | configs/EXP-130.yaml | 2744973 | 0.7min | 2026-08-01 |
| — | rebuild-ceiling | EOE/sealed | val | full | 0.8157 | 0.8064 | 0.8971 | nan | 0.8558 | 0.082 | 5.451 | baseline | e3d7116 | <1min | 2026-08-01 |
| — | gih-in-lane-build | EOE/sealed | val | full | 0.5209 | 0.4963 | 0.5504 | 0.537 | 0.5306 | 1.226 | 14.649 | baseline | e3d7116 | <1min | 2026-08-01 |
| EXP-126 | EXP-126@best | EOE/sealed | val | full | 0.5516 | 0.5513 | 0.5708 | 0.500 | 0.5616 | 0.146 | 14.144 | configs/EXP-126.yaml | e3d7116 | 0 (zero-shot) | 2026-08-01 |
| EXP-126 | EXP-126@best | EOE/sealed | val | none | 0.4702 | 0.4377 | 0.4795 | 0.574 | 0.4780 | 0.146 | 14.965 | configs/EXP-126.yaml | e3d7116 | 0 (zero-shot) | 2026-08-01 |
| EXP-131 | EXP-131@best | TLA/sealed | val | full | 0.7280 | 0.7176 | 0.7533 | 0.286 | 0.7455 | 0.282 | 7.841 | configs/EXP-131.yaml | e3d7116 | 13min | 2026-08-01 |
| EXP-131 | EXP-131@best | EOE/sealed | val | full | 0.6021 | 0.5932 | 0.6335 | 0.574 | 0.6191 | 0.146 | 12.335 | configs/EXP-131.yaml | e3d7116 | 13min | 2026-08-01 |
| EXP-131 | EXP-131@best | EOE/sealed | val | none | 0.5061 | 0.4675 | 0.5319 | 0.500 | 0.5183 | 0.146 | 13.994 | configs/EXP-131.yaml | e3d7116 | 13min | 2026-08-01 |
| EXP-132 | EXP-132@best | TLA/sealed | val | full | 0.7133 | 0.7028 | 0.7355 | 0.321 | 0.7276 | 0.294 | 8.421 | configs/EXP-132.yaml | e3d7116 | 23min | 2026-08-01 |
| EXP-132 | EXP-132@best | EOE/sealed | val | full | 0.6011 | 0.5861 | 0.6314 | 0.537 | 0.6202 | 0.146 | 12.156 | configs/EXP-132.yaml | e3d7116 | 23min | 2026-08-01 |
| EXP-132 | EXP-132@best | EOE/sealed | val | none | 0.5203 | 0.4905 | 0.5464 | 0.407 | 0.5325 | 0.146 | 13.737 | configs/EXP-132.yaml | e3d7116 | 23min | 2026-08-01 |
| EXP-134 | EXP-134@best | TLA/sealed | val | full | 0.7211 | 0.7131 | 0.7516 | 0.214 | 0.7408 | 0.288 | 8.136 | configs/EXP-134.yaml | e3d7116 | 24min | 2026-08-01 |
| EXP-134 | EXP-134@best | EOE/sealed | val | full | 0.6075 | 0.5973 | 0.6421 | 0.611 | 0.6249 | 0.146 | 12.094 | configs/EXP-134.yaml | e3d7116 | 24min | 2026-08-01 |
| EXP-134 | EXP-134@best | EOE/sealed | val | none | 0.5200 | 0.4823 | 0.5432 | 0.519 | 0.5338 | 0.146 | 13.593 | configs/EXP-134.yaml | e3d7116 | 24min | 2026-08-01 |
| EXP-133 | EXP-133@best | TLA/sealed | val | full | 0.7270 | 0.7158 | 0.7548 | 0.286 | 0.7467 | 0.294 | 7.783 | configs/EXP-133.yaml | e3d7116 | 0.4min | 2026-08-01 |
| EXP-135 | EXP-135@best | TLA/sealed | val | full | 0.7278 | 0.7166 | 0.7576 | 0.250 | 0.7484 | 0.294 | 7.786 | configs/EXP-135.yaml | b79aa9d | 2.4h | 2026-08-01 |
| EXP-135 | EXP-135@best | EOE/sealed | val | full | 0.6022 | 0.5904 | 0.6328 | 0.556 | 0.6181 | 0.146 | 12.187 | configs/EXP-135.yaml | b79aa9d | 2.4h | 2026-08-01 |
| EXP-135 | EXP-135@best | EOE/sealed | val | none | 0.5082 | 0.4689 | 0.5231 | 0.519 | 0.5186 | 0.146 | 13.797 | configs/EXP-135.yaml | b79aa9d | 2.4h | 2026-08-01 |
| EXP-136 | EXP-136@best | TLA/sealed | val | full | 0.7279 | 0.7150 | 0.7559 | 0.321 | 0.7478 | 0.285 | 7.683 | configs/EXP-136.yaml | b79aa9d | 8min | 2026-08-01 |
| EXP-136 | EXP-136@best | EOE/sealed | val | full | 0.6022 | 0.5895 | 0.6305 | 0.556 | 0.6175 | 0.146 | 12.158 | configs/EXP-136.yaml | b79aa9d | 8min | 2026-08-01 |
| EXP-136 | EXP-136@best | EOE/sealed | val | none | 0.5162 | 0.4752 | 0.5373 | 0.500 | 0.5277 | 0.146 | 13.538 | configs/EXP-136.yaml | b79aa9d | 8min | 2026-08-01 |
| EXP-137 | EXP-137@best | TLA/sealed | val | full | 0.7267 | 0.7153 | 0.7551 | 0.321 | 0.7464 | 0.307 | 7.799 | configs/EXP-137.yaml | b79aa9d | 14min | 2026-08-01 |
| EXP-137 | EXP-137@best | EOE/sealed | val | full | 0.5976 | 0.5815 | 0.6321 | 0.611 | 0.6153 | 0.146 | 12.140 | configs/EXP-137.yaml | b79aa9d | 14min | 2026-08-01 |
| EXP-137 | EXP-137@best | EOE/sealed | val | none | 0.5190 | 0.4778 | 0.5333 | 0.556 | 0.5292 | 0.146 | 13.407 | configs/EXP-137.yaml | b79aa9d | 14min | 2026-08-01 |

## HOB — the sandbag (2026-09-05): zero-shot on a never-seen set

17lands dropped HOB on 2026-09-05; HOB is quarantined from every corpus, so
every model row here is a true holdout (journal 2026-09-05). `day3` = the
first three days of data (2026-08-11..13). The three MSH rows close the
EXP-033 provenance gap flagged on 2026-07-31 (MSH is IN-corpus for EXP-033).

| EXP | Model | Set(s) | Split | Stats | Top-1 | Top-3 | Expert Top-1 | Skill-gap | NLL | Config | SHA | Wall-clock | Date |
|-----|-------|--------|-------|-------|-------|-------|--------------|-----------|-----|--------|-----|------------|------|
| EXP-033 | EXP-033-zeroshot@best | MSH | val | none | 0.5391 | 0.8644 | 0.5366 | -0.0015 | 1.1744 | configs/EXP-033.yaml | 3f7449b | 98min | 2026-09-05 |
| EXP-033 | EXP-033-zeroshot@best | MSH | val | week1 | 0.6620 | 0.9353 | 0.6858 | +0.0366 | 0.8919 | configs/EXP-033.yaml | 3f7449b | 98min | 2026-09-05 |
| EXP-033 | EXP-033-zeroshot@best | MSH | val | full | 0.6691 | 0.9386 | 0.6955 | +0.0403 | 0.8678 | configs/EXP-033.yaml | 3f7449b | 98min | 2026-09-05 |
| — | rarity-first | HOB | val | none | 0.3238 | 0.6435 | 0.3255 | +0.0028 | 12.9846 | baseline | 3f7449b | <1min | 2026-09-05 |
| — | gih-greedy | HOB | val | week1 | 0.3561 | 0.7184 | 0.3844 | +0.0409 | 1.7869 | baseline | 3f7449b | <1min | 2026-09-05 |
| — | gih-greedy | HOB | val | full | 0.3555 | 0.7144 | 0.3848 | +0.0412 | 1.7874 | baseline | 3f7449b | <1min | 2026-09-05 |
| — | gih-in-lane | HOB | val | full | 0.4182 | 0.7958 | 0.4448 | +0.0358 | 2.6827 | baseline | 3f7449b | <1min | 2026-09-05 |
| — | alsa-greedy | HOB | val | full | 0.4511 | 0.7908 | 0.4637 | +0.0115 | 1.5177 | baseline | 3f7449b | <1min | 2026-09-05 |
| EXP-030 | EXP-030-zeroshot@best | HOB | val | none | 0.5435 | 0.8770 | 0.5358 | -0.0115 | 1.1572 | configs/EXP-030.yaml | 3f7449b | 0 (zero-shot) | 2026-09-05 |
| EXP-030 | EXP-030-zeroshot@best | HOB | val | week1 | 0.6595 | 0.9389 | 0.6754 | +0.0193 | 0.8897 | configs/EXP-030.yaml | 3f7449b | 0 (zero-shot) | 2026-09-05 |
| EXP-033 | EXP-033-zeroshot@best | HOB | val | none | 0.5453 | 0.8766 | 0.5387 | -0.0085 | 1.1548 | configs/EXP-033.yaml | 3f7449b | 0 (zero-shot) | 2026-09-05 |
| EXP-033 | EXP-033-zeroshot@best | HOB | val | day3 | 0.6536 | 0.9358 | 0.6644 | +0.0114 | 0.9041 | configs/EXP-033.yaml | 3f7449b | 0 (zero-shot) | 2026-09-05 |
| EXP-033 | EXP-033-zeroshot@best | HOB | val | week1 | 0.6636 | 0.9408 | 0.6801 | +0.0188 | 0.8793 | configs/EXP-033.yaml | 3f7449b | 0 (zero-shot) | 2026-09-05 |
| EXP-033 | EXP-033-zeroshot@best | HOB | val | full | 0.6675 | 0.9420 | 0.6855 | +0.0217 | 0.8732 | configs/EXP-033.yaml | 3f7449b | 0 (zero-shot) | 2026-09-05 |
| EXP-036 | EXP-036@best | HOB | val | full | 0.6905 | 0.9518 | 0.7100 | +0.0229 | 0.8092 | configs/EXP-036.yaml | 3fa7641 | 16min | 2026-09-05 |
| EXP-034 | EXP-034-zeroshot@best | HOB | val | none | 0.5475 | 0.8792 | 0.5391 | -0.0106 | 1.1483 | configs/EXP-034.yaml | a6d9786 | 306min | 2026-09-05 |
| EXP-034 | EXP-034-zeroshot@best | HOB | val | week1 | 0.6672 | 0.9411 | 0.6831 | +0.0181 | 0.8752 | configs/EXP-034.yaml | a6d9786 | 306min | 2026-09-05 |
| EXP-034 | EXP-034-zeroshot@best | HOB | val | full | 0.6699 | 0.9421 | 0.6889 | +0.0215 | 0.8709 | configs/EXP-034.yaml | a6d9786 | 306min | 2026-09-05 |
| EXP-034 | EXP-034-zeroshot@last | HOB | val | none | 0.5499 | 0.8799 | 0.5421 | -0.0083 | 1.1447 | configs/EXP-034.yaml | a6d9786 | 306min | 2026-09-05 |
| EXP-050 | EXP-050-zeroshot@best | HOB | val | none | 0.5543 | 0.8755 | 0.5513 | -0.0001 | 1.1723 | configs/EXP-050.yaml | 7bd48ca | 108min | 2026-09-05 |
| EXP-050 | EXP-050-zeroshot@best | HOB | val | week1 | 0.6579 | 0.9393 | 0.6686 | +0.0133 | 0.8947 | configs/EXP-050.yaml | 7bd48ca | 108min | 2026-09-05 |
| EXP-050 | EXP-050-zeroshot@best | HOB | val | full | 0.6609 | 0.9405 | 0.6737 | +0.0161 | 0.8862 | configs/EXP-050.yaml | 7bd48ca | 108min | 2026-09-05 |
| EXP-050 | EXP-050-zeroshot@best | MSH | val | full | 0.6727 | 0.9410 | 0.6989 | +0.0405 | 0.8589 | configs/EXP-050.yaml | 7bd48ca | 108min | 2026-09-05 |
| EXP-051 | EXP-051-zeroshot@best | HOB | val | none | 0.5536 | 0.8755 | 0.5509 | -0.0016 | 1.1775 | configs/EXP-051.yaml | 7bd48ca | 108min | 2026-09-05 |
| EXP-051 | EXP-051-zeroshot@best | HOB | val | week1 | 0.6566 | 0.9385 | 0.6732 | +0.0189 | 0.8964 | configs/EXP-051.yaml | 7bd48ca | 108min | 2026-09-05 |
| EXP-051 | EXP-051-zeroshot@best | HOB | val | full | 0.6594 | 0.9395 | 0.6780 | +0.0228 | 0.8884 | configs/EXP-051.yaml | 7bd48ca | 108min | 2026-09-05 |
| EXP-051 | EXP-051-zeroshot@best | MSH | val | full | 0.6687 | 0.9388 | 0.6945 | +0.0399 | 0.8707 | configs/EXP-051.yaml | 7bd48ca | 108min | 2026-09-05 |
| EXP-053 | EXP-053-zeroshot@best | HOB | val | none | 0.5435 | 0.8722 | 0.5384 | -0.0053 | 1.2037 | configs/EXP-053.yaml | 0d0866f | 108min | 2026-09-05 |
| EXP-053 | EXP-053-zeroshot@best | HOB | val | week1 | 0.6560 | 0.9378 | 0.6739 | +0.0190 | 0.9037 | configs/EXP-053.yaml | 0d0866f | 108min | 2026-09-05 |
| EXP-053 | EXP-053-zeroshot@best | HOB | val | full | 0.6634 | 0.9396 | 0.6832 | +0.0211 | 0.8882 | configs/EXP-053.yaml | 0d0866f | 108min | 2026-09-05 |
| EXP-053 | EXP-053-zeroshot@best | MSH | val | full | 0.6690 | 0.9390 | 0.6938 | +0.0382 | 0.8685 | configs/EXP-053.yaml | 0d0866f | 108min | 2026-09-05 |
| EXP-052 | EXP-052-zeroshot@best | HOB | val | none | 0.5509 | 0.8757 | 0.5497 | +0.0001 | 1.1824 | configs/EXP-052.yaml | 8b52fe6 | 112min | 2026-09-06 |
| EXP-052 | EXP-052-zeroshot@best | HOB | val | week1 | 0.6604 | 0.9397 | 0.6764 | +0.0191 | 0.8868 | configs/EXP-052.yaml | 8b52fe6 | 112min | 2026-09-06 |
| EXP-052 | EXP-052-zeroshot@best | HOB | val | full | 0.6647 | 0.9414 | 0.6799 | +0.0177 | 0.8773 | configs/EXP-052.yaml | 8b52fe6 | 112min | 2026-09-06 |
| EXP-052 | EXP-052-zeroshot@best | MSH | val | full | 0.6709 | 0.9401 | 0.6967 | +0.0394 | 0.8619 | configs/EXP-052.yaml | 8b52fe6 | 112min | 2026-09-06 |
| EXP-054 | EXP-054-zeroshot@best | HOB | val | none | 0.5082 | 0.8472 | 0.5043 | -0.0064 | 1.3061 | configs/EXP-054.yaml | e4843ea | 107min | 2026-09-06 |
| EXP-054 | EXP-054-zeroshot@best | HOB | val | week1 | 0.6404 | 0.9298 | 0.6542 | +0.0185 | 0.9402 | configs/EXP-054.yaml | e4843ea | 107min | 2026-09-06 |
| EXP-054 | EXP-054-zeroshot@best | HOB | val | full | 0.6438 | 0.9319 | 0.6589 | +0.0203 | 0.9288 | configs/EXP-054.yaml | e4843ea | 107min | 2026-09-06 |
| EXP-054 | EXP-054-zeroshot@best | MSH | val | full | 0.6676 | 0.9384 | 0.6945 | +0.0419 | 0.8713 | configs/EXP-054.yaml | e4843ea | 107min | 2026-09-06 |
| EXP-055 | EXP-055-zeroshot@best | HOB | val | none | 0.5684 | 0.8869 | 0.5650 | -0.0020 | 1.1230 | configs/EXP-055.yaml | e4843ea | 107min | 2026-09-06 |
| EXP-055 | EXP-055-zeroshot@best | HOB | val | week1 | 0.6672 | 0.9429 | 0.6861 | +0.0197 | 0.8691 | configs/EXP-055.yaml | e4843ea | 107min | 2026-09-06 |
| EXP-055 | EXP-055-zeroshot@best | HOB | val | full | 0.6704 | 0.9442 | 0.6898 | +0.0213 | 0.8608 | configs/EXP-055.yaml | e4843ea | 107min | 2026-09-06 |
| EXP-055 | EXP-055-zeroshot@best | MSH | val | full | 0.6706 | 0.9395 | 0.6965 | +0.0405 | 0.8638 | configs/EXP-055.yaml | e4843ea | 107min | 2026-09-06 |

### HOB deck builder (draft decks, then sealed)

| EXP | Model | Set | Split | Stats | F1 | NB-F1 | Trophy-F1v2 | Win-pref | Win-wtd F1 | Lands-MAE | Basics-L1 | Config | SHA | Wall-clock | Date |
|-----|-------|-----|-------|-------|----|-------|-------------|----------|------------|-----------|-----------|--------|-----|------------|------|
| — | rebuild-ceiling | HOB | val | full | 0.9607 | 0.9499 | 0.9590 | nan | 0.9624 | 0.179 | 0.683 | baseline | 3f7449b | <1min | 2026-09-05 |
| — | gih-in-lane-build | HOB | val | full | 0.8214 | 0.7990 | 0.8291 | 0.567 | 0.8262 | 1.991 | 4.693 | baseline | 3f7449b | <1min | 2026-09-05 |
| EXP-126 | EXP-126@best | HOB | val | none | 0.8424 | 0.7898 | 0.8477 | 0.522 | 0.8452 | 0.205 | 2.255 | configs/EXP-126.yaml | 3f7449b | 0 (zero-shot) | 2026-09-05 |
| EXP-126 | EXP-126@best | HOB | val | full | 0.8794 | 0.8513 | 0.8855 | 0.589 | 0.8846 | 0.203 | 2.327 | configs/EXP-126.yaml | 3f7449b | 0 (zero-shot) | 2026-09-05 |
| EXP-135 | EXP-135@best | HOB | val | full | 0.8821 | 0.8546 | 0.8887 | 0.567 | 0.8872 | 0.217 | 2.273 | configs/EXP-135.yaml | 3f7449b | 0 (zero-shot) | 2026-09-05 |
| — | rebuild-ceiling | HOB/sealed | val | full | 0.8147 | 0.8085 | 0.8949 | nan | 0.8631 | 0.137 | 5.342 | baseline | 3f7449b | <1min | 2026-09-05 |
| — | gih-in-lane-build | HOB/sealed | val | full | 0.5270 | 0.4985 | 0.5617 | 0.500 | 0.5521 | 3.120 | 13.937 | baseline | 3f7449b | <1min | 2026-09-05 |
| EXP-135 | EXP-135@best | HOB/sealed | val | none | 0.5043 | 0.4667 | 0.4957 | 0.477 | 0.4997 | 0.112 | 13.335 | configs/EXP-135.yaml | 3f7449b | 0 (zero-shot) | 2026-09-05 |
| EXP-135 | EXP-135@best | HOB/sealed | val | full | 0.6260 | 0.6033 | 0.6809 | 0.455 | 0.6646 | 0.114 | 10.348 | configs/EXP-135.yaml | 3f7449b | 0 (zero-shot) | 2026-09-05 |
| EXP-138 | EXP-138@best | HOB | val | full | 0.9059 | 0.8792 | 0.9140 | 0.522 | 0.9117 | 0.192 | 1.582 | configs/EXP-138.yaml | e2d25ed | 2min | 2026-09-05 |
