# Data Leakage Verification

- No real image leakage across train / val / test splits
- No augmented leakage into val/test (train size: 573 real -> 573 total)
- Raw augmented folder stats: 93 match val IDs, 92 match test IDs (properly filtered out by get_splits)

## Split Counts

| Split | Images | Unique IDs |
|---|---:|---:|
| Train (Real only) | 573 | 573 |
| Train (Real + Aug) | 573 | 573 |
| Validation | 123 | 123 |
| Test | 123 | 123 |
