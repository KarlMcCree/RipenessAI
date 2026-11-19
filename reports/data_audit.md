# Dataset Audit

Root: `C:\Users\HP\Downloads\RipenessAI\data\real`

## Class distribution

| Class | Images | Share |
|---|---:|---:|
| Green | 212 | 25.9% |
| Semi-ripe | 204 | 24.9% |
| Ripe | 201 | 24.5% |
| Overripe | 203 | 24.8% |
| **Total** | **820** | 100% |

>  Balanced within 1.5×.

## Dimensions

| Size | Count |
|---|---:|
| 256×256 | 819 |
| 223×246 | 1 |

>  2 sizes — pipeline must resize.

## Corrupt files

None. 

## Exact duplicates (MD5)

Found 1 groups:

- `f272af58…` → Semi_ripe_81.jpg, Ripe_6.jpg

> Duplicates across classes = label leak. Remove before training.
