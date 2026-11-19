from pathlib import Path

from src.config import REAL_DIR, AUGMENTED_DIR, REPORTS
from src.data_utils import get_splits

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REPORT_PATH = REPORTS / "leakage_report.md"


def base_id(path: Path) -> str:
    """
    Extract the original image ID:
    - Real files like 'Ripe_148.JPG' -> 'Ripe_148'
    - Augmented files like 'Ripe_148_aug0.jpg' -> 'Ripe_148'
    """
    parts = path.stem.split("_")
    if len(parts) >= 2:
        first, second, *_ = parts
        if second.isdigit():
            return f"{first}_{second}"
    return path.stem


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    lines = ["# Data Leakage Verification\n"]

    # 1. Verify real splits do not share any image IDs
    train_real, val_real, test_real = get_splits(include_augmented=False)

    train_ids = {base_id(Path(p)) for p, _ in train_real}
    val_ids   = {base_id(Path(p)) for p, _ in val_real}
    test_ids  = {base_id(Path(p)) for p, _ in test_real}

    assert not (train_ids & val_ids),  f"REAL leakage: train ∩ val ({train_ids & val_ids})"
    assert not (train_ids & test_ids), f"REAL leakage: train ∩ test ({train_ids & test_ids})"
    assert not (val_ids   & test_ids), f"REAL leakage: val ∩ test ({val_ids & test_ids})"

    msg1 = "No real image leakage across train / val / test splits"
    print(f"✓ {msg1}")
    lines.append(f"- {msg1}")

    # 2. Verify get_splits(include_augmented=True) never leaks val/test IDs into train
    train_full, _, _ = get_splits(include_augmented=True)
    train_full_ids = {base_id(Path(p)) for p, _ in train_full}

    leak_val  = train_full_ids & val_ids
    leak_test = train_full_ids & test_ids

    assert not leak_val,  f"AUGMENTED leakage into val: {leak_val}"
    assert not leak_test, f"AUGMENTED leakage into test: {leak_test}"

    msg2 = f"No augmented leakage into val/test (train size: {len(train_real)} real -> {len(train_full)} total)"
    print(f"✓ {msg2}")
    lines.append(f"- {msg2}")

    # 3. Report raw augmented folder overlap with val/test for awareness
    if AUGMENTED_DIR.is_dir():
        aug_ids = {
            base_id(p) for p in AUGMENTED_DIR.rglob("*")
            if p.suffix.lower() in IMAGE_EXTS
        }
        overlap_val  = aug_ids & val_ids
        overlap_test = aug_ids & test_ids
        msg3 = (
            f"Raw augmented folder stats: {len(overlap_val)} match val IDs, "
            f"{len(overlap_test)} match test IDs (properly filtered out by get_splits)"
        )
        print(f"  {msg3}")
        lines.append(f"- {msg3}")

    lines.extend([
        "\n## Split Counts\n",
        "| Split | Images | Unique IDs |",
        "|---|---:|---:|",
        f"| Train (Real only) | {len(train_real)} | {len(train_ids)} |",
        f"| Train (Real + Aug) | {len(train_full)} | {len(train_full_ids)} |",
        f"| Validation | {len(val_real)} | {len(val_ids)} |",
        f"| Test | {len(test_real)} | {len(test_ids)} |",
    ])

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"→ {REPORT_PATH}")


if __name__ == "__main__":
    main()
