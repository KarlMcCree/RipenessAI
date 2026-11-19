from __future__ import annotations
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .config import DATA_DIR, CLASS_NAMES, REPORTS

IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REPORT_PATH = REPORTS / "data_audit.md"


def find_images(root: Path) -> dict[str, list[Path]]:
    out = {}
    for cls in CLASS_NAMES:
        d = root / cls
        if not d.is_dir():
            print(f"[warn] missing folder: {d}")
            out[cls] = []
        else:
            out[cls] = sorted(p for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    return out


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def audit(root: Path) -> dict:
    per_class = find_images(root)
    counts    = {c: len(v) for c, v in per_class.items()}
    sizes     = Counter()
    per_class_sizes = defaultdict(Counter)
    corrupt   = []
    hashes    = defaultdict(list)

    for cls, paths in per_class.items():
        for p in paths:
            try:
                with Image.open(p) as im:
                    im.verify()                       # catches truncation
                with Image.open(p) as im:             # reopen after verify
                    sizes[im.size] += 1
                    per_class_sizes[cls][im.size] += 1
                hashes[md5(p)].append(p)
            except (UnidentifiedImageError, OSError) as e:
                corrupt.append(f"{p}  ({e})")

    return {
        "counts":    counts,
        "total":     sum(counts.values()),
        "sizes":     sizes,
        "corrupt":   corrupt,
        "duplicates": {h: ps for h, ps in hashes.items() if len(ps) > 1},
    }


def render(stats: dict) -> str:
    L = ["# Dataset Audit\n", f"Root: `{DATA_DIR.resolve()}`\n"]

    # class distribution
    L += ["## Class distribution\n", "| Class | Images | Share |", "|---|---:|---:|"]
    total = stats["total"] or 1
    for c in CLASS_NAMES:
        n = stats["counts"][c]
        L.append(f"| {c} | {n} | {n/total:.1%} |")
    L.append(f"| **Total** | **{stats['total']}** | 100% |\n")

    mx, mn = max(stats["counts"].values(), default=0), min(stats["counts"].values(), default=0)
    if mn and mx / mn > 1.5:
        L.append(f">  Imbalance: {mx/mn:.2f}× ratio. Use class weights.\n")
    else:
        L.append(">  Balanced within 1.5×.\n")

    # dimensions
    L += ["## Dimensions\n", "| Size | Count |", "|---|---:|"]
    for (w, h), n in stats["sizes"].most_common():
        L.append(f"| {w}×{h} | {n} |")
    L.append("")
    if len(stats["sizes"]) == 1:
        L.append(">  Uniform — no resize needed.\n")
    else:
        L.append(f">  {len(stats['sizes'])} sizes — pipeline must resize.\n")

    # corrupt
    L += ["## Corrupt files\n"]
    L += [f"- `{c}`" for c in stats["corrupt"]] if stats["corrupt"] else ["None. ✅"]
    L.append("")

    # duplicates
    L += ["## Exact duplicates (MD5)\n"]
    if stats["duplicates"]:
        L.append(f"Found {len(stats['duplicates'])} groups:\n")
        for h, ps in list(stats["duplicates"].items())[:20]:
            L.append(f"- `{h[:8]}…` → " + ", ".join(p.name for p in ps))
        L.append("\n> Duplicates across classes = label leak. Remove before training.\n")
    else:
        L.append("None. \n")

    return "\n".join(L)


if __name__ == "__main__":
    stats = audit(DATA_DIR)
    REPORT_PATH.write_text(render(stats), encoding="utf-8")
    print(f"Total={stats['total']}  corrupt={len(stats['corrupt'])}  "
          f"duplicate-groups={len(stats['duplicates'])}")
    print(f"→ {REPORT_PATH}")