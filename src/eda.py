from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .config import DATA_DIR, CLASS_NAMES, REPORTS

OUT = REPORTS / "eda"; OUT.mkdir(parents=True, exist_ok=True)
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SAMPLES_PER_CLASS = 6


def collect() -> dict[str, list[Path]]:
    return {
        cls: (sorted(p for p in (DATA_DIR / cls).rglob("*")
                     if p.suffix.lower() in IMAGE_EXTS)
              if (DATA_DIR / cls).is_dir() else [])
        for cls in CLASS_NAMES
    }


def plot_counts(per_class):
    counts = [len(per_class[c]) for c in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(CLASS_NAMES, counts, color="#4C72B0")
    ax.set_title("Class distribution"); ax.set_ylabel("Images")
    for b, n in zip(bars, counts):
        ax.text(b.get_x() + b.get_width()/2, n, str(n), ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(OUT / "class_counts.png"); plt.close(fig)
    return counts


def plot_sizes(per_class):
    sizes = Counter()
    for paths in per_class.values():
        for p in paths:
            try:
                with Image.open(p) as im: sizes[im.size] += 1
            except Exception: pass
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([f"{w}×{h}" for w, h in sizes], list(sizes.values()), color="#55A868")
    ax.set_title("Image dimensions")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(OUT / "size_distribution.png"); plt.close(fig)
    return sizes


def plot_brightness(per_class):
    means = defaultdict(list)
    for cls, paths in per_class.items():
        for p in paths:
            try:
                with Image.open(p).convert("L") as im:
                    means[cls].append(float(np.asarray(im).mean()))
            except Exception: pass
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([means[c] for c in CLASS_NAMES], tick_labels=CLASS_NAMES)
    ax.set_ylabel("Mean grayscale (0–255)")
    ax.set_title("Brightness by class")
    fig.tight_layout(); fig.savefig(OUT / "brightness_by_class.png"); plt.close(fig)
    return {c: float(np.mean(v)) if v else 0.0 for c, v in means.items()}


def plot_sample_grid(per_class):
    fig, axes = plt.subplots(len(CLASS_NAMES), SAMPLES_PER_CLASS,
                             figsize=(SAMPLES_PER_CLASS * 1.8, len(CLASS_NAMES) * 1.8))
    for r, cls in enumerate(CLASS_NAMES):
        for c in range(SAMPLES_PER_CLASS):
            ax = axes[r, c]; ax.axis("off")
            if c < len(per_class[cls]):
                with Image.open(per_class[cls][c]).convert("RGB") as im:
                    ax.imshow(im)
                if c == 0: ax.set_title(cls, loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "sample_grid.png"); plt.close(fig)


def write_md(counts, sizes, brightness):
    total = sum(counts) or 1
    L = ["# EDA\n", "## Class balance\n", "| Class | Count | Share |", "|---|---:|---:|"]
    for c, n in zip(CLASS_NAMES, counts):
        L.append(f"| {c} | {n} | {n/total:.1%} |")
    L.append("")
    mx, mn = max(counts), min(counts)
    L.append(f"> {' Imbalance ' + f'{mx/mn:.2f}×' if mn and mx/mn > 1.5 else '✅ Balanced'}\n")

    L += ["## Dimensions\n", "| Size | Count |", "|---|---:|"]
    for (w, h), n in sizes.most_common():
        L.append(f"| {w}×{h} | {n} |")
    L.append("")

    L += ["## Brightness\n", "| Class | Mean gray |", "|---|---:|"]
    for c in CLASS_NAMES:
        L.append(f"| {c} | {brightness.get(c, 0):.1f} |")
    spread = max(brightness.values()) - min(brightness.values())
    L.append(f"\n> Spread = {spread:.1f} "
             f"({' may encode class' if spread > 25 else ' small'})\n")

    L += ["## Figures\n",
          "![counts](class_counts.png)\n![sizes](size_distribution.png)\n"
          "![brightness](brightness_by_class.png)\n![samples](sample_grid.png)\n"]

    (OUT / "eda.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    pc = collect()
    c  = plot_counts(pc)
    s  = plot_sizes(pc)
    b  = plot_brightness(pc)
    plot_sample_grid(pc)
    write_md(c, s, b)
    print(f"→ {OUT}/eda.md")