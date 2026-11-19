import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.config import CLASS_NAMES, REPORTS, MODELS
from src.data_utils import get_splits, make_dataset

GALLERY_N = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline", "mobilenet"], default="mobilenet")
    args = ap.parse_args()

    tag = args.model
    title_name = "MobileNetV2" if tag == "mobilenet" else "Baseline CNN"

    model = tf.keras.models.load_model(MODELS / f"{tag}.keras")
    _, _, test = get_splits(include_augmented=False)
    test_ds = make_dataset(test)

    y_true = np.array([l for _, l in test])
    y_pred = model.predict(test_ds, verbose=0).argmax(1)

    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=3, zero_division=0
    )
    (REPORTS / f"error_analysis_{tag}.txt").write_text(report, encoding="utf-8")

    # --- confusion matrix ---
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45)
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{title_name} confusion matrix")
    fig.tight_layout()
    fig.savefig(REPORTS / f"error_confusion_{tag}.png")
    plt.close(fig)

    # --- misclassification pairs ---
    pairs = {}
    for t, p in zip(y_true, y_pred):
        if t != p:
            key = (CLASS_NAMES[t], CLASS_NAMES[p])
            pairs[key] = pairs.get(key, 0) + 1
    (REPORTS / f"error_pairs_{tag}.json").write_text(
        json.dumps({f"{t}->{p}": n for (t, p), n in pairs.items()}, indent=2),
        encoding="utf-8",
    )

    # --- gallery of misclassified images ---
    wrong = [
        (path, lbl, int(pred))
        for (path, lbl), pred in zip(test, y_pred)
        if lbl != pred
    ]
    wrong = wrong[:GALLERY_N]
    if wrong:
        cols = 4
        rows = (len(wrong) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = np.atleast_2d(axes)
        for i in range(rows * cols):
            ax = axes[i // cols, i % cols]
            ax.axis("off")
            if i < len(wrong):
                path, true_lbl, pred_lbl = wrong[i]
                img = tf.keras.utils.load_img(path, target_size=(224, 224))
                ax.imshow(img)
                ax.set_title(
                    f"T:{CLASS_NAMES[true_lbl]}\nP:{CLASS_NAMES[pred_lbl]}",
                    fontsize=8,
                )
        fig.tight_layout()
        fig.savefig(REPORTS / f"error_gallery_{tag}.png")
        plt.close(fig)

    # --- markdown summary ---
    L = [
        f"# Error Analysis — {title_name}\n",
        "## Per-class metrics\n",
        "```",
        report,
        "```\n",
        "## Top misclassification pairs\n",
        "| True → Predicted | Count |",
        "|---|---:|",
    ]
    for (t, p), n in sorted(pairs.items(), key=lambda kv: -kv[1])[:10]:
        L.append(f"| {t} → {p} | {n} |")
    L += [
        "\n## Figures\n",
        f"![confusion](error_confusion_{tag}.png)\n",
        f"![gallery](error_gallery_{tag}.png)\n",
    ]
    (REPORTS / f"error_analysis_{tag}.md").write_text("\n".join(L), encoding="utf-8")

    print(report)
    print(f"\nError analysis for {tag} saved to reports/error_analysis_{tag}.md")


if __name__ == "__main__":
    main()
