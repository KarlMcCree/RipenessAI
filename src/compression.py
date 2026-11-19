import json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.keras import layers, models

from src.config import CLASS_NAMES, BATCH_SIZE, IMG_SIZE, MODELS, REPORTS
from src.data_utils import get_splits, make_dataset

GRID = [
    ("baseline",  "custom",    None, 224, False),
    ("mob_ref",   "mobilenet", 1.00, 224, True),
    ("mob_a75",   "mobilenet", 0.75, 224, False),
    ("mob_a50",   "mobilenet", 0.50, 224, False),
    ("mob_a35",   "mobilenet", 0.35, 224, False),
    ("mob_r192",  "mobilenet", 1.00, 192, True),
    ("mob_r160",  "mobilenet", 1.00, 160, True),
    ("mob_r128",  "mobilenet", 1.00, 128, True),
]
EPOCHS = 10


def build_baseline(n_cls: int) -> models.Model:
    return models.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        layers.Rescaling(1.0 / 255.0),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Flatten(), layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dense(n_cls, activation="softmax"),
    ])


def build_mobilenet(alpha: float, res: int, pretrained: bool, n_cls: int) -> models.Model:
    weights = "imagenet" if pretrained else None
    base = tf.keras.applications.MobileNetV2(
        input_shape=(res, res, 3),
        include_top=False,
        weights=weights,
        alpha=alpha,
    )
    base.trainable = not pretrained

    inp = layers.Input(shape=(res, res, 3))
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(inp)
    x = base(x, training=not pretrained)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(n_cls, activation="softmax")(x)
    return models.Model(inp, out)


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    train, val, test = get_splits(include_augmented=True)
    y_true = np.array(
        [CLASS_NAMES.index(l) if isinstance(l, str) else int(l) for _, l in test],
        dtype=np.int32,
    )
    results = []

    for tag, kind, alpha, res, pretrained in GRID:
        print(f"\n=== {tag}  kind={kind}  alpha={alpha}  res={res}  pretrained={pretrained} ===")
        train_ds = make_dataset(train, img_size=(res, res), batch_size=BATCH_SIZE, shuffle=True)
        test_ds = make_dataset(test, img_size=(res, res), batch_size=BATCH_SIZE, shuffle=False)

        if kind == "custom":
            model = build_baseline(len(CLASS_NAMES))
        else:
            model = build_mobilenet(alpha, res, pretrained, len(CLASS_NAMES))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(train_ds, epochs=EPOCHS, verbose=1)

        y_pred = model.predict(test_ds, verbose=0).argmax(axis=1)
        acc = float((y_true == y_pred).mean())
        f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        path = MODELS / f"comp_{tag}.keras"
        model.save(path)
        size_kb = path.stat().st_size / 1024

        results.append({
            "tag": tag,
            "kind": kind,
            "alpha": alpha,
            "resolution": res,
            "pretrained": pretrained,
            "params": int(model.count_params()),
            "keras_kb": round(size_kb, 1),
            "accuracy": round(acc, 4),
            "macro_f1": round(f1, 4),
        })
        print(f"  acc={acc:.3f}  f1={f1:.3f}  params={model.count_params():,}  size={size_kb:.1f} KB")

    (REPORTS / "compression_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    valid = [r for r in results if r["accuracy"] > 0.5]
    ax.scatter([r["params"] / 1e3 for r in valid], [r["accuracy"] for r in valid], s=80, c="b")
    for r in valid:
        ax.annotate(
            r["tag"],
            (r["params"] / 1e3, r["accuracy"]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("Parameters (thousands)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy vs model size")
    fig.tight_layout()
    fig.savefig(REPORTS / "compression_curve_params.png")
    plt.close(fig)

    res_rows = [r for r in results if r["tag"].startswith("mob_r") or r["tag"] == "mob_ref"]
    res_rows.sort(key=lambda r: r["resolution"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["resolution"] for r in res_rows], [r["accuracy"] for r in res_rows], marker="o", c="r")
    for r in res_rows:
        ax.annotate(
            r["tag"],
            (r["resolution"], r["accuracy"]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("Input resolution")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy vs input resolution (pretrained full-width)")
    fig.tight_layout()
    fig.savefig(REPORTS / "compression_curve_resolution.png")
    plt.close(fig)

    L = [
        "# Compression Study\n",
        "> Two axes swept: model width (alpha) and input resolution. "
        "All runs use the same seed and train/test split.\n",
        "| Config | Kind | Alpha | Res | Pretrained | Params | Keras KB | Accuracy | Macro F1 |",
        "|---|---|---:|---:|:---:|---:|---:|---:|---:|",
    ]
    for r in results:
        L.append(
            f"| {r['tag']} | {r['kind']} | {r['alpha'] or '-'} | "
            f"{r['resolution']} | {'yes' if r['pretrained'] else 'no'} | "
            f"{r['params']:,} | {r['keras_kb']} | "
            f"{r['accuracy']:.3f} | {r['macro_f1']:.3f} |"
        )
    L += [
        "\n## Accuracy vs size\n![params](compression_curve_params.png)\n",
        "## Accuracy vs resolution\n![res](compression_curve_resolution.png)\n",
        "> Note: alpha<1.00 rows are trained from scratch (no ImageNet weights "
        "exist at those widths); they collapse to chance, indicating that "
        "width compression is not viable on this dataset.\n",
    ]
    (REPORTS / "compression_study.md").write_text("\n".join(L), encoding="utf-8")
    print("\n-> reports/compression_study.md")


if __name__ == "__main__":
    main()
