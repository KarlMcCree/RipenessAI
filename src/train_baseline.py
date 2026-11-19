import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import callbacks, layers, models

# FIX 1: Removed "src." so imports work when running from project root
from src.config import BATCH_SIZE, CLASS_NAMES, IMG_SIZE, MODELS, REPORTS
from src.data_utils import get_splits, make_dataset

EPOCHS_HEAD = 15
EPOCHS_FINE = 10


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


def build_mobilenet(n_cls: int) -> tuple[models.Model, tf.keras.Model]:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # Stage 1: Frozen

    inp = layers.Input(shape=(*IMG_SIZE, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(n_cls, activation="softmax")(x)
    return models.Model(inp, out), base


def save_evaluation(model: models.Model, test_pairs, tag: str) -> float:
    REPORTS.mkdir(parents=True, exist_ok=True)
    test_ds = make_dataset(test_pairs, batch_size=BATCH_SIZE, shuffle=False)
    
    y_true = np.array(
        [CLASS_NAMES.index(l) if isinstance(l, str) else int(l) for _, l in test_pairs],
        dtype=np.int32,
    )
    y_pred = model.predict(test_ds, verbose=0).argmax(axis=1)

    report = classification_report(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES, digits=3, zero_division=0,
    )
    (REPORTS / f"{tag}_metrics.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    (REPORTS / f"{tag}_confusion.json").write_text(
        json.dumps({"cm": cm.tolist(), "classes": CLASS_NAMES}, indent=2),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45)
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(REPORTS / f"{tag}_confusion.png")
    plt.close(fig)

    print(f"\n[{tag}]\n{report}")
    return float((y_true == y_pred).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline", "mobilenet"], required=True)
    args = ap.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    train, val, test = get_splits(include_augmented=True)

    print(f"Training on {len(train)} images (Real + Augmented)")
    print(f"Validating on {len(val)} images (Real only)")
    print(f"Testing on {len(test)} images (Real only)\n")

    train_ds = make_dataset(train, batch_size=BATCH_SIZE, shuffle=True)
    val_ds = make_dataset(val, batch_size=BATCH_SIZE, shuffle=False)

    if args.model == "baseline":
        model = build_baseline(len(CLASS_NAMES))
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        h = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD)
        history = h.history
        model.save(MODELS / "baseline.keras")

    else:
        model, base = build_mobilenet(len(CLASS_NAMES))
        print("\n--- STAGE 1: Training 4-Class Head (Frozen Base) ---")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        
        early_stop = callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )
        
        h1 = model.fit(
            train_ds, validation_data=val_ds,
            epochs=EPOCHS_HEAD, callbacks=[early_stop]
        )
        
        
        model.save(MODELS / "mobilenet_stage1_frozen.keras")
        print("\n--- STAGE 2: Fine-Tuning Top Layers ---")
        base.trainable = True
        
        fine_tune_at = 100
        for layer in base.layers[:fine_tune_at]:
            layer.trainable = False
            
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        
        checkpoint_best = callbacks.ModelCheckpoint(
            filepath=str(MODELS / "mobilenet.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        )
        
        h2 = model.fit(
            train_ds, validation_data=val_ds,
            epochs=EPOCHS_FINE,
            callbacks=[early_stop, checkpoint_best],
        )

        history = {k: h1.history[k] + h2.history[k] for k in h1.history}

    (REPORTS / f"{args.model}_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["accuracy"], label="train")
    ax[0].plot(history["val_accuracy"], label="val")
    ax[0].set_title("Accuracy"); ax[0].legend()
    ax[1].plot(history["loss"], label="train")
    ax[1].plot(history["val_loss"], label="val")
    ax[1].set_title("Loss"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(REPORTS / f"{args.model}_curves.png")
    plt.close(fig)

    if args.model == "mobilenet":
        print("\nLoading best fine-tuned weights for evaluation...")
        model = tf.keras.models.load_model(MODELS / "mobilenet.keras")

    acc = save_evaluation(model, test, args.model)
    print(f"\nTest acc = {acc:.3f}")


if __name__ == "__main__":
    main()