from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.config import (
    REAL_DIR,
    AUGMENTED_DIR,
    CLASS_NAMES,
    IMG_SIZE,
    BATCH_SIZE,
    SEED,
    SPLIT,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _scan(root: Path) -> tuple[list[str], list[int]]:
    paths, labels = [], []
    for label, cls in enumerate(CLASS_NAMES):
        d = root / cls
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() in IMAGE_EXTS:
                paths.append(str(p))
                labels.append(int(label))
    return paths, labels


def get_splits(include_augmented: bool = False):
    real_p, real_l = _scan(REAL_DIR)
    if not real_p:
        raise RuntimeError(f"No images found in {REAL_DIR} for classes {CLASS_NAMES}")

    tr_p, rest_p, tr_l, rest_l = train_test_split(
        real_p, real_l, test_size=SPLIT, stratify=real_l, random_state=SEED
    )
    va_p, te_p, va_l, te_l = train_test_split(
        rest_p, rest_l, test_size=0.5, stratify=rest_l, random_state=SEED
    )

    if include_augmented and AUGMENTED_DIR.is_dir():
        aug_p, _ = _scan(AUGMENTED_DIR)
        stem_to_aug: dict[str, list[str]] = {}
        for p in aug_p:
            stem = Path(p).stem.split("_")[0]
            stem_to_aug.setdefault(stem, []).append(p)

        train_stems = {Path(p).stem for p in tr_p}
        extra = [
            p
            for stem, ps in stem_to_aug.items()
            if stem in train_stems
            for p in ps
        ]

        for p in extra:
            cls = Path(p).parent.name
            if cls in CLASS_NAMES:
                tr_p.append(p)
                tr_l.append(int(CLASS_NAMES.index(cls)))

    return (
        list(zip(tr_p, tr_l)),
        list(zip(va_p, va_l)),
        list(zip(te_p, te_l)),
    )


def _load_and_preprocess(path: tf.Tensor, label: tf.Tensor, img_size: tuple[int, int]):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32)
    label = tf.cast(label, tf.int32)
    return img, label


def make_dataset(
    pairs,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle: bool = False,
) -> tf.data.Dataset:
    pairs = list(pairs)
    if not pairs:
        raise ValueError("make_dataset received an empty list of (path, label) pairs.")

    paths = [str(p) for p, _ in pairs]
    labels = [int(CLASS_NAMES.index(l) if isinstance(l, str) else l) for _, l in pairs]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda p, l: _load_and_preprocess(p, l, img_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_representative_dataset(pairs, img_size=IMG_SIZE, n: int = 200):
    for p, _ in list(pairs)[:n]:
        img = tf.keras.utils.load_img(p, target_size=img_size)
        yield [tf.keras.utils.img_to_array(img)[np.newaxis, ...].astype(np.float32)]
