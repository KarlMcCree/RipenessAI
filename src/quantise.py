import json

import numpy as np
import tensorflow as tf

from src.config import MODELS, REPORTS
from src.data_utils import get_splits, make_representative_dataset


def _input_res(keras_path):
    model = tf.keras.models.load_model(keras_path, compile=False)
    shape = model.input_shape  # (None, H, W, 3)
    return int(shape[1]), int(shape[2])


def convert_fp32(keras_path, out_path, _res):
    model = tf.keras.models.load_model(keras_path, compile=False)
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite = conv.convert()
    out_path.write_bytes(tflite)
    return len(tflite) / 1024


def convert_fp16(keras_path, out_path, _res):
    model = tf.keras.models.load_model(keras_path, compile=False)
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    tflite = conv.convert()
    out_path.write_bytes(tflite)
    return len(tflite) / 1024


def convert_int8(keras_path, out_path, res):
    model = tf.keras.models.load_model(keras_path, compile=False)
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: make_representative_dataset(
        get_splits(include_augmented=True)[0], (res, res), n=200)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type  = tf.float32
    conv.inference_output_type = tf.float32
    tflite = conv.convert()
    out_path.write_bytes(tflite)
    return len(tflite) / 1024


def main():
    results = []
    for keras_path in sorted(MODELS.glob("comp_*.keras")):
        tag = keras_path.stem.replace("comp_", "")
        res = _input_res(keras_path)[0]
        print(f"\n{tag}  (input {res}×{res})")

        for mode, fn in [("fp32", convert_fp32),
                         ("fp16", convert_fp16),
                         ("int8", convert_int8)]:
            out = MODELS / f"comp_{tag}_{mode}.tflite"
            try:
                kb = fn(keras_path, out, res)
                results.append({"tag": tag, "mode": mode,
                                "resolution": res, "size_kb": round(kb, 1)})
                print(f"  {mode:5s} {kb:8.1f} KB")
            except Exception as e:
                print(f"  {mode}  FAILED: {e}")

    (REPORTS / "quantisation_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    L = ["# Quantisation Study\n",
         "| Model | Mode | Res | Size (KB) |", "|---|---|---:|---:|"]
    for r in results:
        L.append(f"| {r['tag']} | {r['mode']} | {r['resolution']} | {r['size_kb']} |")
    (REPORTS / "quantisation_study.md").write_text("\n".join(L), encoding="utf-8")
    print("\n→ reports/quantisation_study.md")


if __name__ == "__main__":
    main()