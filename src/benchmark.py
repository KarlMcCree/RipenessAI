import json
import time

import numpy as np
import tensorflow as tf

from src.config import MODELS, REPORTS
from src.data_utils import get_splits, make_dataset

N_RUNS = 50
WARMUP = 5


def _tflite_input_res(interp):
    d = interp.get_input_details()[0]
    shape = d["shape"]  # [1, H, W, 3]
    return int(shape[1]), int(shape[2])


def evaluate(tflite_path, test_pairs):
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    in_det  = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    in_dtype = in_det["dtype"]
    res_h, res_w = _tflite_input_res(interp)

    test_ds = make_dataset(test_pairs, img_size=(res_h, res_w), batch_size=1)
    images, labels = [], []
    for x, y in test_ds:
        images.append(x[0].numpy())
        labels.append(int(y[0]))

    correct = 0
    for img, lbl in zip(images, labels):
        x = img.astype(in_dtype)
        if in_dtype == np.uint8:
            x = np.clip(x, 0, 255).astype(np.uint8)
        interp.set_tensor(in_det["index"], x[np.newaxis, ...])
        interp.invoke()
        pred = int(interp.get_tensor(out_det["index"])[0].argmax())
        correct += (pred == lbl)
    acc = correct / len(labels)

    sample = images[0].astype(in_dtype)
    if in_dtype == np.uint8:
        sample = np.clip(sample, 0, 255).astype(np.uint8)
    sample = sample[np.newaxis, ...]
    for _ in range(WARMUP):
        interp.set_tensor(in_det["index"], sample); interp.invoke()
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        interp.set_tensor(in_det["index"], sample); interp.invoke()
    latency_ms = (time.perf_counter() - t0) / N_RUNS * 1000

    return {
        "size_kb":    round(tflite_path.stat().st_size / 1024, 1),
        "latency_ms": round(latency_ms, 2),
        "accuracy":   round(acc, 4),
        "resolution": res_h,
    }


def main():
    _, _, test = get_splits(include_augmented=False)
    results = []

    for path in sorted(MODELS.glob("comp_*.tflite")):
        name = path.stem.replace("comp_", "")
        base, mode = name.rsplit("_", 1)
        try:
            r = evaluate(path, test)
            r.update({"tag": base, "mode": mode})
            results.append(r)
            print(f"{base:10s} {mode:5s}  res={r['resolution']}  "
                  f"size={r['size_kb']:7.1f} KB  "
                  f"lat={r['latency_ms']:6.2f} ms  acc={r['accuracy']:.3f}")
        except Exception as e:
            print(f"{name}: FAILED — {e}")

    (REPORTS / "benchmark_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    L = ["# Benchmark — Edge Deployment Trade-offs\n",
         f"Measured on host CPU, {N_RUNS} runs after {WARMUP} warmups.\n",
         "| Model | Mode | Res | Size (KB) | Latency (ms) | Accuracy |",
         "|---|---|---:|---:|---:|---:|"]
    for r in sorted(results, key=lambda x: x["size_kb"]):
        L.append(f"| {r['tag']} | {r['mode']} | {r['resolution']} | "
                 f"{r['size_kb']} | {r['latency_ms']} | {r['accuracy']:.3f} |")
    L.append("\n> The 'knee' is the smallest model whose accuracy stays within "
             "~1–2 pts of the reference. Compare rows with accuracy ≥ 0.85 only — "
             "rows at 0.252 failed to learn and should be excluded.\n")
    (REPORTS / "benchmark.md").write_text("\n".join(L), encoding="utf-8")
    print("\n→ reports/benchmark.md")


if __name__ == "__main__":
    main()