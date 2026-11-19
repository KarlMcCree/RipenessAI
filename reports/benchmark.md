# Benchmark — Edge Deployment Trade-offs

Measured on host CPU, 50 runs after 5 warmups.

| Model | Mode | Res | Size (KB) | Latency (ms) | Accuracy |
|---|---|---:|---:|---:|---:|
| mob_a35 | int8 | 224 | 633.1 | 7.71 | 0.260 |
| mob_a35 | fp16 | 224 | 859.7 | 5.74 | 0.260 |
| mob_a50 | int8 | 224 | 974.5 | 13.45 | 0.252 |
| mob_a50 | fp16 | 224 | 1424.8 | 9.73 | 0.252 |
| mob_a35 | fp32 | 224 | 1597.2 | 5.75 | 0.260 |
| mob_a75 | int8 | 224 | 1729.3 | 23.2 | 0.252 |
| mob_r128 | int8 | 128 | 2672.3 | 17.47 | 0.805 |
| mob_r160 | int8 | 160 | 2672.3 | 24.11 | 0.894 |
| mob_r192 | int8 | 192 | 2672.3 | 32.54 | 0.805 |
| mob_ref | int8 | 224 | 2672.3 | 48.08 | 0.894 |
| mob_a75 | fp16 | 224 | 2721.1 | 19.39 | 0.252 |
| mob_a50 | fp32 | 224 | 2727.3 | 10.08 | 0.252 |
| mob_r128 | fp16 | 128 | 4410.0 | 8.34 | 0.878 |
| mob_r160 | fp16 | 160 | 4410.0 | 16.42 | 0.894 |
| mob_r192 | fp16 | 192 | 4410.0 | 28.38 | 0.886 |
| mob_ref | fp16 | 224 | 4410.0 | 40.82 | 0.886 |
| mob_a75 | fp32 | 224 | 5319.9 | 17.23 | 0.252 |
| mob_r128 | fp32 | 128 | 8697.7 | 19.25 | 0.878 |
| mob_r160 | fp32 | 160 | 8697.7 | 17.98 | 0.894 |
| mob_r192 | fp32 | 192 | 8697.7 | 26.5 | 0.886 |
| mob_ref | fp32 | 224 | 8697.7 | 44.44 | 0.886 |
| baseline | int8 | 224 | 10923.2 | 40.99 | 0.878 |
| baseline | fp16 | 224 | 21822.3 | 32.86 | 0.878 |
| baseline | fp32 | 224 | 43635.9 | 33.09 | 0.878 |

> The 'knee' is the smallest model whose accuracy stays within ~1–2 pts of the reference. Compare rows with accuracy ≥ 0.85 only — rows at 0.252 failed to learn and should be excluded.
