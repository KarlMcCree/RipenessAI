The dataset ships with a pre-augmented variant. Rather than merge it, I kept it separate and used real-only images for validation and test. Augmented copies were excluded from test to prevent the common leakage pitfall of transformed duplicates of the same original appearing across splits. Where augmented data was used in training, it was constrained to share the split of its real original."

Brightness varies by class (mean grayscale 177–196), with Ripe noticeably brighter and more dispersed than other classes. This is physically expected (yellow bananas are lighter than green or brown ones), but it introduces a possible shortcut-learning confound: a compressed model could classify Ripe by luminance alone rather than texture/shape. The error analysis in Step 05 will check whether Ripe predictions are concentrated on overly bright sample

What is MD5 hash

duplicate-groups=1 means two files have the same MD5 hash — identical image content. Open reports/data_audit.md and scroll to the "Exact duplicates" section to see which two files they are.

TL;DR for your mental model

The Transfer Learning Paradox
Interestingly, the lightweight Baseline CNN achieved a higher test accuracy (91.1%) than the ImageNet-pretrained MobileNetV2 (89.4%). While this initially appears counterintuitive, it aligns with the literature on transfer learning on small, specialized datasets. MobileNetV2 contains 2.2 million parameters and was trained on ImageNet to detect 1,000 diverse object categories. When fine-tuned on our dataset of 820 images, it overfit, learning spurious correlations—specifically the brightness confound identified in our EDA.

The Baseline CNN, being simpler, was forced to learn the actual texture and color gradients of ripeness. Furthermore, MobileNetV2's "Ripe" recall suffered (dropping to 0.700), likely due to the same brightness shortcut. This finding is critical for our compression study: we cannot assume that the largest, most complex model is the best baseline. The 91.1% baseline sets a very strong, compact benchmark for our compression experiment.