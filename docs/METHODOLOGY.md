# Methodology

1. **Duplicate removal.** MD5 hash of every file; identical files are kept once. Dataset 1: 10,581 files became 3,888 unique images (63.3% were copies). In the original split 97% of test images had a copy in training.
2. **Group-aware split.** `GroupShuffleSplit` 70/15/15, seed 42, so identical images never fall on both sides. Train 1,116 fractured + 1,604 normal; validation 247 + 337; test 243 + 341.
3. **Preprocessing.** Resize to 224 px. Optional: CLAHE (clip 2.0, 8x8 tiles), 3x3 median filter.
4. **Model.** DenseNet121 with ImageNet weights, new 2-class head. Class-weighted cross-entropy, AdamW (lr 1e-4, weight decay 1e-4), cosine schedule, 15 epochs, batch 32. The epoch with the best validation AUC is kept.
5. **Threshold.** Chosen on the validation set to give 95% sensitivity.
6. **Unsupported-image check.** Ledoit-Wolf covariance of pooled penultimate features on training images; limit = 99th percentile of validation distances times a margin (0.9).
7. **Explanation.** Grad-CAM on `denseblock4`, jet overlay with weight 0.4.
8. **External test.** A second dataset never used for training in the first experiment; then both datasets pooled for training and evaluated per dataset.
9. **Network experiments.** 21 channel settings (SNR, bandwidth, JPEG quality, packet loss) on a labelled set; measured accuracy, bytes, unsupported rate and time.
