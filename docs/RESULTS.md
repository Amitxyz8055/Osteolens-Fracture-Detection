# Results

| Test | Result |
|---|---|
| Internal clean test, 584 images | TP 239, FN 4, TN 340, FP 1; sensitivity 98.4%, specificity 99.7%, accuracy 99.1%, AUC 0.9998, threshold 0.8643 |
| Independent dataset, before pooling | AUC 0.834, specificity 55.8% |
| After pooled training | specificity on the second dataset 81.4% |
| JPEG quality 25-50 | 84-89% fewer bytes, no accuracy loss |
| Noise below 30 dB SNR | accuracy drops, file size about 2.8 times larger |
| Bandwidth 0.125 or less | sensitivity drops |
| 5% packet loss | 41% of images flagged unsupported |

Not finished: the preprocessing ablation (CLAHE, median, both) has only its baseline row run. No claim is made about whether preprocessing helps.

Limits: two datasets, one training seed, no clinical validation. The unsupported-image check also rejects about 10% of clean images. Full tables: `results/report_tables.xlsx`.
