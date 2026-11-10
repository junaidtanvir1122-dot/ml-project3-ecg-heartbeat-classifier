## Table 1 – Data

| Item | Value |
|---|---|
| Train / validation / test rows | 78,799 / 8,755 / 21,892 |
| Minimum / maximum signal value | 0.0 / 1.0 |

| Rows per class | N | S | V | F | Q |
|---|---|---|---|---|---|
| Train file (87,554 rows) | 72,471 | 2,223 | 5,788 | 641 | 6,431 |
| Test file (21,892 rows) | 18,118 | 556 | 1,448 | 162 | 1,608 |

## Table 2 – Network and training

| Item | Value |
|---|---|
| Trainable parameters | 32,645 |
| Parameter count by hand | (187×128 + 128) + (128×64 + 64) + (64×5 + 5) = 24,064 + 8,256 + 325 = 32,645 |
| Initial loss (before training) | 1.7174 (full training set); 1.7413 on the first 64 rows (gradient check) |
| Gradient check: max difference W[3] / b[3] | 2.98e-8 / 5.96e-8 |
| Chosen α | 0.5 |
| Best epoch (lowest validation loss) | 23 |
| Training / validation loss at best epoch | 0.0556 / 0.0717 |
| Training / validation accuracy at best epoch | 98.29% / 97.82% |
| Training time and hardware | 127 s (chosen α run), CPU only (no CUDA GPU used) |

## Table 3 – Learning-rate choice

| α | Lowest validation loss | Validation accuracy at that epoch |
|---|---|---|
| 0.01 | 0.1618 | 95.79% |
| 0.1 | 0.0862 | 97.64% |
| 0.5 | 0.0717 | 97.82% |

## Table 4 – Test results

| | N | S | V | F | Q |
|---|---|---|---|---|---|
| Per-class accuracy | 99.54% | 66.73% | 91.44% | 64.81% | 97.64% |

| Metric | Value |
|---|---|
| Test accuracy | 97.77% |
| Balanced accuracy | 84.03% |
| Always-"N" accuracy (Task 6.1) | 82.76% |

Test confusion matrix (rows = true class, columns = predicted class):

| true \ predicted | N | S | V | F | Q |
|---|---|---|---|---|---|
| N | 18034 | 28 | 19 | 10 | 27 |
| S | 176 | 371 | 7 | 0 | 2 |
| V | 102 | 2 | 1324 | 13 | 7 |
| F | 38 | 0 | 18 | 105 | 1 |
| Q | 31 | 2 | 5 | 0 | 1570 |