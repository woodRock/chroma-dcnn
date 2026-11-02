| Rank | Method | Representation | Balanced acc. | \|r\| vs best | p (Bonferroni) | Sig. |
|:----:|--------|----------------|:-------------:|:-------------:|:--------------:|:----:|
| 1 | **ChromaDCNN (chroma_pretrain)** | 2D chromatogram | **0.857 ± 0.117** | — | — | — |
| 2 | SVM (RBF) | chroma-PCA | 0.698 ± 0.102 | 0.685 | 4.59e-08 | Yes |
| 3 | Random Forest | sum spectrum | 0.696 ± 0.113 | 0.668 | 1.07e-07 | Yes |
| 4 | PLS-DA | chroma-PCA | 0.692 ± 0.096 | 0.706 | 1.50e-08 | Yes |
| 5 | ChromaDCNN (from_scratch) | 2D chromatogram | 0.585 ± 0.165 | 0.822 | 1.79e-11 | Yes |
| 6 | SVM (RBF) | max-projection | 0.532 ± 0.120 | 0.936 | 9.11e-15 | Yes |
| 7 | Random Forest | chroma-PCA | 0.530 ± 0.115 | 0.948 | 4.05e-15 | Yes |
| 8 | SVM (RBF) | sum spectrum | 0.493 ± 0.083 | 0.981 | 3.62e-16 | Yes |
| 9 | PLS-DA | max-projection | 0.469 ± 0.107 | 0.974 | 6.38e-16 | Yes |
| 10 | PLS-DA | sum spectrum | 0.429 ± 0.114 | 0.984 | 2.88e-16 | Yes |
| 11 | MLP | max-projection | 0.395 ± 0.123 | 0.991 | 1.68e-16 | Yes |
| 12 | Random Forest | max-projection | 0.382 ± 0.094 | 0.996 | 1.22e-16 | Yes |
| 13 | MLP | chroma-PCA | 0.377 ± 0.159 | 0.963 | 1.31e-15 | Yes |
| 14 | MLP | sum spectrum | 0.369 ± 0.125 | 0.988 | 2.05e-16 | Yes |
