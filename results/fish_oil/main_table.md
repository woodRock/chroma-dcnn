| Rank | Method | Representation | Balanced acc. | \|r\| vs best | p (Bonferroni) | Sig. |
|:----:|--------|----------------|:-------------:|:-------------:|:--------------:|:----:|
| 1 | **ChromaDCNN (chroma_pretrain)** | 2D chromatogram | **0.857 ± 0.094** | — | — | — |
| 2 | SVM (RBF) | chroma-PCA | 0.698 ± 0.102 | 0.742 | 2.09e-09 | Yes |
| 3 | Random Forest | sum spectrum | 0.696 ± 0.113 | 0.712 | 1.05e-08 | Yes |
| 4 | PLS-DA | chroma-PCA | 0.692 ± 0.096 | 0.762 | 6.72e-10 | Yes |
| 5 | ChromaDCNN (from_scratch) | 2D chromatogram | 0.605 ± 0.146 | 0.878 | 5.01e-13 | Yes |
| 6 | SVM (RBF) | max-projection | 0.532 ± 0.120 | 0.972 | 7.03e-16 | Yes |
| 7 | Random Forest | chroma-PCA | 0.530 ± 0.115 | 0.981 | 3.51e-16 | Yes |
| 8 | SVM (RBF) | sum spectrum | 0.493 ± 0.083 | 0.996 | 1.16e-16 | Yes |
| 9 | PLS-DA | max-projection | 0.469 ± 0.107 | 0.988 | 2.20e-16 | Yes |
| 10 | PLS-DA | sum spectrum | 0.429 ± 0.114 | 0.994 | 1.36e-16 | Yes |
| 11 | MLP | max-projection | 0.395 ± 0.123 | 0.997 | 1.07e-16 | Yes |
| 12 | Random Forest | max-projection | 0.382 ± 0.094 | 0.999 | 9.59e-17 | Yes |
| 13 | MLP | chroma-PCA | 0.377 ± 0.159 | 0.976 | 4.97e-16 | Yes |
| 14 | MLP | sum spectrum | 0.369 ± 0.125 | 0.996 | 1.16e-16 | Yes |
