| Rank | Method | Representation | Balanced acc. | \|r\| vs best | p (Bonferroni) | Sig. |
|:----:|--------|----------------|:-------------:|:-------------:|:--------------:|:----:|
| 1 | **ChromaDCNN (chroma_pretrain)** | 2D chromatogram | **0.854 ± 0.101** | — | — | — |
| 2 | SVM (RBF) | chroma-PCA | 0.698 ± 0.102 | 0.711 | 1.13e-08 | Yes |
| 3 | Random Forest | sum spectrum | 0.696 ± 0.113 | 0.684 | 4.91e-08 | Yes |
| 4 | PLS-DA | chroma-PCA | 0.692 ± 0.096 | 0.734 | 3.22e-09 | Yes |
| 5 | ChromaDCNN (from_scratch) | 2D chromatogram | 0.603 ± 0.176 | 0.799 | 7.43e-11 | Yes |
| 6 | SVM (RBF) | max-projection | 0.532 ± 0.120 | 0.960 | 1.68e-15 | Yes |
| 7 | Random Forest | chroma-PCA | 0.530 ± 0.115 | 0.970 | 8.19e-16 | Yes |
| 8 | SVM (RBF) | sum spectrum | 0.493 ± 0.083 | 0.992 | 1.53e-16 | Yes |
| 9 | PLS-DA | max-projection | 0.469 ± 0.107 | 0.983 | 3.15e-16 | Yes |
| 10 | PLS-DA | sum spectrum | 0.429 ± 0.114 | 0.992 | 1.64e-16 | Yes |
| 11 | MLP | max-projection | 0.395 ± 0.123 | 0.996 | 1.21e-16 | Yes |
| 12 | Random Forest | max-projection | 0.382 ± 0.094 | 0.998 | 1.02e-16 | Yes |
| 13 | MLP | chroma-PCA | 0.377 ± 0.159 | 0.970 | 8.21e-16 | Yes |
| 14 | MLP | sum spectrum | 0.369 ± 0.125 | 0.994 | 1.39e-16 | Yes |
