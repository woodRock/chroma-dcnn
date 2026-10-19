## Ablation: Does ChromaDCNN need the RT axis?

**Protocol:** 10 seeds × 5-fold stratified CV (n=50), `from_scratch`,
identical hyperparameters to main experiment.

**1D-mz construction:** sum spectrum tiled across 200 RT bins →
per-bin √+L2 normalisation. Every RT bin is identical; the model
cannot extract RT-axis information.

| Condition | BA (mean ± SD) | Δ vs 2D | p | \|r\| | Sig. |
|-----------|---------------|---------|---|------|------|
| 2D (from scratch) | 0.594 ± 0.168 | — | — | — | — |
| 1D-mz (degenerate) | 0.391 ± 0.155 | -0.203 | 1.99e-07 | 0.603 | Yes |

**Prediction (from saliency analysis):** 1D-mz should drop toward
PLS-DA-on-sum-spectrum (~0.43). Prediction held.

**Interpretation:**
The RT axis carries significant discriminative information. Removing it (1D-mz) causes a statistically significant drop (p=1.99e-07, |r|=0.603). This causally supports the claim that ChromaDCNN exploits the joint RT x m/z structure rather than spectral content alone.

**Paper framing:**
Include as positive evidence for the 2D-structure hypothesis. The drop from 2D to 1D-mz confirms that the RT dimension is load-bearing.
