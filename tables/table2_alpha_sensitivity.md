**Sensitivity of the ranking to the prefix weight $\alpha$**

Spearman $\rho$ of the prefix reweighting, its position-only control and the variant without the head residual $R$, over the frozen grid $\alpha \in \{0, 0.25, 0.5, 1\}$. The six middle columns are Spearman $\rho$ of the prefix score, its position-only control and the variant without $R$; the last three are the development Kendall $\tau_b$, the development score range and the score-tie count of the prefix score. The choice was frozen on the development split alone.

| $\alpha$ | dev prefix | dev position | dev no-$R$ | final prefix | final position | final no-$R$ | dev $\tau_b$ | dev score range | ties |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.969745 | 0.969745 | 0.969745 | 0.965941 | 0.965941 | 0.965941 | 0.8614 | 116 - 13047 | 0 |
| 0.25 | 0.967909 | 0.968293 | 0.967909 | 0.965149 | 0.965269 | 0.965149 | 0.8582 | 122 - 13904 | 0 |
| 0.5 | 0.966649 | 0.966973 | 0.966649 | 0.964200 | 0.963612 | 0.964200 | 0.8566 | 128 - 14760 | 0 |
| 1 | 0.963432 | 0.963900 | 0.963432 | 0.961224 | 0.961608 | 0.961224 | 0.8505 | 140 - 16474 | 0 |

- Every prescribed positive weight has a lower global correlation than the original NSC on both panels, so the development rule selects $\alpha = 0$.
- The score range is the minimum and maximum of the prefix score over the development split. No pair was tied on both the score and the outcome for any setting.
