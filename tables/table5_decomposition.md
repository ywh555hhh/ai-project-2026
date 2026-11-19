**Exact decomposition of the added prefix term**

The added term decomposes exactly into $T = U + P + R$, an FFN-permutation-invariant part, a positional part and a heterogeneous-attention residual. The last column records whether every rank is unchanged when $R$ is removed, for $a \in \{0.25, 0.5, 1\}$. A tiny residual and unchanged ranks mean the structural change carries no extra predictive information on this archive.

| Split | Max identity err. | Swaps | Max swap err. | Max reconstr. err. | Median $\lvert R\rvert/T$ | Max $\lvert R\rvert/T$ | Same ranks w/o $R$ |
|---|---|---|---|---|---|---|---|
| development | 5.35e-13 | 826 | 2.06e-12 | 1.82e-12 | 0.00990% | 0.27260% | True/True/True |
| final | 3.84e-13 | 844 | 3.10e-12 | 1.82e-12 | 0.00786% | 0.23557% | True/True/True |

- The residual contributes a median of about 0.01\% of $T$, so dropping it changes no ranking.
