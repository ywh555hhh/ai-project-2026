**Budgeted architecture selection: mean regret**

Mean regret of the same selections: the selected PPL minus the best feasible PPL in that pool, so zero means the oracle was found and lower is better. The rows match the selection table one for one.

| Split | Cap | $k$ | Pool | NSC = ours | Pos. example | Params | Ref. FLOPs | Random | SD |
|---|---|---|---|---|---|---|---|---|---|
| dev | 10M | 1 | 24 | 1.9408 | 0.0000 | 0.0000 | 1.9408 | 13.2851 | 10.8874 |
| dev | 10M | 3 | 24 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.9054 | 4.3481 |
| dev | 20M | 1 | 65 | 3.1748 | 3.1748 | 2.4933 | 3.1748 | 17.2254 | 11.8259 |
| dev | 20M | 3 | 65 | 0.5650 | 0.5650 | 0.0000 | 0.5650 | 6.2148 | 4.6444 |
| dev | 30M | 1 | 83 | 1.8061 | 1.8061 | 1.8061 | 1.8061 | 16.2464 | 12.4264 |
| dev | 30M | 3 | 83 | 0.0846 | 0.0846 | 1.0745 | 0.0000 | 6.1871 | 6.0560 |
| dev | 50M | 1 | 96 | 0.0886 | 0.0886 | 0.0886 | 0.0886 | 15.3954 | 12.6853 |
| dev | 50M | 3 | 96 | 0.0886 | 0.0886 | 0.0886 | 0.0886 | 6.0426 | 5.8318 |
| final | 10M | 1 | 25 | 0.0000 | 0.0000 | 3.7219 | 0.0000 | 12.1766 | 9.2506 |
| final | 10M | 3 | 25 | 0.0000 | 0.0000 | 3.7219 | 0.0000 | 3.9713 | 4.1380 |
| final | 20M | 1 | 57 | 1.1363 | 1.1363 | 8.4421 | 1.1363 | 15.3630 | 10.5438 |
| final | 20M | 3 | 57 | 0.8499 | 0.8499 | 1.4938 | 0.8499 | 6.2924 | 5.0253 |
| final | 30M | 1 | 79 | 1.6209 | 1.6209 | 0.0000 | 1.6209 | 15.5329 | 10.7638 |
| final | 30M | 3 | 79 | 0.7717 | 0.7717 | 0.0000 | 0.7717 | 5.9301 | 5.5506 |
| final | 50M | 1 | 96 | 1.3715 | 1.3715 | 1.1297 | 1.3715 | 15.3593 | 11.2497 |
| final | 50M | 3 | 96 | 1.3291 | 1.3291 | 1.1297 | 1.3291 | 5.8445 | 5.6723 |

- SD is the standard deviation of the random selection over the 200 trials, not of the regret.
