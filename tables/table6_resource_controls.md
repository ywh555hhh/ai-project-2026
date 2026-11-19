**Successively stricter resource matching**

Restricted $\tau_b$ on successively stricter matched subsets. The prescribed control matches hidden width and total parameters; the FLOPs and depth controls are additional diagnostics added after the final outcomes were available, so they are not causal controls. The last two rows give the diagnostics of the prescribed parameter-matched pairs.

| Split / control | Pairs | Architectures | original NSC | Ref. FLOPs | Params | Prefix $\alpha = 1$ | Pos. ctrl $\alpha = 1$ |
|---|---|---|---|---|---|---|---|
| dev / w+par | 180 | 94 | 0.622222 | 0.622222 | 0.133333 | 0.633333 | 0.622222 |
| dev / w+par+flops | 39 | 47 | 0.179487 | 0.179487 | 0.435897 | 0.230769 | 0.179487 |
| dev / w+par+flops+depth | 15 | 30 | 0.333333 | 0.333333 | 0.466667 | 0.600000 | 0.466667 |
| final / w+par | 185 | 96 | 0.329730 | 0.329730 | 0.275676 | 0.297297 | 0.308108 |
| final / w+par+flops | 45 | 47 | -0.066667 | -0.066667 | 0.377778 | -0.200000 | -0.155556 |
| final / w+par+flops+depth | 18 | 25 | -0.111111 | -0.111111 | 0.555556 | -0.111111 | -0.111111 |

- Matching parameters does not fix FLOPs or depth, so the restricted statistic keeps a strong resource component.
- The FLOPs and depth rows are post-freeze diagnostics and are reported with all remaining pair and architecture counts.
- development: 141 of 141 matched pairs fall below a 0.90 FLOPs ratio (median 0.7115, minimum 0.1863), 161 differ in depth, and NSC correlates with reference FLOPs at $\rho$ = 0.9992.
- final: 140 of 140 matched pairs fall below a 0.90 FLOPs ratio (median 0.7813, minimum 0.1885), 164 differ in depth, and NSC correlates with reference FLOPs at $\rho$ = 0.9996.
