**Benchmark panel and split composition**

The archived GPT-2 / Transformer-XL panel: architectures per split, all pairs, size-controlled pairs, the number of architectures per hidden width (64/128/256/512), the feasible pool size at each parameter cap (10M/20M/30M/50M), the total-parameter range and the outcome field used. The final split is an architecture holdout within the same archive, not an independent benchmark.

| Split | Architectures | All pairs | Restr. pairs | Widths | Pools | Param. range | Label |
|---|---|---|---|---|---|---|---|
| development | 100 | 4950 | 180 | 25/25/26/24 | 24/65/83/96 | 5059475 - 58491823 | valid_ppl |
| final | 100 | 4950 | 185 | 24/26/27/23 | 25/57/79/96 | 3748692 - 59374348 | test_ppl |

- The panel carries one recorded outcome per architecture, without training seeds, tokens or hardware.
