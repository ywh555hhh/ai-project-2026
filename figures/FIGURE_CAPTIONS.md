# Report figures - captions and sources

Ready to paste into the final report (Times New Roman 12 pt, A4, 2.5 cm margins).  Each figure is written as PNG (300 dpi) and as a vector PDF; the PDF is the one to include.

## fig1_alpha_sweep.png / fig1_alpha_sweep.pdf

**Prefix reweighting never improves ranking**

Spearman rank correlation against negative perplexity for the prefix-capacity reweighting $S_a = S_0 + a\sum_l C_l q_l$ and its position-only control, over the frozen grid $a \in \{0, 0.25, 0.5, 1\}$. The dashed line is the original NSC score ($a = 0$); the band is a 95\% paired-architecture bootstrap interval for the prefix score (1000 resamples). Every positive $a$ lowers the correlation on both the development split (validation PPL) and the held-out final split (test PPL), so the development protocol selects $a = 0$. The vertical axis is zoomed.

*Source:* evaluation_summary.json (ranking, frozen.sensitivity_analysis); analysis_summary.json (paired_architecture_bootstrap)

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=0.95\textwidth]{figures/fig1_alpha_sweep.pdf}
    \caption{Spearman rank correlation against negative perplexity for the prefix-capacity reweighting $S_a = S_0 + a\sum_l C_l q_l$ and its position-only control, over the frozen grid $a \in \{0, 0.25, 0.5, 1\}$. The dashed line is the original NSC score ($a = 0$); the band is a 95\% paired-architecture bootstrap interval for the prefix score (1000 resamples). Every positive $a$ lowers the correlation on both the development split (validation PPL) and the held-out final split (test PPL), so the development protocol selects $a = 0$. The vertical axis is zoomed.}
    \label{fig:fig1}
\end{figure}
```

## fig2_ranking_baselines.png / fig2_ranking_baselines.pdf

**Ranking quality of every method and baseline**

Spearman $\rho$ over all pairs for all sixteen evaluated methods, with the size-controlled restricted $\tau_b$ (same hidden width, total-parameter ratio $\geq 0.90$) marked as a diamond. Points are grouped into the original NSC and $a = 0$ variants, the prefix variants ($a > 0$), the positional controls and the resource baselines. The $a = 0$ variants coincide because the selected weight is zero; reported parameters are clearly the weakest baseline on both splits, while reference FLOPs is close to NSC.

*Source:* evaluation_summary.json (ranking)

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=0.95\textwidth]{figures/fig2_ranking_baselines.pdf}
    \caption{Spearman $\rho$ over all pairs for all sixteen evaluated methods, with the size-controlled restricted $\tau_b$ (same hidden width, total-parameter ratio $\geq 0.90$) marked as a diamond. Points are grouped into the original NSC and $a = 0$ variants, the prefix variants ($a > 0$), the positional controls and the resource baselines. The $a = 0$ variants coincide because the selected weight is zero; reported parameters are clearly the weakest baseline on both splits, while reference FLOPs is close to NSC.}
    \label{fig:fig2}
\end{figure}
```

## fig3_mechanism_reorder.png / fig3_mechanism_reorder.pdf

**The score does respond to layer order**

Structural sensitivity of the scores on the 41 synthetic reorderings; these configurations have no trained outcomes and test sensitivity only. Left: the largest absolute score change at $a = 1$ per family. The original NSC changes by exactly 0 on every case because its layer terms are separable and summed. Right: head-only reorderings, which leave every FFN width untouched; the position-only control is blind to them (change exactly 0) while the prefix term is not, which isolates the attention-prefix mechanism.

*Source:* mechanism_check.json (case_details)

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=0.95\textwidth]{figures/fig3_mechanism_reorder.pdf}
    \caption{Structural sensitivity of the scores on the 41 synthetic reorderings; these configurations have no trained outcomes and test sensitivity only. Left: the largest absolute score change at $a = 1$ per family. The original NSC changes by exactly 0 on every case because its layer terms are separable and summed. Right: head-only reorderings, which leave every FFN width untouched; the position-only control is blind to them (change exactly 0) while the prefix term is not, which isolates the attention-prefix mechanism.}
    \label{fig:fig3}
\end{figure}
```

## fig4_nsc_flops_collinearity.png / fig4_nsc_flops_collinearity.pdf

**NSC is almost a restatement of compute**

Recorded perplexity (left) and the original NSC score (right) against reference FLOPs per token, coloured by hidden width, on both splits. The score is nearly a deterministic function of compute ($\rho \approx 0.999$), and so is the recorded outcome, so the score carries little information beyond model size on this archive. Reference FLOPs are a dense-operation estimate, not measured latency.

*Source:* analysis_summary.json (architecture_profiles, resource_diagnostics)

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=0.95\textwidth]{figures/fig4_nsc_flops_collinearity.pdf}
    \caption{Recorded perplexity (left) and the original NSC score (right) against reference FLOPs per token, coloured by hidden width, on both splits. The score is nearly a deterministic function of compute ($\rho \approx 0.999$), and so is the recorded outcome, so the score carries little information beyond model size on this archive. Reference FLOPs are a dense-operation estimate, not measured latency.}
    \label{fig:fig4}
\end{figure}
```

## fig5_selection_regret.png / fig5_selection_regret.pdf

**Budgeted architecture selection under parameter caps**

Finite-pool selection under total-parameter caps of 10M, 20M, 30M and 50M. Top: mean best selected PPL; the dashed line is the best feasible model in each pool. Bottom: mean regret to that oracle. Solid lines are $k = 1$, dashed lines are $k = 3$. The orange band spans the 5-95\% percentile of 200 fixed-seed random-selection trials. All methods choose from the same feasible pool before any outcome is revealed; the variability shown for random selection is over those trials, not over training seeds.

*Source:* analysis_summary.json (splits.*.selection_cases)

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=0.95\textwidth]{figures/fig5_selection_regret.pdf}
    \caption{Finite-pool selection under total-parameter caps of 10M, 20M, 30M and 50M. Top: mean best selected PPL; the dashed line is the best feasible model in each pool. Bottom: mean regret to that oracle. Solid lines are $k = 1$, dashed lines are $k = 3$. The orange band spans the 5-95\% percentile of 200 fixed-seed random-selection trials. All methods choose from the same feasible pool before any outcome is revealed; the variability shown for random selection is over those trials, not over training seeds.}
    \label{fig:fig5}
\end{figure}
```

## fig6_decomposition.png / fig6_decomposition.pdf

**What the added prefix term actually contains**

Exact decomposition of the added term $T = \sum_l C_l q_l$ into an FFN-permutation-invariant part $U$, a positional part $P$ and a heterogeneous-attention residual $R$. The invariant part dominates, the positional part is next, and $R$ contributes a median of about 0.01\% of $T$ (at most about 0.3\%). Because $R$ is negligible, removing it leaves every rank unchanged at $a \in \{0.25, 0.5, 1\}$ on both splits, which is why the structural change does not translate into a better ranking.

*Source:* analysis_summary.json (splits.*.decomposition)

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=0.95\textwidth]{figures/fig6_decomposition.pdf}
    \caption{Exact decomposition of the added term $T = \sum_l C_l q_l$ into an FFN-permutation-invariant part $U$, a positional part $P$ and a heterogeneous-attention residual $R$. The invariant part dominates, the positional part is next, and $R$ contributes a median of about 0.01\% of $T$ (at most about 0.3\%). Because $R$ is negligible, removing it leaves every rank unchanged at $a \in \{0.25, 0.5, 1\}$ on both splits, which is why the structural change does not translate into a better ranking.}
    \label{fig:fig6}
\end{figure}
```

## fig7_scoring_cost.png / fig7_scoring_cost.pdf

**Cold and warm scoring cost per method**

Warm scoring time per method for all 200 configurations, after the Marchenko-Pastur quadrature cache is populated. The dashed line marks the median cold-panel time, which is dominated by the shared quadrature and is therefore nearly identical for every method. Warm time exposes scorer overhead: the without-$R$ diagnostics cost roughly six times the original NSC scorer. These are scorer diagnostics, not model inference latencies, and the timings share one cache, so they are not a fair cold-start comparison between methods.

*Source:* evaluation_summary.json (scoring_cost.by_method)

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=0.95\textwidth]{figures/fig7_scoring_cost.pdf}
    \caption{Warm scoring time per method for all 200 configurations, after the Marchenko-Pastur quadrature cache is populated. The dashed line marks the median cold-panel time, which is dominated by the shared quadrature and is therefore nearly identical for every method. Warm time exposes scorer overhead: the without-$R$ diagnostics cost roughly six times the original NSC scorer. These are scorer diagnostics, not model inference latencies, and the timings share one cache, so they are not a fair cold-start comparison between methods.}
    \label{fig:fig7}
\end{figure}
```
