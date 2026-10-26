# Matched crowding comparison

This report describes synthetic output from the uncalibrated HerdLab simulator. It is not evidence about real market behavior. No parameters were selected in response to these results.

## Design

- Base config: `configs/baseline.yaml`
- Episode pairs: 20 consecutive seeds (`0-19`)
- Low-overlap condition: `agents.feature_overlap = 0`
- High-overlap condition: `agents.feature_overlap = 1`
- The two conditions retain identical family counts, parameter dispersion, base limits, aggregate q_max within each pair, market configuration, and exogenous arrays. Only feature overlap differs.
- In the current simulator, `feature_overlap` changes lag weights for the 12 momentum/reversion agents; the 18 learned agents do not consume this knob.
- Uncertainty: percentile 95% confidence intervals from 10,000 resamples of the 20 complete episode pairs (bootstrap seed 20261026).
- C^pos is averaged over all recorded position states, including the initial state; C^flow is averaged over all order steps.
- Excess realized volatility is `std(simulated returns) - std(exogenous returns)` within an episode (population standard deviations).

## Observed outcome

No forced flow occurred in either condition. High overlap had mean C^pos lower by 0.00147597 and mean C^flow lower by 0.00309075; both paired intervals excluded zero. The drawdown difference was -0.00000014 and the excess-volatility difference was -0.00000001; both intervals included zero. These values describe this simulator configuration only and do not show that greater feature overlap increased crowding or instability.

## Paired results

Differences are high overlap minus low overlap. Intervals that include zero do not establish a directional difference in this finite simulator experiment.

| Metric | Low mean | High mean | Paired difference | 95% episode-bootstrap CI |
| --- | ---: | ---: | ---: | ---: |
| Maximum drawdown | 0.04787769 | 0.04787755 | -0.00000014 | [-0.00000077, +0.00000035] |
| Forced share of absolute agent flow | 0.00000000 | 0.00000000 | +0.00000000 | [+0.00000000, +0.00000000] |
| Mean C^pos | 0.39333392 | 0.39185795 | -0.00147597 | [-0.00238331, -0.00052985] |
| Mean C^flow | 0.40534513 | 0.40225438 | -0.00309075 | [-0.00558926, -0.00080477] |
| Excess realized volatility | 0.00000008 | 0.00000007 | -0.00000001 | [-0.00000003, +0.00000000] |

## Episode pairs

| Seed | DD low | DD high | Forced low | Forced high | C^pos low | C^pos high | C^flow low | C^flow high | Excess vol low | Excess vol high |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.03456115 | 0.03456084 | 0.00000000 | 0.00000000 | 0.41314831 | 0.41021897 | 0.41271425 | 0.40891789 | 0.00000028 | 0.00000025 |
| 1 | 0.06757508 | 0.06757536 | 0.00000000 | 0.00000000 | 0.44675996 | 0.44471764 | 0.42311280 | 0.41485006 | 0.00000153 | 0.00000153 |
| 2 | 0.04048185 | 0.04048264 | 0.00000000 | 0.00000000 | 0.46174392 | 0.46511172 | 0.43477405 | 0.43923130 | 0.00000092 | 0.00000096 |
| 3 | 0.03079970 | 0.03079854 | 0.00000000 | 0.00000000 | 0.45932688 | 0.45696541 | 0.41635230 | 0.41658815 | -0.00000560 | -0.00000565 |
| 4 | 0.04402227 | 0.04402317 | 0.00000000 | 0.00000000 | 0.34114499 | 0.34059415 | 0.36501736 | 0.35808638 | -0.00000311 | -0.00000313 |
| 5 | 0.04742217 | 0.04742394 | 0.00000000 | 0.00000000 | 0.35675974 | 0.35388766 | 0.37815418 | 0.37658750 | -0.00000045 | -0.00000047 |
| 6 | 0.05509075 | 0.05508957 | 0.00000000 | 0.00000000 | 0.46936646 | 0.46772095 | 0.39023751 | 0.38614368 | -0.00000444 | -0.00000441 |
| 7 | 0.13596994 | 0.13597062 | 0.00000000 | 0.00000000 | 0.39956552 | 0.39932437 | 0.45655858 | 0.44095528 | -0.00000113 | -0.00000110 |
| 8 | 0.02346878 | 0.02346379 | 0.00000000 | 0.00000000 | 0.39560925 | 0.38951240 | 0.38414291 | 0.37000095 | -0.00000088 | -0.00000098 |
| 9 | 0.03827861 | 0.03827828 | 0.00000000 | 0.00000000 | 0.39004282 | 0.38623072 | 0.36323648 | 0.35402779 | 0.00000169 | 0.00000167 |
| 10 | 0.03765721 | 0.03765747 | 0.00000000 | 0.00000000 | 0.34964318 | 0.35147174 | 0.38929849 | 0.38994086 | -0.00000069 | -0.00000069 |
| 11 | 0.02370588 | 0.02370629 | 0.00000000 | 0.00000000 | 0.33821276 | 0.33740504 | 0.42147904 | 0.41583147 | 0.00000124 | 0.00000121 |
| 12 | 0.03154775 | 0.03154880 | 0.00000000 | 0.00000000 | 0.44696239 | 0.44556451 | 0.42323298 | 0.42510459 | 0.00000327 | 0.00000328 |
| 13 | 0.04187868 | 0.04187868 | 0.00000000 | 0.00000000 | 0.31734651 | 0.31367616 | 0.41878687 | 0.42251340 | 0.00000051 | 0.00000048 |
| 14 | 0.07610886 | 0.07610897 | 0.00000000 | 0.00000000 | 0.43171564 | 0.42727236 | 0.37400711 | 0.37331200 | 0.00000224 | 0.00000227 |
| 15 | 0.01780955 | 0.01780958 | 0.00000000 | 0.00000000 | 0.31607838 | 0.31527916 | 0.42402775 | 0.41948823 | -0.00000020 | -0.00000020 |
| 16 | 0.04307713 | 0.04307711 | 0.00000000 | 0.00000000 | 0.37888054 | 0.37917323 | 0.42728380 | 0.43008726 | 0.00000194 | 0.00000191 |
| 17 | 0.07110348 | 0.07110313 | 0.00000000 | 0.00000000 | 0.40473490 | 0.40481609 | 0.38941577 | 0.39242255 | -0.00000062 | -0.00000063 |
| 18 | 0.07513200 | 0.07513160 | 0.00000000 | 0.00000000 | 0.38429913 | 0.38441955 | 0.36543789 | 0.36306518 | 0.00000129 | 0.00000125 |
| 19 | 0.02186299 | 0.02186264 | 0.00000000 | 0.00000000 | 0.36533702 | 0.36379717 | 0.44963256 | 0.44793308 | 0.00000380 | 0.00000381 |

Forced share, C^pos, and C^flow use private simulator ground truth solely for this controlled diagnostic comparison; they are not observer inputs.
