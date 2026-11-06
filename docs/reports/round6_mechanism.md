# Round-6 mechanism diagnostics

**Synthetic-simulator disclaimer:** All results are from the uncalibrated synthetic HerdLab simulator. They are not evidence of real-market behavior, funds, or live trading.

## Configuration and execution

`configs/experiment_zero.yaml`; 500 steps, 30 agents; rho=0.25, beta=0.25, max_drawdown=0.05, learned_overlap_mode=lag_weights. Seeds 7000–7029 shared across every condition. Five coupled omega levels plus lag overlap {0,1} × noise sharing {0,1}; diagonal corners reused: 210 unique episodes.

Runtime estimate before execution: 2–4 minutes. Measured invocation wall time 2.09 minutes; retained condition compute 2.08 minutes; 10 workers. Command: `.venv/bin/python -m herdlab.experiments.round6_mechanism --workers 10`. Completed conditions are cached under `runs/round6_mechanism_cache/` with config/source fingerprints and private episode outcomes.

Forced-flow threshold is the explicitly registered existing zero threshold, fixed without fitting on these diagnostic episodes; horizon 10, complete strictly-future windows only. Counts include overlapping windows, not independent events. Forced-order fraction includes nonzero forced buys and sells.

Drawdown breaches count new liquidation schedules triggered with drawdown above the risk limit (including simultaneous leverage breaches); ongoing schedules and cooldown do not add breaches. Fraction ever breached uses those drawdown events. Simultaneously liquidating counts active liquidation decisions, including zero exposure, excluding cooldown. Mean agent maximum drawdown uses each full equity path and its running all-time peak, without the risk-manager cooldown reset. P&L dispersion is the population standard deviation across agents of terminal minus initial equity in currency units; mean return is the mean of those P&Ls divided by initial agent equity. All diagnostics stay private.

OLS includes an intercept. All 95% percentile CIs use 10,000 whole-episode resamples; slope and factorial contrasts jointly resample seed identities across conditions. Slope RNG seeds: 20261106 + outcome row index; factorial seeds: 20261206/20261207. Main effects are high minus low averaged equally over the other knob.

## Condition means and forced-flow gates

| Lag | Noise | forced_order_fraction | drawdown_breaches | fraction_ever_breached | max_simultaneously_liquidating | mean_agent_max_drawdown | pnl_dispersion | mean_agent_return | forced_flow_windows | forced_flow_rate | Total windows | Gate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0 | 0.00529111 | 16.06666667 | 0.22444444 | 1.86666667 | 0.03457333 | 7417.16186381 | 1.00308758 | 94.70000000 | 0.19326531 | 2841 | clears both |
| 0.25 | 0.25 | 0.00229556 | 6.96666667 | 0.11666667 | 1.33333333 | 0.02403112 | 6845.99288309 | 1.08738012 | 43.73333333 | 0.08925170 | 1312 | clears both |
| 0.5 | 0.5 | 0.00035111 | 1.06666667 | 0.03000000 | 0.56666667 | 0.01592450 | 5936.69608568 | 1.17686460 | 6.63333333 | 0.01353741 | 199 | clears both |
| 0.75 | 0.75 | 0.00007778 | 0.23333333 | 0.00777778 | 0.20000000 | 0.01279543 | 5344.63028998 | 1.30315087 | 1.40000000 | 0.00285714 | 42 | uninformative (<1%) |
| 1 | 1 | 0.00003333 | 0.10000000 | 0.00333333 | 0.10000000 | 0.01128617 | 4872.79223216 | 1.46123732 | 0.46666667 | 0.00095238 | 14 | uninformative (<1%), underpowered (<30) |
| 0 | 1 | 0.00004444 | 0.13333333 | 0.00444444 | 0.13333333 | 0.01137677 | 4846.22600885 | 1.45242077 | 0.93333333 | 0.00190476 | 28 | uninformative (<1%), underpowered (<30) |
| 1 | 0 | 0.00487778 | 14.76666667 | 0.21444444 | 2.03333333 | 0.03345236 | 7432.96985991 | 1.01300841 | 89.96666667 | 0.18360544 | 2699 | clears both |

Sweep levels with fewer than 30 forced-flow windows: 1.

These gates flag the forced-flow label. The registered continuous-outcome slope and knob rules below are reported even for sparse windows; this avoids reintroducing the round-5 direction-rule problem. No observer is fitted in item 2, so the replication rule is not applicable.

## Slopes with 95% CIs

| Outcome | Slope | Lower | Upper |
| --- | ---: | ---: | ---: |
| forced_order_fraction | -0.0050933333 | -0.0058791333 | -0.0043990889 |
| drawdown_breaches | -15.4666666667 | -17.8266666667 | -13.3466666667 |
| fraction_ever_breached | -0.2204444444 | -0.2342222222 | -0.2066666667 |
| max_simultaneously_liquidating | -1.8666666667 | -2.0533333333 | -1.6800000000 |
| mean_agent_max_drawdown | -0.0231240070 | -0.0259688544 | -0.0205811354 |
| pnl_dispersion | -2636.0407425670 | -3021.6009813175 | -2251.1373715941 |
| mean_agent_return | 0.4528280921 | 0.4043114150 | 0.5022048708 |
| forced_flow_windows | -92.3200000000 | -106.2000000000 | -79.3733333333 |
| forced_flow_rate | -0.1884081633 | -0.2164646259 | -0.1625571429 |

## Factorial main effects on forced-order fraction

| Setting | Effect | Lower | Upper |
| --- | ---: | ---: | ---: |
| lag overlap | -0.0002122222 | -0.0005378056 | 0.0000977778 |
| noise sharing | -0.0050455556 | -0.0056744722 | -0.0044755556 |

## Rule outcomes (verbatim)

Forced-selling direction: decreases.

M1: supported (association, not cause).

M2: supported (association, not cause).

M3: supported (association, not cause).

lag overlap: inconclusive.

noise sharing: drives the decline in this simulator.

M1–M3 may hold together and describe associations only. Knob attribution is the sole causal interpretation allowed here and is limited to these interventions in this simulator.

## Deviations, definitions and open questions

No equation, label, threshold, population, seed, or parameter was tuned after viewing results. Missing noise_sharing retains the historical mode-dependent coupling and RNG draw order; frozen full-length pre-change stress hashes cover all five omega levels. New logging does not affect decisions. No later step was implemented.

For research-lead review: the registration does not specify breach episode versus exceedance-step counts, reset versus all-time drawdown, terminal versus pathwise P&L dispersion, or whether zero-exposure liquidation decisions count as liquidating. The explicit operational definitions above were fixed before execution. Confirm them before interpreting M1–M3. The general 1%/30-window gates are displayed without suppressing the specifically registered continuous-outcome rules; no replication claim is made.

## Validation

`.venv/bin/pytest -q`: `189 passed in 30.12s`. Verified all seven cache fingerprints, 210 seed/condition rows, all 11 reported CIs and verbatim rule outcomes. Coupled-sweep forced-flow counts reproduce round 5 exactly: 2841/1312/199/42/14. Full-length pre-change hashes and explicit diagonal compatibility tests pass.
