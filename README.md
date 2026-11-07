# HerdLab

HerdLab simulates a population of competing trading agents, records what an outsider could see (prices, volume, executed imbalance, liquidity), and tests whether an observer model can use that public record to infer hidden crowding and predict synchronized selling.

Every result here comes from an uncalibrated synthetic simulator. None of it is evidence about real markets, and this repository contains no live-trading code.

## Setup

```bash
python -m venv .venv
```

```bash
.venv/bin/pip install -e . pytest
```

```bash
.venv/bin/pytest -q
```

Run the latest experiments. Each writes its report to `docs/reports/`. On 10 CPU cores the population run took about 26 minutes, the mechanism sweep 2 minutes and the scaled-signal grid about 125 minutes. Finished conditions are cached under `runs/`:

```bash
.venv/bin/python -m herdlab.experiments.round6_population
```

```bash
.venv/bin/python -m herdlab.experiments.round6_mechanism --workers 10
```

```bash
.venv/bin/python -m herdlab.experiments.round6_scaled_grid
```

## How it works

Each agent turns a return forecast into a target position and sends an order toward it. Net orders move the price through an aggregate impact model, and agents that breach a drawdown limit are forced to liquidate. Momentum, mean-reversion, ridge, LightGBM and MLP agents share this loop.

Per-agent positions and orders stay in a private record that is used only to build labels. Observers (logistic regression, LightGBM, a strategy-replication baseline and a GRU) see public features only. The main target is whether a large episode of agent selling occurs in the next 10 steps, with the threshold fit on training episodes only.

Equations and design choices are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Designs are pre-registered in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) before each run, and negative results stay in the log.

## Results so far

| Question | Result | Main caveat |
|---|---|---|
| Does the GRU generalize to an agent architecture it never trained on? (round 3) | PR-AUC 0.66 [0.41, 0.79] against 0.04 for the trivial reference, at a 1.5% base rate | LightGBM reaches 0.60, and agents saw only public prices |
| Does private information make inference harder? (rounds 4 to 6) | It depends on signal size. With signals scaled to each family's forecast (round 6), information makes inference harder at β = 0.5 (gain change −0.26 [−0.35, −0.13] at ρ = 0.5), easier at β = 1 (+0.09 [+0.07, +0.12]), and has no established direction at β = 0.25 | The large round-5 effect at β = 0.25 (−0.38) shrank to −0.07 [−0.22, +0.07] once scaling was fixed. At β = 1 the scaling itself fails, because learned agents' forecasts shrink. Noise alone degrades inference at every β |
| Does shared agent input raise crowding? (H1, rounds 4 and 5) | Yes: mean position alignment rises from 0.25 to 0.83 and the event rate from 0.03% to 14% as overlap goes from 0 to 1 | Forced selling doesn't follow. Under a 5% drawdown limit, forced-flow windows fall from 2,841 to 14 as overlap rises |
| Why does forced selling fall as overlap rises? (round 6) | Overlap had two parts, shared lag weights and shared signal noise. Sharing the noise causes the drop (effect on forced-order fraction −0.0050 [−0.0057, −0.0045]); lag overlap has no established effect | With independent noise, forced sellers are mostly agents whose noise happened to lose. This doesn't show that crowding protects agents |
| Which agents make crowding visible? (round 6) | Removing momentum agents lowers GRU PR-AUC from 0.96 to 0.73, reversion to 0.80 and MLP to 0.90. Removing linear or GBDT agents has no established effect | One market setting, and no run removes reversion alone from the low-scoring momentum/linear/MLP mix |
| Can the observer predict forced selling? (round 5) | Weakly: GRU PR-AUC is 0.09–0.18 at base rates of 1.6–6.2% in the stress conditions | These conditions tighten the drawdown limit, because forced flow almost disappears otherwise |

Removing price impact cuts GRU PR-AUC from 0.95 to 0.26, so the observer depends on the price footprint of agent flow. Disabling forced deleveraging barely changes it (0.95), so the default label does not track liquidations closely.

The agent population matters more than the seeds. Swapping seeds moves GRU PR-AUC by at most 0.035, while swapping the population moves it by 0.16 to 0.19 (about 0.78 for momentum, linear and MLP agents against about 0.96 for the five-family mix). Round 6 points to the missing reversion agents as the likely cause. Every number above belongs to one selected market setting.

Full tables are in [docs/reports/experiment_zero_round3.md](docs/reports/experiment_zero_round3.md), [docs/reports/experiment_zero_round5.md](docs/reports/experiment_zero_round5.md), [docs/reports/round5_reconcile.md](docs/reports/round5_reconcile.md), [docs/reports/overlap_sweep_stress.md](docs/reports/overlap_sweep_stress.md), [docs/reports/round6_population.md](docs/reports/round6_population.md), [docs/reports/round6_mechanism.md](docs/reports/round6_mechanism.md) and [docs/reports/round6_scaled_grid.md](docs/reports/round6_scaled_grid.md).

## Next steps

1. Test crowding with a shared strategy and independent noise, since shared noise is what removed forced selling in round 6.
2. Remove reversion agents alone from the momentum/linear/MLP mix to confirm what drives the population gap.
3. Work out why learned agents' forecasts shrink under strong private signals, which breaks the per-family scaling at β = 1.
4. Test on real data. Binance perpetual futures come first, because liquidation data gives a partly observable label for forced flow.

## Concepts behind the project

- **Liquidity spirals:** margin calls force sales, which worsen liquidity and trigger more margin calls. Brunnermeier and Pedersen, "Market Liquidity and Funding Liquidity" (2009).
- **Crowded-trade limits:** arbitrageurs can't easily tell how many others hold the same trade. Stein, "Presidential Address: Sophisticated Investors and Market Efficiency" (2009).
- **Market impact:** Kyle, "Continuous Auctions and Insider Trading" (1985), and Bouchaud, Farmer and Lillo, "How Markets Slowly Digest Changes in Supply and Demand" (2009).
- **Crowding from public data:** Lou and Polk, "Comomentum: Inferring Arbitrage Activity from Return Correlations" (2022). This is a natural public-data baseline.
- **Flow toxicity:** Easley, López de Prado and O'Hara, "Flow Toxicity and Liquidity in a High-Frequency World" (2012), and the critique by Andersen and Bondarenko, "VPIN and the Flash Crash" (2014).
- **Self-exciting flow:** Hawkes processes measure how much trading is triggered by earlier trading. Hardiman, Bercot and Bouchaud, "Critical Reflexivity in Financial Markets" (2013).
- **Endogenous risk:** risk created by traders reacting to each other. Danielsson and Shin, "Endogenous Risk" (2003).
- **Identifiability:** when does public data determine the hidden state at all? This project's ablations touch the question, but it hasn't been studied directly.

## Failures worth studying

| Event | Relevance |
|---|---|
| Quant Quake, August 2007 | Similar quant funds unwound together. Khandani and Lo, "What Happened to the Quants in August 2007?" (2011) |
| Crypto crash, October 10 2025 | About $19B of reported liquidations, with auto-deleveraging closing positions. See the [FTI writeup](https://www.fticonsulting.com/zh-cn/china/insights/articles/crypto-crash-october-2025-leverage-met-liquidity) and [Amberdata's analysis](https://blog.amberdata.io/leverage-liquidations-the-31b-deleveraging). Accounts of the cause are disputed |
| LTCM, 1998 | Other firms held similar trades, so the exit disappeared when LTCM needed it |
| Volmageddon, February 2018 | Short-volatility products had to rebalance mechanically, which made their flow partly predictable |
| Yen carry unwind, August 2024 | A crowded leveraged trade unwound across assets |
| Flash Crash, May 2010 | Liquidity vanished, which is a different mechanism from crowding. CFTC and SEC, "Findings Regarding the Market Events of May 6, 2010" |
| Archegos, 2021 | Concentration was hidden across several prime brokers, so even the banks couldn't see the whole position |
| Portfolio insurance, 1987 | Many investors following one rule created the selling they meant to protect against |

## Videos

These came from a YouTube search and haven't been reviewed for accuracy:

- [Lessons from Long-Term Capital Management](https://www.youtube.com/watch?v=U47rMAyMCAw)
- [The 1998 Collapse of Long-Term Capital Management](https://www.youtube.com/watch?v=zPIexh2vOQo)
- [The Failure of Long Term Capital Management](https://www.youtube.com/watch?v=Ft4TMtamMKs)
- [The 2018 Volmageddon Crash: How Short Volatility Blew Up](https://www.youtube.com/watch?v=xhKY1k_0_wg)
- [What Volmageddon Taught Me About Market Risk](https://www.youtube.com/watch?v=C8Jjshvm084)
