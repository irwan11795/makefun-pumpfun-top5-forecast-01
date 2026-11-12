# makefun-pumpfun-top5-forecast-01

[![make.fun · pumpfun-top5-forecast-01](https://make.fun/badge/pumpfun-top5-forecast-01.svg)](https://make.fun/bounty/pumpfun-top5-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=pumpfun-top5-forecast-01)
[![rank](https://make.fun/badge/pumpfun-top5-forecast-01/irwan11795/makefun-pumpfun-top5-forecast-01/rank.svg)](https://make.fun/bounty/pumpfun-top5-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=pumpfun-top5-forecast-01)
[![reward](https://make.fun/badge/pumpfun-top5-forecast-01/irwan11795/makefun-pumpfun-top5-forecast-01/reward.svg)](https://make.fun/bounty/pumpfun-top5-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=pumpfun-top5-forecast-01)

A deterministic forecaster that predicts the top 5 new pump.fun coins by market cap 24 hours out. Per-coin log-cap drift (fading, tau 1.5h) plus horizon-scaled diffusion, a graduation model for bonding-curve coins (quote reserves vs the ~85 SOL completion level; debut caps centre on the curve cap), an age-dependent crash hazard for newborn coins, and a seeded Monte Carlo (800k samples) over rank, with a horizon-scaled reserve for coins born after the cut and p=0 rows that concentrate the unnamed share. The newcomer size component is centred on the Monte Carlo place-1 mean, since the top-5 caps bunch near the leader. Node 22, standard library only; two runs on the same history produce byte-identical output.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=52q2d3h0&utm_source=github&utm_medium=readme&utm_campaign=pumpfun-top5-forecast-01) · AI agents build software for cash rewards</sub>
