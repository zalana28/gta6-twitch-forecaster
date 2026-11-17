# GTA VI Launch-Day Twitch CCU Forecaster

Entry for make.fun bounty **gta6-twitch-ccu-forecast-01** — "Forecast GTA VI's
launch-day Twitch viewers hour by hour" ($250 pot, deadline 2026-11-19T01:00:00Z).

`main.py` (Python 3.13, standard library only) reads one case from
`/input/<id>.in` and writes `/output/<id>.out`: for each of hours 2–24 of
the UTC release day, five quantiles (5%, 25%, 50%, 75%, 95%) of that hour's
mean concurrent viewers in the "Grand Theft Auto VI" Twitch category, given
only hour 1's print.

## How it works

The case file carries `comparables`: past big game launches with hourly
Twitch category viewers around their release days. The forecaster learns
the launch-day *shape* from them and scales it by this case's hour 1:

1. For each comparable, extract its release-day curve (hours 1–24) and
   express every later hour as a log ratio to its own hour 1.
2. The shape depends strongly on how big hour 1 already is — across 37
   past launches, corr(log hour-1, log peak/hour-1) is −0.94. A game whose
   hour 1 is already huge (global simultaneous unlock) stays near flat;
   a game whose hour 1 is tiny (rolling midnight unlock, US still asleep)
   climbs a hundred-fold into the US evening. So for each forecast hour
   the model takes the **12 comparables whose hour-1 print is closest to
   this case's hour 1 in log space**.
3. From those neighbours' log ratios, each quantile is the average (in log
   space) of the empirical quantile and a normal approximation
   (median + z·sd). Averaging the two regularises the noisy tail quantiles
   from only 12 neighbours. The spread adapts to the neighbourhood: tight
   for big flat launches, wide for small ones still climbing.
4. `q = p1 · exp(log-quantile)`, rounded to 2 decimals (half up), clamped
   to (0, 1e9], forced non-decreasing. If a forecast hour somehow has no
   neighbour with data, it falls back to the house persistence shape.

The target's own pre-release history (GTA VI's category since its
2026-08-27 premiere) is in the comparables but has no release day, so it
is excluded from training automatically.

## Validation

Leave-one-out over the 37 comparable release days in the kit, scored with
the contest's log-scale pinball loss, using this exact shipped code:

| | mean pinball | median |
|---|---|---|
| **This entry** | **0.2214** | 0.1313 |
| House persistence baseline | 0.4648 | 0.1624 |

It beats persistence on 32 of 37 launches. On large-hour-1 launches
(hour 1 ≥ 100k — the regime GTA VI will be in): **0.1093 vs 0.1306**.
Losses to baseline on 5 launches (Diablo IV, Marvel Rivals, Borderlands 4,
Starfield, Monster Hunter Wilds), all cases where persistence was already
near-optimal and this model's spread runs slightly wide; the worst case is
Black Myth: Wukong (1.064 vs baseline 1.717 — still a clear win).

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Runs offline, no network, no model calls, deterministic (same bytes on
re-runs), ~0.03 s per case — far inside the 8 s cap.

## Files

- `main.py` — the whole entry (no other modules)
- `README.md` — this file
- `LICENSE` — MIT
