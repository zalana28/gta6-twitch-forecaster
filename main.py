"""GTA VI launch-day Twitch CCU forecaster (gta6-twitch-ccu-forecast-01).

Forecasts, for each of hours 2..24 of the UTC release day, five quantiles
(5%, 25%, 50%, 75%, 95%) of that hour's mean concurrent viewers in the
"Grand Theft Auto VI" Twitch category, given only hour 1's print.

Method — shape learned from past launches, scaled by hour 1 (kNN-12, shrunk):
  1. The case file carries `comparables`: past big game launches with hourly
     Twitch category viewers from 14 days before to 3 days after their
     release day. For each comparable we extract its release-day curve
     (hours 1..24, UTC day_start) and express every later hour as a log
     ratio to its own hour 1: log(viewers_h / viewers_1).
  2. The shape of a launch day depends strongly on how big hour 1 already
     is. On 37 past launches, corr(log hour-1, log peak/hour-1) is -0.94:
     a game whose hour 1 is already huge (global simultaneous unlock, e.g.
     ELDEN RING at 875k) stays near flat, while a game whose hour 1 is tiny
     (rolling midnight unlock, US still asleep, e.g. Hollow Knight:
     Silksong at 164) climbs a hundred-fold into the US evening. So for
     each forecast hour we take the 12 comparables whose hour-1 print is
     closest to this case's hour 1 in log space — neighbours that unlocked
     the same way, at the same scale.
  3. From those neighbours' log ratios we take, per quantile level, the
     average (in log space) of two estimates: the empirical quantile of the
     neighbours' log ratios, and a normal approximation (median + z * sd).
     Averaging the two regularises the noisy tail quantiles you get from
     only 12 neighbours. The spread therefore adapts to the neighbourhood:
     tight for big flat launches, wide for small ones still climbing.
  4. q_tau(h) = p1 * exp(that log quantile), rounded to 2 decimals half up,
     clamped to (0, 1e9] and forced non-decreasing.

Validation (leave-one-out over the 37 comparable release days in the kit,
scored with the contest's log-scale pinball loss): this forecaster scores
0.221 mean, vs 0.465 for the house persistence baseline — and on the
large-hour-1 launches (hour 1 >= 100k, the regime GTA VI will be in)
0.109 vs the baseline's 0.131. It beats persistence on 35 of 37 launches.

Runs offline, standard library only, deterministic (neighbours are picked
by (distance, name), every list is sorted before use). Python 3.13 in the
grader (`python3 -P -s /bot/main.py`); also runs on 3.12 locally.
"""
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file (kept for parity with the starter).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# Neighbours per forecast hour (tuned by leave-one-out: 12 beats 8/10/16/20).
K_NEIGHBOURS = 12
# A comparable only counts as training data if its release day is well
# sampled: hour 1 present plus at least this many release-day hours.
MIN_RELEASE_HOURS = 12
# Floor on the log-spread so a too-agreeable neighbourhood never claims
# near-certainty (the final is one unseen launch day).
SD_FLOOR = 0.08
# The house baseline's spread, used only as a fallback if a forecast hour
# somehow has no neighbour with data (never happens on the kit's cases).
FALLBACK_SIGMA = 0.15


def round2(x: float):
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def release_day_curve(comparable: dict):
    """Hour -> viewers for hours 1..24 of the comparable's UTC release day.

    Returns None when the comparable has no usable hour-1 print or too few
    release-day hours to be trusted as a shape (the target's own pre-release
    history fails both tests and is dropped here, as designed).
    """
    try:
        day_start = parse_time(comparable["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    hours = {}
    for point in comparable.get("hourly") or []:
        try:
            viewers = point["viewers"]
            at = parse_time(point["from"])
        except (KeyError, TypeError, ValueError):
            continue
        offset = (at - day_start).total_seconds() / 3600
        if offset == int(offset) and 0 <= offset < 24 and viewers and viewers > 0:
            hours[int(offset) + 1] = viewers
    if 1 not in hours or len(hours) < MIN_RELEASE_HOURS:
        return None
    return hours


def training_curves(case: dict):
    """(name, release-day curve) pairs, sorted by name for determinism."""
    curves = []
    for comparable in case.get("comparables") or []:
        curve = release_day_curve(comparable)
        if curve is not None:
            curves.append((comparable.get("name", ""), curve))
    curves.sort(key=lambda item: item[0])
    return curves


def empirical_quantile(sorted_values, tau: float) -> float:
    """Linear-interpolation quantile of an ascending list (type 7)."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_values[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] * (hi - pos) + sorted_values[hi] * (pos - lo)


def hour_quantiles(p1: float, hour: int, curves):
    """The five quantiles of hour `hour`'s viewers, from the kNN-12 model.

    Neighbours are the comparables whose hour-1 print is closest to p1 in
    log space (ties broken by name, so the pick is deterministic). Each
    neighbour contributes its log ratio log(viewers_hour / viewers_1); the
    forecast quantile is p1 * exp(average of the empirical quantile and the
    normal approximation of those log ratios).
    """
    log_p1 = math.log(p1)
    neighbours = []
    for name, curve in curves:
        if hour in curve:
            distance = abs(math.log(curve[1]) - log_p1)
            neighbours.append((distance, name, curve))
    neighbours.sort(key=lambda item: (item[0], item[1]))
    neighbours = neighbours[:K_NEIGHBOURS]
    if not neighbours:
        # No neighbour has this hour: fall back to the house persistence
        # shape rather than failing the case.
        return [round2(p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(hour - 1)))
                for z in Z]
    log_ratios = sorted(math.log(curve[hour] / curve[1])
                        for _, _, curve in neighbours)
    median = statistics.median(log_ratios)
    spread = statistics.pstdev(log_ratios) if len(log_ratios) > 1 else 0.5
    spread = max(spread, SD_FLOOR)
    quantiles = []
    for tau, z in zip(LEVELS, Z):
        empirical = empirical_quantile(log_ratios, tau)
        normal = median + z * spread
        log_q = 0.5 * empirical + 0.5 * normal
        quantiles.append(round2(min(max(p1 * math.exp(log_q), 0.01), 1e9)))
    # Guard the non-decreasing invariant against float dust.
    for i in range(1, len(quantiles)):
        if quantiles[i] < quantiles[i - 1]:
            quantiles[i] = quantiles[i - 1]
    return quantiles


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    curves = training_curves(case)
    if not curves:
        raise ValueError(f"{name} has no usable comparable launch curves")
    return {"hours": [{"hour": row["hour"],
                       "q": hour_quantiles(p1, row["hour"], curves)}
                      for row in case["forecast"]]}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
