"""pumpfun-top5-forecast-01 entry: Monte Carlo forecast of the top 5 by market cap.

Python 3.13, stdlib only, no network, deterministic (fixed seed).
Model per answer read (horizon h hours after the history ends):
  - each named coin: log10 cap = now + damped momentum + shared market shock + idiosyncratic noise (sd grows with h)
  - newcomers (coins born after the cut): Poisson count growing with h, log10 cap ~ Normal below the current #5
  - place probabilities = share of simulations where a coin lands on that place (newcomers stay unnamed)
  - sizes = smoothed order-statistic histogram mixed with a small uniform floor
ponytail: parameters are priors, not tuned (practice history is only 6 marks); upgrade = fit on final dataset.
"""
import json
import math
import os
import random
from datetime import datetime

IN = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUT = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
N_SIM = 1500  # per read
TOP_POOL = 60  # coins simulated
SHRINK = 0.92  # never put full mass on named coins
FLOOR_BIN = 0.002


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def poisson(rng, lam):
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


def main():
    task = json.load(open(os.path.join(IN, "task.json")))
    marks = []
    for line in open(os.path.join(IN, "history.ndjson")):
        line = line.strip()
        if line:
            m = json.loads(line)
            if m.get("ok") and m.get("rows"):
                marks.append(m)
    if not marks:
        raise SystemExit("no usable mark")
    places = task["places"]
    edges = task["size"]["edges"]
    nbins = len(edges) + 1
    cut = ts(task["history_until"])
    horizons = [max((ts(r) - cut) / 3600.0, 0.05) for r in task["reads"]]

    def cap(r):
        return math.log10(int(r["market_cap_lamports"])) - 9

    last = marks[-1]
    rows = sorted(last["rows"], key=lambda r: r["rank"])[:TOP_POOL]
    prev = {}  # mint -> list of (t, logcap)
    for m in marks[-4:]:
        t = ts(m["at"])
        for r in m["rows"]:
            prev.setdefault(r["mint"], []).append((t, cap(r)))
    t_last = ts(last["at"])
    coins = []
    for r in rows:
        s = cap(r)
        pts = prev.get(r["mint"], [])
        slope = 0.0  # log10 per hour over the last marks
        if len(pts) >= 2 and pts[-1][0] > pts[0][0]:
            slope = (pts[-1][1] - pts[0][1]) / ((pts[-1][0] - pts[0][0]) / 3600.0)
        coins.append((r["mint"], s, slope))
    s5 = sorted((c[1] for c in coins), reverse=True)[min(places, len(coins)) - 1]

    rng = random.Random(20261112)
    cnt = {c[0]: [0.0] * places for c in coins}
    size_hist = [[0.0] * nbins for _ in range(places)]
    total_sims = 0

    def bin_of(x):
        if x < edges[0]:
            return 0
        for i in range(1, len(edges)):
            if x < edges[i]:
                return i
        return nbins - 1

    for h in horizons:
        sd_i = 0.22 + 0.05 * min(h, 12)  # idiosyncratic
        sd_m = 0.08 + 0.01 * min(h, 12)  # shared market
        lam_new = 0.5 * min(h, 12)
        for _ in range(N_SIM):
            shock = rng.gauss(0, sd_m)
            sims = []
            for mint, s, slope in coins:
                drift = max(-0.3, min(0.3, 0.2 * slope * min(h, 2.0)))
                sims.append((s + drift + shock + rng.gauss(0, sd_i), mint))
            for _k in range(poisson(rng, lam_new)):
                sims.append((s5 - 0.2 + rng.gauss(0, 0.5) + shock, None))
            sims.sort(key=lambda x: -x[0])
            for k in range(min(places, len(sims))):
                v, mint = sims[k]
                if mint is not None:
                    cnt[mint][k] += 1
                # smooth the size histogram: spread each draw over +-1 bin triangularly
                b = bin_of(v)
                size_hist[k][b] += 0.5
                if b > 0:
                    size_hist[k][b - 1] += 0.25
                if b < nbins - 1:
                    size_hist[k][b + 1] += 0.25
            total_sims += 1

    probs = []
    for mint, _s, _sl in coins:
        for k in range(places):
            p = math.floor(cnt[mint][k] / total_sims * SHRINK * 1e6) / 1e6
            if p >= 0.004:
                probs.append({"mint": mint, "place": k + 1, "p": p})
    # keep each place's sum <= 1 and at most 100 rows (SHRINK already guarantees the sum)
    sizes = []
    for k in range(places):
        tot = sum(size_hist[k]) or 1.0
        for b in range(nbins):
            p = math.floor((size_hist[k][b] / tot * (1 - FLOOR_BIN * nbins) + FLOOR_BIN) * 1e6) / 1e6
            sizes.append({"place": k + 1, "bin": b, "p": p})

    top5, used = [], set()
    for k in range(places):
        best = max((c for c in coins if c[0] not in used), key=lambda c: cnt[c[0]][k])
        top5.append(best[0])
        used.add(best[0])

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, task.get("output", "forecast.json")), "w") as f:
        json.dump({"probabilities": probs, "top5": top5, "sizes": sizes}, f, separators=(",", ":"))
        f.write("\n")


if __name__ == "__main__":
    main()
