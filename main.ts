// pumpfun-top5-forecast-01 — deterministic top-5 forecaster for pump.fun coins.
// Reads /input/history.ndjson + /input/task.json, writes /output/forecast.json.
// Node 22 stdlib only. No network, no wall clock, no Math.random: the same
// history always produces byte-identical output (fixed-seed Monte Carlo).
//
// Model (log10 market cap in SOL, per coin):
//  - pool coins: log cap follows a drift that fades with tau=1.5h (hot coins
//    cool off) plus diffusion scaled by horizon; per-coin drift/vol from OLS
//    on the coin's own marks, shrunk for short series.
//  - curve coins: can only reach the top 5 by graduating; graduation chance
//    from quote reserves vs the ~85 SOL completion level and the inflow rate;
//    a debuting coin's cap centres on its curve cap (~0.7x, wide sd) because
//    observed curve->pool transitions mostly land at or below the curve cap.
//  - age-dependent crash hazard: a newborn curve coin can rug (hazard
//    lambda0=0.15/h at birth, decaying with age tau=0.5h, floor 0.02/h);
//    a crash costs ~1.0 log10 of cap.
//  - rank probabilities: seeded Monte Carlo (mulberry32, 800k samples) over
//    candidates; sample count chosen so the tail probabilities are stable.
//  - unnamed reserve per place grows with horizon (newcomers are born after
//    the cut; in practice ~3 of 9 coins that ever held top-5 were newcomers)
//    up to a 0.65 cap; the reserve also carries coins the model cannot name.
//  - size bins: per-place mixture of the named coins' predicted cap
//    distributions plus a newcomer component centred on the Monte Carlo
//    mean cap at that place (placeMeanCap[k]), sd 0.6.
//  - p=0 rows: likely-chart coins with negligible MC P(place) are named at
//    p=0 (max 100 rows/place, format-compliant) so the unnamed share
//    concentrates on the coins that could actually place.
//
// v4 (2026-11-12): v3 REGRESSED live (rank 213/343, log -38.3, size -19.19
// vs v1 rank 67, log -35.92, size -16.92) even though the 3-cut proxy said
// v3 beats v1 by +7.09. Lesson: the proxy cuts (truth 10-30 min after the
// cut) do not represent the live reads (0-60 min after history_until);
// the short-horizon tuning — narrow newcomer sd 0.35, centre at the
// place-1 level, reserve 0.65 — overshot the live size distribution.
// v4 reverts ONLY the size parameters to v1's (centre placeMeanCap[k],
// sd 0.6, reserve base 0.45) and keeps the order-side improvements
// (crash hazard, p=0 zero rows, 800k samples). Live v4: rank 93/372,
// log -35.88 (order -18.67, size -17.21) — order +0.34 vs v1, size -0.29.
//
// v5 (2026-11-12): v4's size is -0.29 behind v1's. The remaining
// size-relevant differences from v1 are SIZE_SD_MULT 1.15 (widens every
// named coin's survive-path sd 15%) and the crash parts in the size
// mixture. v5 sets SIZE_SD_MULT back to 1.0 to isolate the widening:
// if v5's live size recovers to ~-16.92, the widening was the cost.

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? "/input";
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? "/output";

// ---- tunable parameters (calibrated on the dev history, 2026-11-11) ----
const TAU_H = 1.5;            // drift persistence (hours)
const CRASH_LAMBDA0 = 0.15;   // crash hazard per hour for a newborn coin
const CRASH_TAU_AGE = 0.5;    // hazard decays with coin age (hours)
const CRASH_FLOOR = 0.02;     // residual crash hazard per hour
const CRASH_DROP = 1.0;       // log10 cap lost in a crash
const CRASH_SD = 0.5;         // sd of a crashed coin's log10 cap
const NEWCOMER_CENTER_OFF = 0.06; // newcomer size centre falls off per place from place 1
const SIZE_SD_MULT = 1.0;     // no widening: v4's 1.15 cost live size (v4 -17.21 vs v1 -16.92)
const REAL_P_MIN = 0.0005;    // smallest real-p row kept; below this the coin joins the unnamed pool
const ZERO_MAX_P = 0.001;     // max MC P(place k) for a coin to be zeroed out at that place
const ROW_CAP = 100;          // format cap: at most 100 rows per place
const REAL_ROW_MAX = 60;      // cap on real-p rows per place (rest of the rows go to p=0)
const BASE_SD = 0.15;         // base log10 sd
const YOUNG_EXTRA_SD = 0.15;  // extra sd for single-observation coins
const STALE_DECAY_PER_H = 0.03; // mean decay per hour since last seen
const STALE_SD_PER_H = 0.08;    // extra sd per hour since last seen
const NEWCOMER_BASE = 0.45;   // unnamed reserve at horizon 0
const NEWCOMER_PER_H = 0.025; // unnamed reserve growth per horizon hour
const NEWCOMER_MAX = 0.65;
const DEBUT_SD = 0.55;        // sd of a fresh pool debut around the curve cap (log10)
const DEBUT_LOG_FACTOR = -0.15; // median debut cap = curve cap x 10^-0.15 (~0.7x)
const GRAD_TARGET_QR = 85;    // SOL of quote reserves to complete the curve
const NEWCOMER_SIZE_SD = 0.6;
const MC_SAMPLES = 800000;
const SEED = 0x5eed1234;
const CANDIDATE_MAX = 250;
const LAMPORTS_PER_SOL = 1e9;

type Mark = { at: string; rows: Row[] | null };
type Row = {
  rank: number; mint: string; market_cap_lamports: number; venue: string;
  created_at: string; complete?: boolean; mayhem?: boolean;
  quote_reserves_lamports?: number;
};
type Coin = {
  mint: string;
  pts: { t: number; y: number; venue: string; qr: number }[];
  createdAt: number; // hours since t0
  mean: number; sd: number; gradProb: number; survival: number; mayhem: boolean;
  lastVenue: string; lastT: number; lastY: number;
};

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
// Normal draws via Box-Muller on the seeded stream (spare cached, deterministic).
function makeRandn(rand: () => number) {
  let spare: number | null = null;
  return (): number => {
    if (spare !== null) { const v = spare; spare = null; return v; }
    let u = 0, v = 0, s = 0;
    do { u = rand() * 2 - 1; v = rand() * 2 - 1; s = u * u + v * v; } while (s >= 1 || s === 0);
    const m = Math.sqrt(-2 * Math.log(s) / s);
    spare = v * m;
    return u * m;
  };
}
// Standard normal CDF (Abramowitz & Stegun 7.1.26, |err| < 1.5e-7).
function normCdf(x: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989422804014327 * Math.exp(-x * x / 2);
  let p = d * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  if (x > 0) p = 1 - p;
  return p;
}
function floor6(x: number): number { return Math.floor(x * 1e6) / 1e6; }

function main(): void {
  const task = JSON.parse(readFileSync(`${INPUT_DIR}/task.json`, "utf8"));
  const t0 = Date.parse(task.t0) / 3600000;
  const historyUntil = Date.parse(task.history_until) / 3600000;
  const historyUntilRel = historyUntil - t0; // hours since t0 (same frame as per-coin times)
  const reads: string[] = task.reads;
  const meanRead = reads.reduce((s: number, r: string) => s + Date.parse(r), 0) / reads.length / 3600000;
  const horizon = Math.min(24, Math.max(0.1, meanRead - historyUntil)); // hours
  const places: number = task.places ?? 5;
  const edges: number[] = task.size.edges;
  const nBins = edges.length + 1;

  const marks: Mark[] = readFileSync(`${INPUT_DIR}/history.ndjson`, "utf8")
    .split("\n").filter((l) => l.trim().length > 0)
    .map((l) => JSON.parse(l))
    .filter((m: any) => m.ok === true && Array.isArray(m.rows));

  const coins = new Map<string, Coin>();
  for (const m of marks) {
    const mt = Date.parse(m.at) / 3600000 - t0;
    for (const r of m.rows ?? []) {
      const capSol = Number(r.market_cap_lamports) / LAMPORTS_PER_SOL;
      if (!(capSol > 0)) continue;
      let c = coins.get(r.mint);
      if (!c) {
        c = { mint: r.mint, pts: [], createdAt: Date.parse(r.created_at) / 3600000 - t0, mean: 0, sd: 0, gradProb: 0, survival: 1, mayhem: false, lastVenue: r.venue, lastT: mt, lastY: 0 };
        coins.set(r.mint, c);
      }
      c.pts.push({ t: mt, y: Math.log10(capSol), venue: r.venue, qr: Number(r.quote_reserves_lamports ?? 0) / LAMPORTS_PER_SOL });
      c.lastVenue = r.venue; c.lastT = mt; c.lastY = Math.log10(capSol);
      if (r.mayhem) c.mayhem = true;
    }
  }

  // ---- per-coin predicted distribution at the answer horizon ----
  const all: Coin[] = [];
  for (const c of coins.values()) {
    const n = c.pts.length;
    const tLast = c.lastT, yLast = c.lastY;
    // OLS slope of log cap vs time (per hour), shrunk for short series.
    let slope = 0, vol = 0.28;
    if (n >= 2) {
      const mt = c.pts.reduce((s, p) => s + p.t, 0) / n;
      const my = c.pts.reduce((s, p) => s + p.y, 0) / n;
      let num = 0, den = 0;
      for (const p of c.pts) { num += (p.t - mt) * (p.y - my); den += (p.t - mt) * (p.t - mt); }
      slope = den > 1e-9 ? num / den : 0;
      slope = Math.max(-0.5, Math.min(0.5, slope)) * (n / (n + 3));
      if (n >= 3) {
        let ss = 0;
        for (const p of c.pts) { const fit = my + (num / den) * (p.t - mt); ss += (p.y - fit) * (p.y - fit); }
        vol = Math.sqrt(ss / Math.max(1, n - 2)) / Math.sqrt(Math.max(1e-9, c.pts[n - 1].t - c.pts[0].t));
        vol = Math.max(0.08, Math.min(0.5, vol));
      }
    }
    const staleH = Math.max(0, historyUntilRel - tLast);
    const driftTerm = slope * TAU_H * (1 - Math.exp(-horizon / TAU_H));
    c.mean = yLast + driftTerm - STALE_DECAY_PER_H * staleH;
    let sd2 = vol * vol * horizon + BASE_SD * BASE_SD + (STALE_SD_PER_H * staleH) ** 2;
    if (n < 2) sd2 += YOUNG_EXTRA_SD * YOUNG_EXTRA_SD;
    c.sd = Math.sqrt(sd2);
    // Graduation chance for curve coins (only path to the top 5).
    c.gradProb = 0;
    if (c.lastVenue === "curve") {
      const qrLast = c.pts[c.pts.length - 1].qr;
      let qrSlope = 0;
      if (n >= 2) {
        const mt = c.pts.reduce((s, p) => s + p.t, 0) / n;
        const mq = c.pts.reduce((s, p) => s + p.qr, 0) / n;
        let num = 0, den = 0;
        for (const p of c.pts) { num += (p.t - mt) * (p.qr - mq); den += (p.t - mt) * (p.t - mt); }
        qrSlope = den > 1e-9 ? num / den : 0;
      }
      if (qrSlope > 0 && qrLast < GRAD_TARGET_QR) {
        const tComplete = (GRAD_TARGET_QR - qrLast) / qrSlope;
        c.gradProb = Math.min(0.85, 1 - Math.exp(-horizon / Math.max(0.25, tComplete)));
      } else if (qrLast >= GRAD_TARGET_QR) {
        c.gradProb = 0.85;
      } else {
        c.gradProb = Math.min(0.2, 0.03 * horizon);
      }
    }
    // Age-dependent crash hazard: a newborn coin can rug; survivors get safer.
    // Hazard integrated over the horizon: floor*H + lambda0*tau*(e^{-age/tau} - e^{-(age+H)/tau}).
    const ageH = Math.max(0, historyUntilRel - c.createdAt);
    const haz = CRASH_FLOOR * horizon + CRASH_LAMBDA0 * CRASH_TAU_AGE * (Math.exp(-ageH / CRASH_TAU_AGE) - Math.exp(-(ageH + horizon) / CRASH_TAU_AGE));
    c.survival = Math.exp(-haz);
    all.push(c);
  }

  // ---- candidates: top by predicted mean ----
  all.sort((a, b) => b.mean - a.mean || (a.mint < b.mint ? -1 : 1));
  const cand = all.slice(0, Math.min(CANDIDATE_MAX, all.length));
  cand.sort((a, b) => (a.mint < b.mint ? -1 : 1)); // fixed draw order

  // ---- seeded Monte Carlo over rank ----
  const rand = mulberry32(SEED);
  const randn = makeRandn(rand);
  const counts = new Map<string, number[]>(); // mint -> per-place counts
  for (const c of cand) counts.set(c.mint, new Array(places).fill(0));
  const placeMeanCap: number[] = new Array(places).fill(0);
  for (let s = 0; s < MC_SAMPLES; s++) {
    // top-5 by sampled log cap (insertion into a small sorted list)
    const top: { mint: string; v: number }[] = [];
    for (const c of cand) {
      let v: number;
      if (rand() > c.survival) {
        // Crashed (rug): cap collapses far below the top-5 level.
        v = c.lastY - CRASH_DROP + CRASH_SD * randn();
      } else if (c.lastVenue === "curve" && c.gradProb > 0) {
        // Graduating: the debut pool cap centres on the curve cap (~0.7x),
        // wide because a hot debut can run up fast after migration.
        if (rand() < c.gradProb) v = c.lastY + DEBUT_LOG_FACTOR + DEBUT_SD * randn();
        else v = c.mean + c.sd * randn();
      } else {
        v = c.mean + c.sd * randn();
      }
      if (top.length < places || v > top[top.length - 1].v) {
        let i = top.length;
        while (i > 0 && top[i - 1].v < v) i--;
        top.splice(i, 0, { mint: c.mint, v });
        if (top.length > places) top.pop();
      }
    }
    for (let k = 0; k < top.length; k++) {
      counts.get(top[k].mint)![k]++;
      placeMeanCap[k] += top[k].v;
    }
  }
  for (let k = 0; k < places; k++) placeMeanCap[k] /= MC_SAMPLES;

  // ---- probabilities with newcomer reserve ----
  // Row budget per place is 100 (format cap). Real-p rows go to the coins the
  // Monte Carlo puts at that place; the remaining slots go to p=0 rows on the
  // coins most likely to be on the chart but almost certain not to take this
  // place: the rules score a p=0 row as 0 for its coin and take it out of the
  // place's unnamed share, so the share concentrates on the coins left
  // unnamed (newcomers and movers). A p=0 row on a coin that does take the
  // place scores the floor, so only coins with negligible MC P(place) qualify.
  const newcomerReserve = Math.min(NEWCOMER_MAX, NEWCOMER_BASE + NEWCOMER_PER_H * horizon);
  const namedMass = 1 - newcomerReserve;
  const probabilities: { mint: string; place: number; p: number }[] = [];
  const placeNamed = new Map<number, { mint: string; p: number }[]>(); // normalized to sum 1
  // Likely-chart coins: top by predicted mean (pool coins first — a mayhem
  // coin still on its curve has its rows dropped by the scorer, so it cannot
  // shrink the unnamed share).
  const chartLikely = [...all]
    .sort((a, b) => (a.lastVenue === "pool" ? 0 : 1) - (b.lastVenue === "pool" ? 0 : 1) || b.mean - a.mean || (a.mint < b.mint ? -1 : 1))
    .slice(0, 150);
  for (let k = 0; k < places; k++) {
    const rows: { mint: string; p: number }[] = [];
    let total = 0;
    for (const c of cand) {
      const cnt = counts.get(c.mint)![k];
      if (cnt > 0) { rows.push({ mint: c.mint, p: cnt / MC_SAMPLES }); total += cnt / MC_SAMPLES; }
    }
    rows.sort((a, b) => b.p - a.p || (a.mint < b.mint ? -1 : 1));
    const named: { mint: string; p: number }[] = [];
    const realRows: { mint: string; p: number }[] = [];
    let keptFrac = 0;
    for (const r of rows) {
      const p = floor6((r.p / total) * namedMass);
      if (p >= REAL_P_MIN && realRows.length < REAL_ROW_MAX) {
        realRows.push({ mint: r.mint, p });
        named.push({ mint: r.mint, p: r.p / total });
        keptFrac += r.p / total;
      }
    }
    // Renormalize the size fractions over the kept rows.
    if (keptFrac > 0) for (const n of named) n.p /= keptFrac;
    const used = new Set(realRows.map((r) => r.mint));
    const zeroRows: { mint: string; p: number }[] = [];
    for (const c of chartLikely) {
      if (realRows.length + zeroRows.length >= ROW_CAP) break;
      if (used.has(c.mint)) continue;
      if (c.mayhem && c.lastVenue === "curve") continue;
      const pk = (counts.get(c.mint)?.[k] ?? 0) / MC_SAMPLES;
      if (pk < ZERO_MAX_P) zeroRows.push({ mint: c.mint, p: 0 });
    }
    for (const r of realRows) probabilities.push({ mint: r.mint, place: k + 1, p: r.p });
    for (const r of zeroRows) probabilities.push({ mint: r.mint, place: k + 1, p: 0 });
    placeNamed.set(k, named);
  }
  probabilities.sort((a, b) => a.place - b.place || b.p - a.p || (a.mint < b.mint ? -1 : 1));

  // ---- top5: highest predicted mean cap ----
  const top5 = all.slice(0, places).map((c) => c.mint);

  // ---- sizes: per-place mixture over named coins + newcomer component ----
  // The top 5 bunch near the leader (spread 0.16-0.35 log10 in the practice data)
  // and newcomers enter near the top level, so the newcomer component is centred
  // on the place-1 MC level with a small per-place fall-off, not on the old
  // coins' low place means.
  const sizes: { place: number; bin: number; p: number }[] = [];
  for (let k = 0; k < places; k++) {
    const named = placeNamed.get(k)!;
    const binP = new Array(nBins).fill(0);
    for (const { mint, p } of named) {
      const c = coins.get(mint)!;
      const w = p * namedMass;
      // mixture: survive path (curve: no-grad vs debut) + crash path
      const surv = c.survival;
      const parts: { mean: number; sd: number; w: number }[] =
        c.lastVenue === "curve" && c.gradProb > 0
          ? [{ mean: c.mean, sd: c.sd * SIZE_SD_MULT, w: w * surv * (1 - c.gradProb) }, { mean: c.lastY + DEBUT_LOG_FACTOR, sd: DEBUT_SD, w: w * surv * c.gradProb }, { mean: c.lastY - CRASH_DROP, sd: CRASH_SD, w: w * (1 - surv) }]
          : [{ mean: c.mean, sd: c.sd * SIZE_SD_MULT, w: w * surv }, { mean: c.lastY - CRASH_DROP, sd: CRASH_SD, w: w * (1 - surv) }];
      for (const part of parts) {
        if (part.w <= 0) continue;
        for (let b = 0; b < nBins; b++) {
          const lo = b === 0 ? -Infinity : edges[b - 1];
          const hi = b === nBins - 1 ? Infinity : edges[b];
          binP[b] += part.w * (normCdf((hi - part.mean) / part.sd) - normCdf((lo - part.mean) / part.sd));
        }
      }
    }
    const newcomerCenter = placeMeanCap[k]; // v1 centre: MC mean cap at this place (v2/v3 recenter to place-1 level overshot live: v3 size -19.19 vs v1 -16.92)
    for (let b = 0; b < nBins; b++) {
      const lo = b === 0 ? -Infinity : edges[b - 1];
      const hi = b === nBins - 1 ? Infinity : edges[b];
      binP[b] += newcomerReserve * (normCdf((hi - newcomerCenter) / NEWCOMER_SIZE_SD) - normCdf((lo - newcomerCenter) / NEWCOMER_SIZE_SD));
    }
    for (let b = 0; b < nBins; b++) {
      const p = floor6(binP[b]);
      if (p > 0) sizes.push({ place: k + 1, bin: b, p });
    }
  }

  mkdirSync(OUTPUT_DIR, { recursive: true });
  writeFileSync(`${OUTPUT_DIR}/${task.output ?? "forecast.json"}`, `${JSON.stringify({ probabilities, top5, sizes })}\n`);
}

main();
