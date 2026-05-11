#!/usr/bin/env node
/**
 * Batch experiment runner for Evolving Kelvin-Helmholtz Patterns.
 * CPU-based vorticity-streamfunction simulation — no browser/WebGL needed.
 *
 * Usage:
 *   node batch_fluid.mjs [--gens 30] [--seeds 10] [--topos none,ring,star,fc] [--out results.csv]
 */

import { writeFileSync } from 'fs';

// ── CLI args ────────────────────────────────────────────────

function parseArgs() {
  const args = process.argv.slice(2);
  const opts = { gens: 30, seeds: 10, topos: ['none', 'ring', 'star', 'fc'], out: 'kh_results.csv' };
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--gens')  opts.gens  = parseInt(args[++i]);
    if (args[i] === '--seeds') opts.seeds = parseInt(args[++i]);
    if (args[i] === '--topos') opts.topos = args[++i].split(',');
    if (args[i] === '--out')   opts.out   = args[++i];
  }
  return opts;
}

// ── Configuration ───────────────────────────────────────────

const CONFIG = {
  simSize: 128,
  simSteps: 500,
  warmupIters: 300,
  jacobiIters: 40,
  dt: 0.005,
  numIslands: 4,
  popSize: 8,
  eliteCount: 2,
  tournamentSize: 3,
  migrationInterval: 5,
  mutationSigma: { nu: 0.0008, vShear: 0.3, delta: 0.008, aPert: 0.03, kPert: 0.5 },
  initRange: {
    nu:     [0.0003, 0.003],
    vShear: [1.5, 4.0],
    delta:  [0.03, 0.08],
    aPert:  [0.15, 0.4],
    kPert:  [2, 6]
  }
};

const PARAM_RANGES = {
  nu:     [0.0001, 0.005],
  vShear: [1.0, 5.0],
  delta:  [0.02, 0.12],
  aPert:  [0.05, 0.5],
  kPert:  [1, 8]
};

// ── PRNG ────────────────────────────────────────────────────

function mulberry32(seed) {
  return function() {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

let rng = Math.random;

// ── Utilities ───────────────────────────────────────────────

function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

function gaussianRandom() {
  let u, v, s;
  do { u = rng() * 2 - 1; v = rng() * 2 - 1; s = u*u + v*v; } while (s >= 1 || s === 0);
  return u * Math.sqrt(-2 * Math.log(s) / s);
}

function randRange(lo, hi) { return lo + rng() * (hi - lo); }

function shannonEntropy(histogram, total) {
  let e = 0;
  for (let i = 0; i < histogram.length; i++) {
    if (histogram[i] > 0) { const p = histogram[i] / total; e -= p * Math.log2(p); }
  }
  return e;
}

// ── CPU Vorticity-Streamfunction Simulator ──────────────────

class Simulator {
  constructor(size) {
    this.size = size;
    const n = size * size;
    // Vorticity: ping-pong buffers
    this.omega0 = new Float32Array(n);
    this.omega1 = new Float32Array(n);
    // Streamfunction: ping-pong buffers for Jacobi
    this.psi0 = new Float32Array(n);
    this.psi1 = new Float32Array(n);
  }

  evaluate(ind) {
    const s = this.size;
    const n = s * s;
    const dx = 1.0 / s;
    const dx2 = dx * dx;
    const { nu, vShear, delta, aPert, kPert } = ind;
    const TWO_PI = 6.283185307;

    // ── Initialize vorticity (double shear layer) ──
    for (let iy = 0; iy < s; iy++) {
      const y = (iy + 0.5) / s;  // cell-centered
      for (let ix = 0; ix < s; ix++) {
        const x = (ix + 0.5) / s;
        const arg1 = (y - 0.25) / delta;
        const arg2 = (y - 0.75) / delta;
        const sech2_1 = sech2(arg1);
        const sech2_2 = sech2(arg2);
        const pert = 1.0 + aPert * Math.sin(TWO_PI * kPert * x);
        this.omega0[iy * s + ix] = -(vShear / delta) * (sech2_1 - sech2_2) * pert;
      }
    }

    // Clear streamfunction
    this.psi0.fill(0);
    this.psi1.fill(0);

    let srcOmega = this.omega0, dstOmega = this.omega1;
    let srcPsi = this.psi0, dstPsi = this.psi1;

    // ── Warmup: Poisson solve from scratch ──
    for (let iter = 0; iter < CONFIG.warmupIters; iter++) {
      jacobiStep(srcOmega, srcPsi, dstPsi, s, dx2);
      [srcPsi, dstPsi] = [dstPsi, srcPsi];
    }

    // ── Simulation timesteps ──
    for (let step = 0; step < CONFIG.simSteps; step++) {
      // Poisson solve (Jacobi iterations, warm-started)
      for (let iter = 0; iter < CONFIG.jacobiIters; iter++) {
        jacobiStep(srcOmega, srcPsi, dstPsi, s, dx2);
        [srcPsi, dstPsi] = [dstPsi, srcPsi];
      }

      // Semi-Lagrangian advection + explicit diffusion
      advectDiffuse(srcOmega, srcPsi, dstOmega, s, dx, nu, CONFIG.dt);
      [srcOmega, dstOmega] = [dstOmega, srcOmega];
    }

    // Compute fitness from final vorticity
    return this._fitness(srcOmega, ind);
  }

  _fitness(omega, ind) {
    const s = this.size;
    const n = s * s;
    const bins = 64;
    const hist = new Float32Array(bins);
    let alive = 0;
    const omMax = ind.vShear / ind.delta;

    for (let i = 0; i < n; i++) {
      const normOm = Math.min(Math.abs(omega[i]) / omMax, 1.0);
      hist[Math.min(Math.floor(normOm * bins), bins - 1)]++;
      if (normOm > 0.02 && normOm < 0.95) alive++;
    }

    const entropy = shannonEntropy(hist, n) / Math.log2(bins);
    const aliveRatio = alive / n;
    const alivePenalty = 4 * aliveRatio * (1 - aliveRatio);

    // Mixing score: penalize x-uniform patterns (horizontal bars = no swirls)
    let xVar = 0;
    for (let iy = 0; iy < s; iy++) {
      let rowSum = 0;
      for (let ix = 0; ix < s; ix++) rowSum += Math.abs(omega[iy * s + ix]);
      const rowMean = rowSum / s;
      let rv = 0;
      for (let ix = 0; ix < s; ix++) rv += (Math.abs(omega[iy * s + ix]) - rowMean) ** 2;
      xVar += rv / s;
    }
    xVar /= (s * omMax * omMax);
    const mixingScore = Math.min(xVar * 200, 1.0);

    return entropy * (0.3 + 0.7 * alivePenalty) * (0.3 + 0.7 * mixingScore);
  }
}

// ── Fluid solver kernels ────────────────────────────────────

function sech2(x) {
  x = Math.max(-20, Math.min(20, x));
  const ex = Math.exp(x);
  const ch = (ex + 1.0 / ex) * 0.5;
  return 1.0 / (ch * ch);
}

/** One Jacobi iteration for ∇²ψ = -ω */
function jacobiStep(omega, psiSrc, psiDst, s, dx2) {
  for (let iy = 0; iy < s; iy++) {
    const ym = ((iy - 1 + s) % s) * s;
    const yc = iy * s;
    const yp = ((iy + 1) % s) * s;
    for (let ix = 0; ix < s; ix++) {
      const xm = (ix - 1 + s) % s;
      const xp = (ix + 1) % s;
      const idx = yc + ix;

      const pE = psiSrc[yc + xp];
      const pW = psiSrc[yc + xm];
      const pN = psiSrc[yp + ix];
      const pS = psiSrc[ym + ix];

      psiDst[idx] = (pE + pW + pN + pS + dx2 * omega[idx]) / 4.0;
    }
  }
}

/** Semi-Lagrangian advection + explicit viscous diffusion */
function advectDiffuse(omegaSrc, psi, omegaDst, s, dx, nu, dt) {
  const invTwoDx = 1.0 / (2.0 * dx);
  const invDx2 = 1.0 / (dx * dx);

  for (let iy = 0; iy < s; iy++) {
    const ym = ((iy - 1 + s) % s) * s;
    const yc = iy * s;
    const yp = ((iy + 1) % s) * s;

    for (let ix = 0; ix < s; ix++) {
      const xm = (ix - 1 + s) % s;
      const xp = (ix + 1) % s;
      const idx = yc + ix;

      // Velocity from streamfunction
      const u = (psi[yp + ix] - psi[ym + ix]) * invTwoDx;   // ∂ψ/∂y
      const v = -(psi[yc + xp] - psi[yc + xm]) * invTwoDx;  // -∂ψ/∂x

      // Semi-Lagrangian backtrace (in grid coordinates)
      // Position in [0, s) grid space
      let bx = ix + 0.5 - u * dt * s;  // u is in domain coords, multiply by s for grid
      let by = iy + 0.5 - v * dt * s;

      // Periodic wrapping
      bx = ((bx % s) + s) % s;
      by = ((by % s) + s) % s;

      // Bilinear interpolation
      const ix0 = Math.floor(bx);
      const iy0 = Math.floor(by);
      const fx = bx - ix0;
      const fy = by - iy0;

      const ix1 = (ix0 + 1) % s;
      const iy1 = (iy0 + 1) % s;
      const wx0 = ix0 % s;
      const wy0 = iy0 % s;

      const omegaAdv =
        omegaSrc[wy0 * s + wx0] * (1 - fx) * (1 - fy) +
        omegaSrc[wy0 * s + ix1] * fx * (1 - fy) +
        omegaSrc[iy1 * s + wx0] * (1 - fx) * fy +
        omegaSrc[iy1 * s + ix1] * fx * fy;

      // Explicit viscous diffusion (on current field)
      const oC = omegaSrc[idx];
      const lap = (omegaSrc[yc + xp] + omegaSrc[yc + xm] +
                   omegaSrc[yp + ix] + omegaSrc[ym + ix] - 4.0 * oC) * invDx2;

      omegaDst[idx] = clamp(omegaAdv + dt * nu * lap, -200, 200);
    }
  }
}

// ── Individual ──────────────────────────────────────────────

class Individual {
  constructor(nu, vShear, delta, aPert, kPert) {
    this.nu = nu; this.vShear = vShear; this.delta = delta;
    this.aPert = aPert; this.kPert = kPert;
    this.fitness = 0;
  }

  static random() {
    const r = CONFIG.initRange;
    return new Individual(randRange(...r.nu), randRange(...r.vShear), randRange(...r.delta), randRange(...r.aPert), randRange(...r.kPert));
  }

  clone() {
    const c = new Individual(this.nu, this.vShear, this.delta, this.aPert, this.kPert);
    c.fitness = this.fitness; return c;
  }

  mutate() {
    const s = CONFIG.mutationSigma;
    const R = PARAM_RANGES;
    this.nu     = clamp(this.nu     + gaussianRandom() * s.nu,     R.nu[0],     R.nu[1]);
    this.vShear = clamp(this.vShear + gaussianRandom() * s.vShear, R.vShear[0], R.vShear[1]);
    this.delta  = clamp(this.delta  + gaussianRandom() * s.delta,  R.delta[0],  R.delta[1]);
    this.aPert  = clamp(this.aPert  + gaussianRandom() * s.aPert,  R.aPert[0],  R.aPert[1]);
    this.kPert  = clamp(this.kPert  + gaussianRandom() * s.kPert,  R.kPert[0],  R.kPert[1]);
  }

  static crossover(a, b) {
    return new Individual(
      rng() < 0.5 ? a.nu : b.nu,
      rng() < 0.5 ? a.vShear : b.vShear,
      rng() < 0.5 ? a.delta : b.delta,
      rng() < 0.5 ? a.aPert : b.aPert,
      rng() < 0.5 ? a.kPert : b.kPert
    );
  }
}

// ── Island ──────────────────────────────────────────────────

class Island {
  constructor(id) {
    this.id = id;
    this.population = Array.from({length: CONFIG.popSize}, () => Individual.random());
  }
  get best() { return this.population.reduce((a, b) => a.fitness > b.fitness ? a : b); }
  get worst() { return this.population.reduce((a, b) => a.fitness < b.fitness ? a : b); }
  tournamentSelect() {
    let best = null;
    for (let i = 0; i < CONFIG.tournamentSize; i++) {
      const c = this.population[Math.floor(rng() * this.population.length)];
      if (!best || c.fitness > best.fitness) best = c;
    }
    return best;
  }
  evolve() {
    this.population.sort((a, b) => b.fitness - a.fitness);
    const next = [];
    for (let i = 0; i < CONFIG.eliteCount; i++) next.push(this.population[i].clone());
    while (next.length < CONFIG.popSize) {
      const child = Individual.crossover(this.tournamentSelect(), this.tournamentSelect());
      child.mutate();
      next.push(child);
    }
    this.population = next;
  }
}

// ── Migration ───────────────────────────────────────────────

const TOPOLOGIES = {
  none: () => [],
  ring: (n) => Array.from({length: n}, (_, i) => [i, (i+1) % n]),
  star: (n) => Array.from({length: n-1}, (_, i) => [0, i+1]),
  fc:   (n) => { const e = []; for (let i = 0; i < n; i++) for (let j = i+1; j < n; j++) e.push([i, j]); return e; }
};

function migrate(islands, topology) {
  const edges = TOPOLOGIES[topology](islands.length);
  const migrants = islands.map(isl => isl.best.clone());
  for (const [a, b] of edges) {
    const wA = islands[a].worst, wB = islands[b].worst;
    if (migrants[b].fitness > wA.fitness) {
      const idx = islands[a].population.indexOf(wA);
      islands[a].population[idx] = migrants[b].clone();
    }
    if (migrants[a].fitness > wB.fitness) {
      const idx = islands[b].population.indexOf(wB);
      islands[b].population[idx] = migrants[a].clone();
    }
  }
}

// ── Diversity Metrics ───────────────────────────────────────

function normalizeParams(ind) {
  return ['nu', 'vShear', 'delta', 'aPert', 'kPert'].map(
    k => (ind[k] - PARAM_RANGES[k][0]) / (PARAM_RANGES[k][1] - PARAM_RANGES[k][0])
  );
}

function euclideanDist(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += (a[i] - b[i]) ** 2;
  return Math.sqrt(sum);
}

function avgPairwiseDist(individuals) {
  if (individuals.length < 2) return 0;
  const normed = individuals.map(normalizeParams);
  let total = 0, count = 0;
  for (let i = 0; i < normed.length; i++) {
    for (let j = i + 1; j < normed.length; j++) { total += euclideanDist(normed[i], normed[j]); count++; }
  }
  return total / count;
}

function centroid(individuals) {
  const normed = individuals.map(normalizeParams);
  const dim = normed[0].length;
  const c = new Array(dim).fill(0);
  for (const p of normed) for (let i = 0; i < dim; i++) c[i] += p[i];
  for (let i = 0; i < dim; i++) c[i] /= normed.length;
  return c;
}

function computeDiversity(isls) {
  const allInds = isls.flatMap(isl => isl.population);
  const total = avgPairwiseDist(allInds);
  const withinValues = isls.map(isl => avgPairwiseDist(isl.population));
  const within = withinValues.reduce((a, b) => a + b, 0) / withinValues.length;
  const centroids = isls.map(isl => centroid(isl.population));
  let bTotal = 0, bCount = 0;
  for (let i = 0; i < centroids.length; i++) {
    for (let j = i + 1; j < centroids.length; j++) { bTotal += euclideanDist(centroids[i], centroids[j]); bCount++; }
  }
  return { total, within, between: bCount > 0 ? bTotal / bCount : 0 };
}

// ── Main ────────────────────────────────────────────────────

async function run() {
  const opts = parseArgs();
  const simulator = new Simulator(CONFIG.simSize);
  const rows = [];

  // Time a single evaluation for estimate
  const t0 = performance.now();
  rng = mulberry32(42);
  const testInd = Individual.random();
  simulator.evaluate(testInd);
  const evalMs = performance.now() - t0;
  rng = Math.random;

  const totalEvals = opts.topos.length * opts.seeds * opts.gens * CONFIG.numIslands * CONFIG.popSize;
  const estMinutes = (totalEvals * evalMs / 1000 / 60).toFixed(1);
  console.log(`Single eval: ${evalMs.toFixed(0)}ms | Total: ${totalEvals} evals | Est: ${estMinutes} min`);
  console.log(`Config: ${opts.topos.join(',')} × ${opts.seeds} seeds × ${opts.gens} gens`);
  console.log('---');

  const startTime = performance.now();
  let runIndex = 0;
  const totalRuns = opts.topos.length * opts.seeds;

  for (const topo of opts.topos) {
    for (let seed = 1; seed <= opts.seeds; seed++) {
      rng = mulberry32(seed);
      const islands = Array.from({length: CONFIG.numIslands}, (_, i) => new Island(i));

      for (let gen = 0; gen < opts.gens; gen++) {
        // Evaluate
        for (const isl of islands) {
          for (const ind of isl.population) {
            ind.fitness = simulator.evaluate(ind);
          }
        }

        // Record metrics
        const diversity = computeDiversity(islands);
        const allFit = islands.flatMap(isl => isl.population.map(ind => ind.fitness));
        rows.push({
          topology: topo, seed, generation: gen,
          bestFitness: Math.max(...allFit),
          meanFitness: allFit.reduce((a, b) => a + b, 0) / allFit.length,
          diversityTotal: diversity.total,
          diversityBetween: diversity.between,
          diversityWithin: diversity.within
        });

        // Evolve + migrate
        for (const isl of islands) isl.evolve();
        if (topo !== 'none' && gen > 0 && gen % CONFIG.migrationInterval === 0) {
          migrate(islands, topo);
        }
      }

      runIndex++;
      const elapsed = ((performance.now() - startTime) / 1000).toFixed(0);
      const best = Math.max(...islands.flatMap(isl => isl.population.map(ind => ind.fitness)));
      console.log(`[${runIndex}/${totalRuns}] ${topo} seed=${seed} ${elapsed}s best=${best.toFixed(3)}`);
    }
  }

  rng = Math.random;

  // Write CSV
  const header = 'topology,seed,generation,best_fitness,mean_fitness,diversity_total,diversity_between,diversity_within';
  const csv = header + '\n' + rows.map(r =>
    `${r.topology},${r.seed},${r.generation},${r.bestFitness.toFixed(6)},${r.meanFitness.toFixed(6)},${r.diversityTotal.toFixed(6)},${r.diversityBetween.toFixed(6)},${r.diversityWithin.toFixed(6)}`
  ).join('\n');
  writeFileSync(opts.out, csv);

  const totalTime = ((performance.now() - startTime) / 1000 / 60).toFixed(1);
  console.log(`\nDone. ${rows.length} rows → ${opts.out} (${totalTime} min)`);

  // Summary
  for (const topo of opts.topos) {
    const finalRows = rows.filter(r => r.topology === topo && r.generation === opts.gens - 1);
    const mean = (key) => (finalRows.reduce((a, r) => a + r[key], 0) / finalRows.length).toFixed(3);
    console.log(`  ${topo.padEnd(5)} diversity=${mean('diversityTotal')} best=${mean('bestFitness')}`);
  }
}

run();
