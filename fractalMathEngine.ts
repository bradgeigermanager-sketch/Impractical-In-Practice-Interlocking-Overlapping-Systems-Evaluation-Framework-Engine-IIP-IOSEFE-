/**
 * fractalMathEngine.ts
 *
 * Fractal Impracticality Analysis Math Core for FRA‑01
 * ----------------------------------------------------
 * Implements:
 *  - Fractal Dimension (box-counting)
 *  - Self-Similarity Index (SSI)
 *  - Boundary Roughness
 *  - Fractal Instability Index (FII)
 */

export interface ScalePattern {
  scale: number;              // epsilon or resolution
  features: number[];         // feature vector at this scale
}

export interface FractalMetrics {
  fractalDimension: number | null;
  selfSimilarityIndex: number | null;
  boundaryRoughness: number | null;
  fractalInstabilityIndex: number | null;
}

// ------------------------------------------------------------
// Utility
// ------------------------------------------------------------

function safeLog(x: number): number {
  if (x <= 0) throw new Error(`Log undefined for non-positive value: ${x}`);
  return Math.log(x);
}

export function euclideanDistance(a: number[], b: number[]): number {
  if (a.length !== b.length) throw new Error("Vectors must have same length");
  return Math.sqrt(a.reduce((sum, x, i) => sum + (x - b[i]) ** 2, 0));
}

export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) throw new Error("Vectors must have same length");

  const dot = a.reduce((sum, x, i) => sum + x * b[i], 0);
  const na = Math.sqrt(a.reduce((s, x) => s + x * x, 0));
  const nb = Math.sqrt(b.reduce((s, x) => s + x * x, 0));

  if (na === 0 || nb === 0) return 0;
  return dot / (na * nb);
}

// ------------------------------------------------------------
// 1. Fractal Dimension (Box-Counting)
// ------------------------------------------------------------

export function fractalDimensionBoxCount(
  scales: number[],
  counts: number[]
): number | null {
  if (scales.length !== counts.length) {
    throw new Error("scales and counts must have same length");
  }
  if (scales.length < 2) return null;

  const xs: number[] = [];
  const ys: number[] = [];

  for (let i = 0; i < scales.length; i++) {
    const s = scales[i];
    const c = counts[i];
    if (s > 0 && c > 0) {
      xs.push(safeLog(1 / s));
      ys.push(safeLog(c));
    }
  }

  if (xs.length < 2) return null;

  const n = xs.length;
  const meanX = xs.reduce((a, b) => a + b, 0) / n;
  const meanY = ys.reduce((a, b) => a + b, 0) / n;

  let num = 0;
  let den = 0;

  for (let i = 0; i < n; i++) {
    num += (xs[i] - meanX) * (ys[i] - meanY);
    den += (xs[i] - meanX) ** 2;
  }

  if (den === 0) return null;
  return num / den;
}

// ------------------------------------------------------------
// 2. Self-Similarity Index (SSI)
// ------------------------------------------------------------

export function selfSimilarityIndex(
  patterns: ScalePattern[],
  similarityFn: (a: number[], b: number[]) => number = cosineSimilarity
): number | null {
  if (patterns.length < 2) return null;

  const sims: number[] = [];

  for (let i = 0; i < patterns.length - 1; i++) {
    const p1 = patterns[i];
    const p2 = patterns[i + 1];
    sims.push(similarityFn(p1.features, p2.features));
  }

  if (sims.length === 0) return null;
  return sims.reduce((a, b) => a + b, 0) / sims.length;
}

// ------------------------------------------------------------
// 3. Boundary Roughness
// ------------------------------------------------------------

export function boundaryRoughness(
  boundaryLengths: number[],
  referenceLength?: number
): number | null {
  if (boundaryLengths.length < 2) return null;

  const ref =
    referenceLength ??
    boundaryLengths.reduce((a, b) => a + b, 0) / boundaryLengths.length;

  if (ref <= 0) return null;

  const normalized = boundaryLengths
    .filter((b) => b > 0)
    .map((b) => b / ref);

  if (normalized.length < 2) return null;

  const mean =
    normalized.reduce((a, b) => a + b, 0) / normalized.length;

  const variance =
    normalized.reduce((sum, x) => sum + (x - mean) ** 2, 0) /
    (normalized.length - 1);

  const std = Math.sqrt(variance);
  if (mean === 0) return null;

  return std / mean; // coefficient of variation
}

// ------------------------------------------------------------
// 4. Fractal Instability Index (FII)
// ------------------------------------------------------------

export function fractalInstabilityIndex(
  dF: number | null,
  ssi: number | null,
  bR: number | null,
  alpha = 1.0,
  beta = 1.0,
  gamma = 1.0
): number | null {
  if (dF === null && ssi === null && bR === null) return null;

  const d = dF ?? 0;
  const s = ssi ?? 0;
  const b = bR ?? 0;

  return alpha * d + beta * s + gamma * b;
}

// ------------------------------------------------------------
// 5. High-level aggregator
// ------------------------------------------------------------

export function computeFractalMetrics(
  scales: number[],
  counts: number[],
  patterns: ScalePattern[],
  boundaryLengths: number[],
  alpha = 1.0,
  beta = 1.0,
  gamma = 1.0
): FractalMetrics {
  const fractalDimension = fractalDimensionBoxCount(scales, counts);
  const selfSimilarity = selfSimilarityIndex(patterns);
  const roughness = boundaryRoughness(boundaryLengths);
  const instability = fractalInstabilityIndex(
    fractalDimension,
    selfSimilarity,
    roughness,
    alpha,
    beta,
    gamma
  );

  return {
    fractalDimension,
    selfSimilarityIndex: selfSimilarity,
    boundaryRoughness: roughness,
    fractalInstabilityIndex: instability
  };
}
