"""DEP2-CX1 empirical test: is a DIODE depth value z-depth (along the optical axis) or ray distance?

Data: the 40 pinned DIODE views of the workshop, read and SHA-256-verified with the carried samples.py
(fetch_corpus on the cached ETH byte ranges). Intrinsics: DIODE devkit intrinsics.txt (diode-dataset/diode-devkit
@8b1765b7d801a5f5e2877c434ffe164e62ce8c90): [fx, fy, cx, cy] = [886.81, 927.06, 512, 384].

Hypotheses for a valid pixel (u, v) with value D and normalised ray q = ((u-cx)/fx, (v-cy)/fy, 1):
  (a) z-depth:       P = D * q
  (b) ray distance:  P = D * q / |q|
A real plane is exactly planar under the true hypothesis; under the wrong one it bends by the factor |q| (up to
~1.19 at this camera's corners), most visibly in regions that span a wide range of off-axis angles.

Declared, hypothesis-symmetric region rule (no hand picking):
  1. Pixels: mask & finite & 0.6 <= D <= 350 m (the workshop's scoring support), sampled on a fixed stride-4 grid.
  2. Under EACH hypothesis separately, sequential RANSAC (seed 0, 1,500 iterations) extracts up to 4 planes: a point
     is an inlier when its point-to-plane distance is below 1% of its range |P|; a plane is kept when it has at
     least 3% of the sampled pixels and its inliers span at least 15 degrees of viewing angle; the inliers are then
     removed and the next plane is searched.
  3. Every region found under either hypothesis is evaluated under BOTH: a least-squares (SVD) plane is fitted to
     the region's 3D points, and the residual is reported as RMS(point-to-plane distance) / mean range.
  4. Per region: ratio = residual(ray) / residual(z). Ratio > 1 favours z-depth. Regions whose span is small carry
     little information (both hypotheses nearly coincide near a single viewing direction); the span is reported.
Sensitivity: devkit issue #20 disputes the published field of view, so the whole evaluation (step 3, on the
regions from steps 1-2 at the published intrinsics) is repeated with the focal lengths scaled by 0.8-1.3.

Usage: python -I planarity_test.py <carried dir> <corpus cache> <out dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

FX, FY, CX, CY = 886.81, 927.06, 512.0, 384.0
STRIDE, ITERS, MAX_PLANES, MIN_SHARE, MIN_SPAN_DEG, REL_TOL = 4, 1500, 4, 0.03, 15.0, 0.01
FOCAL_SCALES = (0.8, 0.9, 1.0, 1.1, 1.2, 1.3)


def rays(shape, scale=1.0):
    v, u = np.mgrid[0 : shape[0] : STRIDE, 0 : shape[1] : STRIDE]
    q = np.stack([(u - CX) / (FX * scale), (v - CY) / (FY * scale), np.ones_like(u, dtype=float)], axis=-1)
    return u, v, q


def points(depth, q, hypothesis):
    p = depth[..., None] * q
    return p / np.linalg.norm(q, axis=-1, keepdims=True) if hypothesis == "ray" else p


def plane_lsq(p):
    centre = p.mean(axis=0)
    _, _, vt = np.linalg.svd(p - centre, full_matrices=False)
    normal = vt[-1]
    distance = (p - centre) @ normal
    return float(np.sqrt(np.mean(distance**2)) / np.mean(np.linalg.norm(p, axis=1)))


def ransac(p, rng):
    n = len(p)
    best, best_count = None, 0
    rng_norm = np.linalg.norm(p, axis=1)
    for _ in range(ITERS):
        a, b, c = p[rng.choice(n, 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-12:
            continue
        normal /= length
        inliers = np.abs((p - a) @ normal) < REL_TOL * rng_norm
        count = int(inliers.sum())
        if count > best_count:
            best, best_count = inliers, count
    return best


def span_deg(q_region):
    unit = q_region / np.linalg.norm(q_region, axis=1, keepdims=True)
    centre = unit.mean(axis=0)
    centre /= np.linalg.norm(centre)
    angles = np.degrees(np.arccos(np.clip(unit @ centre, -1, 1)))
    return float(2 * np.percentile(angles, 95))


def main() -> int:
    carried, cache, out = (Path(a).resolve() for a in sys.argv[1:4])
    sys.path.insert(0, str(carried))
    from depth_reference import samples

    records = samples.read_corpus(samples.fetch_corpus(cache_dir=cache))  # SHA-256 verified
    out.mkdir(parents=True, exist_ok=True)
    per_image, regions_all = [], []
    for record in records:
        depth = np.asarray(record["depth"], dtype=float)
        mask = np.asarray(record["mask"])
        u, v, q = rays(depth.shape)
        d = depth[::STRIDE, ::STRIDE]
        m = mask[::STRIDE, ::STRIDE] & np.isfinite(d) & (d >= 0.6) & (d <= 350)
        dq, qq = d[m], q[m]
        regions = []
        for hypothesis in ("z", "ray"):
            rng = np.random.default_rng(0)
            remaining = np.ones(len(dq), dtype=bool)
            for _ in range(MAX_PLANES):
                index = np.flatnonzero(remaining)
                if len(index) < 3:
                    break
                p = points(dq[index], qq[index], hypothesis)
                inliers = ransac(p, rng)
                if inliers is None or inliers.sum() < MIN_SHARE * len(dq):
                    break
                chosen = index[inliers]
                remaining[chosen] = False
                span = span_deg(qq[chosen])
                if span >= MIN_SPAN_DEG:
                    regions.append({"found_under": hypothesis, "pixels": chosen})
        rows = []
        for k, region in enumerate(regions):
            sel = region["pixels"]
            row = {"image": record["id"], "domain": record.get("domain"), "region": k,
                   "found_under": region["found_under"], "n_points": int(len(sel)),
                   "span_deg": round(span_deg(qq[sel]), 2)}
            for scale in FOCAL_SCALES:
                _, _, qs = rays(depth.shape, scale)
                qs = qs[m][sel]
                rz = plane_lsq(points(dq[sel], qs, "z"))
                rr = plane_lsq(points(dq[sel], qs, "ray"))
                row[f"f{scale}"] = {"rel_rms_z": rz, "rel_rms_ray": rr, "ratio_ray_over_z": rr / rz}
            rows.append(row)
        regions_all += rows
        ratios = [r["f1.0"]["ratio_ray_over_z"] for r in rows]
        per_image.append({
            "image": record["id"], "domain": record.get("domain"), "regions": len(rows),
            "median_rel_rms_z": float(np.median([r["f1.0"]["rel_rms_z"] for r in rows])) if rows else None,
            "median_rel_rms_ray": float(np.median([r["f1.0"]["rel_rms_ray"] for r in rows])) if rows else None,
            "median_ratio_ray_over_z": float(np.median(ratios)) if rows else None,
            "regions_favouring_z": int(sum(x > 1 for x in ratios)),
        })
        print(json.dumps(per_image[-1]), flush=True)
    summary = {}
    for scale in FOCAL_SCALES:
        ratios = np.array([r[f"f{scale}"]["ratio_ray_over_z"] for r in regions_all])
        wide = np.array([r[f"f{scale}"]["ratio_ray_over_z"] for r in regions_all if r["span_deg"] >= 30])
        summary[str(scale)] = {
            "regions": int(len(ratios)), "favour_z": int((ratios > 1).sum()),
            "median_ratio_ray_over_z": float(np.median(ratios)),
            "geometric_mean_ratio": float(np.exp(np.mean(np.log(ratios)))),
            "wide_regions_span_ge_30deg": int(len(wide)), "wide_favour_z": int((wide > 1).sum()),
            "wide_median_ratio": float(np.median(wide)) if len(wide) else None,
            "median_rel_rms_z": float(np.median([r[f"f{scale}"]["rel_rms_z"] for r in regions_all])),
            "median_rel_rms_ray": float(np.median([r[f"f{scale}"]["rel_rms_ray"] for r in regions_all])),
        }
    result = {"intrinsics": {"fx": FX, "fy": FY, "cx": CX, "cy": CY,
                             "source": "diode-dataset/diode-devkit@8b1765b7d801a5f5e2877c434ffe164e62ce8c90 intrinsics.txt"},
              "rule": {"stride": STRIDE, "ransac_iterations": ITERS, "max_planes_per_hypothesis": MAX_PLANES,
                       "min_share": MIN_SHARE, "min_span_deg": MIN_SPAN_DEG, "inlier_tolerance": "1% of range",
                       "seed": 0},
              "summary_by_focal_scale": summary, "per_image": per_image, "regions": regions_all}
    (out / "planarity_results.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
