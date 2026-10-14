"""Post-hoc stratification of planarity_results.json (declared filters: the better hypothesis's relative RMS below
0.3% = clearly planar; span >= 30 degrees = wide). Writes stratified_summary.json."""
import json
import numpy as np
d = json.load(open("planarity_results.json"))
R = d["regions"]
out = {}
def stats(sel, key):
    r = np.array([x[key]["ratio_ray_over_z"] for x in sel])
    if not len(r):
        return {"n": 0}
    return {"n": int(len(r)), "favour_z": int((r > 1).sum()), "median_ratio_ray_over_z": float(np.median(r)),
            "geometric_mean": float(np.exp(np.log(r).mean())), "q10": float(np.quantile(r, .1)),
            "median_rel_rms_z": float(np.median([x[key]["rel_rms_z"] for x in sel])),
            "median_rel_rms_ray": float(np.median([x[key]["rel_rms_ray"] for x in sel]))}
for scale in ("0.8", "0.9", "1.0", "1.1", "1.2", "1.3"):
    key = f"f{scale}"
    planar = [x for x in R if min(x[key]["rel_rms_z"], x[key]["rel_rms_ray"]) < 0.003]
    out[scale] = {"all": stats(R, key), "clearly_planar": stats(planar, key),
                  "clearly_planar_wide": stats([x for x in planar if x["span_deg"] >= 30], key),
                  "clearly_planar_found_under_ray": stats([x for x in planar if x["found_under"] == "ray"], key),
                  "found_under_z": stats([x for x in R if x["found_under"] == "z"], key),
                  "found_under_ray": stats([x for x in R if x["found_under"] == "ray"], key),
                  "indoors": stats([x for x in R if x["domain"] == "indoors"], key),
                  "outdoor": stats([x for x in R if x["domain"] == "outdoor"], key)}
json.dump(out, open("stratified_summary.json", "w"), indent=1)
for s, v in out.items():
    print(s, {k: (v[k]["n"], v[k].get("favour_z"), round(v[k].get("median_ratio_ray_over_z", 0), 2)) for k in v})
