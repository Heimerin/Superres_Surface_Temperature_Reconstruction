import warnings
from pathlib import Path

import numpy as np 
import pandas as pd 
import xarray as xr
from scipy.ndimage import shift as nd_shift

from diurnal_harmonic import ECO_CUBE, PIXEL_M, design, load_geoloc, shift_masked, time_keys

warnings.filterwarnings("ignore", message="Consolidated metadata")
 
MODEL_NPZ = Path("results/diurnal/diurnal_model.npz")
RESID_CSV = Path("results/diurnal/scene_residuals.csv")
N_SCENES = 10
STRIPE_K = 3.0
NORMAL_K = 1.0
Z_MAX = 4.0
 
 
def outlier_scenes(resid):
    resid["bin"] = (resid["hour"] // 4).clip(0, 5).astype(int)
    grouped = resid.groupby("bin")["rms"]
    median = grouped.transform("median")
    mad = grouped.transform(lambda s: 1.4826 * (s - s.median()).abs().median())
    resid["z"] = (resid["rms"] - median) / mad
    out = resid[resid["z"] > Z_MAX].sort_values("time")
    print(f"Sceny z nietypowo dużą resztą (z > {Z_MAX}): {len(out)} z {len(resid)}")
    print(out[["time", "hour", "night", "rms", "z"]].round(2).to_string(index=False))
 
 
def main():
    eco = xr.open_zarr(ECO_CUBE)
    model = np.load(MODEL_NPZ)
    name = str(model["model"])
    coef, b, good = model["coef"], model["b"], model["good"]
    hours, season = model["hours"], model["season"]
    k_of = {t: k for k, t in enumerate(pd.DatetimeIndex(model["times"]).floor("s"))}
    j_of = {t: j for j, t in enumerate(time_keys(eco.time.values))}
    geoloc = load_geoloc()
 
    resid = pd.read_csv(RESID_CSV)
    resid["key"] = time_keys(resid["time"])
    outlier_scenes(resid)
 
    counts = {"pas": np.zeros((8, 4)), "reszta": np.zeros((8, 4))}
    lst_vals = {"pas": [], "reszta": []}
    for key in resid.sort_values("rms", ascending=False)["key"].head(N_SCENES):
        j, k = j_of[key], k_of[key]
        frame = eco[["LST", "QC"]].isel(time=j).load()
        east, north = geoloc.get(key, (0.0, 0.0))
        lst = shift_masked(frame.LST.values, east, north).ravel()
        qc = nd_shift(frame.QC.values, (north / PIXEL_M, -east / PIXEL_M), order=0,
                      mode="constant", cval=0).ravel().astype(int)
        a = (design([hours[k]], [season[k]], name) @ coef.T)[0]
        res = lst - b[k] - a
        ok = good & np.isfinite(res)
        res = res - np.median(res[ok])
        groups = {"pas": ok & (res > STRIPE_K), "reszta": ok & (np.abs(res) < NORMAL_K)}
        for g, m in groups.items():
            for f in range(8):
                counts[g][f] += np.bincount((qc[m] >> (2 * f)) & 3, minlength=4)
            lst_vals[g].append(lst[m])
 
    rows = []
    for f in range(8):
        for v in range(4):
            rows.append({"bity": f"{2 * f}-{2 * f + 1}", "wartość": v,
                         "pas": counts["pas"][f, v] / counts["pas"][f].sum(),
                         "reszta": counts["reszta"][f, v] / counts["reszta"][f].sum()})
    table = pd.DataFrame(rows)
    table = table[(table["pas"] > 0.001) | (table["reszta"] > 0.001)]
    print(f"\nRozkład pól QC w {N_SCENES} najgorszych scenach: piksele w pasach (reszta > {STRIPE_K} K) "
          f"i piksele typowe (|reszta| < {NORMAL_K} K)")
    print(table.round(3).to_string(index=False))
 
    for g, vals in lst_vals.items():
        v = np.concatenate(vals)
        print(f"LST [{g}]: {len(v)} pikseli, percentyle 1/50/99: {np.percentile(v, [1, 50, 99]).round(1)} K")
 
 
if __name__ == "__main__":
    main()
