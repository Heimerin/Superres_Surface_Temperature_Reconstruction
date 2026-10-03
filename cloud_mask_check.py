from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import distance_transform_edt

CUBE_PATH = Path("data/ecostress/krakow_ecostress_70m.zarr")
MAX_DIST = 40
REF_DIST = 30
FRAC_RANGE = (0.3, 0.9)
MIN_REF_PIXELS = 5000

ds = xr.open_zarr(CUBE_PATH)
partial = ((ds.frac_clear >= FRAC_RANGE[0]) & (ds.frac_clear <= FRAC_RANGE[1])).values
subset = ds[["LST", "cloud"]].isel(time=np.flatnonzero(partial))
is_night = subset.is_night.values

CONTROL = False
rng = np.random.default_rng(0)
perm = np.arange(subset.sizes["time"])
for group in (is_night, ~is_night):
    idx = np.flatnonzero(group)
    perm[idx] = rng.permutation(idx)

rows = []
for i in range(subset.sizes["time"]):
    lst = subset.LST.isel(time=i).values
    cloud = subset.cloud.isel(time=perm[i] if CONTROL else i).values == 1
    if not cloud.any():
        continue
    dist = distance_transform_edt(~cloud)
    valid = np.isfinite(lst)
    ref = valid & (dist > REF_DIST)
    if ref.sum() < MIN_REF_PIXELS:
        continue
    anomaly = lst - np.median(lst[ref])
    for d in range(1, MAX_DIST + 1):
        ring = valid & (dist >= d) & (dist < d + 1)
        if ring.any():
            rows.append({"scene": i, "night": bool(is_night[i]), "dist_px": d,
                         "anomaly": float(np.median(anomaly[ring]))})

df = pd.DataFrame(rows)
summary = df.groupby(["dist_px", "night"])["anomaly"].median().unstack("night")
summary = summary.rename(columns={False: "dzień", True: "noc"})
print(f"Scen użytych: dzień {df[~df.night].scene.nunique()}, noc {df[df.night].scene.nunique()}")
print(summary.round(2).to_string())

ax = summary.plot(marker="o")
ax.axhline(0, color="gray", lw=0.8)
ax.set_xlabel("odległość od najbliższego piksela chmury [px po 70 m]")
ax.set_ylabel("mediana anomalii LST [K]")
plt.savefig(f"cloud_edge_profile{'_control' if CONTROL else ''}.png", dpi=150, bbox_inches="tight")
summary.to_csv("inventory/cloud_edge_bias.csv")