import warnings
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt
warnings.filterwarnings("ignore", message="Consolidated metadata")

CUBE_PATH = Path("data/ecostress/krakow_ecostress_70m.zarr")
MAX_STORED_DIST = 255
TRAIN_BUFFER = {"noc": 5, "dzień": 10}
MIN_VALID = 0.3

ds = xr.open_zarr(CUBE_PATH)
is_night = ds.is_night.values
n = ds.sizes["time"]

dist = np.empty((n, ds.sizes["y"], ds.sizes["x"]), dtype="uint8")
frac_train = np.empty(n)
for i in range(n):
    cloud = ds.cloud.isel(time=i).values == 1
    d = distance_transform_edt(~cloud) if cloud.any() else np.full(cloud.shape, MAX_STORED_DIST)
    dist[i] = np.minimum(np.floor(d), MAX_STORED_DIST).astype("uint8")
    lst = ds.LST.isel(time=i).values
    buffer = TRAIN_BUFFER["noc"] if is_night[i] else TRAIN_BUFFER["dzień"]
    frac_train[i] = np.mean(np.isfinite(lst) & (dist[i] >= buffer))

new = xr.Dataset(
    {"cloud_dist": (("time", "y", "x"), dist)},
    coords={"time": ds.time, "y": ds.y, "x": ds.x},
)
new["cloud_dist"].attrs = {
    "units": "px (70 m)",
    "description": "odległość do najbliższego piksela chmury, 0 = chmura, obcięta do 255",
}
new.attrs = dict(ds.attrs)
new.chunk({"time": 1}).to_zarr(CUBE_PATH, mode="a")

frac_clear = ds.frac_clear.values
for label, group in (("noc", is_night), ("dzień", ~is_night)):
    before = int((frac_clear[group] >= MIN_VALID).sum())
    after = int((frac_train[group] >= MIN_VALID).sum())
    print(f"{label}: przeloty z >= {MIN_VALID:.0%} ważnych pikseli: bez marginesu {before}, "
          f"z marginesem {TRAIN_BUFFER[label]} px: {after}")