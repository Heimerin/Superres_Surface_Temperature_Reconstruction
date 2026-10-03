import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from lst_common import (LS_BAD_BITS, aggregate_lst, block_mean, decode, grid_transform,
                        spectral_indices)

warnings.filterwarnings("ignore", message="Consolidated metadata")

LS_CUBE = Path("data/landsat/krakow_landsat_30m.zarr")
OUT_CUBE = Path("data/landsat/krakow_wald_pyramid.zarr")
FACTOR = 3
RES = 30.0
TEST_FROM_YEAR = 2025
BANDS = ["blue", "green", "red", "nir08", "swir16", "swir22"]
CUBE_BATCH = 20


def process_scene(frame, platform):
    h = frame.sizes["y"] // FACTOR ** 2 * FACTOR ** 2
    w = frame.sizes["x"] // FACTOR ** 2 * FACTOR ** 2
    frame = frame.isel(y=slice(0, h), x=slice(0, w))

    raw = {"lst": decode(frame.lwir11).values, "emis": decode(frame.emis).values,
           "drad": decode(frame.drad).values}
    raw.update({b: decode(frame[b]).values for b in BANDS})
    clear = (frame.qa_pixel.values & LS_BAD_BITS) == 0
    for arr in raw.values():
        clear &= np.isfinite(arr)
    raw = {k: np.where(clear, v, np.nan) for k, v in raw.items()}

    lst90, emis90, drad90 = aggregate_lst(raw["lst"], raw["emis"], raw["drad"], platform, FACTOR)
    lst270, emis270, _ = aggregate_lst(lst90, emis90, drad90, platform, FACTOR)
    naive90 = block_mean(raw["lst"], FACTOR)
    naive270 = block_mean(naive90, FACTOR)
    refl90 = {b: block_mean(raw[b], FACTOR) for b in BANDS}
    refl270 = {b: block_mean(refl90[b], FACTOR) for b in BANDS}

    level90 = {"lst": lst90, "lst_naive": naive90, "emis": emis90, **refl90, **spectral_indices(refl90)}
    level270 = {"lst": lst270, "lst_naive": naive270, "emis": emis270, **refl270,
                **spectral_indices(refl270)}
    return level90, level270


def level_coords(tf, n_y, n_x, factor):
    step = RES * factor
    x = tf.c + step * (np.arange(n_x) + 0.5)
    y = tf.f - step * (np.arange(n_y) + 0.5)
    return y, x


def main():
    ls = xr.open_zarr(LS_CUBE)
    tf = grid_transform(ls)
    needed = ["lwir11", "emis", "drad", "qa_pixel", *BANDS]
    n = ls.sizes["time"]
    rad_vs_naive = []

    for start in range(0, n, CUBE_BATCH):
        idx = range(start, min(start + CUBE_BATCH, n))
        stacks90, stacks270 = {}, {}
        for i in idx:
            frame = ls[needed].isel(time=i).load()
            level90, level270 = process_scene(frame, int(ls.platform.values[i]))
            for k, v in level90.items():
                stacks90.setdefault(k, []).append(v.astype("float32"))
            for k, v in level270.items():
                stacks270.setdefault(k, []).append(v.astype("float32"))
            diff = level270["lst"] - level270["lst_naive"]
            if np.isfinite(diff).any():
                rad_vs_naive.append((np.nanmean(np.abs(diff)), np.nanpercentile(diff, 99)))

        n90 = stacks90["lst"][0].shape
        n270 = stacks270["lst"][0].shape
        y90, x90 = level_coords(tf, *n90, FACTOR)
        y270, x270 = level_coords(tf, *n270, FACTOR ** 2)
        times = ls.time.values[list(idx)]
        lst270 = np.stack(stacks270["lst"])

        ds = xr.Dataset(
            {**{f"{k}_90": (("time", "y90", "x90"), np.stack(v)) for k, v in stacks90.items()},
             **{f"{k}_270": (("time", "y270", "x270"), np.stack(v)) for k, v in stacks270.items()}},
            coords={
                "time": times, "y90": y90, "x90": x90, "y270": y270, "x270": x270,
                "platform": ("time", ls.platform.values[list(idx)]),
                "is_test": ("time", pd.DatetimeIndex(times).year.to_numpy() >= TEST_FROM_YEAR),
                "frac_valid_270": ("time", np.isfinite(lst270).mean(axis=(1, 2))),
            },
        )
        ds.attrs.update({
            "crs": "EPSG:32634",
            "levels_m": [RES * FACTOR, RES * FACTOR ** 2],
            "aggregation": "blokowa srednia radiancji powierzchni (emis*B(T) + (1-emis)*drad), stale K1/K2 TIRS B10",
            "source": str(LS_CUBE),
        })
        ds = ds.chunk({"time": 1})
        if start == 0:
            ds.to_zarr(OUT_CUBE, mode="w")
        else:
            ds.to_zarr(OUT_CUBE, append_dim="time")
        print(f"  piramida: {idx.stop}/{n}")

    stats = np.array(rad_vs_naive)
    print(f"Radiancja vs średnia temperatur (270 m): mediana po scenach średniej |różnicy| "
          f"{np.median(stats[:, 0]):.3f} K, mediana 99. percentyla {np.median(stats[:, 1]):.3f} K")


if __name__ == "__main__":
    main()