from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from lst_common import ECO_EVAL_BUFFER, find_pairs, grid_transform, landsat_lst, to_eco_grid

LS_CUBE = Path("data/landsat/krakow_landsat_30m.zarr")
ECO_CUBE = Path("data/ecostress/krakow_ecostress_70m.zarr")
OUT_CSV = Path("inventory/krakow_coregistration.csv")
PAIR_MAX_MINUTES = 60
MAX_SHIFT = 4
MIN_COMMON = 2000


def shifted_pair(a, b, dy, dx):
    h, w = a.shape
    a_part = a[max(0, -dy):min(h, h - dy), max(0, -dx):min(w, w - dx)]
    b_part = b[max(0, dy):min(h, h + dy), max(0, dx):min(w, w + dx)]
    return a_part, b_part


def match_stats(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < MIN_COMMON:
        return np.nan, np.nan
    x, y = a[ok], b[ok]
    anom = (y - y.mean()) - (x - x.mean())
    return np.corrcoef(x, y)[0, 1], np.sqrt((anom ** 2).mean())


def refine(values, k):
    if k == 0 or k == len(values) - 1:
        return 0.0
    left, center, right = values[k - 1], values[k], values[k + 1]
    denom = left - 2 * center + right
    if not np.isfinite(denom) or denom == 0:
        return 0.0
    return 0.5 * (left - right) / denom


def estimate_shift(ls_img, eco_img):
    shifts = range(-MAX_SHIFT, MAX_SHIFT + 1)
    r = np.full((len(shifts), len(shifts)), np.nan)
    for iy, dy in enumerate(shifts):
        for ix, dx in enumerate(shifts):
            r[iy, ix] = match_stats(*shifted_pair(ls_img, eco_img, dy, dx))[0]
    iy, ix = np.unravel_index(np.nanargmax(r), r.shape)
    at_edge = iy in (0, len(shifts) - 1) or ix in (0, len(shifts) - 1)
    dy = shifts[iy] + refine(r[:, ix], iy)
    dx = shifts[ix] + refine(r[iy, :], ix)
    return dy, dx, shifts[iy], shifts[ix], at_edge


def main():
    ls = xr.open_zarr(LS_CUBE)
    eco = xr.open_zarr(ECO_CUBE)
    pixel = grid_transform(eco).a
    rows = []
    for i, j, dt in find_pairs(ls, eco, PAIR_MAX_MINUTES):
        ls_img = to_eco_grid(landsat_lst(ls.isel(time=i)).values, ls, eco)
        frame = eco.isel(time=j)
        keep = frame.cloud_dist.values >= ECO_EVAL_BUFFER["dzień"]
        eco_img = np.where(keep, frame.LST.values, np.nan)
        r0, rmse0 = match_stats(ls_img, eco_img)
        if np.isnan(r0):
            continue
        dy, dx, int_dy, int_dx, at_edge = estimate_shift(ls_img, eco_img)
        r_best, rmse_best = match_stats(*shifted_pair(ls_img, eco_img, int_dy, int_dx))
        rows.append({
            "landsat": str(ls.time.values[i])[:16],
            "dt_min": round(dt, 1),
            "shift_east_m": dx * pixel,
            "shift_north_m": -dy * pixel,
            "r0": r0,
            "r_best": r_best,
            "anom_rmse0": rmse0,
            "anom_rmse_best": rmse_best,
            "view_zenith": float(np.nanmedian(frame.view_zenith.values)),
            "at_edge": at_edge,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()