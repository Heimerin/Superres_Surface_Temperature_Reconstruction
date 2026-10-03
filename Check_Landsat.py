from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from affine import Affine
from rasterio.warp import Resampling, reproject

LS_CUBE = Path("data/landsat/krakow_landsat_30m.zarr")
ECO_CUBE = Path("data/ecostress/krakow_ecostress_70m.zarr")
LS_BAD_BITS = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)
PAIR_MAX_MINUTES = 60
ECO_DAY_BUFFER = 25
MIN_COVERAGE = 0.99
MIN_COMMON = 2000
OUT_DIR = Path("figures")

CRS = "EPSG:32634"


def grid_transform(ds):
    x = ds.x.values
    y = ds.y.values
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    return Affine(dx, 0.0, x[0] - dx / 2, 0.0, dy, y[0] - dy / 2)

def decode(da):
    data = da.astype("float32")
    nodata = da.attrs.get("nodata")
    if nodata is not None:
        data = data.where(da != nodata)
    return data * da.attrs["scale"] + da.attrs["offset"]


def landsat_lst(frame):
    good = (frame.qa_pixel & LS_BAD_BITS) == 0
    return decode(frame.lwir11).where(good)


def stretch(band):
    lo, hi = np.nanpercentile(band, [2, 98])
    return np.clip((band - lo) / (hi - lo), 0, 1)


def plot_example(ls):
    months = pd.to_datetime(ls.time.values).month
    summer = np.flatnonzero((months >= 6) & (months <= 8))
    i = summer[np.argmax(ls.frac_clear.values[summer])]
    frame = ls.isel(time=i).load()
    lst = landsat_lst(frame).values
    rgb = np.dstack([stretch(decode(frame[b]).values) for b in ("red", "green", "blue")])
    print(f"Przykład: {str(frame.time.values)[:16]}, LST {np.nanmin(lst):.1f}-{np.nanmax(lst):.1f} K, "
          f"średnia {np.nanmean(lst):.1f} K")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im = axes[0].imshow(lst, cmap="inferno")
    fig.colorbar(im, ax=axes[0], label="LST [K]")
    axes[1].imshow(np.nan_to_num(rgb))
    for ax, title in zip(axes, ("Landsat LST", "RGB")):
        ax.set_title(f"{title} {str(frame.time.values)[:10]}")
        ax.axis("off")
    fig.savefig(OUT_DIR / "landsat_example.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def to_eco_grid(arr, ls, eco):
    kwargs = {
        "src_transform": grid_transform(ls),
        "src_crs": CRS,
        "dst_transform": grid_transform(eco),
        "dst_crs": CRS,
        "resampling": Resampling.average,
    }
    shape = (eco.sizes["y"], eco.sizes["x"])
    mean = np.full(shape, np.nan, dtype="float32")
    reproject(arr.astype("float32"), mean, src_nodata=np.nan, dst_nodata=np.nan, **kwargs)
    coverage = np.zeros(shape, dtype="float32")
    reproject(np.isfinite(arr).astype("float32"), coverage, **kwargs)
    return np.where(coverage >= MIN_COVERAGE, mean, np.nan)


def compare_pairs(ls, eco):
    ls_times = pd.to_datetime(ls.time.values)
    eco_times = pd.to_datetime(eco.time.values)
    rows, samples = [], []
    for i, t in enumerate(ls_times):
        offsets = (eco_times - t).total_seconds()
        j = int(np.argmin(np.abs(offsets)))
        dt = offsets[j] / 60
        if abs(dt) > PAIR_MAX_MINUTES:
            continue
        ls_on_eco = to_eco_grid(landsat_lst(ls.isel(time=i)).values, ls, eco)
        frame = eco.isel(time=j)
        eco_lst = frame.LST.values
        ok = np.isfinite(ls_on_eco) & np.isfinite(eco_lst) & (frame.cloud_dist.values >= ECO_DAY_BUFFER)
        n = int(ok.sum())
        if n < MIN_COMMON:
            print(f"  {t:%Y-%m-%d}: tylko {n} wspólnych pikseli, pomijam")
            continue
        a, b = ls_on_eco[ok], eco_lst[ok]
        diff = b - a
        anom = (b - b.mean()) - (a - a.mean())
        rows.append({"landsat": f"{t:%Y-%m-%d %H:%M}", "dt_min": round(dt, 1), "n": n,
                     "bias_eco_minus_ls": diff.mean(), "rmse": np.sqrt((diff ** 2).mean()),
                     "anom_rmse": np.sqrt((anom ** 2).mean()), "r": np.corrcoef(a, b)[0, 1]})
        samples.append((f"{t:%Y-%m-%d}", a, b))

    if not rows:
        print("Brak par do porównania")
        return
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 6))
    for label, a, b in samples:
        ax.scatter(a, b, s=2, alpha=0.3, label=label)
    lims = [min(s[1].min() for s in samples), max(s[1].max() for s in samples)]
    ax.plot(lims, lims, color="gray", lw=1)
    ax.set_xlabel("Landsat LST uśrednione do 70 m [K]")
    ax.set_ylabel("ECOSTRESS LST [K]")
    ax.legend(markerscale=5)
    fig.savefig(OUT_DIR / "landsat_ecostress_pairs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    ls = xr.open_zarr(LS_CUBE)
    eco = xr.open_zarr(ECO_CUBE)
    plot_example(ls)
    compare_pairs(ls, eco)


if __name__ == "__main__":
    main()