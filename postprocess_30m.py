import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.warp import transform as warp_transform

from lst_common import CRS, block_mean, grid_transform, psf_blur

matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="Consolidated metadata")

LS_CUBE = Path("data/landsat/krakow_landsat_30m.zarr")
RAW_DIR = Path("results/maps30/raw")
OUT_DIR = Path("results/maps30")
FIG_DIR = Path("figures")
PIXEL_M = 30.0
TIRS_SIGMA_M = 72.0
FACTOR = 3
CENTER_LONLAT = (19.9373, 50.0617)
CROP = 360


def rmse(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(((a[ok] - b[ok]) ** 2).mean()))


def save_geotiff(path, arr, transform):
    profile = {"driver": "GTiff", "height": arr.shape[0], "width": arr.shape[1], "count": 1,
               "dtype": "float32", "crs": CRS, "transform": transform, "nodata": np.nan,
               "compress": "deflate"}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype("float32"), 1)


def center_window(ls):
    xs, ys = warp_transform("EPSG:4326", CRS, [CENTER_LONLAT[0]], [CENTER_LONLAT[1]])
    row = int(np.argmin(np.abs(ls.y.values - ys[0])))
    col = int(np.argmin(np.abs(ls.x.values - xs[0])))
    return slice(row - CROP // 2, row + CROP // 2), slice(col - CROP // 2, col + CROP // 2)


def stretch(band):
    lo, hi = np.nanpercentile(band, [2, 98])
    return np.nan_to_num(np.clip((band - lo) / (hi - lo), 0, 1))


def plot_center(date, panels, rgb, win):
    vmin, vmax = np.nanpercentile(panels["Landsat 30 m (produkt)"][win], [2, 98])
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    axes[0].imshow(rgb[win])
    axes[0].set_title("RGB 30 m")
    for ax, (name, arr) in zip(axes[1:], panels.items()):
        im = ax.imshow(arr[win], cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title(name)
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im, ax=axes.tolist(), shrink=0.6, label="LST [K]")
    fig.suptitle(f"Centrum Krakowa, {date}")
    fig.savefig(FIG_DIR / f"maps30_center_{date}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(exist_ok=True)
    ls = xr.open_zarr(LS_CUBE)
    transform = grid_transform(ls)
    win = center_window(ls)
    sigma_px = TIRS_SIGMA_M / PIXEL_M
    rows = []

    for path in sorted(RAW_DIR.glob("*.npz")):
        date = path.stem
        with np.load(path) as npz:
            data = {k: npz[k] for k in npz.files}
        panels = {"Landsat 30 m (produkt)": data["product"]}
        for key in sorted(k for k in data if k.startswith("model_")):
            name = key.removeprefix("model_")
            lst30 = data[key]
            sensor = lst30 if name.startswith("A") else psf_blur(lst30, sigma_px)
            rows.append({"date": date, "model": name,
                         "spojnosc_90m": rmse(block_mean(sensor, FACTOR), data["lst_90"]),
                         "zgodnosc_z_produktem_30m": rmse(sensor, data["product"])})
            save_geotiff(OUT_DIR / f"{name}_{date}.tif", lst30, transform)
            panels[f"U-Net {name}"] = lst30
            if not name.startswith("A"):
                panels[f"U-Net {name} widziany przez sensor"] = sensor
        rgb = np.dstack([stretch(data[b]) for b in ("red", "green", "blue")])
        plot_center(date, dict(list(panels.items())[:5]), rgb, win)
        print(f"  {date}: mapy i rysunek zapisane")

    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()