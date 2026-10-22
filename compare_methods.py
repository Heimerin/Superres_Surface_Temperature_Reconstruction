from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

from lst_common import ring_index, spectrum, valid_tiles

matplotlib.use("Agg")

COARSE = "lst_270psf"
SUFFIX = "_psf" if COARSE == "lst_270psf" else ""
BASELINE_DIR = Path(f"results/baselines{SUFFIX}/preds")
UNET_DIR = Path("results/unet/preds")
PYRAMID = Path("data/landsat/krakow_wald_pyramid.zarr")
OUT_DIR = Path("figures")
BASELINES = ["brak_wyostrzania", "tsharp", "lgbm_global"]
UNET_RUNS = [f"A{SUFFIX}", f"B{SUFFIX}"]
TILE = 64
PIXEL_M = 90.0
COARSE_M = 270.0
FRC_THRESHOLD = 0.5
CENTER_LONLAT = (19.9373, 50.0617)
CROP = 120


def load_date(date):
    with np.load(BASELINE_DIR / f"{date}.npz") as npz:
        data = {"truth": npz["truth"], **{m: npz[m] for m in BASELINES}}
    for run in UNET_RUNS:
        path = UNET_DIR / run / f"{date}.npz"
        if not path.exists():
            return None
        with np.load(path) as npz:
            data[f"unet_{run}"] = npz[f"unet_{run}"]
            data[f"unet_{run}_sensor"] = npz[f"unet_{run}_sensor"]
    return data


def spectral_comparison(dates, methods):
    window = np.outer(np.hanning(TILE), np.hanning(TILE))
    rings = ring_index(TILE)
    n_rings = TILE // 2
    auto_truth = np.zeros(n_rings)
    auto = {m: np.zeros(n_rings) for m in methods}
    cross = {m: np.zeros(n_rings) for m in methods}
    n_tiles = 0
    for date in dates:
        data = load_date(date)
        stack = np.stack([data["truth"]] + [data[m] for m in methods])
        for y, x in valid_tiles(stack, TILE):
            f_truth = spectrum(data["truth"][y:y + TILE, x:x + TILE], window)
            auto_truth += np.bincount(rings, np.abs(f_truth) ** 2)[:n_rings]
            for m in methods:
                f_m = spectrum(data[m][y:y + TILE, x:x + TILE], window)
                auto[m] += np.bincount(rings, np.abs(f_m) ** 2)[:n_rings]
                cross[m] += np.bincount(rings, (f_m * np.conj(f_truth)).real)[:n_rings]
            n_tiles += 1

    freq = np.arange(n_rings) / (TILE * PIXEL_M) * 1000
    band = freq > 1000 / (2 * COARSE_M)
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for m in methods:
        frc = cross[m] / np.sqrt(auto[m] * auto_truth)
        below = np.flatnonzero((frc < FRC_THRESHOLD) & (freq > 0))
        f_cut = freq[below[0]] if len(below) else np.nan
        rows.append({"method": m, "eff_res_m": 1000 / (2 * f_cut) if np.isfinite(f_cut) else np.nan,
                     "power_ratio_detail": auto[m][band].sum() / auto_truth[band].sum(),
                     "frc_detail": cross[m][band].sum() / np.sqrt(auto[m][band].sum() * auto_truth[band].sum())})
        axes[0].plot(freq[1:], frc[1:], label=m)
        axes[1].plot(freq[1:], (auto[m] / auto_truth)[1:], label=m)
    for ax in axes:
        ax.axvline(1000 / (2 * COARSE_M), color="gray", ls="--", lw=0.8)
        ax.set_xlabel("częstość przestrzenna [cykle/km]")
        ax.legend(fontsize=8)
    axes[0].axhline(FRC_THRESHOLD, color="gray", lw=0.8)
    axes[0].set_ylabel("FRC z prawdą 90 m")
    axes[1].set_ylabel("moc metody / moc prawdy")
    axes[1].set_yscale("log")
    fig.savefig(OUT_DIR / f"compare_spectral{SUFFIX}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Kafle: {n_tiles}")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def center_index():
    ds = xr.open_zarr(PYRAMID)
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:32634", always_xy=True).transform(*CENTER_LONLAT)
    return int(np.argmin(np.abs(ds.y90.values - y))), int(np.argmin(np.abs(ds.x90.values - x)))


def visual_comparison(dates, methods):
    row, col = center_index()
    y0, x0 = row - CROP // 2, col - CROP // 2
    win = (slice(y0, y0 + CROP), slice(x0, x0 + CROP))
    best_date, best_frac = None, -1.0
    for date in dates:
        frac = np.isfinite(load_date(date)["truth"][win]).mean()
        if frac > best_frac:
            best_date, best_frac = date, frac
    data = load_date(best_date)
    panels = ["truth"] + methods
    vmin, vmax = np.nanpercentile(data["truth"][win], [2, 98])
    fig, axes = plt.subplots(2, (len(panels) + 1) // 2, figsize=(4 * ((len(panels) + 1) // 2), 8))
    for ax, name in zip(axes.ravel(), panels):
        im = ax.imshow(data[name][win], cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="LST [K]")
    fig.suptitle(f"Centrum Krakowa, {best_date} (pokrycie {best_frac:.0%})")
    fig.savefig(OUT_DIR / f"compare_visual{SUFFIX}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    dates = sorted(p.stem for p in BASELINE_DIR.glob("*.npz"))
    dates = [d for d in dates if load_date(d) is not None]
    methods = BASELINES + [m for run in UNET_RUNS for m in (f"unet_{run}", f"unet_{run}_sensor")]
    print(f"Sceny testowe: {len(dates)}")
    spectral_comparison(dates, methods)
    visual_comparison(dates, methods)


if __name__ == "__main__":
    main()