import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from rasterio.warp import transform as warp_transform

from diurnal_harmonic import (ECO_CUBE, FIG_DIR, anomaly_map, design, fit, load_bias_table,
                              load_ecostress, load_geoloc)
from lst_common import CRS

matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="Consolidated metadata")

MODELS = ["K2", "K2+S", "K2xS"]
POINTS = {
    "Rynek (centrum)": (19.9373, 50.0617),
    "Las Wolski": (19.855, 50.060),
    "Bagry (woda)": (19.999, 50.0335),
    "pola na płn.-wsch.": (20.20, 50.15),
    "huta (przemysł)": (20.11, 50.08),
}
SEASONS = {"1 V": -1.0, "16 VII": 0.0, "30 IX": 1.0}
COLORS = {"1 V": "tab:green", "16 VII": "tab:orange", "30 IX": "tab:brown"}
STEP = 0.25
HOURS = np.arange(0, 24 + STEP / 2, STEP)
WIN = 1


def point_index(eco, lon, lat):
    xs, ys = warp_transform("EPSG:4326", CRS, [lon], [lat])
    row = int(np.argmin(np.abs(eco.y.values - ys[0])))
    col = int(np.argmin(np.abs(eco.x.values - xs[0])))
    return row, col


def curve(coef, good, model, shape, row, col, season):
    X = design(HOURS, np.full(HOURS.size, season), model)
    idx = [r * shape[1] + c for r in range(row - WIN, row + WIN + 1)
           for c in range(col - WIN, col + WIN + 1)]
    idx = [i for i in idx if good[i]]
    return (X @ coef[idx].T).mean(axis=1)


def plot_points(fits, points, shape):
    fig, axes = plt.subplots(2, len(points), figsize=(4 * len(points), 8), sharex=True,
                             sharey="col", layout="constrained")
    rows = []
    for r, model in enumerate(["K2+S", "K2xS"]):
        for c, (name, (row, col)) in enumerate(points.items()):
            ax = axes[r, c]
            ax.plot(HOURS, curve(*fits["K2"], "K2", shape, row, col, 0.0), color="gray", lw=1,
                    label="K2 (bez sezonu)")
            for label, s in SEASONS.items():
                cv = curve(*fits[model], model, shape, row, col, s)
                ax.plot(HOURS, cv, color=COLORS[label], label=label)
                rows.append({"punkt": name, "model": model, "dzień": label,
                             "02:00": cv[int(2 / STEP)], "14:00": cv[int(14 / STEP)]})
            ax.axhline(0, color="black", lw=0.5)
            ax.set_title(f"{name}\n{model}", fontsize=10)
            ax.set_xticks([0, 6, 12, 18, 24])
    axes[0, 0].legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("czas słoneczny [h]")
    for ax in axes[:, 0]:
        ax.set_ylabel("anomalia LST [K]")
    fig.savefig(FIG_DIR / "seasonal_models_points.png", dpi=130)
    plt.close(fig)
    table = pd.DataFrame(rows).pivot_table(index=["punkt", "model"], columns="dzień",
                                           values=["02:00", "14:00"])
    print(table.round(2).to_string())


def plot_change_maps(fits, points, shape):
    changes = {}
    for model in ["K2+S", "K2xS"]:
        coef, good = fits[model]
        for h in (2, 14):
            changes[(model, h)] = (anomaly_map(coef, good, model, h, 1.0, shape)
                                   - anomaly_map(coef, good, model, h, -1.0, shape))
    lim = np.nanpercentile(np.abs(np.stack(list(changes.values()))), 98)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for ax, ((model, h), arr) in zip(axes.ravel(), changes.items()):
        im = ax.imshow(arr, cmap="BrBG_r", vmin=-lim, vmax=lim)
        for name, (row, col) in points.items():
            ax.plot(col, row, "k+", ms=8)
        ax.set_title(f"{model}: zmiana anomalii o {h:02d}:00 (30 IX minus 1 V)")
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="[K]")
    fig.savefig(FIG_DIR / "seasonal_models_change.png", dpi=130)
    plt.close(fig)


def main():
    eco = xr.open_zarr(ECO_CUBE)
    y, w, _, meta, shape = load_ecostress(eco, load_geoloc(), load_bias_table())
    fits = {}
    for model in MODELS:
        coef, _, _, good, _ = fit(y, w, design(meta["hour"], meta["season"], model), bins=meta["bin"].values)
        fits[model] = (coef, good)
        print(f"  dopasowano {model}")
    points = {name: point_index(eco, *lonlat) for name, lonlat in POINTS.items()}
    plot_points(fits, points, shape)
    plot_change_maps(fits, points, shape)


if __name__ == "__main__":
    main()