import io
import warnings
import zipfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from lst_common import ECO_EVAL_BUFFER

matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="Consolidated metadata")

ERA_DIR = Path("data/era5land")
ECO_CUBE = Path("data/ecostress/krakow_ecostress_70m.zarr")
LS_CUBE = Path("data/landsat/krakow_landsat_30m.zarr")
PYRAMID = Path("data/landsat/krakow_wald_pyramid.zarr")
OUT_CSV = Path("results/era5_scene_means.csv")
OUT_FIG = Path("figures/era5_vs_lst.png")
MIN_VALID_PX = 5000


def open_era_parts(path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if name.endswith(".nc"):
                    yield xr.open_dataset(io.BytesIO(zf.read(name)), engine="h5netcdf")
    else:
        yield xr.open_dataset(path, engine="h5netcdf")


def region_means(ds):
    time_name = "valid_time" if "valid_time" in ds.dims else "time"
    ds = ds.rename({time_name: "time"})
    names = [v for v in ("skt", "t2m") if v in ds]
    means = ds[names].mean(dim=["latitude", "longitude"]).load().reset_coords(drop=True)
    return means.to_dataframe()[names]


def load_era5():
    frames = []
    for path in sorted(ERA_DIR.glob("era5land_*.nc")):
        for ds in open_era_parts(path):
            with ds:
                frames.append(region_means(ds))
    era = pd.concat(frames).groupby(level=0).first().sort_index()
    return era.dropna()


def era_at(era, times):
    x = era.index.values.astype("datetime64[s]").astype("int64")
    t = pd.to_datetime(times).values.astype("datetime64[s]").astype("int64")
    inside = (t >= x[0]) & (t <= x[-1])
    out = {c: np.where(inside, np.interp(t, x, era[c].values), np.nan) for c in era.columns}
    return pd.DataFrame(out)


def ecostress_means():
    eco = xr.open_zarr(ECO_CUBE)
    is_night = eco.is_night.values
    rows = []
    for j in range(eco.sizes["time"]):
        frame = eco.isel(time=j)
        buffer = ECO_EVAL_BUFFER["noc" if is_night[j] else "dzień"]
        lst = np.where(frame.cloud_dist.values >= buffer, frame.LST.values, np.nan)
        n = int(np.isfinite(lst).sum())
        if n < MIN_VALID_PX:
            continue
        rows.append({"sensor": "ECOSTRESS", "time": eco.time.values[j], "night": bool(is_night[j]),
                     "solar_hour": float(eco.solar_hour.values[j]), "lst_mean": float(np.nanmean(lst)),
                     "n": n})
    return pd.DataFrame(rows)


def landsat_means():
    pyr = xr.open_zarr(PYRAMID)
    ls = xr.open_zarr(LS_CUBE)
    solar = pd.Series(ls.solar_hour.values, index=pd.to_datetime(ls.time.values))
    rows = []
    for i in range(pyr.sizes["time"]):
        lst = pyr.lst_90.isel(time=i).values
        n = int(np.isfinite(lst).sum())
        if n < MIN_VALID_PX:
            continue
        t = pd.Timestamp(pyr.time.values[i])
        rows.append({"sensor": "Landsat", "time": t, "night": False,
                     "solar_hour": float(solar.get(t, np.nan)), "lst_mean": float(np.nanmean(lst)),
                     "n": n})
    return pd.DataFrame(rows)


def main():
    era = load_era5()
    print(f"ERA5-Land: {len(era)} godzin, {era.index.min()} - {era.index.max()}")
    scenes = pd.concat([ecostress_means(), landsat_means()], ignore_index=True)
    scenes = pd.concat([scenes, era_at(era, scenes["time"])], axis=1).dropna(subset=["skt"])
    scenes["diff_skt"] = scenes["lst_mean"] - scenes["skt"]
    scenes["group"] = scenes["sensor"] + np.where(scenes["night"], " noc", " dzień")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    scenes.to_csv(OUT_CSV, index=False)

    summary = scenes.groupby("group").apply(lambda g: pd.Series({
        "n": len(g),
        "r_skt": np.corrcoef(g["lst_mean"], g["skt"])[0, 1],
        "bias_K": g["diff_skt"].mean(),
        "std_diff_K": g["diff_skt"].std(),
        "std_lst_K": g["lst_mean"].std(),
    }), include_groups=False)
    print(summary.round(2).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for name, g in scenes.groupby("group"):
        axes[0].scatter(g["skt"], g["lst_mean"], s=10, alpha=0.6, label=name)
        axes[1].scatter(g["solar_hour"], g["diff_skt"], s=10, alpha=0.6, label=name)
    lims = [scenes[["skt", "lst_mean"]].min().min(), scenes[["skt", "lst_mean"]].max().max()]
    axes[0].plot(lims, lims, color="gray", lw=1)
    axes[0].set_xlabel("ERA5-Land skin temperature, średnia regionu [K]")
    axes[0].set_ylabel("średnia LST sceny [K]")
    axes[1].axhline(0, color="gray", lw=1)
    axes[1].set_xlabel("lokalny czas słoneczny [h]")
    axes[1].set_ylabel("LST sceny - ERA5-Land [K]")
    for ax in axes:
        ax.legend(fontsize=8)
    OUT_FIG.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()