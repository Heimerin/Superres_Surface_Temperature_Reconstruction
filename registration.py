from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from lst_common import (ECO_EVAL_BUFFER, find_pairs, gradient_magnitude, grid_transform,
                        landsat_lst, register, to_eco_grid)

LS_CUBE = Path("data/landsat/krakow_landsat_30m.zarr")
ECO_CUBE = Path("data/ecostress/krakow_ecostress_70m.zarr")
DIRECT_CSV = Path("inventory/krakow_coregistration.csv")
TEMPLATE_PATH = Path("data/landsat/krakow_lst_template_70m.npy")
OUT_CSV = Path("data/ecostress/krakow_ecostress_shifts.csv")
OUT_DIR = Path("figures")
TEMPLATE_MIN_CLEAR = 0.5
MAX_SHIFT = 4
MIN_COMMON = 2000
PAIR_MAX_MINUTES = 60


def build_template(ls, eco):
    if TEMPLATE_PATH.exists():
        return np.load(TEMPLATE_PATH)
    idx = np.flatnonzero(ls.frac_clear.values >= TEMPLATE_MIN_CLEAR)
    print(f"Wzorzec: mediana z {len(idx)} scen Landsata")
    stack = np.stack([to_eco_grid(landsat_lst(ls.isel(time=i)).values, ls, eco) for i in idx])
    template = np.nanmedian(stack, axis=0)
    np.save(TEMPLATE_PATH, template)
    return template


def register_all(eco, template_grad, pixel):
    is_night = eco.is_night.values
    rows = []
    for j in range(eco.sizes["time"]):
        frame = eco.isel(time=j)
        buffer = ECO_EVAL_BUFFER["noc" if is_night[j] else "dzień"]
        img = np.where(frame.cloud_dist.values >= buffer, frame.LST.values, np.nan)
        res = register(template_grad, gradient_magnitude(img), MAX_SHIFT, MIN_COMMON)
        if res is not None:
            rows.append({
                "time": frame.time.values,
                "night": bool(is_night[j]),
                "shift_east_m": res["dx_px"] * pixel,
                "shift_north_m": -res["dy_px"] * pixel,
                "r0": res["r0"],
                "r_best": res["r_best"],
                "peak_contrast": res["r_best"] - res["r_median"],
                "at_edge": res["at_edge"],
            })
        if (j + 1) % 50 == 0:
            print(f"  zarejestrowano {j + 1}/{eco.sizes['time']}")
    return pd.DataFrame(rows)


def compare_with_direct(ls, eco, shifts):
    direct = pd.read_csv(DIRECT_CSV)
    eco_time = {str(ls.time.values[i])[:16]: eco.time.values[j]
                for i, j, _ in find_pairs(ls, eco, PAIR_MAX_MINUTES)}
    direct["time"] = pd.to_datetime(direct["landsat"].map(eco_time))
    merged = direct.merge(shifts, on="time", suffixes=("_direct", "_grad"))
    merged["diff_east"] = merged["shift_east_m_grad"] - merged["shift_east_m_direct"]
    merged["diff_north"] = merged["shift_north_m_grad"] - merged["shift_north_m_direct"]
    columns = ["landsat", "shift_east_m_direct", "shift_east_m_grad",
               "shift_north_m_direct", "shift_north_m_grad", "diff_east", "diff_north"]
    print("\nWalidacja: przesunięcia bezpośrednie vs gradientowe [m]")
    print(merged[columns].round(1).to_string(index=False))
    rms = np.sqrt((merged["diff_east"] ** 2 + merged["diff_north"] ** 2).mean())
    print(f"RMS różnicy wektorów przesunięcia: {rms:.1f} m")


def summarize(shifts):
    shifts["shift_m"] = np.hypot(shifts["shift_east_m"], shifts["shift_north_m"])
    print("\nPodsumowanie wg pory doby")
    summary = shifts.groupby("night").agg(
        n=("shift_m", "size"),
        median_shift_m=("shift_m", "median"),
        frac_over_35m=("shift_m", lambda s: (s > 35).mean()),
        median_r_best=("r_best", "median"),
        median_contrast=("peak_contrast", "median"),
        at_edge=("at_edge", "sum"),
    )
    print(summary.rename(index={False: "dzień", True: "noc"}).round(2).to_string())

    fig, ax = plt.subplots(figsize=(6, 6))
    for night, label in ((False, "dzień"), (True, "noc")):
        part = shifts[shifts["night"] == night]
        ax.scatter(part["shift_east_m"], part["shift_north_m"], s=8, alpha=0.5, label=label)
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("przesunięcie na wschód [m]")
    ax.set_ylabel("przesunięcie na północ [m]")
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    fig.savefig(OUT_DIR / "ecostress_shifts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    ls = xr.open_zarr(LS_CUBE)
    eco = xr.open_zarr(ECO_CUBE)
    template_grad = gradient_magnitude(build_template(ls, eco))
    shifts = register_all(eco, template_grad, grid_transform(eco).a)
    shifts.to_csv(OUT_CSV, index=False)
    compare_with_direct(ls, eco, shifts)
    summarize(shifts)


if __name__ == "__main__":
    main()