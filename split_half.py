import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from diurnal_harmonic import (BIN_LABELS, ECO_CUBE, FIG_DIR, GAIN_MODELS, anomaly_map, design, fit,
                              load_bias_table, load_ecostress, load_geoloc)

matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="Consolidated metadata")

MODEL = "K2+Sdg"
HOURS = [2, 14]


def week_parity(meta):
    t = pd.DatetimeIndex(meta["time"])
    week = t.year.to_numpy() * 100 + t.isocalendar()["week"].to_numpy().astype(int)
    order = {wk: i for i, wk in enumerate(np.unique(week))}
    return np.array([order[wk] % 2 == 0 for wk in week])


def year_split(meta):
    return pd.DatetimeIndex(meta["time"]).year.to_numpy() <= 2024


def half_maps(y, w, meta, mask, shape, bins):
    X = design(meta["hour"][mask], meta["season"][mask], MODEL)
    coef, _, _, good, _ = fit(y[mask], w[mask], X, bins=bins, gain=MODEL in GAIN_MODELS)
    return {h: anomaly_map(coef, good, MODEL, h, 0.0, shape) for h in HOURS}


def robust_sd(x):
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def compare(name, maps_a, maps_b):
    rows, tails, panels = [], [], []
    for h in HOURS:
        a, b = maps_a[h], maps_b[h]
        ok = np.isfinite(a) & np.isfinite(b)
        d = np.where(ok, a - b, np.nan)
        d -= np.nanmedian(d)
        ad = np.abs(d)
        core = ok & (ad <= np.nanpercentile(ad, 99))
        signal = np.nanstd(np.where(ok, (a + b) / 2, np.nan))
        unc = robust_sd(d) / 2
        rows.append({"podział": name, "godzina": f"{h:02d}:00",
                     "piksele": ok.mean(),
                     "wzór_std_K": signal,
                     "niepewność_RMS_K": np.sqrt(np.nanmean(d ** 2)) / 2,
                     "niepewność_odporna_K": unc,
                     "korelacja": np.corrcoef(a[ok], b[ok])[0, 1],
                     "korelacja_bez_1%": np.corrcoef(a[core], b[core])[0, 1],
                     "sygnał/szum": signal / unc})
        q = np.nanpercentile(ad, [50, 90, 99, 99.9, 100])
        tails.append({"podział": name, "godzina": f"{h:02d}:00",
                      **{f"|A-B| p{p}": v for p, v in zip(["50", "90", "99", "99.9", "max"], q)}})
        panels.append((h, a, b, ad / 2))
    return rows, tails, panels


def plot(name, panels):
    fig, axes = plt.subplots(len(panels), 3, figsize=(16, 5 * len(panels)), layout="constrained")
    for row, (h, a, b, unc) in zip(axes, panels):
        lim = np.nanpercentile(np.abs(np.stack([a, b])), 98)
        for ax, arr, title in ((row[0], a, "połowa A"), (row[1], b, "połowa B")):
            im = ax.imshow(arr, cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.set_title(f"{title}, {h:02d}:00")
        fig.colorbar(im, ax=row[:2].tolist(), shrink=0.8, label="anomalia [K]")
        im2 = row[2].imshow(unc, cmap="magma", vmin=0, vmax=np.nanpercentile(unc, 98))
        row[2].set_title(f"|A - B| / 2, {h:02d}:00")
        fig.colorbar(im2, ax=row[2], shrink=0.8, label="[K]")
        for ax in row:
            ax.axis("off")
    fig.suptitle(f"Test dwóch połówek: {name}, model {MODEL}")
    fig.savefig(FIG_DIR / f"split_half_{name}.png", dpi=120)
    plt.close(fig)


def main():
    eco = xr.open_zarr(ECO_CUBE)
    y, w, _, meta, shape = load_ecostress(eco, load_geoloc(), load_bias_table())
    splits = {"tygodnie": week_parity(meta), "lata": year_split(meta)}
    bins = meta["bin"].values
    rows, tails = [], []
    for name, mask in splits.items():
        counts = pd.crosstab(np.where(mask, "A", "B"), meta["bin"].map(dict(enumerate(BIN_LABELS))))
        print(f"\nPodział '{name}': sceny w połówkach wg pory doby\n{counts.to_string()}")
        maps_a = half_maps(y, w, meta, mask, shape, bins[mask])
        maps_b = half_maps(y, w, meta, ~mask, shape, bins[~mask])
        r, t, panels = compare(name, maps_a, maps_b)
        rows += r
        tails += t
        plot(name, panels)
    print("\n" + pd.DataFrame(rows).round(3).to_string(index=False))
    print("\n" + pd.DataFrame(tails).round(2).to_string(index=False))


if __name__ == "__main__":
    main()