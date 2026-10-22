import warnings
from pathlib import Path
 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
 
from lst_common import ECO_TRAIN_BUFFER, LS_BAD_BITS, keep_mask, to_eco_grid

matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="Consolidated metadata")

ECO_CUBE = Path("data/ecostress/krakow_ecostress_70m.zarr")
LS_CUBE = Path("data/landsat/krakow_landsat_30m.zarr")
OUT_NPZ = Path("results/obs_counts_70m.npz")
OUT_FIG = Path("figures/obs_counts.png")
YEARS = (2023, 2026)
MONTHS = (5, 9)
BIN_EDGES = np.arange(0, 25, 4)
BIN_LABELS = [f"{a:02d}-{b:02d}" for a, b in zip(BIN_EDGES[:-1], BIN_EDGES[1:])]
NIGHT_BINS = [0, 5]
MIN_PER_BIN = 2

 
def select(times):
    t = pd.DatetimeIndex(times)
    return ((t.year >= YEARS[0]) & (t.year <= YEARS[1])
            & (t.month >= MONTHS[0]) & (t.month <= MONTHS[1]))


 
def ecostress_counts(eco):
    idx = np.flatnonzero(select(eco.time.values) & keep_mask(eco.time.values, "ECOSTRESS"))
    hours = eco.solar_hour.values
    night = eco.is_night.values
    counts = np.zeros((len(BIN_LABELS), eco.sizes["y"], eco.sizes["x"]), dtype="int32")
    scenes = np.zeros(len(BIN_LABELS), dtype=int)
    for j in idx:
        frame = eco.isel(time=j)
        buffer = ECO_TRAIN_BUFFER["noc" if night[j] else "dzień"]
        valid = np.isfinite(frame.LST.values) & (frame.cloud_dist.values >= buffer)
        b = min(int(hours[j] // 4), len(BIN_LABELS) - 1)
        counts[b] += valid
        scenes[b] += 1
    return counts, scenes, len(idx)
 
def landsat_counts(ls, eco):
    idx = np.flatnonzero(select(ls.time.values) & keep_mask(ls.time.values, "Landsat"))
    total = np.zeros((ls.sizes["y"], ls.sizes["x"]), dtype="float32")
    for i in idx:
        frame = ls[["qa_pixel", "lwir11"]].isel(time=i)
        qa = frame.qa_pixel.values
        total += ((qa & LS_BAD_BITS) == 0) & (frame.lwir11.values != 0)
    return to_eco_grid(total, ls, eco), len(idx)
 
def summarize(counts, scenes, ls_count):
    inside = counts.sum(axis=0) > 0
    rows = []
    for b, label in enumerate(BIN_LABELS):
        c = counts[b][inside]
        rows.append({"pora": label, "sceny": scenes[b], "mediana": np.median(c),
                     "p10": np.percentile(c, 10), "p90": np.percentile(c, 90),
                     f"udzial_>={MIN_PER_BIN}": (c >= MIN_PER_BIN).mean()})
    print(pd.DataFrame(rows).round(2).to_string(index=False))
 
    covered = (counts >= MIN_PER_BIN).sum(axis=0)[inside]
    night_ok = (counts[NIGHT_BINS].sum(axis=0) >= MIN_PER_BIN)[inside]
    print(f"\nLiczba przedziałów pory doby z >= {MIN_PER_BIN} obserwacjami (udział pikseli):")
    print((pd.Series(covered).value_counts(normalize=True).sort_index()).round(3).to_string())
    print(f"Piksele z >= 4 przedziałami i nocą: {(night_ok & (covered >= 4)).mean():.1%}")
    ls_inside = ls_count[inside & np.isfinite(ls_count)]
    print(f"Landsat: mediana {np.median(ls_inside):.1f}, p10 {np.percentile(ls_inside, 10):.1f}")
    return inside, covered
 
 
def plot(counts, ls_count, inside, covered):
    vmax = np.percentile(counts[:, inside], 99)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.ravel()
    for b, label in enumerate(BIN_LABELS):
        im = axes[b].imshow(np.where(inside, counts[b], np.nan), vmin=0, vmax=vmax, cmap="viridis")
        axes[b].set_title(f"ECOSTRESS {label} h")
    axes[6].imshow(np.where(inside, ls_count, np.nan), vmin=0, cmap="viridis")
    axes[6].set_title("Landsat (ok. 10:50 h)")
    for ax in axes[:7]:
        ax.axis("off")
    fig.colorbar(im, ax=axes[:6].tolist(), shrink=0.6, label="liczba obserwacji")
    axes[7].hist(covered, bins=np.arange(-0.5, 7))
    axes[7].set_xlabel(f"przedziały z >= {MIN_PER_BIN} obserwacjami")
    axes[7].set_ylabel("liczba pikseli")
    OUT_FIG.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    plt.close(fig)
 
 
def main():
    eco = xr.open_zarr(ECO_CUBE)
    ls = xr.open_zarr(LS_CUBE)
    counts, scenes, n_eco = ecostress_counts(eco)
    ls_count, n_ls = landsat_counts(ls, eco)
    print(f"Sceny: ECOSTRESS {n_eco}, Landsat {n_ls} (lata {YEARS[0]}-{YEARS[1]}, miesiące {MONTHS[0]}-{MONTHS[1]})")
    inside, covered = summarize(counts, scenes, ls_count)
    plot(counts, ls_count, inside, covered)
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ, counts=counts, landsat=ls_count, bin_labels=np.array(BIN_LABELS))
 
 
if __name__ == "__main__":
    main()

