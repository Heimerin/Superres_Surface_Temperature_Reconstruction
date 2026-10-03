from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

PRED_DIR = Path("results/baselines/preds")
OUT_DIR = Path("figures")
PIXEL_M = 90.0
SIGMAS = np.round(np.arange(0.0, 2.01, 0.1), 2)
MIN_WEIGHT = 0.99
METHODS = ["brak_wyostrzania", "tsharp", "lgbm_scena", "lgbm_global", "lgbm_global_proj"]


def masked_blur(img, mask, sigma):
    if sigma == 0:
        return img, mask
    weight = gaussian_filter(mask.astype(float), sigma)
    blurred = gaussian_filter(np.where(mask, img, 0.0), sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        return blurred / weight, mask & (weight >= MIN_WEIGHT)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    sse = {m: np.zeros(len(SIGMAS)) for m in METHODS}
    count = np.zeros(len(SIGMAS))

    for path in sorted(PRED_DIR.glob("*.npz")):
        with np.load(path) as npz:
            truth = npz["truth"]
            preds = {m: npz[m] for m in METHODS}
        base = np.isfinite(truth)
        for p in preds.values():
            base &= np.isfinite(p)
        for k, sigma in enumerate(SIGMAS):
            valid = None
            for m, p in preds.items():
                blurred, ok = masked_blur(p, base, sigma)
                valid = ok if valid is None else valid
                sse[m][k] += ((blurred[valid] - truth[valid]) ** 2).sum()
            count[k] += valid.sum()

    rows = []
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in METHODS:
        rmse = np.sqrt(sse[m] / count)
        k = int(np.argmin(rmse))
        rows.append({"method": m, "rmse_sigma0": rmse[0], "best_sigma_px": SIGMAS[k],
                     "best_sigma_m": SIGMAS[k] * PIXEL_M, "rmse_best": rmse[k]})
        ax.plot(SIGMAS * PIXEL_M, rmse, marker="o", ms=3, label=m)
    ax.set_xlabel("sigma dodatkowego rozmycia predykcji [m]")
    ax.set_ylabel("RMSE względem prawdy 90 m [K]")
    ax.legend(fontsize=8)
    fig.savefig(OUT_DIR / "psf_estimate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()