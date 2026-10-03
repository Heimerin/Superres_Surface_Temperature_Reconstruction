from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PRED_DIR = Path("results/baselines/preds")
OUT_DIR = Path("figures")
TILE = 64
PIXEL_M = 90.0
COARSE_M = 270.0
FRC_THRESHOLD = 0.5
METHODS = ["brak_wyostrzania", "tsharp", "lgbm_scena", "lgbm_global", "lgbm_global_proj"]


def valid_tiles(stack, tile):
    ok = np.isfinite(stack).all(axis=0)
    h, w = ok.shape
    return [(y, x) for y in range(0, h - tile + 1, tile) for x in range(0, w - tile + 1, tile)
            if ok[y:y + tile, x:x + tile].all()]


def ring_index(n):
    fy, fx = np.indices((n, n)) - n // 2
    return np.hypot(fx, fy).round().astype(int).ravel()


def spectrum(img, window):
    return np.fft.fftshift(np.fft.fft2((img - img.mean()) * window)).ravel()


def main():
    OUT_DIR.mkdir(exist_ok=True)
    window = np.outer(np.hanning(TILE), np.hanning(TILE))
    rings = ring_index(TILE)
    n_rings = TILE // 2
    auto_truth = np.zeros(n_rings)
    auto = {m: np.zeros(n_rings) for m in METHODS}
    cross = {m: np.zeros(n_rings) for m in METHODS}
    n_tiles = 0

    for path in sorted(PRED_DIR.glob("*.npz")):
        with np.load(path) as npz:
            names = ["truth"] + METHODS
            stack = np.stack([npz[k] for k in names])
        for y, x in valid_tiles(stack, TILE):
            tiles = {k: stack[i, y:y + TILE, x:x + TILE] for i, k in enumerate(names)}
            f_truth = spectrum(tiles["truth"], window)
            auto_truth += np.bincount(rings, np.abs(f_truth) ** 2)[:n_rings]
            for m in METHODS:
                f_m = spectrum(tiles[m], window)
                auto[m] += np.bincount(rings, np.abs(f_m) ** 2)[:n_rings]
                cross[m] += np.bincount(rings, (f_m * np.conj(f_truth)).real)[:n_rings]
            n_tiles += 1

    freq = np.arange(n_rings) / (TILE * PIXEL_M) * 1000
    nyq_coarse = 1000 / (2 * COARSE_M)
    band = (freq > nyq_coarse) & (freq > 0)
    print(f"Kafle {TILE}x{TILE} px bez braków danych: {n_tiles}")

    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for m in METHODS:
        frc = cross[m] / np.sqrt(auto[m] * auto_truth)
        ratio = auto[m] / auto_truth
        below = np.flatnonzero((frc < FRC_THRESHOLD) & (freq > 0))
        f_cut = freq[below[0]] if len(below) else np.nan
        rows.append({
            "method": m,
            "f_frc05_cyc_km": f_cut,
            "eff_res_m": 1000 / (2 * f_cut) if np.isfinite(f_cut) else np.nan,
            "power_ratio_detail": auto[m][band].sum() / auto_truth[band].sum(),
            "frc_detail": cross[m][band].sum() / np.sqrt(auto[m][band].sum() * auto_truth[band].sum()),
        })
        axes[0].plot(freq[1:], frc[1:], label=m)
        axes[1].plot(freq[1:], ratio[1:], label=m)

    for ax in axes:
        ax.axvline(nyq_coarse, color="gray", ls="--", lw=0.8)
        ax.set_xlabel("częstość przestrzenna [cykle/km]")
        ax.legend(fontsize=8)
    axes[0].axhline(FRC_THRESHOLD, color="gray", lw=0.8)
    axes[0].set_ylabel("FRC z prawdą 90 m")
    axes[1].set_ylabel("moc metody / moc prawdy")
    axes[1].set_yscale("log")
    fig.savefig(OUT_DIR / "effective_resolution_baselines.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()