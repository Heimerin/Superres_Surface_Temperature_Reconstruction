from pathlib import Path

import numpy as np
import pandas as pd

SHIFTS_CSV = Path("data/ecostress/krakow_ecostress_shifts.csv")
OUT_CSV = Path("data/ecostress/krakow_ecostress_geoloc.csv")
MIN_CONTRAST = 0.1
REG_SIGMA_M = 21.0


def robust_sigma(values):
    return 1.4826 * np.median(np.abs(values - np.median(values)))


def main():
    d = pd.read_csv(SHIFTS_CSV)
    reliable = (~d["night"]) & (~d["at_edge"]) & (d["peak_contrast"] > MIN_CONTRAST)
    ref = d[reliable]

    center_east = ref["shift_east_m"].median()
    center_north = ref["shift_north_m"].median()
    sigma_east = robust_sigma(ref["shift_east_m"])
    sigma_north = robust_sigma(ref["shift_north_m"])
    print(f"Wiarygodne rejestracje dzienne: {reliable.sum()} z {(~d['night']).sum()}")
    print(f"Środek rozkładu: wschód {center_east:.1f} m, północ {center_north:.1f} m")
    print(f"Rozrzut (odporne sigma): wschód {sigma_east:.1f} m, północ {sigma_north:.1f} m")

    out = pd.DataFrame({"time": d["time"], "night": d["night"]})
    out["method"] = np.where(reliable, "registered", "prior")
    out["shift_east_m"] = np.where(reliable, d["shift_east_m"], center_east)
    out["shift_north_m"] = np.where(reliable, d["shift_north_m"], center_north)
    out["sigma_east_m"] = np.where(reliable, REG_SIGMA_M, sigma_east)
    out["sigma_north_m"] = np.where(reliable, REG_SIGMA_M, sigma_north)
    out.to_csv(OUT_CSV, index=False)

    counts = out.groupby(["night", "method"]).size().unstack(fill_value=0)
    print(counts.rename(index={False: "dzień", True: "noc"}).to_string())


if __name__ == "__main__":
    main()