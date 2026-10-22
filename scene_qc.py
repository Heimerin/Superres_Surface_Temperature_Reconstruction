from pathlib import Path

import numpy as np
import pandas as pd

IN_CSV = Path("results/era5_scene_means.csv")
OUT_CSV = Path("results/qc_scene_outliers.csv")
Z_MAX = 5.0
FULL = {"ECOSTRESS": 252340, "Landsat": 151200}


def split_of(row):
    if row["sensor"] != "Landsat":
        return "-"
    year = row["time"].year
    return "test" if year >= 2025 else ("val" if year == 2024 else "train")


def main():
    d = pd.read_csv(IN_CSV, parse_dates=["time"])
    d["frac"] = d["n"] / d["sensor"].map(FULL)
    grouped = d.groupby("group")["diff_skt"]
    median = grouped.transform("median")
    mad = grouped.transform(lambda s: 1.4826 * (s - s.median()).abs().median())
    d["z"] = (d["diff_skt"] - median) / mad

    out = d[d["z"].abs() > Z_MAX].copy()
    out["split"] = out.apply(split_of, axis=1)
    out = out.sort_values("z")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    columns = ["group", "time", "lst_mean", "skt", "diff_skt", "z", "frac", "split"]
    print(f"Sceny odstające (|z| > {Z_MAX}): {len(out)} z {len(d)}")
    print(out[columns].round(2).to_string(index=False))

    clean = d[d["z"].abs() <= Z_MAX]
    print("\nRozrzut po odjęciu ERA5-Land bez scen odstających:")
    print(clean.groupby("group")["diff_skt"].agg(["count", "std"]).round(2).to_string())


if __name__ == "__main__":
    main()