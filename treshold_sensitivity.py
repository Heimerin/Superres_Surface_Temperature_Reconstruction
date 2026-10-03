from pathlib import Path

import pandas as pd

DATA_DIR = Path("inventory")
AOI_NAME = "krakow"
THRESHOLDS = [0.3, 0.5, 0.8]
TOLERANCES_MIN = [30, 60]


def load(name, key):
    df = pd.read_csv(DATA_DIR / f"{AOI_NAME}_{name}.csv")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, format="ISO8601")
    return df.drop_duplicates(subset=[key], keep="last")


eco_inv = load("ecostress", "id")
eco = load("ecostress_clear", "overpass")
ls = load("landsat_clear", "id")

pairs = pd.read_csv(DATA_DIR / f"{AOI_NAME}_pairs.csv")
pairs["overpass"] = pairs["id_eco"].map(eco_inv.set_index("id")["overpass"])
pairs = pairs.merge(ls[["id", "frac_clear"]].rename(columns={"id": "id_ls", "frac_clear": "clear_ls"}),
                    on="id_ls", how="left")
pairs = pairs.merge(eco[["overpass", "frac_clear"]].rename(columns={"frac_clear": "clear_eco"}),
                    on="overpass", how="left")

raw = eco_inv.drop_duplicates("overpass")
print("Wszystkie przeloty ECOSTRESS (przed filtrem chmur): rok x pora doby")
print(pd.crosstab(raw["datetime"].dt.year, raw["period"]).to_string())

for t in THRESHOLDS:
    ok = eco[eco["frac_clear"] >= t]["period"].value_counts()
    n_ls = (ls["frac_clear"] >= t).sum()
    line = f"\nPróg {t:.0%}: Landsat {n_ls}, ECOSTRESS dzień {ok.get('dzień', 0)}, noc {ok.get('noc', 0)}"
    for tol in TOLERANCES_MIN:
        mask = (pairs["dt_min"].abs() <= tol) & (pairs["clear_ls"] >= t) & (pairs["clear_eco"] >= t)
        line += f", pary do {tol} min: {mask.sum()}"
    print(line)