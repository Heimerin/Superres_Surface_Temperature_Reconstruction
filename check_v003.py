from pathlib import Path

import earthaccess
import pandas as pd

AOI = (19.70, 49.93, 20.30, 50.18)
TILE = "34UDA"
TEMPORAL = ("2018-07-01", "2026-10-11")
MONTHS = range(4, 11)
V2_INV = Path("inventory/krakow_ecostress.csv")
WINDOW = (pd.Timestamp("2024-06-25", tz="UTC"), pd.Timestamp("2024-07-04", tz="UTC"))


def attributes(umm):
    return {a.get("Name"): (a.get("Values") or [None])[0] for a in umm.get("AdditionalAttributes", [])}


def search_v003():
    granules = earthaccess.search_data(short_name="ECO_L2T_LSTE", version="003", bounding_box=AOI,
                                       temporal=TEMPORAL, count=-1)
    rows = []
    for g in granules:
        umm = g["umm"]
        name = umm["GranuleUR"]
        parts = name.split("_")
        rows.append({"id": name, "time": umm["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"],
                     "overpass": "_".join(parts[3:5]), "tile": TILE if f"_{TILE}_" in name else "inny",
                     "granule": g, **attributes(umm)})
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    return df[df["time"].dt.month.isin(MONTHS)]


def main():
    v3 = search_v003()
    v3_tile = v3[v3["tile"] == TILE]
    v2 = pd.read_csv(V2_INV, dtype={"overpass": str})
    v2["time"] = pd.to_datetime(v2["datetime"], utc=True, format="ISO8601")
    v2_tile = v2[v2["tile"] == TILE]

    counts = pd.DataFrame({
        "v002": v2_tile.drop_duplicates("overpass").groupby(v2_tile["time"].dt.year).size(),
        "v003": v3_tile.drop_duplicates("overpass").groupby(v3_tile["time"].dt.year).size(),
    }).fillna(0).astype(int)
    print(f"Przeloty nad kaflem {TILE} (miesiące IV-X):")
    print(counts.to_string())

    sample = v3_tile.iloc[-1]
    layers = [link.rsplit("/", 1)[-1].replace(sample["id"] + "_", "") for link in sample["granule"].data_links()]
    print(f"\nWarstwy w {sample['id']}:\n  {layers}")
    attr_cols = [c for c in v3.columns if c not in ("id", "time", "overpass", "tile", "granule")]
    print(f"\nAtrybuty dodatkowe w metadanych: {attr_cols}")

    keys = [c for c in attr_cols if any(k in c.lower() for k in ("obstruct", "geoloc", "quality"))]
    window = v3_tile[(v3_tile["time"] >= WINDOW[0]) & (v3_tile["time"] < WINDOW[1])]
    print(f"\nPrzeloty v003 w oknie 25.06-03.07.2024: {window['overpass'].nunique()}")
    if len(window):
        print(window[["time"] + keys].to_string(index=False))


if __name__ == "__main__":
    main()