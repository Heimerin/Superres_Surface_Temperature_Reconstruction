from pathlib import Path

import earthaccess
import matplotlib.pyplot as plt
import pandas as pd
import planetary_computer 
import pystac_client
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time
from astropy.utils import iers

iers.conf.auto_download=False  # Disable auto-download of IERS data to avoid network issues
iers.conf.iers_defraded_accuracy = "warn"

AOIS = {
    "krakow": (19.70, 49.93, 20.30, 50.18),
    "gzm": (18.60, 50.15, 19.30, 50.45),
}


DATE_START = "2018-07-01"
DATE_END = "2026-09-30"
MONTHS = [4,5,6,7,8,9,10]
PAIR_TOLERANCE = pd.Timedelta(minutes=60)
OUT_DIR = Path("inventory")
PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"

def search_landsat(bbox):
    catalog = pystac_client.Client.open(PC_STAC, modifier=planetary_computer.sign_inplace)
    search = catalog.search(

        collections=["landsat-c2-l2"],
        bbox=bbox,
        datetime=f"{DATE_START}/{DATE_END}",
        #query={"platform": {"in": ["landsat-8"]}},
    )
    rows = []
    for item in search.items():
        p = item.properties
        rows.append({
            "id": item.id,
            "datetime": p["datetime"],
            "platform": p.get("platform"),
            "cloud_cover": p.get("eo:cloud_cover"),
            "path": p.get("landsat:wrs_path"),
            "row": p.get("landsat:wrs_row"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, format="ISO8601")
        df = df[df["id"].str.startswith(("LC08", "LC09"))].reset_index(drop=True)
    return df 

def cloud_from_umm(umm):
    if "CloudCover" in umm:
        return umm["CloudCover"]
    for attr in umm.get("AdditionalAttributes", []):
        if "cloud" in attr.get("Name", "").lower():
            try:
                return float(attr["Values"][0])
            except (ValueError, KeyError, IndexError):
                return None
    return None

def search_ecostress(bbox):
    granules = earthaccess.search_data(
        short_name="ECO_L2T_LSTE",
        version="002",
        bounding_box=bbox,
        temporal=(DATE_START, DATE_END),
        count=-1,
    )
    rows = []
    for g in granules:
        umm = g["umm"]
        name = umm["GranuleUR"]
        parts = name.split("_")
        parsed = len(parts) > 5
        rows.append({
            "id": name,
            "datetime": umm["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"],
            "orbit": parts[3] if parsed else None,
            "scene": parts[4] if parsed else None,
            "tile": parts[5] if parsed else None,
            "cloud_cover": cloud_from_umm(umm),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, format="ISO8601")
    df["overpass"] = df["orbit"].fillna("") + "_" + df["scene"].fillna("")
    missing = df["orbit"].isna()
    df.loc[missing, "overpass"] = df.loc[missing, "datetime"].dt.floor("5min").astype(str)
    return df

def add_solar_geometry(df, bbox):
    lon = (bbox[0] + bbox[2]) / 2
    lat = (bbox[1] + bbox[3]) / 2
    location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg)
    times = Time(df["datetime"].dt.tz_convert(None).to_numpy())
    frame = AltAz(obstime=times, location=location)
    df["sun_alt"] = get_sun(times).transform_to(frame).alt.deg
    utc_hours = df["datetime"].dt.hour + df["datetime"].dt.minute / 60
    df["solar_hour"] = (utc_hours + lon / 15) % 24
    df["period"] = pd.cut(
        df["sun_alt"], bins=[-90, -6, 0, 90], labels=["noc", "zmierzch/świt", "dzień"]
    )
    df["month"] = df["datetime"].dt.month
    return df


def find_pairs(landsat, eco):
    ls = landsat.sort_values("datetime")[["id", "datetime", "cloud_cover"]]
    ec = eco.sort_values("datetime")[["id", "datetime", "cloud_cover", "period"]]
    ec = ec.assign(datetime_eco=ec["datetime"])
    pairs = pd.merge_asof(
        ls, ec,
        on="datetime",
        direction="nearest",
        tolerance=PAIR_TOLERANCE,
        suffixes=("_ls", "_eco"),
    )
    pairs = pairs.dropna(subset=["id_eco"]).copy()
    pairs["dt_min"] = (pairs["datetime_eco"] - pairs["datetime"]).dt.total_seconds() / 60
    return pairs

def summarize(name, landsat, eco, pairs):
    eco_unique = eco.drop_duplicates(subset=["overpass"])
    print(f"\n=== {name} ===")
    print(f"Landsat: {len(landsat)} scen, w tym <20% chmur: {(landsat['cloud_cover'] < 20).sum()}")
    print(f"ECOSTRESS: {len(eco)} kafli, {len(eco_unique)} unikalnych przelotów")
    print(eco_unique["period"].value_counts().to_string())
    print(f"Pary Landsat-ECOSTRESS (tolerancja {PAIR_TOLERANCE}): {len(pairs)}")
 
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(eco_unique["solar_hour"], bins=24, range=(0, 24), alpha=0.7, label="ECOSTRESS")
    axes[0].hist(landsat["solar_hour"], bins=24, range=(0, 24), alpha=0.7, label="Landsat")
    axes[0].set_xlabel("lokalny średni czas słoneczny [h]")
    axes[0].set_ylabel("liczba przelotów")
    axes[0].legend()
 
    table = pd.crosstab(eco_unique["month"], eco_unique["period"])
    table.plot(kind="bar", stacked=True, ax=axes[1])
    axes[1].set_xlabel("miesiąc")
    axes[1].set_title("ECOSTRESS: przeloty wg miesiąca i pory doby")
 
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}_inventory.png", dpi=150)
    plt.close(fig)
 
 
def main():
    OUT_DIR.mkdir(exist_ok=True)
    for name, bbox in AOIS.items():
        landsat = search_landsat(bbox)
        eco = search_ecostress(bbox)
        if landsat.empty or eco.empty:
            print(f"{name}: brak wyników (Landsat: {len(landsat)}, ECOSTRESS: {len(eco)})")
            continue
 
        landsat = add_solar_geometry(landsat, bbox)
        eco = add_solar_geometry(eco, bbox)
        landsat = landsat[landsat["month"].isin(MONTHS)]
        eco = eco[eco["month"].isin(MONTHS)]
        pairs = find_pairs(landsat, eco)
 
        landsat.to_csv(OUT_DIR / f"{name}_landsat.csv", index=False)
        eco.to_csv(OUT_DIR / f"{name}_ecostress.csv", index=False)
        pairs.to_csv(OUT_DIR / f"{name}_pairs.csv", index=False)
        summarize(name, landsat, eco, pairs)
 
 
if __name__ == "__main__":
    main()

