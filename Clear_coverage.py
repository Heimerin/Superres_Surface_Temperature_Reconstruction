import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import earthaccess
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

AOI_NAME = "krakow"
AOI_BBOX = (19.70, 49.93, 20.30, 50.18)
ECO_EPSG = 32634
ECO_RES = 70.0
ECO_BASE = "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/ECO_L2T_LSTE.002"
LANDSAT_MAX_SCENE_CLOUD = 60
LANDSAT_BAD_BITS = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)
CLEAR_THRESHOLD = 0.8
PAIR_MAX_MINUTES = 30
N_WORKERS = 8
DATA_DIR = Path("inventory")
PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"

ECO_COLUMNS = ["overpass", "datetime", "period", "solar_hour", "month",
               "frac_lst", "frac_cloud", "frac_clear", "error"]
LS_COLUMNS = ["id", "datetime", "path", "row", "cloud_cover", "frac_clear", "error"]

cookie = str(Path.home() / ".urs_cookies")
os.environ.update({
    "GDAL_HTTP_NETRC": "YES",
    "GDAL_HTTP_COOKIEFILE": cookie,
    "GDAL_HTTP_COOKIEJAR": cookie,
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": "TIF",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "2",
})


def save_rows(rows, path, columns):
    if not rows:
        return
    df = pd.DataFrame(rows).reindex(columns=columns)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def load_results(path, key):
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, format="ISO8601")
    return df.drop_duplicates(subset=[key], keep="last")


def eco_url(granule_id, layer):
    return f"{ECO_BASE}/{granule_id}/{granule_id}_{layer}.tif"


def eco_aoi_grid():
    left, bottom, right, top = transform_bounds("EPSG:4326", f"EPSG:{ECO_EPSG}", *AOI_BBOX)
    width = int(np.ceil((right - left) / ECO_RES))
    height = int(np.ceil((top - bottom) / ECO_RES))
    return (left, bottom, right, top), (height, width)


def read_eco_layer(url, bounds, shape, fill):
    with rasterio.open(url) as src:
        if src.crs.to_epsg() != ECO_EPSG:
            return None
        window = from_bounds(*bounds, transform=src.transform)
        data = src.read(1, window=window, boundless=True, fill_value=fill, out_shape=shape)
    return data.astype("float32")


def process_eco_overpass(granule_ids, bounds, shape):
    lst = np.full(shape, np.nan, dtype="float32")
    cloud = np.full(shape, np.nan, dtype="float32")
    for gid in granule_ids:
        new_lst = read_eco_layer(eco_url(gid, "LST"), bounds, shape, np.nan)
        new_cloud = read_eco_layer(eco_url(gid, "cloud"), bounds, shape, 255)
        if new_lst is None or new_cloud is None:
            continue
        new_cloud[new_cloud > 1] = np.nan
        lst = np.where(np.isnan(lst), new_lst, lst)
        cloud = np.where(np.isnan(cloud), new_cloud, cloud)
    has_lst = np.isfinite(lst) & (lst > 150)
    cloudy = cloud == 1
    return {
        "frac_lst": float(has_lst.mean()),
        "frac_cloud": float(cloudy.mean()),
        "frac_clear": float((has_lst & ~cloudy).mean()),
    }


def eco_clear_fractions():
    eco = pd.read_csv(DATA_DIR / f"{AOI_NAME}_ecostress.csv", parse_dates=["datetime"])
    out_path = DATA_DIR / f"{AOI_NAME}_ecostress_clear.csv"
    previous = load_results(out_path, "overpass")
    done = set(previous.loc[previous["error"].isna(), "overpass"]) if not previous.empty else set()

    meta = eco.drop_duplicates("overpass").set_index("overpass")
    meta = meta[["datetime", "period", "solar_hour", "month"]]
    groups = {k: g["id"].tolist() for k, g in eco.groupby("overpass") if k not in done}
    bounds, shape = eco_aoi_grid()
    print(f"ECOSTRESS: {len(groups)} przelotów do sprawdzenia, {len(done)} już gotowych")

    rows = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(process_eco_overpass, ids, bounds, shape): k
                   for k, ids in groups.items()}
        for i, fut in enumerate(as_completed(futures), 1):
            key = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:
                result = {"error": str(exc)[-300:]}
            rows.append({"overpass": key, **meta.loc[key].to_dict(), **result})
            if i % 50 == 0 or i == len(futures):
                save_rows(rows, out_path, ECO_COLUMNS)
                rows = []
                print(f"  ECOSTRESS {i}/{len(futures)}")
    return load_results(out_path, "overpass")


def landsat_clear_fraction(item):
    href = item.assets["qa_pixel"].href
    with rasterio.open(href) as src:
        bounds = transform_bounds("EPSG:4326", src.crs, *AOI_BBOX)
        window = from_bounds(*bounds, transform=src.transform)
        qa = src.read(1, window=window, boundless=True, fill_value=1)
    clear = (qa & LANDSAT_BAD_BITS) == 0
    return {"frac_clear": float(clear.mean())}


def landsat_clear_fractions():
    ls = pd.read_csv(DATA_DIR / f"{AOI_NAME}_landsat.csv", parse_dates=["datetime"])
    ls = ls[ls["cloud_cover"] <= LANDSAT_MAX_SCENE_CLOUD]
    out_path = DATA_DIR / f"{AOI_NAME}_landsat_clear.csv"
    previous = load_results(out_path, "id")
    done = set(previous.loc[previous["error"].isna(), "id"]) if not previous.empty else set()
    todo = ls[~ls["id"].isin(done)].set_index("id")
    print(f"Landsat: {len(todo)} scen do sprawdzenia, {len(done)} już gotowych")
    if todo.empty:
        return previous

    catalog = pystac_client.Client.open(PC_STAC, modifier=planetary_computer.sign_inplace)
    ids = todo.index.tolist()
    items = {}
    for start in range(0, len(ids), 100):
        search = catalog.search(collections=["landsat-c2-l2"], ids=ids[start:start + 100])
        items.update({item.id: item for item in search.items()})

    rows = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(landsat_clear_fraction, items[i]): i for i in ids if i in items}
        for n, fut in enumerate(as_completed(futures), 1):
            scene_id = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:
                result = {"error": str(exc)[:200]}
            meta = todo.loc[scene_id, ["datetime", "path", "row", "cloud_cover"]].to_dict()
            rows.append({"id": scene_id, **meta, **result})
            if n % 50 == 0 or n == len(futures):
                save_rows(rows, out_path, LS_COLUMNS)
                rows = []
                print(f"  Landsat {n}/{len(futures)}")
    return load_results(out_path, "id")


def clean_pairs(eco_clear, ls_clear):
    pairs = pd.read_csv(DATA_DIR / f"{AOI_NAME}_pairs.csv", parse_dates=["datetime", "datetime_eco"])
    eco = pd.read_csv(DATA_DIR / f"{AOI_NAME}_ecostress.csv")
    to_overpass = eco.set_index("id")["overpass"]
    pairs["overpass"] = pairs["id_eco"].map(to_overpass)
    pairs = pairs.merge(ls_clear[["id", "frac_clear"]].rename(columns={"id": "id_ls", "frac_clear": "clear_ls"}),
                        on="id_ls", how="left")
    pairs = pairs.merge(eco_clear[["overpass", "frac_clear"]].rename(columns={"frac_clear": "clear_eco"}),
                        on="overpass", how="left")
    good = pairs[(pairs["dt_min"].abs() <= PAIR_MAX_MINUTES)
                 & (pairs["clear_ls"] >= CLEAR_THRESHOLD)
                 & (pairs["clear_eco"] >= CLEAR_THRESHOLD)]
    good.to_csv(DATA_DIR / f"{AOI_NAME}_pairs_clear.csv", index=False)
    return good


def summarize(eco_clear, ls_clear, good_pairs):
    eco_ok = eco_clear[eco_clear["frac_clear"] >= CLEAR_THRESHOLD].copy()
    ls_ok = ls_clear[ls_clear["frac_clear"] >= CLEAR_THRESHOLD]
    eco_ok["year"] = eco_ok["datetime"].dt.year
    n_errors = eco_clear["error"].notna().sum() + ls_clear["error"].notna().sum()

    print(f"\n=== {AOI_NAME}: próg czystości {CLEAR_THRESHOLD:.0%} ===")
    print(f"Landsat: {len(ls_ok)} czystych scen z {len(ls_clear)} sprawdzonych")
    print(f"ECOSTRESS: {len(eco_ok)} czystych przelotów z {len(eco_clear)} sprawdzonych")
    print(eco_ok["period"].value_counts().to_string())
    print("\nECOSTRESS czyste przeloty: rok x pora doby")
    print(pd.crosstab(eco_ok["year"], eco_ok["period"]).to_string())
    print(f"\nCzyste pary Landsat-ECOSTRESS (do {PAIR_MAX_MINUTES} min): {len(good_pairs)}")
    print(f"Błędy odczytu: {n_errors}")


def main():
    earthaccess.login(persist=True)
    eco_clear = eco_clear_fractions()
    ls_clear = landsat_clear_fractions()
    good_pairs = clean_pairs(eco_clear, ls_clear)
    summarize(eco_clear, ls_clear, good_pairs)


if __name__ == "__main__":
    main()