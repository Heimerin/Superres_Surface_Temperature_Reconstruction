import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import warnings

import earthaccess
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds
from rasterio.windows import transform as window_transform

warnings.filterwarnings("ignore", message="Consolidated metadata")

AOI_NAME = "krakow"
AOI_BBOX = (19.70, 49.93, 20.30, 50.18)
REFERENCE_TILE = "34UDA"
ECO_EPSG = 32634
MIN_CLEAR = 0.3
LAYERS = {
    "LST": "float32",
    "LST_err": "float32",
    "EmisWB": "float32",
    "view_zenith": "float32",
    "QC": "uint16",
    "cloud": "uint8",
    "water": "uint8",
}
N_WORKERS = 4
CUBE_BATCH = 50
INV_DIR = Path("inventory")
RAW_DIR = Path("data/ecostress/raw")
CUBE_PATH = Path(f"data/ecostress/{AOI_NAME}_ecostress_70m.zarr")
INDEX_PATH = Path(f"data/ecostress/{AOI_NAME}_ecostress_index.csv")
ECO_BASE = "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/ECO_L2T_LSTE.002"

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


def eco_url(granule_id, layer):
    return f"{ECO_BASE}/{granule_id}/{granule_id}_{layer}.tif"


def load_csv(path, key):
    df = pd.read_csv(path, dtype={"overpass": str})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, format="ISO8601")
    return df.drop_duplicates(subset=[key], keep="last")


def select_overpasses():
    inv = load_csv(INV_DIR / f"{AOI_NAME}_ecostress.csv", "id")
    clear = load_csv(INV_DIR / f"{AOI_NAME}_ecostress_clear.csv", "overpass")
    clear = clear[clear["error"].isna() & (clear["frac_clear"] >= MIN_CLEAR)]
    ref = inv[inv["tile"] == REFERENCE_TILE][["overpass", "id"]]
    ref = ref.rename(columns={"id": "granule_id"})
    sel = clear.merge(ref, on="overpass", how="left")
    missing = sel["granule_id"].isna().sum()
    print(f"Wybrane przeloty: {len(sel)}, w tym bez kafla {REFERENCE_TILE}: {missing}")
    sel=sel.sort_values("granule_id")
    sel = sel.dropna(subset=["granule_id"]).drop_duplicates("overpass", keep="last")
    columns = ["overpass", "granule_id", "datetime", "period", "solar_hour", "frac_clear"]
    return sel[columns].sort_values("datetime").reset_index(drop=True)


def reference_grid(granule_id):
    with rasterio.open(eco_url(granule_id, "LST")) as src:
        if src.crs.to_epsg() != ECO_EPSG:
            raise ValueError(f"Kafel {REFERENCE_TILE} ma układ {src.crs}, oczekiwano EPSG:{ECO_EPSG}")
        left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, *AOI_BBOX)
        b = src.bounds
        if left < b.left or right > b.right or bottom < b.bottom or top > b.top:
            raise ValueError(f"Obszar {AOI_NAME} wychodzi poza kafel {REFERENCE_TILE}: {b}")
        w = from_bounds(left, bottom, right, top, transform=src.transform)
        col0 = int(np.floor(w.col_off))
        row0 = int(np.floor(w.row_off))
        col1 = int(np.ceil(w.col_off + w.width))
        row1 = int(np.ceil(w.row_off + w.height))
        window = Window(col0, row0, col1 - col0, row1 - row0)
        return window, window_transform(window, src.transform), src.transform


def read_layer(granule_id, layer, dtype, window, tile_transform):
    with rasterio.open(eco_url(granule_id, layer)) as src:
        if not src.transform.almost_equals(tile_transform):
            raise ValueError(f"Inna siatka pikseli: {granule_id} ({layer})")
        data = src.read(1, window=window)
        nodata = src.nodata
        scale, offset = src.scales[0], src.offsets[0]
    if np.dtype(dtype).kind != "f":
        return data.astype(dtype)
    data = data.astype("float32")
    if nodata is not None:
        data[data == nodata] = np.nan
    if scale != 1 or offset != 0:
        data = data * scale + offset
    return data.astype("float32")


def save_npz(path, arrays):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def fetch_overpass(overpass, granule_id, window, tile_transform):
    for attempt in range(3):
        try:
            arrays = {layer: read_layer(granule_id, layer, dtype, window, tile_transform)
                      for layer, dtype in LAYERS.items()}
            break
        except rasterio.errors.RasterioIOError:
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))
    save_npz(RAW_DIR / f"{overpass}.npz", arrays)


def download_all(sel, window, tile_transform):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    todo = sel[~sel["overpass"].map(lambda k: (RAW_DIR / f"{k}.npz").exists())]
    print(f"Do pobrania: {len(todo)}, już pobrane: {len(sel) - len(todo)}")
    failures = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(fetch_overpass, r.overpass, r.granule_id, window, tile_transform): r.overpass
                   for r in todo.itertuples()}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                fut.result()
            except Exception as exc:
                failures.append((futures[fut], str(exc)[-300:]))
            if i % 25 == 0 or i == len(futures):
                print(f"  pobrano {i}/{len(futures)}, błędy: {len(failures)}")
    for overpass, message in failures[:10]:
        print(f"  {overpass}: {message}")
    return failures


def build_cube(sel, grid_transform, shape):
    available = sel[sel["overpass"].map(lambda k: (RAW_DIR / f"{k}.npz").exists())]
    available = available.reset_index(drop=True)
    height, width = shape
    x = grid_transform.c + grid_transform.a * (np.arange(width) + 0.5)
    y = grid_transform.f + grid_transform.e * (np.arange(height) + 0.5)

    for start in range(0, len(available), CUBE_BATCH):
        batch = available.iloc[start:start + CUBE_BATCH]
        stacks = {layer: [] for layer in LAYERS}
        for key in batch["overpass"]:
            with np.load(RAW_DIR / f"{key}.npz") as npz:
                for layer in LAYERS:
                    stacks[layer].append(npz[layer])

        ds = xr.Dataset(
            {layer: (("time", "y", "x"), np.stack(arrs)) for layer, arrs in stacks.items()},
            coords={
                "time": batch["datetime"].dt.tz_convert(None).to_numpy(),
                "y": y,
                "x": x,
                "solar_hour": ("time", batch["solar_hour"].to_numpy()),
                "frac_clear": ("time", batch["frac_clear"].to_numpy()),
                "is_night": ("time", (batch["period"] == "noc").to_numpy()),
            },
        )
        ds = ds.chunk({"time": 1})
        ds.attrs.update({
            "crs": f"EPSG:{ECO_EPSG}",
            "transform": list(grid_transform)[:6],
            "source": "ECOSTRESS ECO_L2T_LSTE v002",
            "tile": REFERENCE_TILE,
            "aoi_bbox_wgs84": list(AOI_BBOX),
        })
        if start == 0:
            ds.to_zarr(CUBE_PATH, mode="w")
        else:
            ds.to_zarr(CUBE_PATH, append_dim="time")
        print(f"  kostka: {min(start + CUBE_BATCH, len(available))}/{len(available)}")

    available.to_csv(INDEX_PATH, index=False)
    print(f"Kostka zapisana: {CUBE_PATH} ({len(available)} przelotów, {height}x{width} px)")


def main():
    earthaccess.login(persist=True)
    sel = select_overpasses()
    window, grid_transform, tile_transform = reference_grid(sel["granule_id"].iloc[0])
    print(f"Siatka: {window.height}x{window.width} px, piksel {grid_transform.a:.1f} m, "
          f"lewy górny róg ({grid_transform.c:.1f}, {grid_transform.f:.1f})")
    download_all(sel, window, tile_transform)
    build_cube(sel, grid_transform, (window.height, window.width))


if __name__ == "__main__":
    main()