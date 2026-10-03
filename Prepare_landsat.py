import warnings

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
import xarray as xr
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds
from rasterio.windows import transform as window_transform
warnings.filterwarnings("ignore", message="Consolidated metadata")

AOI_NAME = "krakow"
AOI_BBOX = (19.70, 49.93, 20.30, 50.18)
LS_EPSG = 32634
RES = 30.0
MIN_CLEAR = 0.3
ASSETS = [
    "lwir11", "qa", "emis", "trad", "urad", "drad", "atran", "cdist", "qa_pixel",
    "blue", "green", "red", "nir08", "swir16", "swir22",
]
N_WORKERS = 4
CUBE_BATCH = 10
INV_DIR = Path("inventory")
RAW_DIR = Path("data/landsat/raw")
CUBE_PATH = Path(f"data/landsat/{AOI_NAME}_landsat_30m.zarr")
INDEX_PATH = Path(f"data/landsat/{AOI_NAME}_landsat_index.csv")
PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
EXCLUDE = {"LC08_187_20221028"}


def load_csv(path, key):
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, format="ISO8601")
    return df.drop_duplicates(subset=[key], keep="last")


def select_scenes():
    inv = load_csv(INV_DIR / f"{AOI_NAME}_landsat.csv", "id")
    clear = load_csv(INV_DIR / f"{AOI_NAME}_landsat_clear.csv", "id")
    clear = clear[clear["error"].isna()]
    parts = clear["id"].str.split("_")
    clear["acquisition"] = parts.str[0] + "_" + parts.str[2].str[:3] + "_" + parts.str[3]
    clear["platform"] = parts.str[0].str[-1].astype(int)
    good = clear.groupby("acquisition")["frac_clear"].max()
    good = good[good >= MIN_CLEAR].index
    good=good[~good.isin(EXCLUDE)]
    sel = clear[clear["acquisition"].isin(good)]
    sel = sel.merge(inv[["id", "solar_hour"]], on="id", how="left")
    print(f"Wybrane akwizycje: {len(good)}, sceny: {len(sel)}")
    return sel.sort_values("datetime").reset_index(drop=True)


def fetch_items(ids):
    catalog = pystac_client.Client.open(PC_STAC)
    items = {}
    for start in range(0, len(ids), 100):
        search = catalog.search(collections=["landsat-c2-l2"], ids=ids[start:start + 100])
        items.update({item.id: item for item in search.items()})
    return items


def band_scale(asset):
    bands = asset.extra_fields.get("raster:bands") or asset.extra_fields.get("bands") or [{}]
    b = bands[0]
    return {
        "scale": float(b.get("scale", b.get("raster:scale", 1.0))),
        "offset": float(b.get("offset", b.get("raster:offset", 0.0))),
    }


def asset_metadata(item):
    meta = {}
    for key in ASSETS:
        asset = item.assets[key]
        with rasterio.open(planetary_computer.sign(asset.href)) as src:
            nodata = src.nodata
        meta[key] = {**band_scale(asset), "nodata": None if nodata is None else float(nodata)}
    return meta


def reference_grid(item):
    with rasterio.open(planetary_computer.sign(item.assets["lwir11"].href)) as src:
        if src.crs.to_epsg() != LS_EPSG:
            raise ValueError(f"{item.id}: układ {src.crs}, oczekiwano EPSG:{LS_EPSG}")
        left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, *AOI_BBOX)
        w = from_bounds(left, bottom, right, top, transform=src.transform)
        col0 = int(np.floor(w.col_off))
        row0 = int(np.floor(w.row_off))
        col1 = int(np.ceil(w.col_off + w.width))
        row1 = int(np.ceil(w.row_off + w.height))
        window = Window(col0, row0, col1 - col0, row1 - row0)
        return window_transform(window, src.transform), (row1 - row0, col1 - col0)


def scene_window(src_transform, grid_transform, shape):
    if abs(src_transform.a - RES) > 1e-6 or abs(src_transform.e + RES) > 1e-6:
        raise ValueError(f"Nieoczekiwany rozmiar piksela: {src_transform.a}, {src_transform.e}")
    col = (grid_transform.c - src_transform.c) / RES
    row = (src_transform.f - grid_transform.f) / RES
    if abs(col - round(col)) > 1e-6 or abs(row - round(row)) > 1e-6:
        raise ValueError("Siatka sceny przesunięta względem siatki kostki")
    return Window(round(col), round(row), shape[1], shape[0])


def read_scene(item, grid_transform, shape):
    arrays = {}
    for key in ASSETS:
        with rasterio.open(planetary_computer.sign(item.assets[key].href)) as src:
            if src.crs.to_epsg() != LS_EPSG:
                raise ValueError(f"{item.id}: układ {src.crs}")
            window = scene_window(src.transform, grid_transform, shape)
            if key == "qa_pixel":
                fill = 1
            else:
                fill = src.nodata if src.nodata is not None else 0
            arrays[key] = src.read(1, window=window, boundless=True, fill_value=fill)
    return arrays


def mosaic(scenes):
    result = scenes[0]
    for other in scenes[1:]:
        take = ((result["qa_pixel"] & 1) == 1) & ((other["qa_pixel"] & 1) == 0)
        for key in ASSETS:
            result[key] = np.where(take, other[key], result[key])
    return result


def save_npz(path, arrays):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def fetch_acquisition(acquisition, scene_ids, items, grid_transform, shape):
    for attempt in range(3):
        try:
            scenes = [read_scene(items[s], grid_transform, shape) for s in scene_ids]
            break
        except rasterio.errors.RasterioIOError:
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))
    save_npz(RAW_DIR / f"{acquisition}.npz", mosaic(scenes))


def download_all(sel, items, grid_transform, shape):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    groups = sel.groupby("acquisition")["id"].apply(list)
    todo = {k: v for k, v in groups.items() if not (RAW_DIR / f"{k}.npz").exists()}
    print(f"Do pobrania: {len(todo)}, już pobrane: {len(groups) - len(todo)}")
    failures = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(fetch_acquisition, k, ids, items, grid_transform, shape): k
                   for k, ids in todo.items()}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                fut.result()
            except Exception as exc:
                failures.append((futures[fut], str(exc)[-300:]))
            if i % 10 == 0 or i == len(futures):
                print(f"  pobrano {i}/{len(futures)}, błędy: {len(failures)}")
    for acquisition, message in failures[:10]:
        print(f"  {acquisition}: {message}")
    return failures


def build_cube(sel, meta, grid_transform, shape):
    acq = (sel.groupby("acquisition")
              .agg(datetime=("datetime", "min"), platform=("platform", "first"),
                   path=("path", "first"), solar_hour=("solar_hour", "first"),
                   frac_clear=("frac_clear", "max"))
              .reset_index()
              .sort_values("datetime"))
    acq = acq[acq["acquisition"].map(lambda k: (RAW_DIR / f"{k}.npz").exists())]
    acq = acq.reset_index(drop=True)
    height, width = shape
    x = grid_transform.c + grid_transform.a * (np.arange(width) + 0.5)
    y = grid_transform.f + grid_transform.e * (np.arange(height) + 0.5)

    for start in range(0, len(acq), CUBE_BATCH):
        batch = acq.iloc[start:start + CUBE_BATCH]
        stacks = {key: [] for key in ASSETS}
        for k in batch["acquisition"]:
            with np.load(RAW_DIR / f"{k}.npz") as npz:
                for key in ASSETS:
                    stacks[key].append(npz[key])

        ds = xr.Dataset(
            {key: (("time", "y", "x"), np.stack(arrs)) for key, arrs in stacks.items()},
            coords={
                "time": batch["datetime"].dt.tz_convert(None).to_numpy(),
                "y": y,
                "x": x,
                "platform": ("time", batch["platform"].to_numpy()),
                "path": ("time", batch["path"].to_numpy()),
                "solar_hour": ("time", batch["solar_hour"].to_numpy()),
                "frac_clear": ("time", batch["frac_clear"].to_numpy()),
            },
        )
        ds = ds.chunk({"time": 1})
        for key in ASSETS:
            ds[key].attrs = meta[key]
        ds.attrs.update({
            "crs": f"EPSG:{LS_EPSG}",
            "transform": list(grid_transform)[:6],
            "source": "Landsat 8/9 Collection 2 Level-2 (Planetary Computer)",
            "aoi_bbox_wgs84": list(AOI_BBOX),
        })
        if start == 0:
            ds.to_zarr(CUBE_PATH, mode="w")
        else:
            ds.to_zarr(CUBE_PATH, append_dim="time")
        print(f"  kostka: {min(start + CUBE_BATCH, len(acq))}/{len(acq)}")

    acq.to_csv(INDEX_PATH, index=False)
    print(f"Kostka zapisana: {CUBE_PATH} ({len(acq)} akwizycji, {height}x{width} px)")


def main():
    sel = select_scenes()
    items = fetch_items(sel["id"].tolist())
    missing = set(sel["id"]) - set(items)
    if missing:
        print(f"Brak w katalogu: {len(missing)} scen, pomijam")
        sel = sel[sel["id"].isin(items)]

    first = items[sel["id"].iloc[0]]
    grid_transform, shape = reference_grid(first)
    meta = asset_metadata(first)
    print(f"Siatka: {shape[0]}x{shape[1]} px, piksel {grid_transform.a:.1f} m, "
          f"lewy górny róg ({grid_transform.c:.1f}, {grid_transform.f:.1f})")
    for key, m in meta.items():
        print(f"  {key:9s} scale={m['scale']:g} offset={m['offset']:g} nodata={m['nodata']}")

    download_all(sel, items, grid_transform, shape)
    build_cube(sel, meta, grid_transform, shape)


if __name__ == "__main__":
    main()