from pathlib import Path

import numpy as np
import pandas as pd
from affine import Affine
from rasterio.warp import Resampling, reproject
from scipy.ndimage import gaussian_filter, sobel

CRS = "EPSG:32634"
LS_BAD_BITS = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)
ECO_TRAIN_BUFFER = {"noc": 5, "dzień": 10}
ECO_EVAL_BUFFER = {"noc": 13, "dzień": 25}
QC_OUTLIERS = Path("results/qc_scene_outliers.csv")
BAD_PERIODS = {"ECOSTRESS": [("2024-06-25", "2024-07-04")]}


def decode(da):
    data = da.astype("float32")
    nodata = da.attrs.get("nodata")
    if nodata is not None:
        data = data.where(da != nodata)
    return data * da.attrs["scale"] + da.attrs["offset"]


def grid_transform(ds):
    x = ds.x.values
    y = ds.y.values
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    return Affine(dx, 0.0, x[0] - dx / 2, 0.0, dy, y[0] - dy / 2)


def landsat_lst(frame):
    good = (frame.qa_pixel & LS_BAD_BITS) == 0
    return decode(frame.lwir11).where(good)


def to_eco_grid(arr, ls, eco, min_coverage=0.99):
    kwargs = {
        "src_transform": grid_transform(ls),
        "src_crs": CRS,
        "dst_transform": grid_transform(eco),
        "dst_crs": CRS,
        "resampling": Resampling.average,
    }
    shape = (eco.sizes["y"], eco.sizes["x"])
    mean = np.full(shape, np.nan, dtype="float32")
    reproject(arr.astype("float32"), mean, src_nodata=np.nan, dst_nodata=np.nan, **kwargs)
    coverage = np.zeros(shape, dtype="float32")
    reproject(np.isfinite(arr).astype("float32"), coverage, **kwargs)
    return np.where(coverage >= min_coverage, mean, np.nan)


def find_pairs(ls, eco, max_minutes):
    ls_times = pd.to_datetime(ls.time.values)
    eco_times = pd.to_datetime(eco.time.values)
    pairs = []
    for i, t in enumerate(ls_times):
        offsets = (eco_times - t).total_seconds()
        j = int(np.argmin(np.abs(offsets)))
        if abs(offsets[j]) <= max_minutes * 60:
            pairs.append((i, j, offsets[j] / 60))
    return pairs


def gradient_magnitude(img):
    return np.hypot(sobel(img, axis=0), sobel(img, axis=1))


def shifted_pair(a, b, dy, dx):
    h, w = a.shape
    a_part = a[max(0, -dy):min(h, h - dy), max(0, -dx):min(w, w - dx)]
    b_part = b[max(0, dy):min(h, h + dy), max(0, dx):min(w, w + dx)]
    return a_part, b_part


def correlation(a, b, min_common):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < min_common:
        return np.nan
    return np.corrcoef(a[ok], b[ok])[0, 1]


def parabola_offset(values, k):
    if k == 0 or k == len(values) - 1:
        return 0.0
    left, center, right = values[k - 1], values[k], values[k + 1]
    denom = left - 2 * center + right
    if not np.isfinite(denom) or denom == 0:
        return 0.0
    return 0.5 * (left - right) / denom


def register(reference, image, max_shift, min_common):
    shifts = range(-max_shift, max_shift + 1)
    r = np.full((len(shifts), len(shifts)), np.nan)
    for iy, dy in enumerate(shifts):
        for ix, dx in enumerate(shifts):
            r[iy, ix] = correlation(*shifted_pair(reference, image, dy, dx), min_common)
    if np.all(np.isnan(r)):
        return None
    iy, ix = np.unravel_index(np.nanargmax(r), r.shape)
    center = max_shift
    return {
        "dy_px": shifts[iy] + parabola_offset(r[:, ix], iy),
        "dx_px": shifts[ix] + parabola_offset(r[iy, :], ix),
        "r0": r[center, center],
        "r_best": r[iy, ix],
        "r_median": np.nanmedian(r),
        "at_edge": iy in (0, len(shifts) - 1) or ix in (0, len(shifts) - 1),
    }


TIRS_K = {8: (774.8853, 1321.0789), 9: (799.0284, 1329.2405)}


def planck_radiance(temperature, platform):
    k1, k2 = TIRS_K[platform]
    return k1 / np.expm1(k2 / temperature)


def planck_temperature(radiance, platform):
    k1, k2 = TIRS_K[platform]
    return k2 / np.log(k1 / radiance + 1)


def block_mean(arr, factor):
    h = arr.shape[0] // factor * factor
    w = arr.shape[1] // factor * factor
    blocks = arr[:h, :w].reshape(h // factor, factor, w // factor, factor)
    return blocks.mean(axis=(1, 3))


def aggregate_lst(lst, emis, drad, platform, factor):
    surface = emis * planck_radiance(lst, platform) + (1 - emis) * drad
    surface_c = block_mean(surface, factor)
    emis_c = block_mean(emis, factor)
    drad_c = block_mean(drad, factor)
    emitted = (surface_c - (1 - emis_c) * drad_c) / emis_c
    return planck_temperature(emitted, platform), emis_c, drad_c


def normalized_difference(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        nd = (a - b) / (a + b)
    return np.where(np.isfinite(nd), nd, np.nan)


def spectral_indices(refl):
    return {
        "ndvi": normalized_difference(refl["nir08"], refl["red"]),
        "ndbi": normalized_difference(refl["swir16"], refl["nir08"]),
        "mndwi": normalized_difference(refl["green"], refl["swir16"]),
        "albedo": (0.356 * refl["blue"] + 0.130 * refl["red"] + 0.373 * refl["nir08"]
                   + 0.085 * refl["swir16"] + 0.072 * refl["swir22"] - 0.0018),
    }


def psf_blur(img, sigma_px, min_weight=0.99):
    mask = np.isfinite(img)
    weight = gaussian_filter(mask.astype(float), sigma_px)
    blurred = gaussian_filter(np.where(mask, img, 0.0), sigma_px)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = blurred / weight
    return np.where(weight >= min_weight, out, np.nan)


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


def excluded_times(sensor):
    if not QC_OUTLIERS.exists():
        return []
    d = pd.read_csv(QC_OUTLIERS)
    times = pd.to_datetime(d.loc[d["sensor"] == sensor, "time"], format="ISO8601")
    return list(times.dt.floor("s"))


def keep_mask(times, sensor):
    t = pd.DatetimeIndex(times).floor("s")
    keep = ~t.isin(excluded_times(sensor))
    for start, end in BAD_PERIODS.get(sensor, []):
        keep &= ~((t >= pd.Timestamp(start)) & (t < pd.Timestamp(end)))
    return keep