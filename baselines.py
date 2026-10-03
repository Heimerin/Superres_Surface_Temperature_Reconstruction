import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter

from lst_common import block_mean

warnings.filterwarnings("ignore", message="Consolidated metadata")

PYRAMID = Path("data/landsat/krakow_wald_pyramid.zarr")
OUT_DIR = Path("results/baselines")
FACTOR = 3
MIN_VALID = 0.3
MIN_EVAL = 2000
SAMPLES_PER_SCENE = 20000
PIXEL_M = 90.0
TIRS_SIGMA_M = 72.0
FEATURES = ["emis", "blue", "green", "red", "nir08", "swir16", "swir22",
            "ndvi", "ndbi", "mndwi", "albedo"]
INDEX_FEATURES = ["ndvi", "ndbi", "mndwi", "albedo"]
LGBM_SCENE = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31,
              "min_child_samples": 20, "verbose": -1}
LGBM_GLOBAL = {"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 63,
               "min_child_samples": 50, "colsample_bytree": 0.8, "verbose": -1}


def upsample(coarse):
    return np.repeat(np.repeat(coarse, FACTOR, axis=0), FACTOR, axis=1)


def load_scene(ds, i):
    frame = ds.isel(time=i).load()
    names = FEATURES + ["lst"]
    coarse = {n: frame[f"{n}_270"].values for n in names}
    fine = {n: frame[f"{n}_90"].values for n in names}
    return coarse, fine


def stack_features(d, names):
    return np.stack([d[n] for n in names], axis=-1)


def predict_grid(model, X):
    out = np.full(X.shape[:2], np.nan, dtype="float32")
    ok = np.isfinite(X).all(axis=-1)
    if ok.any():
        out[ok] = model.predict(X[ok])
    return out


def psf_blur(img, sigma_px, min_weight=0.99):
    mask = np.isfinite(img)
    weight = gaussian_filter(mask.astype(float), sigma_px)
    blurred = gaussian_filter(np.where(mask, img, 0.0), sigma_px)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = blurred / weight
    return np.where(weight >= min_weight, out, np.nan)


def residual_correction(pred_fine, fit_coarse, lst_coarse):
    return pred_fine + upsample(lst_coarse - fit_coarse)


def project_to_coarse(pred_fine, lst_coarse):
    return pred_fine + upsample(lst_coarse - block_mean(pred_fine, FACTOR))


def fractional_cover(ndvi, ndvi_min, ndvi_max):
    scaled = np.clip((ndvi_max - ndvi) / (ndvi_max - ndvi_min), 0, 1)
    return 1 - scaled ** 0.625


def tsharp(coarse, fine):
    lo, hi = np.nanpercentile(coarse["ndvi"], [1, 99])
    fc_coarse = fractional_cover(coarse["ndvi"], lo, hi)
    fc_fine = fractional_cover(fine["ndvi"], lo, hi)
    ok = np.isfinite(fc_coarse) & np.isfinite(coarse["lst"])
    slope, intercept = np.polyfit(fc_coarse[ok], coarse["lst"][ok], 1)
    return residual_correction(intercept + slope * fc_fine,
                               intercept + slope * fc_coarse, coarse["lst"])


def lgbm_scene(coarse, fine):
    Xc = stack_features(coarse, FEATURES)
    ok = np.isfinite(coarse["lst"]) & np.isfinite(Xc).all(axis=-1)
    model = lgb.LGBMRegressor(**LGBM_SCENE).fit(Xc[ok], coarse["lst"][ok])
    fit_coarse = predict_grid(model, Xc)
    pred_fine = predict_grid(model, stack_features(fine, FEATURES))
    return residual_correction(pred_fine, fit_coarse, coarse["lst"])


def global_features(coarse, fine):
    up = {n: upsample(coarse[n]) for n in INDEX_FEATURES + ["lst"]}
    columns = [up["lst"]] + [fine[n] for n in FEATURES] + [fine[n] - up[n] for n in INDEX_FEATURES]
    return np.stack(columns, axis=-1), up["lst"]


def train_global(ds, train_idx, rng):
    X_parts, y_parts = [], []
    for i in train_idx:
        coarse, fine = load_scene(ds, i)
        X, up = global_features(coarse, fine)
        y = fine["lst"] - up
        ok = np.isfinite(y) & np.isfinite(X).all(axis=-1)
        rows = np.flatnonzero(ok.ravel())
        if len(rows) > SAMPLES_PER_SCENE:
            rows = rng.choice(rows, SAMPLES_PER_SCENE, replace=False)
        X_parts.append(X.reshape(-1, X.shape[-1])[rows])
        y_parts.append(y.ravel()[rows])
    X_train, y_train = np.concatenate(X_parts), np.concatenate(y_parts)
    print(f"Model globalny: {len(y_train)} próbek z {len(train_idx)} scen")
    return lgb.LGBMRegressor(**LGBM_GLOBAL).fit(X_train, y_train)


def evaluate(preds, truth):
    common = np.isfinite(truth)
    for p in preds.values():
        common &= np.isfinite(p)
    if common.sum() < MIN_EVAL:
        return None
    rows = {}
    for name, p in preds.items():
        d = p[common] - truth[common]
        rows[name] = {"rmse": float(np.sqrt((d ** 2).mean())), "bias": float(d.mean())}
    return int(common.sum()), rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "preds").mkdir(exist_ok=True)
    ds = xr.open_zarr(PYRAMID)
    usable = ds.frac_valid_270.values >= MIN_VALID
    is_test = ds.is_test.values
    train_idx = np.flatnonzero(usable & ~is_test)
    test_idx = np.flatnonzero(usable & is_test)
    print(f"Sceny: trening {len(train_idx)}, test {len(test_idx)}")

    model_global = train_global(ds, train_idx, np.random.default_rng(0))

    records = []
    for i in test_idx:
        coarse, fine = load_scene(ds, i)
        X, up = global_features(coarse, fine)
        global_pred = up + predict_grid(model_global, X)
        preds = {
            "brak_wyostrzania": up,
            "tsharp": tsharp(coarse, fine),
            "lgbm_scena": lgbm_scene(coarse, fine),
            "lgbm_global": global_pred,
            "lgbm_global_proj": project_to_coarse(global_pred, coarse["lst"]),
        }
        result = evaluate(preds, fine["lst"])
        if result is None:
            continue
        n, rows = result
        blurred = {k: psf_blur(v, TIRS_SIGMA_M / PIXEL_M) for k, v in preds.items()}
        result_psf = evaluate(blurred, fine["lst"])
        date = str(ds.time.values[i])[:10]
        for name, m in rows.items():
            record = {"date": date, "n": n, "method": name, **m}
            if result_psf is not None:
                record["rmse_psf"] = result_psf[1][name]["rmse"]
                record["bias_psf"] = result_psf[1][name]["bias"]
            records.append(record)
        np.savez_compressed(OUT_DIR / "preds" / f"{date}.npz", truth=fine["lst"], **preds)

    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "per_scene.csv", index=False)
    summary = df.groupby("method").agg(
        rmse_median=("rmse", "median"), rmse_mean=("rmse", "mean"),
        bias_median=("bias", "median"), rmse_psf_median=("rmse_psf", "median"),
        scenes=("date", "nunique"),
    ).sort_values("rmse_psf_median")
    print(f"\nWyniki na {df['date'].nunique()} scenach testowych (90 m z 270 m):")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()