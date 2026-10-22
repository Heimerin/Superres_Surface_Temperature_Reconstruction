import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from scipy.ndimage import shift as nd_shift

from lst_common import (CRS, ECO_EVAL_BUFFER, ECO_TRAIN_BUFFER, grid_transform, keep_mask,
                        landsat_lst, to_eco_grid)

matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="Consolidated metadata")

ECO_CUBE = Path("data/ecostress/krakow_ecostress_70m.zarr")
LS_CUBE = Path("data/landsat/krakow_landsat_30m.zarr")
GEOLOC_CSV = Path("data/ecostress/krakow_ecostress_geoloc.csv")
BIAS_CSV = Path("inventory/cloud_edge_bias.csv")
OUT_DIR = Path("results/diurnal")
FIG_DIR = Path("figures")
YEARS = (2023, 2026)
MONTHS = (5, 9)
PIXEL_M = 70.0
MODELS = ["K0", "K2", "K2+S", "K2xS", "K2+Sd", "K2g", "K2xSg", "K2+Sdg"]
GAIN_MODELS = {"K2g", "K2xSg", "K2+Sdg"}
BASE_MODEL = {"K2g": "K2", "K2xSg": "K2xS", "K2+Sdg": "K2+Sd"}
FINAL_MODEL = "K2+Sdg"
G_PRIOR = 0.05
G_TOL = 5e-3
N_ITER_GAIN = 80
COLUMNS = ["bez_anomalii", "K0", "K2", "K2+S", "K2xS", "K2+Sd", "K2_gs", "K2xS_gs", "K2+Sd_gs",
           "K2g_gk", "K2g_gs", "K2xSg_gk", "K2xSg_gs", "K2+Sdg_gk", "K2+Sdg_gs"]
LATITUDE = 50.06
N_FOLDS = 5
SEASON_CENTER = 197
SEASON_HALF = 76
MIN_OBS = 8
COVER_GROUPS = [(0, 5), (1, 4), (2, 3)]
MIN_PER_GROUP = 2
MIN_EVAL_PX = 5000
N_ITER = 30
TOL = 1e-3
RIDGE = 1e-3
SIGMA_MIN = 0.3
ROBUST = True
HUBER_C = 1.5
SCALE_SAMPLE = 7
MAP_HOURS = [2, 6, 10, 14, 18, 22]
BIN_LABELS = ["00-04", "04-08", "08-12", "12-16", "16-20", "20-24"]
N_WORST = 3


def select(times):
    t = pd.DatetimeIndex(times)
    return ((t.year >= YEARS[0]) & (t.year <= YEARS[1])
            & (t.month >= MONTHS[0]) & (t.month <= MONTHS[1]))


def time_keys(times):
    return pd.DatetimeIndex(pd.to_datetime(times, format="ISO8601")).floor("s")


def season_feature(times):
    doy = pd.DatetimeIndex(times).dayofyear.to_numpy()
    return (doy - SEASON_CENTER) / SEASON_HALF


def load_geoloc():
    g = pd.read_csv(GEOLOC_CSV)
    keys = time_keys(g["time"])
    return dict(zip(keys, zip(g["shift_east_m"], g["shift_north_m"])))


def load_bias_table():
    df = pd.read_csv(BIAS_CSV, index_col=0)
    table = {}
    for col in ("dzień", "noc"):
        lut = np.zeros(256)
        lut[df.index.values.astype(int)] = np.nan_to_num(df[col].values)
        table[col] = lut
    return table


def shift_masked(img, east_m, north_m):
    s = (north_m / PIXEL_M, -east_m / PIXEL_M)
    valid = np.isfinite(img)
    moved = nd_shift(np.where(valid, img, 0.0), s, order=1, mode="constant", cval=0.0)
    weight = nd_shift(valid.astype(float), s, order=1, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = moved / weight
    return np.where(weight > 0.999, out, np.nan)


def load_ecostress(eco, geoloc, bias):
    idx = np.flatnonzero(select(eco.time.values) & keep_mask(eco.time.values, "ECOSTRESS"))
    keys = time_keys(eco.time.values)
    n, shape = len(idx), (eco.sizes["y"], eco.sizes["x"])
    y = np.full((n, shape[0] * shape[1]), np.nan, dtype="float32")
    w = np.zeros_like(y)
    m_eval = np.zeros(y.shape, dtype=bool)
    for k, j in enumerate(idx):
        frame = eco[["LST", "LST_err", "cloud_dist"]].isel(time=j).load()
        period = "noc" if eco.is_night.values[j] else "dzień"
        dist = frame.cloud_dist.values
        lst = frame.LST.values
        ok_train = np.isfinite(lst) & (dist >= ECO_TRAIN_BUFFER[period])
        ok_eval = np.isfinite(lst) & (dist >= ECO_EVAL_BUFFER[period])
        sigma = np.maximum(np.nan_to_num(frame.LST_err.values, nan=1.0), SIGMA_MIN)
        weight = 1.0 / (sigma ** 2 + bias[period][dist] ** 2)
        east, north = geoloc.get(keys[j], (0.0, 0.0))
        lst_s = shift_masked(np.where(ok_train, lst, np.nan), east, north)
        w_s = shift_masked(np.where(ok_train, weight, np.nan), east, north)
        e_s = shift_masked(np.where(ok_eval, 1.0, np.nan), east, north)
        y[k] = lst_s.ravel()
        w[k] = np.nan_to_num(w_s).ravel() * np.isfinite(lst_s).ravel()
        m_eval[k] = np.isfinite(e_s).ravel()
    meta = pd.DataFrame({
        "time": eco.time.values[idx],
        "hour": eco.solar_hour.values[idx],
        "night": eco.is_night.values[idx],
    })
    meta["bin"] = np.clip((meta["hour"] // 4).astype(int), 0, 5)
    meta["season"] = season_feature(meta["time"])
    return y, w, m_eval, meta, shape


def load_landsat(ls, eco):
    idx = np.flatnonzero(select(ls.time.values) & keep_mask(ls.time.values, "Landsat"))
    y = np.stack([to_eco_grid(landsat_lst(ls.isel(time=i)).values, ls, eco).ravel() for i in idx])
    return y.astype("float32"), ls.solar_hour.values[idx], season_feature(ls.time.values[idx])


def solar_weight(hours, season):
    doy = np.asarray(season, dtype=float) * SEASON_HALF + SEASON_CENTER
    decl = np.deg2rad(23.44) * np.sin(2 * np.pi * (284 + doy) / 365)
    lat = np.deg2rad(LATITUDE)
    hour_angle = 2 * np.pi * (np.asarray(hours, dtype=float) - 12) / 24
    elev = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    return np.maximum(0.0, elev)


def design(hours, season, model):
    model = BASE_MODEL.get(model, model)
    phase = 2 * np.pi * np.asarray(hours, dtype=float) / 24
    s = np.asarray(season, dtype=float)
    K = 0 if model == "K0" else 2
    cols = [np.ones_like(phase)]
    for k in range(1, K + 1):
        cols += [np.cos(k * phase), np.sin(k * phase)]
    if model in ("K2+S", "K2xS"):
        cols.append(s)
    if model == "K2xS":
        cols += [s * np.cos(phase), s * np.sin(phase)]
    if model == "K2+Sd":
        cols.append(s * solar_weight(hours, season))
    return np.stack(cols, axis=1)


def normal_matrix(w, X):
    X32 = X.astype("float32")
    P = X32.shape[1]
    A = np.empty((w.shape[1], P, P), dtype="float32")
    for i in range(P):
        for j in range(i, P):
            A[:, i, j] = A[:, j, i] = w.T @ (X32[:, i] * X32[:, j])
    return A


def coverage(valid, bins):
    bins = np.asarray(bins)
    ok = np.ones(valid.shape[1], dtype=bool)
    for group in COVER_GROUPS:
        ok &= valid[np.isin(bins, group)].sum(axis=0) >= MIN_PER_GROUP
    return ok


def scene_sums(y0, a, wg):
    S = np.zeros((len(y0), 5))
    for k in range(len(y0)):
        wk, yk, ak = (v[k].astype(np.float64) for v in (wg, y0, a))
        S[k] = [wk.sum(), (wk * yk).sum(), (wk * ak).sum(), (wk * ak * ak).sum(), (wk * yk * ak).sum()]
    return S.T


def scene_level(y0, a, wg, gain):
    sw, sy, sa, saa, sya = scene_sums(y0, a, wg)
    if not gain:
        return (sy - sa) / sw, np.ones(len(sw))
    lam = G_PRIOR * np.median(saa)
    det = sw * (saa + lam) - sa ** 2
    b = (sy * (saa + lam) - sa * (sya + lam)) / det
    g = (sw * (sya + lam) - sa * sy) / det
    return b, g


def fit(y, w, X, bins=None, gain=False, robust=ROBUST):
    valid = w > 0
    y0 = np.where(valid, y, 0.0).astype("float32")
    good = valid.sum(axis=0) >= MIN_OBS
    if bins is not None:
        good &= coverage(valid, bins)
    ok = valid & good
    reg = np.diag([0.0] + [RIDGE] * (X.shape[1] - 1))
    w_eff = w
    coef = np.zeros((y.shape[1], X.shape[1]))
    a = np.zeros_like(y0)
    b, g = scene_level(y0, a, w_eff * good, False)
    n_iter = N_ITER_GAIN if gain else N_ITER
    X32 = X.astype("float32")
    sample = ok.ravel()[::SCALE_SAMPLE]
    for it in range(n_iter):
        Xg = (X * g[:, None]).astype("float32")
        A = normal_matrix(w_eff, Xg)
        rhs = (w_eff * (y0 - b.astype("float32")[:, None])).T @ Xg
        coef[good] = np.linalg.solve(A[good] + reg, rhs[good][..., None])[..., 0]
        coef[good] -= coef[good].mean(axis=0)
        a = X32 @ coef.T.astype("float32")
        b_new, g_new = scene_level(y0, a, w_eff * good, gain)
        if gain:
            m = g_new.mean()
            g_new /= m
            coef *= m
            a *= m
        delta = np.abs(b_new - b).max()
        dg = np.abs(g_new - g).max()
        b, g = b_new, g_new
        if robust:
            pred = b.astype("float32")[:, None] + g.astype("float32")[:, None] * a
            z = np.abs(y0 - pred) * np.sqrt(w)
            scale = 1.4826 * np.median(z.ravel()[::SCALE_SAMPLE][sample])
            w_eff = (w * np.minimum(1.0, HUBER_C * scale / np.maximum(z, 1e-6))).astype("float32")
        if delta < TOL and dg < G_TOL and (not robust or it >= 3):
            break
    else:
        print(f"    uwaga: brak zbieżności po {n_iter} iteracjach (zmiana b {delta:.4f} K, zmiana g {dg:.4f})")
    return coef, b, g, good, it + 1


def typical_gain(g, bins):
    bins = np.asarray(bins)
    return np.array([np.median(g[bins == k]) if np.any(bins == k) else 1.0
                     for k in range(len(BIN_LABELS))])


def hour_bin(hours):
    return np.clip((np.asarray(hours) // 4).astype(int), 0, len(BIN_LABELS) - 1)


def scene_rmse(y, mask, a, good, gain=False):
    out = []
    for k in range(len(y)):
        ok = mask[k] & good & np.isfinite(y[k])
        if ok.sum() < MIN_EVAL_PX:
            out.append(np.nan)
            continue
        yk, ak = y[k][ok].astype(np.float64), a[k][ok].astype(np.float64)
        if gain:
            g, b = np.polyfit(ak, yk, 1)
            e = yk - b - g * ak
        else:
            e = yk - ak
            e = e - e.mean()
        out.append(float(np.sqrt((e ** 2).mean())))
    return np.array(out)


def score_columns(model):
    if model in GAIN_MODELS:
        return [(f"{model}_gk", "gk"), (f"{model}_gs", "gs")]
    if model in ("K2", "K2xS", "K2+Sd"):
        return [(model, "plain"), (f"{model}_gs", "gs")]
    return [(model, "plain")]


def score(y, mask, a, good, kind, gk, bins):
    if kind == "gk":
        return scene_rmse(y, mask, a * gk[bins][:, None], good)
    return scene_rmse(y, mask, a, good, gain=kind == "gs")


def cv_folds(meta):
    t = pd.DatetimeIndex(meta["time"])
    week = t.year.to_numpy() * 100 + t.isocalendar()["week"].to_numpy().astype(int)
    fold_of_week = {wk: i % N_FOLDS for i, wk in enumerate(np.unique(week))}
    return np.array([fold_of_week[wk] for wk in week])


def cross_validate(y, w, m_eval, meta):
    folds = cv_folds(meta)
    rmse = {m: np.full(len(meta), np.nan) for m in COLUMNS}
    for f in range(N_FOLDS):
        train, test = folds != f, folds == f
        bins = meta["bin"].values
        for model in MODELS:
            coef, _, g, good, _ = fit(y[train], w[train], design(meta["hour"][train], meta["season"][train], model),
                                      bins=bins[train], gain=model in GAIN_MODELS)
            gk = typical_gain(g, bins[train])
            a = design(meta["hour"][test], meta["season"][test], model) @ coef.T
            for col, kind in score_columns(model):
                rmse[col][test] = score(y[test], m_eval[test], a, good, kind, gk, bins[test])
        zeros = np.zeros((test.sum(), y.shape[1]), dtype="float32")
        rmse["bez_anomalii"][test] = scene_rmse(y[test], m_eval[test], zeros, good)
        print(f"  fold {f + 1}/{N_FOLDS}: {test.sum()} scen testowych")
    table = pd.DataFrame(rmse)
    table["pora"] = [BIN_LABELS[k] for k in meta["bin"]]
    summary = table.groupby("pora").median()
    summary.insert(0, "sceny", table.groupby("pora").size())
    summary.loc["średnia przedziałów"] = summary.mean()
    summary.loc["średnia przedziałów", "sceny"] = summary["sceny"].iloc[:-1].sum()
    return summary


def full_fits(y, w, meta, ls_y, ls_hours, ls_season):
    fits, ls_rmse = {}, {}
    for model in MODELS:
        coef, b, g, good, n_it = fit(y, w, design(meta["hour"], meta["season"], model),
                                     bins=meta["bin"].values, gain=model in GAIN_MODELS)
        fits[model] = (coef, b, g, good)
        print(f"  {model}: dopasowane piksele {good.mean():.1%}, iteracje {n_it}")
        gk = typical_gain(g, meta["bin"].values)
        a_ls = design(ls_hours, ls_season, model) @ coef.T
        for col, kind in score_columns(model):
            ls_rmse[col] = float(np.nanmedian(score(ls_y, np.isfinite(ls_y), a_ls, good, kind, gk,
                                                    hour_bin(ls_hours))))
    zeros = np.zeros_like(ls_y)
    ls_rmse["bez_anomalii"] = float(np.nanmedian(scene_rmse(ls_y, np.isfinite(ls_y), zeros, good)))
    return fits, ls_rmse


def residual_diagnostics(y, w, meta, model, coef, b, g, good, shape):
    a = (g[:, None] * (design(meta["hour"], meta["season"], model) @ coef.T)).astype("float32")
    rows = []
    for k in range(len(y)):
        ok = (w[k] > 0) & good
        if ok.sum() < MIN_EVAL_PX:
            continue
        e = y[k][ok] - b[k] - a[k][ok]
        rows.append({"k": k, "time": meta["time"].iloc[k], "hour": meta["hour"].iloc[k],
                     "night": bool(meta["night"].iloc[k]), "rms": float(np.sqrt(((e - e.mean()) ** 2).mean()))})
    diag = pd.DataFrame(rows).sort_values("rms", ascending=False)
    diag.drop(columns="k").to_csv(OUT_DIR / "scene_residuals.csv", index=False)
    worst = pd.concat([diag[diag["night"]].head(N_WORST), diag[~diag["night"]].head(N_WORST)])
    print(worst.drop(columns="k").round({"hour": 2, "rms": 2}).to_string(index=False))

    fig, axes = plt.subplots(2, N_WORST, figsize=(5 * N_WORST, 8), layout="constrained")
    for ax, (_, r) in zip(axes.ravel(), worst.iterrows()):
        k = int(r["k"])
        res = np.where((w[k] > 0) & good, y[k] - b[k] - a[k], np.nan).reshape(shape)
        im = ax.imshow(res, cmap="RdBu_r", vmin=-4, vmax=4)
        ax.set_title(f"{str(r['time'])[:16]}  {r['hour']:.1f} h  RMS {r['rms']:.2f} K", fontsize=9)
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="reszta [K]")
    fig.savefig(FIG_DIR / "diurnal_residuals_worst.png", dpi=130)
    plt.close(fig)


def anomaly_map(coef, good, model, hour, season, shape):
    x = design([hour], [season], model)
    return np.where(good, (x @ coef.T)[0], np.nan).reshape(shape)


def save_geotiff(path, arr, transform):
    profile = {"driver": "GTiff", "height": arr.shape[0], "width": arr.shape[1], "count": 1,
               "dtype": "float32", "crs": CRS, "transform": transform, "nodata": np.nan,
               "compress": "deflate"}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype("float32"), 1)


def plot_gain(ax, g, meta):
    night = meta["night"].to_numpy()
    for sel, color, label in ((~night, "tab:orange", "dzień"), (night, "tab:blue", "noc")):
        ax.scatter(meta["season"].to_numpy()[sel], g[sel], s=12, color=color, label=label)
    ax.axhline(1, color="black", lw=0.5)
    ax.set_xticks([-1, 0, 1], ["1 V", "16 VII", "30 IX"])
    ax.set_ylabel("wzmocnienie g")
    ax.set_title("wzmocnienie kontrastów sceny")
    ax.legend(fontsize=8)


def final_maps(model, coef, b, g, good, meta, shape, transform):
    gk = typical_gain(g, meta["bin"].values) if model in GAIN_MODELS else np.ones(len(BIN_LABELS))
    maps = {h: anomaly_map(coef, good, model, h, 0.0, shape) * gk[hour_bin(h)] for h in MAP_HOURS}
    contrast = maps[2] - maps[14]
    lim = np.nanpercentile(np.abs(np.stack(list(maps.values()))), 98)

    fig, axes = plt.subplots(2, 4, figsize=(20, 9), layout="constrained")
    axes = axes.ravel()
    for ax, h in zip(axes, MAP_HOURS):
        im = ax.imshow(maps[h], cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(f"anomalia o {h:02d}:00, połowa lipca" + (f", g = {gk[hour_bin(h)]:.2f}" if model in GAIN_MODELS else ""))
    fig.colorbar(im, ax=axes[:6].tolist(), shrink=0.6, label="anomalia LST [K]")
    clim = np.nanpercentile(np.abs(contrast), 98)
    im2 = axes[6].imshow(contrast, cmap="PuOr_r", vmin=-clim, vmax=clim)
    axes[6].set_title("anomalia 02:00 minus 14:00")
    fig.colorbar(im2, ax=axes[6], shrink=0.8, label="[K]")
    if "S" in model and model not in GAIN_MODELS:
        seasonal = anomaly_map(coef, good, model, 14, 1.0, shape) - anomaly_map(coef, good, model, 14, -1.0, shape)
        slim = np.nanpercentile(np.abs(seasonal), 98)
        im3 = axes[7].imshow(seasonal, cmap="BrBG_r", vmin=-slim, vmax=slim)
        axes[7].set_title("anomalia 14:00: 30 IX minus 1 V")
        fig.colorbar(im3, ax=axes[7], shrink=0.8, label="[K]")
    for ax in axes[:7]:
        ax.axis("off")
    if model in GAIN_MODELS:
        plot_gain(axes[7], g, meta)
    elif "S" not in model:
        axes[7].axis("off")
    fig.suptitle(f"Dobowa anomalia LST, model {model}")
    fig.savefig(FIG_DIR / "diurnal_anomaly_maps.png", dpi=130)
    plt.close(fig)

    for h in (2, 14):
        save_geotiff(OUT_DIR / f"anomaly_{h:02d}h_70m.tif", maps[h], transform)
    save_geotiff(OUT_DIR / "anomaly_02h_minus_14h_70m.tif", contrast, transform)
    np.savez_compressed(OUT_DIR / "diurnal_model.npz", model=model, coef=coef, good=good, b=b, g=g,
                        hours=meta["hour"].to_numpy(), season=meta["season"].to_numpy(),
                        times=meta["time"].to_numpy().astype("datetime64[s]"), shape=np.array(shape))
    scenes = meta[["time", "hour", "night", "season"]].assign(b=b, g=g)
    scenes.to_csv(OUT_DIR / "scene_gain.csv", index=False)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    eco = xr.open_zarr(ECO_CUBE)
    ls = xr.open_zarr(LS_CUBE)
    y, w, m_eval, meta, shape = load_ecostress(eco, load_geoloc(), load_bias_table())
    ls_y, ls_hours, ls_season = load_landsat(ls, eco)
    print(f"Sceny ECOSTRESS: {len(meta)}, Landsat (tylko do walidacji): {len(ls_y)}, "
          f"dopasowanie odporne: {ROBUST} (c = {HUBER_C})")

    print("\nWalidacja krzyżowa (całe tygodnie odkładane razem):")
    summary = cross_validate(y, w, m_eval, meta)
    fits, ls_rmse = full_fits(y, w, meta, ls_y, ls_hours, ls_season)
    summary.loc["Landsat 10-11"] = pd.Series(ls_rmse)
    summary.loc["Landsat 10-11", "sceny"] = len(ls_y)
    order = ["sceny"] + COLUMNS
    print("\nMediana RMSE anomalii na scenach testowych [K] (poziom sceny odjęty):")
    print("_gk: typowe wzmocnienie dla pory doby (mediana z treningu), _gs: wzmocnienie dopasowane do sceny testowej")
    print(summary[order].round(3).T.to_string())
    summary[order].to_csv(OUT_DIR / "validation.csv")

    coef, b, g, good = fits[FINAL_MODEL]
    print(f"\nModel końcowy: {FINAL_MODEL}")
    print(pd.Series(g).groupby(meta["night"].map({True: "noc", False: "dzień"}).to_numpy())
          .describe()[["count", "mean", "std", "min", "50%", "max"]].round(2).to_string())
    print("\nNajgorzej dopasowane sceny (3 nocne, 3 dzienne):")
    residual_diagnostics(y, w, meta, FINAL_MODEL, coef, b, g, good, shape)
    final_maps(FINAL_MODEL, coef, b, g, good, meta, shape, grid_transform(eco))


if __name__ == "__main__":
    main()