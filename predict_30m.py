import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

from lst_common import LS_BAD_BITS, decode, spectral_indices
from unet_train import FEATURES, INDEX_FEATURES, UNet, upsample

warnings.filterwarnings("ignore", message="Consolidated metadata")

LS_CUBE = Path("data/landsat/krakow_landsat_30m.zarr")
PYRAMID = Path("data/landsat/krakow_wald_pyramid.zarr")
MODEL_DIR = Path("results/unet")
RAW_DIR = Path("results/maps30/raw")
FACTOR = 3
BANDS = ["blue", "green", "red", "nir08", "swir16", "swir22"]
TILE = 384
MARGIN = 48


def load_model(name, device):
    ckpt = torch.load(MODEL_DIR / f"unet_{name}.pt", map_location=device, weights_only=False)
    model = UNet(cin=len(ckpt["mean"]) + 1).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["mean"], ckpt["std"]


def build_inputs(ls_frame, pyr_frame):
    h, w = pyr_frame.sizes["y90"] * FACTOR, pyr_frame.sizes["x90"] * FACTOR
    frame = ls_frame.isel(y=slice(0, h), x=slice(0, w))
    raw = {"emis": decode(frame.emis).values, **{b: decode(frame[b]).values for b in BANDS}}
    product = decode(frame.lwir11).values
    clear = (frame.qa_pixel.values & LS_BAD_BITS) == 0
    for arr in list(raw.values()) + [product]:
        clear &= np.isfinite(arr)
    fine = {k: np.where(clear, v, np.nan) for k, v in raw.items()}
    fine.update(spectral_indices(fine))

    lst_90 = pyr_frame["lst_90"].values
    up_lst = upsample(lst_90)
    channels = [up_lst - np.nanmean(lst_90)]
    channels += [fine[f] for f in FEATURES]
    channels += [fine[f] - upsample(pyr_frame[f"{f}_90"].values) for f in INDEX_FEATURES]
    extras = {"lst_90": lst_90, "product": np.where(clear, product, np.nan),
              "red": fine["red"], "green": fine["green"], "blue": fine["blue"]}
    return np.stack(channels).astype("float32"), up_lst, extras


def normalize(x, mean, std):
    x = (x - mean[:, None, None]) / std[:, None, None]
    valid = np.isfinite(x).all(axis=0)
    x = np.where(valid, x, 0.0)
    return np.concatenate([x, valid[None]]).astype("float32"), valid


@torch.no_grad()
def predict_tiled(model, inp, device):
    _, h, w = inp.shape
    step = TILE - 2 * MARGIN
    ph, pw = (-h) % step, (-w) % step
    padded = np.pad(inp, ((0, 0), (MARGIN, MARGIN + ph), (MARGIN, MARGIN + pw)))
    out = np.zeros((h + ph, w + pw), dtype="float32")
    for y in range(0, h + ph, step):
        for x in range(0, w + pw, step):
            tile = torch.from_numpy(padded[:, y:y + TILE, x:x + TILE])[None].to(device)
            r = model(tile)[0].cpu().numpy()
            out[y:y + step, x:x + step] = r[MARGIN:MARGIN + step, MARGIN:MARGIN + step]
    return out[:h, :w]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["A_psf", "B_psf"])
    parser.add_argument("--n-scenes", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ls = xr.open_zarr(LS_CUBE)
    pyr = xr.open_zarr(PYRAMID)
    models = {name: load_model(name, device) for name in args.models}

    months = pd.DatetimeIndex(pyr.time.values).month
    candidates = np.flatnonzero(pyr.is_test.values & (months >= 6) & (months <= 8))
    chosen = candidates[np.argsort(pyr.frac_valid_270.values[candidates])[::-1][:args.n_scenes]]

    for i in chosen:
        t = pyr.time.values[i]
        date = str(t)[:10]
        x, up_lst, extras = build_inputs(ls.sel(time=t).load(), pyr.isel(time=i).load())
        preds = {}
        for name, (model, mean, std) in models.items():
            inp, valid = normalize(x, mean, std)
            preds[f"model_{name}"] = np.where(valid, up_lst + predict_tiled(model, inp, device), np.nan)
        np.savez_compressed(RAW_DIR / f"{date}.npz", **extras, **preds)
        print(f"  {date}: predykcje zapisane")


if __name__ == "__main__":
    main()