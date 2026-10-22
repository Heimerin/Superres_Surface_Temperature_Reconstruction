import argparse
import copy
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr

from lst_common import keep_mask, psf_blur, ring_index, spectrum, valid_tiles

warnings.filterwarnings("ignore", message="Consolidated metadata")

PYRAMID = Path("data/landsat/krakow_wald_pyramid.zarr")
BASELINE_PREDS = {"lst_270": Path("results/baselines/preds"),
                  "lst_270psf": Path("results/baselines_psf/preds")}
OUT_DIR = Path("results/unet")
FACTOR = 3
PIXEL_M = 90.0
MIN_VALID = 0.3
VAL_YEAR = 2024
PATCH = 96
MIN_PATCH_VALID = 0.5
FEATURES = ["emis", "blue", "green", "red", "nir08", "swir16", "swir22",
            "ndvi", "ndbi", "mndwi", "albedo"]
INDEX_FEATURES = ["ndvi", "ndbi", "mndwi", "albedo"]
GUIDE_CHANNELS = [8, 9, 10, 11]
TILE = 64
COARSE_M = 270.0


def upsample(a):
    return np.repeat(np.repeat(a, FACTOR, axis=0), FACTOR, axis=1)


def load_scene(ds, i, coarse_var):
    frame = ds.isel(time=i).load()
    up_lst = upsample(frame[coarse_var].values)
    channels = [up_lst - np.nanmean(frame[coarse_var].values)]
    channels += [frame[f"{f}_90"].values for f in FEATURES]
    channels += [frame[f"{f}_90"].values - upsample(frame[f"{f}_270"].values) for f in INDEX_FEATURES]
    return {
        "date": str(frame.time.values)[:10],
        "x": np.stack(channels).astype("float32"),
        "up": up_lst.astype("float32"),
        "truth": frame["lst_90"].values.astype("float32"),
    }


def channel_stats(scenes):
    n_ch = scenes[0]["x"].shape[0]
    total, total_sq, count = np.zeros(n_ch), np.zeros(n_ch), np.zeros(n_ch)
    for s in scenes:
        x = s["x"].reshape(n_ch, -1).astype("float64")
        ok = np.isfinite(x)
        total += np.where(ok, x, 0).sum(axis=1)
        total_sq += np.where(ok, x ** 2, 0).sum(axis=1)
        count += ok.sum(axis=1)
    mean = total / count
    std = np.sqrt(total_sq / count - mean ** 2)
    return mean.astype("float32"), np.maximum(std, 1e-6).astype("float32")


def prepare(scenes, mean, std):
    for s in scenes:
        x = (s.pop("x") - mean[:, None, None]) / std[:, None, None]
        valid_in = np.isfinite(x).all(axis=0)
        x = np.where(valid_in, x, 0.0)
        s["input"] = np.concatenate([x, valid_in[None]]).astype("float32")
        s["valid_in"] = valid_in
        s["target"] = s["truth"] - s["up"]
        s["valid"] = valid_in & np.isfinite(s["target"])


def sample_batch(scenes, batch, rng, device):
    inputs, targets, masks, ups = [], [], [], []
    while len(inputs) < batch:
        s = scenes[rng.integers(len(scenes))]
        h, w = s["valid"].shape
        y0 = rng.integers(0, (h - PATCH) // FACTOR + 1) * FACTOR
        x0 = rng.integers(0, (w - PATCH) // FACTOR + 1) * FACTOR
        win = (slice(y0, y0 + PATCH), slice(x0, x0 + PATCH))
        mask = s["valid"][win]
        if mask.mean() < MIN_PATCH_VALID:
            continue
        inputs.append(s["input"][:, win[0], win[1]])
        targets.append(np.nan_to_num(s["target"][win]))
        masks.append(mask.astype("float32"))
        ups.append(np.nan_to_num(s["up"][win]))
    to_t = lambda a: torch.from_numpy(np.stack(a)).to(device)
    return to_t(inputs), to_t(targets), to_t(masks), to_t(ups)


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
    )


class UNet(nn.Module):
    def __init__(self, cin, base=32):
        super().__init__()
        self.enc1 = conv_block(cin, base)
        self.enc2 = conv_block(base, base * 2)
        self.enc3 = conv_block(base * 2, base * 4)
        self.mid = conv_block(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = conv_block(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        m = self.mid(F.max_pool2d(e3, 2))
        d3 = self.dec3(torch.cat([self.up3(m), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)[:, 0]


def block_project(r):
    block_mean = F.avg_pool2d(r[:, None], FACTOR)
    return r - F.interpolate(block_mean, scale_factor=FACTOR, mode="nearest")[:, 0]


def gaussian_kernel(sigma_px, device):
    radius = int(np.ceil(3 * sigma_px))
    t = torch.arange(-radius, radius + 1, dtype=torch.float32)
    g = torch.exp(-0.5 * (t / sigma_px) ** 2)
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None].to(device), radius


def blur(img, kernel, radius):
    padded = F.pad(img[:, None], (radius, radius, radius, radius), mode="replicate")
    return F.conv2d(padded, kernel)[:, 0]


def erode(mask, radius):
    size = 2 * radius + 1
    out = 1 - F.max_pool2d(1 - mask[:, None], size, stride=1, padding=radius)[:, 0]
    out[:, :radius] = 0
    out[:, -radius:] = 0
    out[:, :, :radius] = 0
    out[:, :, -radius:] = 0
    return out


def masked_mse(pred, target, mask):
    return ((pred - target) ** 2 * mask).sum() / mask.sum().clamp(min=1)


def edge_aware_tv(r, guide, mask):
    dxr = (r[:, :, 1:] - r[:, :, :-1]).abs()
    dyr = (r[:, 1:] - r[:, :-1]).abs()
    dxg = (guide[:, :, :, 1:] - guide[:, :, :, :-1]).abs().mean(dim=1)
    dyg = (guide[:, :, 1:] - guide[:, :, :-1]).abs().mean(dim=1)
    g0 = torch.cat([dxg.flatten(), dyg.flatten()]).median().detach() + 1e-6
    mx = mask[:, :, 1:] * mask[:, :, :-1]
    my = mask[:, 1:] * mask[:, :-1]
    total = (dxr * torch.exp(-dxg / g0) * mx).sum() + (dyr * torch.exp(-dyg / g0) * my).sum()
    return total / (mx.sum() + my.sum()).clamp(min=1)


@torch.no_grad()
def predict_scene(model, s, project, device):
    _, h, w = s["input"].shape
    inp = torch.from_numpy(s["input"])[None].to(device)
    inp = F.pad(inp, (0, (-w) % 8, 0, (-h) % 8))
    r = model(inp)[:, :h, :w]
    if project:
        r = block_project(r)
    r = r[0].cpu().numpy()
    return np.where(s["valid_in"], s["up"] + r, np.nan)


def scene_score(pred, truth, sigma_px, use_psf):
    p = psf_blur(pred, sigma_px) if use_psf else pred
    ok = np.isfinite(p) & np.isfinite(truth)
    return float(np.sqrt(((p[ok] - truth[ok]) ** 2).mean()))


def train(model, train_scenes, val_scenes, args, project, device):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps)
    sigma_px = args.sigma_m / PIXEL_M
    kernel, radius = gaussian_kernel(sigma_px, device)
    rng = np.random.default_rng(0)
    best_score, best_state = np.inf, None

    for step in range(1, args.steps + 1):
        model.train()
        inp, target, mask, up = sample_batch(train_scenes, args.batch, rng, device)
        r = model(inp)
        if project:
            r = block_project(r)
        if args.variant == "A":
            loss = masked_mse(r, target, mask)
        else:
            sensor = blur(up, kernel, radius) - up + blur(r, kernel, radius)
            loss = masked_mse(sensor, target, erode(mask, radius))
            loss = loss + args.lambda_reg * edge_aware_tv(r, inp[:, GUIDE_CHANNELS], mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            scores = [scene_score(predict_scene(model, s, project, device), s["truth"],
                                  sigma_px, use_psf=args.variant == "B") for s in val_scenes]
            score = float(np.median(scores))
            print(f"  krok {step}: strata {loss.item():.4f}, walidacja {score:.3f} K")
            if score < best_score:
                best_score, best_state = score, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return best_score


def detail_power_ratio(pairs):
    window = np.outer(np.hanning(TILE), np.hanning(TILE))
    rings = ring_index(TILE)
    n_rings = TILE // 2
    freq = np.arange(n_rings) / (TILE * PIXEL_M) * 1000
    band = freq > 1000 / (2 * COARSE_M)
    power_pred, power_truth = 0.0, 0.0
    for pred, truth in pairs:
        for y, x in valid_tiles(np.stack([truth, pred]), TILE):
            f_truth = spectrum(truth[y:y + TILE, x:x + TILE], window)
            f_pred = spectrum(pred[y:y + TILE, x:x + TILE], window)
            power_truth += np.bincount(rings, np.abs(f_truth) ** 2)[:n_rings][band].sum()
            power_pred += np.bincount(rings, np.abs(f_pred) ** 2)[:n_rings][band].sum()
    return power_pred / power_truth if power_truth > 0 else np.nan


def validation_summary(model, val_scenes, args, project, device):
    sigma_px = args.sigma_m / PIXEL_M
    preds = [predict_scene(model, s, project, device) for s in val_scenes]
    truths = [s["truth"] for s in val_scenes]
    summary = {
        "name": args.variant + args.tag,
        "variant": args.variant,
        "lambda_reg": args.lambda_reg,
        "sigma_m": args.sigma_m,
        "project": project,
        "coarse": args.coarse,
        "val_rmse": float(np.median([scene_score(p, t, sigma_px, False) for p, t in zip(preds, truths)])),
        "val_rmse_psf": float(np.median([scene_score(p, t, sigma_px, True) for p, t in zip(preds, truths)])),
        "val_detail_power": float(detail_power_ratio(list(zip(preds, truths)))),
    }
    with open(OUT_DIR / f"val_{summary['name']}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Walidacja: RMSE {summary['val_rmse']:.3f} K, RMSE z PSF {summary['val_rmse_psf']:.3f} K, "
          f"moc szczegółu / prawda {summary['val_detail_power']:.2f}")


def evaluate_test(model, test_scenes, args, project, device):
    sigma_px = args.sigma_m / PIXEL_M
    name = args.variant + args.tag
    pred_dir = OUT_DIR / "preds" / name
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in test_scenes:
        pred = predict_scene(model, s, project, device)
        sensor = psf_blur(pred, sigma_px)
        mask = np.isfinite(s["truth"]) & np.isfinite(pred)
        baseline_path = BASELINE_PREDS[args.coarse] / f"{s['date']}.npz"
        if baseline_path.exists():
            with np.load(baseline_path) as npz:
                for key in npz.files:
                    mask &= np.isfinite(npz[key])
        truth = np.where(mask, s["truth"], np.nan)
        rows.append({"date": s["date"],
                     "rmse": scene_score(np.where(mask, pred, np.nan), truth, sigma_px, False),
                     "rmse_psf": scene_score(np.where(mask, pred, np.nan), truth, sigma_px, True)})
        np.savez_compressed(pred_dir / f"{s['date']}.npz", truth=s["truth"],
                            **{f"unet_{name}": pred, f"unet_{name}_sensor": sensor})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / f"test_{name}.csv", index=False)
    print(f"\nTest ({len(df)} scen): RMSE mediana {df['rmse'].median():.3f} K, "
          f"RMSE z PSF mediana {df['rmse_psf'].median():.3f} K")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["A", "B"], required=True)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sigma-m", type=float, default=72.0)
    parser.add_argument("--lambda-reg", type=float, default=0.02)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--project", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tag", default="")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--coarse", default="lst_270", choices=["lst_270", "lst_270psf"])
    args = parser.parse_args()
    if args.project is None:
        project = args.variant == "A" and args.coarse == "lst_270"
    else:
        project = args.project

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ds = xr.open_zarr(PYRAMID)
    years = pd.DatetimeIndex(ds.time.values).year.to_numpy()
    usable = (ds.frac_valid_270.values >= MIN_VALID) & keep_mask(ds.time.values, "Landsat")
    is_test = ds.is_test.values
    split = {
        "train": np.flatnonzero(usable & ~is_test & (years != VAL_YEAR)),
        "val": np.flatnonzero(usable & ~is_test & (years == VAL_YEAR)),
        "test": np.flatnonzero(usable & is_test),
    }
    print(f"Wariant {args.variant}{args.tag}, wejście: {args.coarse}, projekcja: {project}, "
          f"lambda: {args.lambda_reg}, urządzenie: {device}")
    print({k: len(v) for k, v in split.items()})

    scenes = {k: [load_scene(ds, i, args.coarse) for i in v] for k, v in split.items()}
    mean, std = channel_stats(scenes["train"])
    for part in scenes.values():
        prepare(part, mean, std)

    model = UNet(cin=scenes["train"][0]["input"].shape[0]).to(device)
    print(f"Parametry sieci: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} mln")
    ckpt_path = OUT_DIR / f"unet_{args.variant}{args.tag}.pt"
    if args.eval_only:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
    else:
        best = train(model, scenes["train"], scenes["val"], args, project, device)
        print(f"Najlepsza walidacja: {best:.3f} K")
        torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std, "args": vars(args)},
                   ckpt_path)
    model.eval()
    validation_summary(model, scenes["val"], args, project, device)
    evaluate_test(model, scenes["test"], args, project, device)


if __name__ == "__main__":
    main()