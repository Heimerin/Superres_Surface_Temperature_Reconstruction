import warnings
from pathlib import Path

import numpy as np
import xarray as xr

from lst_common import block_mean, psf_blur

warnings.filterwarnings("ignore", message="Consolidated metadata")

PYRAMID = Path("data/landsat/krakow_wald_pyramid.zarr")
FACTOR = 3
PIXEL_M = 90.0
TIRS_SIGMA_M = 72.0


def main():
    ds = xr.open_zarr(PYRAMID)
    sigma_truth = TIRS_SIGMA_M / PIXEL_M
    sigma_total = FACTOR * sigma_truth
    sigma_add = float(np.sqrt(sigma_total ** 2 - sigma_truth ** 2))
    print(f"Sigma prawdy {sigma_truth:.2f} px, docelowa {sigma_total:.2f} px, "
          f"dodatkowe rozmycie {sigma_add:.2f} px ({sigma_add * PIXEL_M:.0f} m)")

    n = ds.sizes["time"]
    out = np.empty((n, ds.sizes["y270"], ds.sizes["x270"]), dtype="float32")
    for i in range(n):
        truth = ds["lst_90"].isel(time=i).values
        out[i] = block_mean(psf_blur(truth, sigma_add), FACTOR)

    new = xr.Dataset(
        {"lst_270psf": (("time", "y270", "x270"), out)},
        coords={"time": ds.time, "y270": ds.y270, "x270": ds.x270},
    )
    new["lst_270psf"].attrs = {
        "description": "lst_90 rozmyte dodatkowa PSF gaussowska i usrednione blokami 3x3",
        "sigma_add_px90": sigma_add,
    }
    new.attrs = dict(ds.attrs)
    new.chunk({"time": 1}).to_zarr(PYRAMID, mode="a")

    old = ds["lst_270"].values
    diff = out - old
    print(f"Ważne piksele 270 m: dawniej {np.isfinite(old).mean():.1%}, teraz {np.isfinite(out).mean():.1%}")
    print(f"Różnica względem średniej blokowej: RMS {np.sqrt(np.nanmean(diff ** 2)):.3f} K")


if __name__ == "__main__":
    main()