import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path("results/unet")
OUT_PATH = Path("figures/l_curve.png")
SIGMA_M = 72.0


def main():
    rows = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("val_*.json"))]
    df = pd.DataFrame(rows)
    if "coarse" not in df:
        df["coarse"] = "lst_270"
    df["coarse"] = df["coarse"].fillna("lst_270")
    df = df.sort_values(["coarse", "variant", "lambda_reg"])
    columns = ["name", "coarse", "variant", "lambda_reg", "val_rmse", "val_rmse_psf", "val_detail_power"]
    print(df[columns].round(3).to_string(index=False))

    curve = df[(df["variant"] == "B") & (df["sigma_m"] == SIGMA_M) & (~df["project"].astype(bool))
               & (df["coarse"] == "lst_270")]
    curve = curve.sort_values("lambda_reg")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(curve["val_detail_power"], curve["val_rmse_psf"], marker="o", label="wariant B")
    for _, r in curve.iterrows():
        ax.annotate(f"λ={r['lambda_reg']:g}", (r["val_detail_power"], r["val_rmse_psf"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.axvline(1.0, color="gray", ls="--", lw=0.8, label="moc szczegółu prawdy")
    ax.set_xscale("log")
    ax.set_xlabel("moc szczegółu predykcji / moc szczegółu prawdy (walidacja)")
    ax.set_ylabel("RMSE z PSF na walidacji [K]")
    ax.legend(fontsize=8)
    OUT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":  
    main()