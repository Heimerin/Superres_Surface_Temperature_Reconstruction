from pathlib import Path

import cdsapi

OUT_DIR = Path("data/era5land")
AREA = [50.25, 19.6, 49.85, 20.4]
VARIABLES = ["skin_temperature", "2m_temperature"]
YEARS = range(2018, 2027)
MONTHS = range(4, 11)
LAST = (2026, 9)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    for year in YEARS:
        for month in MONTHS:
            if (year, month) > LAST:
                continue
            target = OUT_DIR / f"era5land_{year}_{month:02d}.nc"
            if target.exists():
                continue
            request = {
                "variable": VARIABLES,
                "year": str(year),
                "month": f"{month:02d}",
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": AREA,
                "data_format": "netcdf",
                "download_format": "unarchived",
            }
            print(f"Pobieram {year}-{month:02d}")
            tmp = target.with_suffix(".tmp")
            client.retrieve("reanalysis-era5-land", request, str(tmp))
            tmp.replace(target)


if __name__ == "__main__":
    main()