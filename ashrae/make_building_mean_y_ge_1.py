import argparse
from pathlib import Path

import pandas as pd


# Buildings intentionally excluded in the historical dataset build.
DEFAULT_EXCLUDE_IDS = [
    53,
    83,
    182,
    186,
    252,
    314,
    329,
    579,
    604,
    691,
    737,
    821,
    848,
    849,
    853,
    856,
    857,
    1110,
    1446,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create train.cleaned_isamu_matt.building_mean_y_ge_1.csv "
            "from train.cleaned_isamu_matt.csv."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("ashrae/train.cleaned_isamu_matt.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ashrae/train.cleaned_isamu_matt.building_mean_y_ge_1.csv"),
    )
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument(
        "--exclude-ids",
        type=int,
        nargs="*",
        default=DEFAULT_EXCLUDE_IDS,
        help="Optional explicit building_id exclusions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    sums: dict[int, float] = {}
    counts: dict[int, int] = {}

    # Pass 1: building-level mean(meter_reading)
    for chunk in pd.read_csv(
        args.input,
        usecols=["building_id", "meter_reading"],
        dtype={"building_id": "int16", "meter_reading": "float32"},
        chunksize=args.chunk_size,
    ):
        agg = chunk.groupby("building_id", observed=True)["meter_reading"].agg(["sum", "count"])
        for bid, row in agg.iterrows():
            b = int(bid)
            sums[b] = sums.get(b, 0.0) + float(row["sum"])
            counts[b] = counts.get(b, 0) + int(row["count"])

    exclude = set(args.exclude_ids or [])
    keep_ids = {
        bid
        for bid in sums
        if (sums[bid] / counts[bid]) >= args.threshold and bid not in exclude
    }

    # Pass 2: filter rows by keep_ids
    if args.output.exists():
        args.output.unlink()

    first = True
    kept_rows = 0
    cols = ["building_id", "meter", "timestamp", "meter_reading", "site_id", "square_feet"]
    dtypes = {
        "building_id": "int16",
        "meter": "int8",
        "meter_reading": "float32",
        "site_id": "int8",
        "square_feet": "float32",
    }
    for chunk in pd.read_csv(args.input, usecols=cols, dtype=dtypes, chunksize=args.chunk_size):
        out = chunk[chunk["building_id"].isin(keep_ids)]
        kept_rows += int(len(out))
        out.to_csv(args.output, mode="w" if first else "a", index=False, header=first)
        first = False

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Threshold: mean meter_reading >= {args.threshold}")
    print(f"Excluded building IDs: {sorted(exclude)}")
    print(f"Buildings kept: {len(keep_ids)}")
    print(f"Rows kept: {kept_rows}")


if __name__ == "__main__":
    main()

