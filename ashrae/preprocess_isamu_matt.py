import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# Community-derived site timezone offsets used in many ASHRAE solutions.
# local_time = weather_timestamp + offset_hours
DEFAULT_SITE_OFFSETS = {
    0: -5,
    1: 0,
    2: -7,
    3: -5,
    4: -8,
    5: 0,
    6: -5,
    7: -5,
    8: -5,
    9: -6,
    10: -7,
    11: -5,
    12: 0,
    13: -6,
    14: -5,
    15: -5,
}


@dataclass
class Config:
    input_dir: Path
    output_dir: Path
    constant_streak_min_len: int
    spike_rolling_window: int
    spike_mad_mult: float
    spike_abs_floor: float
    site_consensus_min_count: int
    site_consensus_min_frac: float
    use_site_consensus: bool


def _parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="ASHRAE preprocessing inspired by Isamu & Matt write-up."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("ashrae"))
    parser.add_argument("--output-dir", type=Path, default=Path("ashrae"))
    parser.add_argument("--constant-streak-min-len", type=int, default=168)
    parser.add_argument("--spike-rolling-window", type=int, default=5)
    parser.add_argument("--spike-mad-mult", type=float, default=8.0)
    parser.add_argument("--spike-abs-floor", type=float, default=5000.0)
    parser.add_argument("--site-consensus-min-count", type=int, default=8)
    parser.add_argument("--site-consensus-min-frac", type=float, default=0.05)
    parser.add_argument("--use-site-consensus", action="store_true")
    args = parser.parse_args()
    return Config(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        constant_streak_min_len=args.constant_streak_min_len,
        spike_rolling_window=args.spike_rolling_window,
        spike_mad_mult=args.spike_mad_mult,
        spike_abs_floor=args.spike_abs_floor,
        site_consensus_min_count=args.site_consensus_min_count,
        site_consensus_min_frac=args.site_consensus_min_frac,
        use_site_consensus=args.use_site_consensus,
    )


def _validate_required_inputs(input_dir: Path) -> None:
    required = ["train.csv", "weather_train.csv", "building_metadata.csv"]
    missing = [name for name in required if not (input_dir / name).exists()]
    if missing:
        missing_joined = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing required input file(s) in '{input_dir}': {missing_joined}. "
            "Place ASHRAE CSV files in that folder or pass --input-dir pointing to them."
        )


def clean_weather(weather_path: Path, out_path: Path) -> pd.DataFrame:
    weather = pd.read_csv(weather_path)
    weather["timestamp"] = pd.to_datetime(weather["timestamp"])
    weather["offset_hours"] = weather["site_id"].map(DEFAULT_SITE_OFFSETS).fillna(0).astype(int)
    weather["timestamp"] = weather["timestamp"] + pd.to_timedelta(weather["offset_hours"], unit="h")
    weather = weather.drop(columns=["offset_hours"])
    weather = weather.sort_values(["site_id", "timestamp"], kind="mergesort")

    numeric_cols = [
        "air_temperature",
        "cloud_coverage",
        "dew_temperature",
        "precip_depth_1_hr",
        "sea_level_pressure",
        "wind_direction",
        "wind_speed",
    ]
    numeric_cols = [c for c in numeric_cols if c in weather.columns]
    weather[numeric_cols] = weather.groupby("site_id", group_keys=False)[numeric_cols].apply(
        lambda g: g.interpolate(method="linear", limit_direction="both")
    )
    weather.to_csv(out_path, index=False)
    return weather


def _compute_streaming_flags(
    chunk: pd.DataFrame,
    const_state: dict[tuple[int, int], tuple[np.float32, int]],
    diff_state: dict[tuple[int, int], np.float32],
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray]:
    const_mask = np.zeros(len(chunk), dtype=bool)
    spike_mask = np.zeros(len(chunk), dtype=bool)
    for (bld, meter), idx in chunk.groupby(["building_id", "meter"], sort=False, observed=True).groups.items():
        idx_arr = np.asarray(idx, dtype=np.int64)
        vals = chunk.loc[idx_arr, "meter_reading"].to_numpy(dtype=np.float32, copy=False)
        key = (int(bld), int(meter))

        prev_const_val, prev_run_len = const_state.get(key, (np.float32(np.nan), 0))
        run = np.empty(vals.shape[0], dtype=np.int32)
        if np.isnan(prev_const_val):
            run[0] = 1
        else:
            run[0] = prev_run_len + 1 if vals[0] == prev_const_val else 1
        for i in range(1, vals.shape[0]):
            run[i] = run[i - 1] + 1 if vals[i] == vals[i - 1] else 1
        const_mask[idx_arr] = run >= cfg.constant_streak_min_len
        const_state[key] = (vals[-1], int(run[-1]))

        prev_diff_val = diff_state.get(key, np.float32(np.nan))
        diffs = np.empty(vals.shape[0], dtype=np.float32)
        if np.isnan(prev_diff_val):
            diffs[0] = 0.0
        else:
            diffs[0] = abs(vals[0] - prev_diff_val)
        if vals.shape[0] > 1:
            diffs[1:] = np.abs(np.diff(vals))
        spike_mask[idx_arr] = diffs > cfg.spike_abs_floor
        diff_state[key] = vals[-1]

    return const_mask, spike_mask


def _iter_train_chunks(train_path: Path, chunk_size: int = 250_000):
    return pd.read_csv(
        train_path,
        usecols=["building_id", "meter", "timestamp", "meter_reading"],
        dtype={
            "building_id": "int16",
            "meter": "int8",
            "meter_reading": "float32",
        },
        chunksize=chunk_size,
    )


def _attach_meta_and_basic_fixes(chunk: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    chunk["timestamp"] = pd.to_datetime(chunk["timestamp"])
    chunk = chunk.merge(meta, on="building_id", how="left", copy=False)
    site0_mask = (chunk["site_id"] == 0) & (chunk["meter"] == 0)
    chunk.loc[site0_mask, "meter_reading"] = chunk.loc[site0_mask, "meter_reading"] * 0.2931
    return chunk


def _build_consensus_keys(train_path: Path, meta: pd.DataFrame, cfg: Config) -> set[tuple[int, int]]:
    if not cfg.use_site_consensus:
        return set()

    const_state: dict[tuple[int, int], tuple[np.float32, int]] = {}
    diff_state: dict[tuple[int, int], np.float32] = {}
    agg: dict[tuple[int, int], list[int]] = {}

    for chunk in _iter_train_chunks(train_path):
        chunk = _attach_meta_and_basic_fixes(chunk, meta)
        const_mask, spike_mask = _compute_streaming_flags(chunk, const_state, diff_state, cfg)
        initial = const_mask | spike_mask

        key_df = pd.DataFrame(
            {
                "site_id": chunk["site_id"].astype("int16"),
                "ts_i64": chunk["timestamp"].astype("int64"),
                "initial": initial.astype("int8"),
            }
        )
        grouped = key_df.groupby(["site_id", "ts_i64"], observed=True).agg(
            anomaly_count=("initial", "sum"),
            total_count=("initial", "size"),
        )
        for (site_id, ts_i64), row in grouped.iterrows():
            key = (int(site_id), int(ts_i64))
            if key not in agg:
                agg[key] = [0, 0]
            agg[key][0] += int(row["anomaly_count"])
            agg[key][1] += int(row["total_count"])

    consensus: set[tuple[int, int]] = set()
    for key, (anom, total) in agg.items():
        frac = anom / total if total else 0.0
        if anom >= cfg.site_consensus_min_count and frac >= cfg.site_consensus_min_frac:
            consensus.add(key)
    return consensus


def _write_cleaned_train(
    train_path: Path,
    meta: pd.DataFrame,
    consensus: set[tuple[int, int]],
    cfg: Config,
    cleaned_train_path: Path,
    report_path: Path,
) -> None:
    const_state: dict[tuple[int, int], tuple[np.float32, int]] = {}
    diff_state: dict[tuple[int, int], np.float32] = {}
    stats = {
        "rows_total": 0,
        "anomaly_constant_streak": 0,
        "anomaly_spike": 0,
        "site_time_consensus": 0,
        "anomaly_removed": 0,
        "rows_kept": 0,
    }

    if cleaned_train_path.exists():
        cleaned_train_path.unlink()
    first_chunk = True

    for chunk in _iter_train_chunks(train_path):
        chunk = _attach_meta_and_basic_fixes(chunk, meta)
        const_mask, spike_mask = _compute_streaming_flags(chunk, const_state, diff_state, cfg)
        ts_i64 = chunk["timestamp"].astype("int64").to_numpy()
        site_arr = chunk["site_id"].astype("int16").to_numpy()
        consensus_mask = np.fromiter(
            ((int(site_arr[i]), int(ts_i64[i])) in consensus for i in range(len(chunk))),
            dtype=bool,
            count=len(chunk),
        )
        removed = const_mask | spike_mask | consensus_mask
        kept = ~removed

        stats["rows_total"] += int(len(chunk))
        stats["anomaly_constant_streak"] += int(const_mask.sum())
        stats["anomaly_spike"] += int(spike_mask.sum())
        stats["site_time_consensus"] += int(consensus_mask.sum())
        stats["anomaly_removed"] += int(removed.sum())
        stats["rows_kept"] += int(kept.sum())

        out_chunk = chunk.loc[kept, ["building_id", "meter", "timestamp", "meter_reading", "site_id", "square_feet"]]
        out_chunk.to_csv(cleaned_train_path, mode="w" if first_chunk else "a", index=False, header=first_chunk)
        first_chunk = False

    report_df = pd.DataFrame({"rule": list(stats.keys()), "row_count": list(stats.values())})
    report_df.to_csv(report_path, index=False)


def main() -> None:
    cfg = _parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    _validate_required_inputs(cfg.input_dir)

    train_path = cfg.input_dir / "train.csv"
    weather_path = cfg.input_dir / "weather_train.csv"
    bmeta_path = cfg.input_dir / "building_metadata.csv"

    cleaned_weather_path = cfg.output_dir / "weather_train.cleaned.csv"
    cleaned_train_path = cfg.output_dir / "train.cleaned_isamu_matt.csv"
    report_path = cfg.output_dir / "train.cleaned_isamu_matt.anomaly_report.csv"

    print("Loading building metadata...")
    building = pd.read_csv(
        bmeta_path,
        usecols=["building_id", "site_id", "square_feet"],
        dtype={
            "building_id": "int16",
            "site_id": "int8",
            "square_feet": "float32",
        },
    )

    print("Cleaning weather (timezone shift + interpolation)...")
    clean_weather(weather_path, cleaned_weather_path)

    print("Pass 1/2: finding site-time consensus anomalies...")
    consensus = _build_consensus_keys(train_path, building, cfg)
    if cfg.use_site_consensus:
        print(f"Consensus keys found: {len(consensus):,}")
    else:
        print("Site-time consensus disabled.")

    print("Pass 2/2: filtering train and writing cleaned file...")
    _write_cleaned_train(train_path, building, consensus, cfg, cleaned_train_path, report_path)

    print("Done.")
    print(f"Saved: {cleaned_train_path}")
    print(f"Saved: {cleaned_weather_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
