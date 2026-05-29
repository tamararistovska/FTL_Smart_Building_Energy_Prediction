"""
run_tree_baselines.py

Run non-deep-learning baselines on existing notebook splits:
- Centralized global model (train on pooled source/warm clients)
- Local-only model (one model per client)
for:
- XGBoost
- LightGBM

Expected globals from notebook:
- client_data: dict[bid] -> {"train": df, "test": df, ...}
- usable_bids: list of building ids
- cold_set: set of cold-start building ids (optional; defaults empty)
- source_bids: list of warm/source building ids (optional; defaults usable_bids)
- TARGET: target column name (e.g., "y")
- FEATURES or (NUM_FEATURES + CAT_FEATURES)

Outputs in globals:
- df_local_xgb, df_central_xgb
- df_local_lgbm, df_central_lgbm

Saved CSVs:
- saved_results/{model_label}_tree_centralized_xgb_seed{seed}.csv
- saved_results/{model_label}_tree_local_only_xgb_seed{seed}.csv
- saved_results/{model_label}_tree_centralized_lgbm_seed{seed}.csv
- saved_results/{model_label}_tree_local_only_lgbm_seed{seed}.csv
- saved_results/{model_label}_tree_baseline_summary_seed{seed}.csv
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---- model imports with clear guidance if missing ----
try:
    from xgboost import XGBRegressor
except Exception as e:
    raise ImportError(
        "XGBoost is not installed. Install in your notebook env with: pip install xgboost"
    ) from e

try:
    from lightgbm import LGBMRegressor
except Exception as e:
    raise ImportError(
        "LightGBM is not installed. Install in your notebook env with: pip install lightgbm"
    ) from e


def _resolve_features(features: list[str] | None = None) -> list[str]:
    if features is not None and len(features) > 0:
        return list(features)

    if "FEATURES" in globals() and isinstance(FEATURES, (list, tuple)) and len(FEATURES) > 0:
        return list(FEATURES)

    num_feats = globals().get("NUM_FEATURES")
    cat_feats = globals().get("CAT_FEATURES")
    if isinstance(num_feats, (list, tuple)) and isinstance(cat_feats, (list, tuple)):
        feats = list(num_feats) + list(cat_feats)
        if len(feats) > 0:
            return feats

    raise RuntimeError("Could not resolve FEATURES. Define FEATURES or NUM_FEATURES/CAT_FEATURES first.")


def _metrics_from_log_target(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> tuple[float, float, float, float]:
    mae = float(mean_absolute_error(y_true_log, y_pred_log))
    rmse = float(math.sqrt(mean_squared_error(y_true_log, y_pred_log)))

    y_true_raw = np.expm1(y_true_log)
    y_pred_raw = np.expm1(y_pred_log)

    rmse_raw = float(math.sqrt(mean_squared_error(y_true_raw, y_pred_raw)))
    y_mean_raw = float(np.mean(y_true_raw))
    cvrmse = float((rmse_raw / y_mean_raw * 100.0) if y_mean_raw != 0 else 0.0)

    denom = float(np.sum(np.abs(y_true_raw)))
    wape = float((np.sum(np.abs(y_true_raw - y_pred_raw)) / denom * 100.0) if denom != 0 else 0.0)
    return mae, rmse, cvrmse, wape


def _fit_model(kind: str, X: pd.DataFrame, y: pd.Series, seed: int) -> object:
    if kind == "xgb":
        model = XGBRegressor(
            n_estimators=400,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
        )
    elif kind == "lgbm":
        model = LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            objective="regression",
        )
    else:
        raise ValueError(f"Unsupported model kind: {kind}")

    model.fit(X, y)
    return model


def _per_building_eval_with_model(model: object, bids: list, client_data: dict, features: list[str], target: str, cold_set: set) -> pd.DataFrame:
    rows = []
    for bid in bids:
        dte = client_data[bid]["test"]
        X_te = dte[features]
        y_te = dte[target].values
        y_pred = np.asarray(model.predict(X_te)).reshape(-1)
        mae, rmse, cvrmse, wape = _metrics_from_log_target(y_te, y_pred)
        rows.append((bid, mae, rmse, cvrmse, wape, bid in cold_set))

    return pd.DataFrame(rows, columns=["building_id", "mae", "rmse", "cvrmse", "wape", "cold_start"])


def _run_local_only(kind: str, bids: list, client_data: dict, features: list[str], target: str, cold_set: set, seed: int) -> pd.DataFrame:
    rows = []
    for bid in bids:
        dtr = client_data[bid]["train"]
        dte = client_data[bid]["test"]

        model = _fit_model(kind, dtr[features], dtr[target], seed)
        y_pred = np.asarray(model.predict(dte[features])).reshape(-1)

        mae, rmse, cvrmse, wape = _metrics_from_log_target(dte[target].values, y_pred)
        rows.append((bid, mae, rmse, cvrmse, wape, bid in cold_set))

    return pd.DataFrame(rows, columns=["building_id", "mae", "rmse", "cvrmse", "wape", "cold_start"])


def _save_df(df: pd.DataFrame, name: str, results_dir: str | Path = "saved_results") -> Path:
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return path


def _print_summary(title: str, df: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(f"  Mean MAE (log1p): {df['mae'].mean():.4f}")
    print(f"  Mean RMSE (log1p): {df['rmse'].mean():.4f}")
    print(f"  Mean CVRMSE (raw %): {df['cvrmse'].mean():.2f}%")
    print(f"  Mean WAPE (raw %): {df['wape'].mean():.2f}%")
    if "cold_start" in df.columns and df["cold_start"].any():
        print(f"  Cold-start MAE: {df[df['cold_start']]['mae'].mean():.4f}")
        print(f"  Cold-start CVRMSE: {df[df['cold_start']]['cvrmse'].mean():.2f}%")


def _mean_or_nan(s: pd.Series) -> float:
    return float(s.mean()) if len(s) > 0 else float("nan")


def _summary_row(name: str, df: pd.DataFrame) -> dict[str, float | str]:
    warm_df = df[~df["cold_start"]] if "cold_start" in df.columns else pd.DataFrame()
    cold_df = df[df["cold_start"]] if "cold_start" in df.columns else pd.DataFrame()

    return {
        "Strategy": name,
        "Overall": _mean_or_nan(df["mae"]),
        "Warm": _mean_or_nan(warm_df["mae"]),
        "Cold": _mean_or_nan(cold_df["mae"]),
        "CVRMSE (%)": _mean_or_nan(df["cvrmse"]),
    }


def run_tree_baselines(
    seed: int = 42,
    model_label: str = "model",
    results_dir: str | Path = "saved_results",
    client_data: dict | None = None,
    usable_bids: list | None = None,
    target: str | None = None,
    features: list[str] | None = None,
    cold_set: set | None = None,
    source_bids: list | None = None,
) -> dict[str, pd.DataFrame]:
    client_data = client_data if client_data is not None else globals().get("client_data")
    usable_bids = usable_bids if usable_bids is not None else globals().get("usable_bids")
    target = target if target is not None else globals().get("TARGET")

    if client_data is None or usable_bids is None or target is None:
        raise RuntimeError("Missing required inputs: client_data, usable_bids, target")

    features = _resolve_features(features)

    bids = list(usable_bids)
    cs = set(cold_set if cold_set is not None else globals().get("cold_set", set()))
    src_bids = list(source_bids if source_bids is not None else globals().get("source_bids", bids))

    if len(src_bids) == 0:
        src_bids = bids

    model_label = str(model_label).lower()
    pooled_train = pd.concat([client_data[bid]["train"] for bid in src_bids], ignore_index=True)

    # XGBoost
    print("=" * 72)
    print("Tree Baselines - XGBoost")
    print("=" * 72)
    xgb_global = _fit_model("xgb", pooled_train[features], pooled_train[target], seed)
    globals()["df_central_xgb"] = _per_building_eval_with_model(xgb_global, bids, client_data, features, target, cs)
    globals()["df_local_xgb"] = _run_local_only("xgb", bids, client_data, features, target, cs, seed)

    _save_df(globals()["df_central_xgb"], f"{model_label}_tree_centralized_xgb_seed{int(seed)}", results_dir)
    _save_df(globals()["df_local_xgb"], f"{model_label}_tree_local_only_xgb_seed{int(seed)}", results_dir)

    _print_summary("Centralized XGBoost", globals()["df_central_xgb"])
    _print_summary("Local-only XGBoost", globals()["df_local_xgb"])

    # LightGBM
    print("\n" + "=" * 72)
    print("Tree Baselines - LightGBM")
    print("=" * 72)
    lgbm_global = _fit_model("lgbm", pooled_train[features], pooled_train[target], seed)
    globals()["df_central_lgbm"] = _per_building_eval_with_model(lgbm_global, bids, client_data, features, target, cs)
    globals()["df_local_lgbm"] = _run_local_only("lgbm", bids, client_data, features, target, cs, seed)

    _save_df(globals()["df_central_lgbm"], f"{model_label}_tree_centralized_lgbm_seed{int(seed)}", results_dir)
    _save_df(globals()["df_local_lgbm"], f"{model_label}_tree_local_only_lgbm_seed{int(seed)}", results_dir)

    _print_summary("Centralized LightGBM", globals()["df_central_lgbm"])
    _print_summary("Local-only LightGBM", globals()["df_local_lgbm"])

    globals()["df_tree_baseline_summary"] = pd.DataFrame(
        [
            _summary_row("C. Centralized-XGBoost", globals()["df_central_xgb"]),
            _summary_row("0. Local-only-XGBoost", globals()["df_local_xgb"]),
            _summary_row("C. Centralized-LightGBM", globals()["df_central_lgbm"]),
            _summary_row("0. Local-only-LightGBM", globals()["df_local_lgbm"]),
        ]
    ).sort_values("Overall", ascending=True)
    globals()["df_tree_baseline_summary"].insert(0, "Model", model_label.upper())
    globals()["df_tree_baseline_summary"].insert(1, "Seed", int(seed))
    _save_df(globals()["df_tree_baseline_summary"], f"{model_label}_tree_baseline_summary_seed{int(seed)}", results_dir)
    print("\nTree baseline summary:")
    print(globals()["df_tree_baseline_summary"].to_string(index=False))
    return {
        "df_central_xgb": globals()["df_central_xgb"],
        "df_local_xgb": globals()["df_local_xgb"],
        "df_central_lgbm": globals()["df_central_lgbm"],
        "df_local_lgbm": globals()["df_local_lgbm"],
        "df_tree_baseline_summary": globals()["df_tree_baseline_summary"],
    }


if __name__ == "__main__":
    run_tree_baselines(seed=42)
