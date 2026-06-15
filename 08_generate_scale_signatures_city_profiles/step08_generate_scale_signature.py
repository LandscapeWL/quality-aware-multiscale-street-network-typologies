#!/usr/bin/env python3
"""Generate Step08 scale signatures, city profile tables, and Step09 inputs.

The script is intentionally self-contained so Step08 can be rerun from the
command line without relying on notebook state.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STEP_NAME = "08_generate_scale_signatures_city_profiles"

METRIC_CANDIDATES = [
    "edge_density_km_per_km2",
    "node_density_per_km2",
    "intersection_density_per_km2",
    "average_node_degree",
    "four_way_share",
    "dead_end_share",
    "segment_length_mean",
    "segment_length_median",
    "segment_length_p90",
    "edge_circuity_mean",
    "edge_circuity_median",
    "orientation_entropy",
    "orientation_order",
    "component_count",
    "giant_component_edge_share",
    "betweenness_gini",
    "road_hierarchy_entropy",
]

NETWORK_TYPES = ["drive", "walk"]
CITY_SCALES = ["full_city", "core_5km"]
LOCAL_SCALES = ["hex_2km", "hex_1km"]
LOCAL_STATS = ["mean", "std", "q10", "q25", "q50", "q75", "q90", "n_valid", "valid_share", "edge_unit_share"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root directory.",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def finite_numeric(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_ready(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_ready(v) for v in obj]
    return scalar(obj)


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def find_metric_column(columns: list[str], canonical: str) -> str | None:
    if canonical in columns:
        return canonical
    normalized = {normalize_name(col): col for col in columns}
    return normalized.get(normalize_name(canonical))


def validate_unique(df: pd.DataFrame, keys: list[str], label: str, log: dict[str, Any], strict: bool = True) -> None:
    duplicate_count = int(df.duplicated(keys).sum())
    log.setdefault("unique_key_checks", {})[label] = {
        "keys": keys,
        "duplicate_rows": duplicate_count,
        "unique_keys": int(df[keys].drop_duplicates().shape[0]),
    }
    if strict and duplicate_count:
        raise ValueError(f"{label} has {duplicate_count} duplicate rows for keys {keys}")


def safe_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).map(lambda x: bool(x))


def safe_divide(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return np.nan
    return float(numerator / denominator)


def weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> tuple[float, float, bool]:
    values = values.astype(float)
    weights = weights.astype(float)
    value_mask = np.isfinite(values)
    if not value_mask.any():
        return np.nan, np.nan, False

    x = values[value_mask]
    w = weights[value_mask]
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    w_sum = float(w.sum())
    if w_sum > 0:
        mean = float(np.sum(w * x) / w_sum)
        variance = float(np.sum(w * (x - mean) ** 2) / w_sum)
        return mean, math.sqrt(max(variance, 0.0)), True

    return float(np.mean(x)), float(np.std(x, ddof=0)), False


def quantile_or_nan(values: np.ndarray, q: float) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.quantile(values, q))


def build_city_base(city: pd.DataFrame, pop: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    pop_extra = [col for col in pop.columns if col == "city_id" or col not in city.columns]
    quality_extra = [col for col in quality.columns if col == "city_id" or col not in city.columns]
    base = city.merge(pop[pop_extra], on="city_id", how="left", validate="one_to_one")
    base = base.merge(quality[quality_extra], on="city_id", how="left", validate="one_to_one")
    base["main_analysis_sample"] = base["quality_tier"].eq("core")
    base["sensitivity_sample"] = base["quality_tier"].eq("sensitivity")
    base["excluded_candidate"] = base["quality_tier"].eq("excluded_candidate")
    return base


def map_metrics(city_metrics: pd.DataFrame, local_metrics: pd.DataFrame) -> tuple[list[str], list[str], dict[str, dict[str, str]]]:
    used: list[str] = []
    missing: list[str] = []
    mapping: dict[str, dict[str, str]] = {}
    city_cols = list(city_metrics.columns)
    local_cols = list(local_metrics.columns)
    for metric in METRIC_CANDIDATES:
        city_col = find_metric_column(city_cols, metric)
        local_col = find_metric_column(local_cols, metric)
        if city_col and local_col:
            used.append(metric)
            mapping[metric] = {"city_column": city_col, "local_column": local_col}
        else:
            missing.append(metric)
    return used, missing, mapping


def canonicalize_metric_columns(df: pd.DataFrame, metric_map: dict[str, dict[str, str]], source: str) -> pd.DataFrame:
    out = df.copy()
    key = "city_column" if source == "city" else "local_column"
    for canonical, mapping in metric_map.items():
        actual = mapping[key]
        if actual != canonical:
            out[canonical] = out[actual]
        out[canonical] = finite_numeric(out[canonical])
    return out


def make_city_signature(city_metrics: pd.DataFrame, city_base: pd.DataFrame, used_metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    city_meta_cols = [
        "city_id",
        "city_name_en",
        "country",
        "iso3",
        "region",
        "sample_group",
        "quality_tier",
        "quality_score",
        "quality_weight",
        "main_analysis_sample",
        "sensitivity_sample",
        "excluded_candidate",
    ]
    city_meta = city_base[city_meta_cols].drop_duplicates("city_id")

    cm = city_metrics.merge(city_meta, on="city_id", how="left", suffixes=("", "_base"))
    for _, row in cm.iterrows():
        row_valid = bool(row.get("valid_metric", False))
        for metric in used_metrics:
            raw = row.get(metric, np.nan)
            value = float(raw) if row_valid and pd.notna(raw) and np.isfinite(raw) else np.nan
            rows.append(
                {
                    "city_id": row["city_id"],
                    "city_name_en": row.get("city_name_en_base", row.get("city_name_en")),
                    "country": row.get("country_base", row.get("country")),
                    "iso3": row.get("iso3_base", row.get("iso3")),
                    "region": row.get("region_base", row.get("region")),
                    "sample_group": row.get("sample_group_base", row.get("sample_group")),
                    "quality_tier": row.get("quality_tier"),
                    "quality_score": row.get("quality_score"),
                    "quality_weight": row.get("quality_weight"),
                    "main_analysis_sample": bool(row.get("main_analysis_sample", False)),
                    "sensitivity_sample": bool(row.get("sensitivity_sample", False)),
                    "excluded_candidate": bool(row.get("excluded_candidate", False)),
                    "network_type": row["network_type"],
                    "scale": row["scale"],
                    "metric": metric,
                    "statistic": "value",
                    "value": value,
                    "n_units": 1,
                    "n_signature_units": int(row_valid),
                    "signature_unit_share": float(row_valid),
                    "n_valid": int(pd.notna(value)),
                    "valid_share": float(pd.notna(value)),
                    "edge_unit_share": 0.0,
                    "weighted_mean_used": False,
                    "source_table": "city_morphology_metrics",
                    "scale_valid_metric": row_valid,
                    "missing_flag": bool(pd.isna(value)),
                }
            )
    return pd.DataFrame(rows)


def make_local_signature(
    local_metrics: pd.DataFrame,
    city_base: pd.DataFrame,
    used_metrics: list[str],
    warnings_log: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys = ["city_id", "network_type", "scale"]
    local = local_metrics.copy()
    local["edge_unit"] = safe_bool(local["edge_unit"])
    local["valid_metric_for_signature"] = (
        safe_bool(local.get("valid_metric", pd.Series(False, index=local.index)))
        & safe_bool(local.get("valid_metric_quality", pd.Series(False, index=local.index)))
        & safe_bool(local.get("use_in_local_signature", pd.Series(False, index=local.index)))
    )
    local["use_in_local_clustering"] = safe_bool(local.get("use_in_local_clustering", pd.Series(False, index=local.index)))
    local["population_weight"] = finite_numeric(local.get("population_weight", pd.Series(np.nan, index=local.index)))

    total = (
        local.groupby(keys, observed=True)
        .agg(
            n_units=("unit_id", "size"),
            n_signature_units=("valid_metric_for_signature", "sum"),
            n_clustering_units=("use_in_local_clustering", "sum"),
            edge_unit_share_all=("edge_unit", "mean"),
        )
        .reset_index()
    )
    filtered = local[local["valid_metric_for_signature"]].copy()
    grouped = dict(tuple(filtered.groupby(keys, observed=True)))

    fallback_count = 0
    fallback_examples: list[dict[str, Any]] = []

    for _, meta in total.iterrows():
        group_key = (meta["city_id"], meta["network_type"], meta["scale"])
        group = grouped.get(group_key)
        n_units = int(meta["n_units"])
        n_signature_units = int(meta["n_signature_units"])
        signature_unit_share = safe_divide(float(n_signature_units), float(n_units))
        if group is None:
            group = filtered.iloc[0:0].copy()

        weights_all = finite_numeric(group.get("population_weight", pd.Series(dtype=float))).to_numpy(dtype=float)
        edge_units_all = safe_bool(group.get("edge_unit", pd.Series(dtype=bool))).to_numpy(dtype=bool)

        for metric in used_metrics:
            metric_values = finite_numeric(group.get(metric, pd.Series(dtype=float))).to_numpy(dtype=float)
            finite_mask = np.isfinite(metric_values)
            n_valid = int(finite_mask.sum())
            valid_share = safe_divide(float(n_valid), float(n_units))

            if n_valid:
                x = metric_values[finite_mask]
                w = weights_all[finite_mask] if weights_all.size else np.array([], dtype=float)
                if w.size != x.size:
                    w = np.zeros_like(x)
                mean, std, used_weight = weighted_mean_std(x, w)
                if not used_weight:
                    fallback_count += 1
                    if len(fallback_examples) < 20:
                        fallback_examples.append(
                            {
                                "city_id": group_key[0],
                                "network_type": group_key[1],
                                "scale": group_key[2],
                                "metric": metric,
                            }
                        )
                q10 = quantile_or_nan(x, 0.10)
                q25 = quantile_or_nan(x, 0.25)
                q50 = quantile_or_nan(x, 0.50)
                q75 = quantile_or_nan(x, 0.75)
                q90 = quantile_or_nan(x, 0.90)
                edge_share = float(edge_units_all[finite_mask].mean()) if edge_units_all.size and finite_mask.any() else np.nan
            else:
                mean = std = q10 = q25 = q50 = q75 = q90 = edge_share = np.nan
                used_weight = False

            stat_values = {
                "mean": mean,
                "std": std,
                "q10": q10,
                "q25": q25,
                "q50": q50,
                "q75": q75,
                "q90": q90,
                "n_valid": float(n_valid),
                "valid_share": valid_share,
                "edge_unit_share": edge_share,
            }
            for stat, value in stat_values.items():
                rows.append(
                    {
                        "city_id": group_key[0],
                        "network_type": group_key[1],
                        "scale": group_key[2],
                        "metric": metric,
                        "statistic": stat,
                        "value": value,
                        "n_units": n_units,
                        "n_signature_units": n_signature_units,
                        "signature_unit_share": signature_unit_share,
                        "n_valid": n_valid,
                        "valid_share": valid_share,
                        "edge_unit_share": edge_share,
                        "weighted_mean_used": bool(used_weight and stat in {"mean", "std"}),
                        "source_table": "local_morphology_metrics+local_quality_flags",
                        "scale_valid_metric": bool(n_valid > 0),
                        "missing_flag": bool(pd.isna(value)),
                    }
                )

    if fallback_count:
        warnings_log.append(
            f"Population-weight fallback used for {fallback_count} local metric groups; examples recorded in log."
        )

    local_long = pd.DataFrame(rows)
    city_meta_cols = [
        "city_id",
        "city_name_en",
        "country",
        "iso3",
        "region",
        "sample_group",
        "quality_tier",
        "quality_score",
        "quality_weight",
        "main_analysis_sample",
        "sensitivity_sample",
        "excluded_candidate",
    ]
    local_long = local_long.merge(city_base[city_meta_cols], on="city_id", how="left", validate="many_to_one")
    ordered_cols = city_meta_cols + [
        "network_type",
        "scale",
        "metric",
        "statistic",
        "value",
        "n_units",
        "n_signature_units",
        "signature_unit_share",
        "n_valid",
        "valid_share",
        "edge_unit_share",
        "weighted_mean_used",
        "source_table",
        "scale_valid_metric",
        "missing_flag",
    ]
    return local_long[ordered_cols], {
        "population_weight_fallback_count": fallback_count,
        "population_weight_fallback_examples": fallback_examples,
    }


def add_signature_feature_id(signature_long: pd.DataFrame) -> pd.DataFrame:
    out = signature_long.copy()
    out["signature_feature"] = (
        "sig__"
        + out["network_type"].astype(str)
        + "__"
        + out["scale"].astype(str)
        + "__"
        + out["metric"].astype(str)
        + "__"
        + out["statistic"].astype(str)
    )
    return out


def standardize_wide(
    df: pd.DataFrame,
    feature_cols: list[str],
    tier_col: str,
    z_prefix: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    params: dict[str, Any] = {}
    z_columns: dict[str, pd.Series] = {}
    fit_mask = out[tier_col].eq("core")
    for col in feature_cols:
        values = finite_numeric(out[col])
        fit_values = values[fit_mask].dropna()
        fit_values = fit_values[np.isfinite(fit_values)]
        z_col = f"{z_prefix}{col}"
        if len(fit_values) >= 2:
            mean = float(fit_values.mean())
            std = float(fit_values.std(ddof=0))
            if np.isfinite(std) and std > 0:
                z_columns[z_col] = (values - mean) / std
                status = "ok"
            else:
                z_columns[z_col] = pd.Series(np.nan, index=out.index)
                status = "zero_std"
        else:
            mean = float(fit_values.mean()) if len(fit_values) else np.nan
            std = np.nan
            z_columns[z_col] = pd.Series(np.nan, index=out.index)
            status = "too_few_fit_values"
        params[col] = {
            "mean": mean,
            "std": std,
            "n_fit": int(len(fit_values)),
            "status": status,
        }
    if z_columns:
        out = pd.concat([out, pd.DataFrame(z_columns, index=out.index)], axis=1)
    return out, params


def build_signature_matrix(signature_long: pd.DataFrame, city_base: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    matrix = signature_long.pivot_table(
        index="city_id",
        columns="signature_feature",
        values="value",
        aggfunc="first",
        observed=True,
    )
    matrix.columns.name = None
    matrix = matrix.reset_index()

    meta_cols = [
        "city_id",
        "city_name_en",
        "country",
        "iso3",
        "region",
        "sample_group",
        "morphology_prior",
        "quality_tier",
        "quality_score",
        "quality_weight",
        "quality_confidence_component",
        "main_analysis_sample",
        "sensitivity_sample",
        "excluded_candidate",
    ]
    meta_cols = [col for col in meta_cols if col in city_base.columns]
    matrix = city_base[meta_cols].merge(matrix, on="city_id", how="left", validate="one_to_one")
    feature_cols = [col for col in matrix.columns if col.startswith("sig__")]
    matrix, params = standardize_wide(matrix, feature_cols, "quality_tier", "z__")
    return matrix, params


def attach_signature_z(signature_long: pd.DataFrame, signature_matrix: pd.DataFrame) -> pd.DataFrame:
    z_cols = [col for col in signature_matrix.columns if col.startswith("z__sig__")]
    if not z_cols:
        out = signature_long.copy()
        out["value_z"] = np.nan
        return out
    z_long = signature_matrix[["city_id"] + z_cols].melt(
        id_vars="city_id",
        var_name="z_feature",
        value_name="value_z",
    )
    z_long["signature_feature"] = z_long["z_feature"].str.removeprefix("z__")
    return signature_long.merge(
        z_long[["city_id", "signature_feature", "value_z"]],
        on=["city_id", "signature_feature"],
        how="left",
        validate="one_to_one",
    )


def raw_sig(matrix_row: pd.Series, network: str, scale: str, metric: str, statistic: str) -> float:
    col = f"sig__{network}__{scale}__{metric}__{statistic}"
    value = matrix_row.get(col, np.nan)
    return float(value) if pd.notna(value) and np.isfinite(value) else np.nan


def build_transfer_features(signature_matrix: pd.DataFrame, used_metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    meta_cols = [
        "city_id",
        "city_name_en",
        "country",
        "iso3",
        "region",
        "sample_group",
        "quality_tier",
        "quality_score",
        "quality_weight",
        "main_analysis_sample",
        "sensitivity_sample",
        "excluded_candidate",
    ]
    meta_cols = [col for col in meta_cols if col in signature_matrix.columns]

    def add_row(base: dict[str, Any], network_type: str, metric: str, feature_type: str, value: float, **extra: Any) -> None:
        row = dict(base)
        row.update(
            {
                "network_type": network_type,
                "metric": metric,
                "transfer_feature_type": feature_type,
                "value": value,
            }
        )
        row.update(extra)
        parts = [
            "trans",
            network_type,
            metric,
            feature_type,
            str(extra.get("from_scale", "")),
            str(extra.get("to_scale", "")),
            str(extra.get("statistic", "")),
        ]
        row["transfer_feature"] = "__".join(part for part in parts if part)
        rows.append(row)

    for _, matrix_row in signature_matrix.iterrows():
        base = {col: matrix_row[col] for col in meta_cols}
        for network in NETWORK_TYPES:
            for metric in used_metrics:
                full = raw_sig(matrix_row, network, "full_city", metric, "value")
                core = raw_sig(matrix_row, network, "core_5km", metric, "value")
                hex1_mean = raw_sig(matrix_row, network, "hex_1km", metric, "mean")
                hex2_mean = raw_sig(matrix_row, network, "hex_2km", metric, "mean")

                delta = core - full if np.isfinite(core) and np.isfinite(full) else np.nan
                add_row(
                    base,
                    network,
                    metric,
                    "core_to_full_delta",
                    delta,
                    from_scale="core_5km",
                    to_scale="full_city",
                    statistic="value",
                )

                delta = hex1_mean - hex2_mean if np.isfinite(hex1_mean) and np.isfinite(hex2_mean) else np.nan
                add_row(
                    base,
                    network,
                    metric,
                    "hex1_to_hex2_delta",
                    delta,
                    from_scale="hex_1km",
                    to_scale="hex_2km",
                    statistic="mean",
                )

                for scale, local_mean in [("hex_1km", hex1_mean), ("hex_2km", hex2_mean)]:
                    add_row(
                        base,
                        network,
                        metric,
                        "local_to_full_ratio",
                        safe_divide(local_mean, full),
                        from_scale=scale,
                        to_scale="full_city",
                        statistic="mean_over_value",
                    )

                    q25 = raw_sig(matrix_row, network, scale, metric, "q25")
                    q75 = raw_sig(matrix_row, network, scale, metric, "q75")
                    iqr = q75 - q25 if np.isfinite(q25) and np.isfinite(q75) else np.nan
                    add_row(
                        base,
                        network,
                        metric,
                        "local_iqr",
                        iqr,
                        from_scale=scale,
                        to_scale=scale,
                        statistic="q75_minus_q25",
                    )

                    q10 = raw_sig(matrix_row, network, scale, metric, "q10")
                    q90 = raw_sig(matrix_row, network, scale, metric, "q90")
                    q90_q10 = q90 - q10 if np.isfinite(q10) and np.isfinite(q90) else np.nan
                    add_row(
                        base,
                        network,
                        metric,
                        "local_q90_q10",
                        q90_q10,
                        from_scale=scale,
                        to_scale=scale,
                        statistic="q90_minus_q10",
                    )

                scale_values = np.array([full, core, hex2_mean, hex1_mean], dtype=float)
                valid = scale_values[np.isfinite(scale_values)]
                if valid.size >= 3:
                    mean_abs = abs(float(valid.mean()))
                    heterogeneity = safe_divide(float(valid.std(ddof=0)), mean_abs)
                    stability = safe_divide(1.0, 1.0 + heterogeneity)
                else:
                    heterogeneity = np.nan
                    stability = np.nan
                add_row(
                    base,
                    network,
                    metric,
                    "scale_heterogeneity_score",
                    heterogeneity,
                    from_scale="four_scales",
                    to_scale="four_scales",
                    statistic="cv",
                )
                add_row(
                    base,
                    network,
                    metric,
                    "scale_stability_score",
                    stability,
                    from_scale="four_scales",
                    to_scale="four_scales",
                    statistic="one_over_one_plus_cv",
                )

        for metric in used_metrics:
            for scale, stat in [
                ("full_city", "value"),
                ("core_5km", "value"),
                ("hex_2km", "mean"),
                ("hex_1km", "mean"),
            ]:
                walk = raw_sig(matrix_row, "walk", scale, metric, stat)
                drive = raw_sig(matrix_row, "drive", scale, metric, stat)
                gap = walk - drive if np.isfinite(walk) and np.isfinite(drive) else np.nan
                add_row(
                    base,
                    "walk_minus_drive",
                    metric,
                    "walk_drive_gap",
                    gap,
                    from_scale=scale,
                    to_scale=scale,
                    statistic=stat,
                )

    return pd.DataFrame(rows)


def standardize_long(
    df: pd.DataFrame,
    feature_col: str,
    value_col: str,
    tier_col: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    out[f"{value_col}_z"] = np.nan
    params: dict[str, Any] = {}
    for feature, index in out.groupby(feature_col, observed=True).groups.items():
        values = finite_numeric(out.loc[index, value_col])
        fit_mask = out.loc[index, tier_col].eq("core")
        fit_values = values[fit_mask].dropna()
        fit_values = fit_values[np.isfinite(fit_values)]
        if len(fit_values) >= 2:
            mean = float(fit_values.mean())
            std = float(fit_values.std(ddof=0))
            if np.isfinite(std) and std > 0:
                out.loc[index, f"{value_col}_z"] = (values - mean) / std
                status = "ok"
            else:
                status = "zero_std"
        else:
            mean = float(fit_values.mean()) if len(fit_values) else np.nan
            std = np.nan
            status = "too_few_fit_values"
        params[str(feature)] = {
            "mean": mean,
            "std": std,
            "n_fit": int(len(fit_values)),
            "status": status,
        }
    return out, params


def make_step09_candidates(
    local_metrics: pd.DataFrame,
    city_base: pd.DataFrame,
    used_metrics: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta_cols = [
        "city_id",
        "city_name_en",
        "country",
        "iso3",
        "region",
        "sample_group",
        "quality_tier",
        "quality_score",
        "quality_weight",
        "quality_confidence_component",
        "main_analysis_sample",
        "sensitivity_sample",
        "excluded_candidate",
    ]
    meta_cols = [col for col in meta_cols if col in city_base.columns]
    local = local_metrics.merge(city_base[meta_cols], on="city_id", how="left", validate="many_to_one", suffixes=("", "_city"))
    local["use_in_local_clustering"] = safe_bool(local["use_in_local_clustering"])
    candidates = local[local["use_in_local_clustering"]].copy()
    candidates["step09_main_candidate"] = candidates["network_type"].eq("drive") & candidates["quality_tier"].eq("core")
    candidates["step09_sensitivity_candidate"] = candidates["network_type"].isin(NETWORK_TYPES) & candidates["quality_tier"].isin(
        ["core", "sensitivity"]
    )
    candidates["retained_excluded_candidate"] = candidates["quality_tier"].eq("excluded_candidate")

    standardization: dict[str, Any] = {}
    for network in NETWORK_TYPES:
        for scale in LOCAL_SCALES:
            group_mask = candidates["network_type"].eq(network) & candidates["scale"].eq(scale)
            fit_mask = group_mask & candidates["quality_tier"].eq("core")
            for metric in used_metrics:
                z_col = f"{metric}_z"
                if z_col not in candidates.columns:
                    candidates[z_col] = np.nan
                values = finite_numeric(candidates[metric])
                fit_values = values[fit_mask].dropna()
                fit_values = fit_values[np.isfinite(fit_values)]
                feature_key = f"{network}__{scale}__{metric}"
                if len(fit_values) >= 2:
                    mean = float(fit_values.mean())
                    std = float(fit_values.std(ddof=0))
                    if np.isfinite(std) and std > 0:
                        candidates.loc[group_mask, z_col] = (values[group_mask] - mean) / std
                        status = "ok"
                    else:
                        status = "zero_std"
                else:
                    mean = float(fit_values.mean()) if len(fit_values) else np.nan
                    std = np.nan
                    status = "too_few_fit_values"
                standardization[feature_key] = {
                    "mean": mean,
                    "std": std,
                    "n_fit": int(len(fit_values)),
                    "status": status,
                }

    candidates["metric_missing_count"] = candidates[used_metrics].isna().sum(axis=1)
    candidates["all_clustering_metrics_present"] = candidates["metric_missing_count"].eq(0)

    preferred_cols = [
        "city_id",
        "city_name_en",
        "country",
        "iso3",
        "region",
        "sample_group",
        "quality_tier",
        "quality_score",
        "quality_weight",
        "quality_confidence_component",
        "network_type",
        "scale",
        "unit_id",
        "unit_area_km2",
        "area_km2",
        "valid_area_ratio",
        "edge_unit",
        "center_lon",
        "center_lat",
        "population_weight",
        "built_share",
        "population_sum",
        "valid_metric",
        "valid_metric_quality",
        "local_metric_validity_score",
        "local_quality_flag",
        "local_quality_reason",
        "use_in_local_signature",
        "use_in_local_clustering",
        "step09_main_candidate",
        "step09_sensitivity_candidate",
        "retained_excluded_candidate",
        "metric_missing_count",
        "all_clustering_metrics_present",
    ]
    metric_cols = used_metrics + [f"{metric}_z" for metric in used_metrics]
    final_cols = [col for col in preferred_cols + metric_cols if col in candidates.columns]
    return candidates[final_cols], standardization


def make_description_stats(signature_long: pd.DataFrame) -> pd.DataFrame:
    def q(series: pd.Series, quantile: float) -> float:
        values = finite_numeric(series).dropna()
        if values.empty:
            return np.nan
        return float(values.quantile(quantile))

    grouped = []
    for keys, group in signature_long.groupby(["network_type", "scale", "metric", "statistic", "signature_feature"], observed=True):
        values = finite_numeric(group["value"])
        core_values = finite_numeric(group.loc[group["quality_tier"].eq("core"), "value"])
        row = {
            "network_type": keys[0],
            "scale": keys[1],
            "metric": keys[2],
            "statistic": keys[3],
            "signature_feature": keys[4],
            "n_cities": int(values.notna().sum()),
            "n_missing": int(values.isna().sum()),
            "missing_share": safe_divide(float(values.isna().sum()), float(len(values))),
            "mean": float(values.mean()) if values.notna().any() else np.nan,
            "std": float(values.std(ddof=0)) if values.notna().any() else np.nan,
            "min": float(values.min()) if values.notna().any() else np.nan,
            "q10": q(values, 0.10),
            "q25": q(values, 0.25),
            "q50": q(values, 0.50),
            "q75": q(values, 0.75),
            "q90": q(values, 0.90),
            "max": float(values.max()) if values.notna().any() else np.nan,
            "core_n_cities": int(core_values.notna().sum()),
            "core_mean": float(core_values.mean()) if core_values.notna().any() else np.nan,
            "core_std": float(core_values.std(ddof=0)) if core_values.notna().any() else np.nan,
        }
        grouped.append(row)
    return pd.DataFrame(grouped).sort_values(["network_type", "scale", "metric", "statistic"]).reset_index(drop=True)


def make_quality_check(
    city_base: pd.DataFrame,
    city_metrics: pd.DataFrame,
    local_metrics: pd.DataFrame,
    signature_long: pd.DataFrame,
    step09_candidates: pd.DataFrame,
    used_metrics: list[str],
) -> pd.DataFrame:
    city_network = pd.MultiIndex.from_product([city_base["city_id"], NETWORK_TYPES], names=["city_id", "network_type"]).to_frame(index=False)
    meta_cols = ["city_id", "city_name_en", "country", "iso3", "region", "sample_group", "quality_tier", "quality_score"]
    qc = city_network.merge(city_base[meta_cols], on="city_id", how="left", validate="many_to_one")

    for scale in CITY_SCALES:
        cm = city_metrics[city_metrics["scale"].eq(scale)][["city_id", "network_type", "valid_metric"]].copy()
        cm = cm.rename(columns={"valid_metric": f"{scale}_valid_metric"})
        cm[f"has_{scale}"] = True
        qc = qc.merge(cm, on=["city_id", "network_type"], how="left", validate="one_to_one")
        qc[f"has_{scale}"] = qc[f"has_{scale}"].fillna(False)
        qc[f"{scale}_valid_metric"] = qc[f"{scale}_valid_metric"].fillna(False)

    local = local_metrics.copy()
    local["valid_metric_for_signature"] = (
        safe_bool(local["valid_metric"]) & safe_bool(local["valid_metric_quality"]) & safe_bool(local["use_in_local_signature"])
    )
    local["use_in_local_clustering"] = safe_bool(local["use_in_local_clustering"])
    for scale in LOCAL_SCALES:
        agg = (
            local[local["scale"].eq(scale)]
            .groupby(["city_id", "network_type"], observed=True)
            .agg(
                **{
                    f"{scale}_units": ("unit_id", "size"),
                    f"{scale}_signature_units": ("valid_metric_for_signature", "sum"),
                    f"{scale}_clustering_units": ("use_in_local_clustering", "sum"),
                    f"{scale}_edge_unit_share": ("edge_unit", "mean"),
                }
            )
            .reset_index()
        )
        agg[f"has_{scale}"] = True
        agg[f"{scale}_signature_unit_share"] = agg[f"{scale}_signature_units"] / agg[f"{scale}_units"]
        agg[f"{scale}_clustering_unit_share"] = agg[f"{scale}_clustering_units"] / agg[f"{scale}_units"]
        qc = qc.merge(agg, on=["city_id", "network_type"], how="left", validate="one_to_one")
        qc[f"has_{scale}"] = qc[f"has_{scale}"].fillna(False)
        for col in [f"{scale}_units", f"{scale}_signature_units", f"{scale}_clustering_units"]:
            qc[col] = qc[col].fillna(0).astype(int)

    missing = (
        signature_long.groupby(["city_id", "network_type"], observed=True)
        .agg(
            signature_rows=("signature_feature", "size"),
            signature_missing_features=("missing_flag", "sum"),
        )
        .reset_index()
    )
    missing["signature_valid_feature_share"] = 1 - (missing["signature_missing_features"] / missing["signature_rows"])
    qc = qc.merge(missing, on=["city_id", "network_type"], how="left", validate="one_to_one")

    cand = (
        step09_candidates.groupby(["city_id", "network_type"], observed=True)
        .agg(
            step09_candidate_units=("unit_id", "size"),
            step09_main_candidate_units=("step09_main_candidate", "sum"),
            step09_sensitivity_candidate_units=("step09_sensitivity_candidate", "sum"),
        )
        .reset_index()
    )
    qc = qc.merge(cand, on=["city_id", "network_type"], how="left", validate="one_to_one")
    for col in ["step09_candidate_units", "step09_main_candidate_units", "step09_sensitivity_candidate_units"]:
        qc[col] = qc[col].fillna(0).astype(int)

    qc["four_scale_coverage_count"] = qc[[f"has_{scale}" for scale in CITY_SCALES + LOCAL_SCALES]].sum(axis=1)
    qc["all_four_scales_present"] = qc["four_scale_coverage_count"].eq(4)
    qc["dhaka_or_istanbul_flag"] = qc["city_name_en"].isin(["Dhaka", "Istanbul"])
    qc["walk_core_5km_invalid_dhaka_istanbul_flag"] = (
        qc["dhaka_or_istanbul_flag"] & qc["network_type"].eq("walk") & ~qc["core_5km_valid_metric"]
    )
    qc["excluded_candidate_not_in_main_candidate"] = qc["quality_tier"].eq("excluded_candidate") & qc[
        "step09_main_candidate_units"
    ].eq(0)
    qc["excluded_candidate_main_violation"] = qc["quality_tier"].eq("excluded_candidate") & qc[
        "step09_main_candidate_units"
    ].gt(0)

    def note(row: pd.Series) -> str:
        notes: list[str] = []
        for scale in CITY_SCALES + LOCAL_SCALES:
            if not bool(row.get(f"has_{scale}", False)):
                notes.append(f"missing_{scale}")
        if not bool(row.get("full_city_valid_metric", False)):
            notes.append("invalid_full_city")
        if not bool(row.get("core_5km_valid_metric", False)):
            notes.append("invalid_core_5km")
        for scale in LOCAL_SCALES:
            share = row.get(f"{scale}_signature_unit_share", np.nan)
            if pd.notna(share) and share < 0.50:
                notes.append(f"low_{scale}_signature_share")
        if bool(row.get("walk_core_5km_invalid_dhaka_istanbul_flag", False)):
            notes.append("dhaka_istanbul_walk_core_5km_nan_preserved")
        if bool(row.get("excluded_candidate_not_in_main_candidate", False)):
            notes.append("excluded_candidate_retained_not_main")
        if bool(row.get("excluded_candidate_main_violation", False)):
            notes.append("excluded_candidate_in_main_candidate_error")
        return ";".join(notes)

    qc["anomaly_notes"] = qc.apply(note, axis=1)
    qc["metric_count_used"] = len(used_metrics)
    return qc.sort_values(["city_id", "network_type"]).reset_index(drop=True)


def make_city_profile(
    city_base: pd.DataFrame,
    signature_matrix: pd.DataFrame,
    transfer_features: pd.DataFrame,
    step09_candidates: pd.DataFrame,
    qc: pd.DataFrame,
) -> pd.DataFrame:
    profile = city_base.copy()
    sig_cols = [col for col in signature_matrix.columns if col.startswith("sig__")]
    z_sig_cols = [col for col in signature_matrix.columns if col.startswith("z__sig__")]
    sig_summary = signature_matrix[["city_id"]].copy()
    sig_summary["signature_feature_count"] = len(sig_cols)
    sig_summary["signature_missing_count"] = signature_matrix[sig_cols].isna().sum(axis=1)
    sig_summary["signature_complete_share"] = 1 - sig_summary["signature_missing_count"] / max(len(sig_cols), 1)
    sig_summary["signature_z_feature_count"] = len(z_sig_cols)
    sig_summary["signature_z_missing_count"] = signature_matrix[z_sig_cols].isna().sum(axis=1) if z_sig_cols else 0

    transfer_summary = (
        transfer_features.groupby("city_id", observed=True)
        .agg(
            transfer_feature_count=("transfer_feature", "size"),
            transfer_missing_count=("value", lambda s: int(finite_numeric(s).isna().sum())),
            transfer_z_missing_count=("value_z", lambda s: int(finite_numeric(s).isna().sum())),
        )
        .reset_index()
    )
    transfer_summary["transfer_complete_share"] = 1 - (
        transfer_summary["transfer_missing_count"] / transfer_summary["transfer_feature_count"]
    )

    cand_summary = (
        step09_candidates.groupby("city_id", observed=True)
        .agg(
            step09_candidate_units=("unit_id", "size"),
            step09_main_candidate_units=("step09_main_candidate", "sum"),
            step09_sensitivity_candidate_units=("step09_sensitivity_candidate", "sum"),
            step09_retained_excluded_units=("retained_excluded_candidate", "sum"),
        )
        .reset_index()
    )

    qc_summary = (
        qc.groupby("city_id", observed=True)
        .agg(
            city_network_rows=("network_type", "size"),
            all_four_scales_city_networks=("all_four_scales_present", "sum"),
            city_networks_with_anomaly=("anomaly_notes", lambda s: int((s.astype(str) != "").sum())),
            excluded_candidate_main_violations=("excluded_candidate_main_violation", "sum"),
        )
        .reset_index()
    )

    for table in [sig_summary, transfer_summary, cand_summary, qc_summary]:
        profile = profile.merge(table, on="city_id", how="left", validate="one_to_one")
    count_cols = [
        "step09_candidate_units",
        "step09_main_candidate_units",
        "step09_sensitivity_candidate_units",
        "step09_retained_excluded_units",
        "excluded_candidate_main_violations",
    ]
    for col in count_cols:
        if col in profile.columns:
            profile[col] = profile[col].fillna(0).astype(int)
    return profile


def write_outputs(
    out_dir: Path,
    outputs: dict[str, pd.DataFrame],
    log: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = {
        "scale_signatures_long": "scale_signatures_long.parquet",
        "scale_signatures_matrix": "scale_signatures_matrix.parquet",
        "scale_transfer_features": "scale_transfer_features.parquet",
        "city_profiles_base": "city_profiles_base.parquet",
        "step09_morphotype_candidate_units": "step09_morphotype_candidate_units.parquet",
    }
    csv_files = {
        "scale_signatures_summary_stats": "scale_signatures_summary_stats.csv",
        "scale_signatures_quality_checks": "scale_signatures_quality_checks.csv",
    }

    log["output_paths"] = {}
    log["output_shapes"] = {}
    for key, filename in parquet_files.items():
        path = out_dir / filename
        outputs[key].to_parquet(path, index=False)
        log["output_paths"][key] = str(path)
        log["output_shapes"][key] = {"rows": int(outputs[key].shape[0]), "columns": int(outputs[key].shape[1])}
    for key, filename in csv_files.items():
        path = out_dir / filename
        outputs[key].to_csv(path, index=False, encoding="utf-8-sig")
        log["output_paths"][key] = str(path)
        log["output_shapes"][key] = {"rows": int(outputs[key].shape[0]), "columns": int(outputs[key].shape[1])}

    log_path = out_dir / "08_run_log.json"
    log["output_paths"]["08_run_log"] = str(log_path)
    log_path.write_text(json.dumps(json_ready(log), ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    code_dir = root / "code" / STEP_NAME
    out_dir = root / "data" / STEP_NAME
    start_time = time.time()

    paths = {
        "city_master": root / "data/01_city_boundaries_sample_list/city_sample/city_master.parquet",
        "population_built": root / "data/03_download_population_built_environment_data/city_population_built_environment_metrics.parquet",
        "unit_index": root / "data/05_build_multiscale_spatial_units/spatial_units_index.parquet",
        "city_morphology": root / "data/06_clean_road_networks_calculate_morphology_metrics/city_morphology_metrics.parquet",
        "local_morphology": root / "data/06_clean_road_networks_calculate_morphology_metrics/local_morphology_metrics.parquet",
        "city_quality": root / "data/07_generate_quality_scores_type_confidence/city_quality_scores.parquet",
        "local_quality": root / "data/07_generate_quality_scores_type_confidence/local_quality_flags.parquet",
    }

    log: dict[str, Any] = {
        "step": STEP_NAME,
        "script_path": str(code_dir / Path(__file__).name),
        "root": str(root),
        "run_started_at": now_iso(),
        "input_paths": {key: str(value) for key, value in paths.items()},
        "filter_rules": {
            "city_sample": "Keep all 86 cities; quality_tier == core is the main analysis and standardization fitting sample; sensitivity and excluded_candidate are retained.",
            "local_signature": "valid_metric == True in morphology and quality tables, and use_in_local_signature == True.",
            "step09_output": "Rows with use_in_local_clustering == True are retained; main candidates are drive + core; sensitivity candidates are drive/walk + core/sensitivity.",
            "invalid_city_scale_metrics": "Rows with valid_metric == False are preserved but metric values are set to NaN.",
            "zero_denominators": "Derived ratios with zero or missing denominator remain NaN.",
        },
        "warnings": [],
    }

    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required input {label}: {path}")

    print(f"[{now_iso()}] Reading inputs...")
    city = pd.read_parquet(paths["city_master"])
    pop = pd.read_parquet(paths["population_built"])
    unit_index = pd.read_parquet(paths["unit_index"])
    city_metrics = pd.read_parquet(paths["city_morphology"])
    local_metrics = pd.read_parquet(paths["local_morphology"])
    city_quality = pd.read_parquet(paths["city_quality"])
    local_quality = pd.read_parquet(paths["local_quality"])

    inputs = {
        "city_master": city,
        "population_built": pop,
        "unit_index": unit_index,
        "city_morphology": city_metrics,
        "local_morphology": local_metrics,
        "city_quality": city_quality,
        "local_quality": local_quality,
    }
    log["input_shapes"] = {key: {"rows": int(df.shape[0]), "columns": int(df.shape[1])} for key, df in inputs.items()}

    print(f"[{now_iso()}] Validating keys and coverage...")
    validate_unique(city, ["city_id"], "city_master", log)
    validate_unique(pop, ["city_id"], "population_built", log)
    validate_unique(unit_index, ["city_id", "unit_id", "scale"], "unit_index", log)
    validate_unique(city_metrics, ["city_id", "network_type", "scale"], "city_morphology", log)
    validate_unique(local_metrics, ["city_id", "unit_id", "network_type", "scale"], "local_morphology", log)
    validate_unique(city_quality, ["city_id"], "city_quality", log)
    validate_unique(local_quality, ["city_id", "unit_id", "network_type", "scale"], "local_quality", log)

    city_ids = set(city["city_id"])
    log["coverage_checks"] = {}
    for label, df in inputs.items():
        if "city_id" in df.columns:
            ids = set(df["city_id"])
            log["coverage_checks"][label] = {
                "unique_city_count": int(len(ids)),
                "missing_from_input_vs_master": sorted(city_ids - ids),
                "extra_vs_master": sorted(ids - city_ids),
            }

    city_base = build_city_base(city, pop, city_quality)
    log["quality_tier_counts"] = city_base["quality_tier"].value_counts(dropna=False).to_dict()
    if int(city_base.shape[0]) != 86:
        log["warnings"].append(f"Expected 86 cities, found {city_base.shape[0]}.")

    used_metrics, missing_metrics, metric_mapping = map_metrics(city_metrics, local_metrics)
    if missing_metrics:
        log["warnings"].append(f"Missing candidate metrics: {missing_metrics}")
    log["metric_candidates"] = METRIC_CANDIDATES
    log["used_metrics"] = used_metrics
    log["missing_metrics"] = missing_metrics
    log["metric_column_mapping"] = metric_mapping

    city_metrics = canonicalize_metric_columns(city_metrics, metric_mapping, "city")
    local_metrics = canonicalize_metric_columns(local_metrics, metric_mapping, "local")

    local_quality_cols = [
        "city_id",
        "unit_id",
        "network_type",
        "scale",
        "valid_metric",
        "local_metric_validity_score",
        "local_empty_unit_flag",
        "local_low_edge_count_flag",
        "local_low_node_count_flag",
        "local_high_density_flag",
        "local_circuity_invalid_flag",
        "local_orientation_invalid_flag",
        "local_quality_flag",
        "local_quality_reason",
        "use_in_local_signature",
        "use_in_local_clustering",
    ]
    local_quality_cols = [col for col in local_quality_cols if col in local_quality.columns]
    local = local_metrics.merge(
        local_quality[local_quality_cols],
        on=["city_id", "unit_id", "network_type", "scale"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_quality"),
    )
    if "valid_metric_quality" not in local.columns:
        local["valid_metric_quality"] = local["valid_metric"]
    local_valid_mismatch = int((safe_bool(local["valid_metric"]) != safe_bool(local["valid_metric_quality"])).sum())
    log["local_valid_metric_mismatch_rows"] = local_valid_mismatch
    if local_valid_mismatch:
        log["warnings"].append(f"Local morphology and quality valid_metric disagree on {local_valid_mismatch} rows.")

    print(f"[{now_iso()}] Building city and local scale signatures...")
    city_signature = make_city_signature(city_metrics, city_base, used_metrics)
    local_signature, local_agg_log = make_local_signature(local, city_base, used_metrics, log["warnings"])
    signature_long = pd.concat([city_signature, local_signature], ignore_index=True)
    signature_long = add_signature_feature_id(signature_long)

    print(f"[{now_iso()}] Standardizing signature matrix using core cities...")
    signature_matrix, signature_standardization = build_signature_matrix(signature_long, city_base)
    signature_long = attach_signature_z(signature_long, signature_matrix)

    print(f"[{now_iso()}] Building transfer features...")
    transfer_features = build_transfer_features(signature_matrix, used_metrics)
    transfer_features, transfer_standardization = standardize_long(transfer_features, "transfer_feature", "value", "quality_tier")

    print(f"[{now_iso()}] Building Step09 candidate unit table...")
    step09_candidates, candidate_standardization = make_step09_candidates(local, city_base, used_metrics)

    print(f"[{now_iso()}] Building QC and profile tables...")
    description_stats = make_description_stats(signature_long)
    quality_check = make_quality_check(city_base, city_metrics, local, signature_long, step09_candidates, used_metrics)
    city_profile = make_city_profile(city_base, signature_matrix, transfer_features, step09_candidates, quality_check)

    dhaka_istanbul = quality_check[quality_check["city_name_en"].isin(["Dhaka", "Istanbul"])][
        [
            "city_id",
            "city_name_en",
            "network_type",
            "quality_tier",
            "core_5km_valid_metric",
            "walk_core_5km_invalid_dhaka_istanbul_flag",
            "step09_main_candidate_units",
            "anomaly_notes",
        ]
    ]
    log["dhaka_istanbul_qc"] = dhaka_istanbul.to_dict(orient="records")
    log["excluded_candidate_main_candidate_units"] = int(
        quality_check.loc[quality_check["quality_tier"].eq("excluded_candidate"), "step09_main_candidate_units"].sum()
    )
    log["standardization"] = {
        "sample_rule": "quality_tier == core; missing and infinite values are excluded from fitting.",
        "core_city_count": int(city_base["quality_tier"].eq("core").sum()),
        "signature_feature_count": len([col for col in signature_matrix.columns if col.startswith("sig__")]),
        "signature_params": signature_standardization,
        "transfer_feature_count": int(transfer_features["transfer_feature"].nunique()),
        "transfer_params": transfer_standardization,
        "step09_candidate_metric_params": candidate_standardization,
    }
    log["local_aggregation"] = local_agg_log
    log["step09_candidate_counts"] = {
        "rows": int(step09_candidates.shape[0]),
        "main_candidate_rows": int(step09_candidates["step09_main_candidate"].sum()),
        "sensitivity_candidate_rows": int(step09_candidates["step09_sensitivity_candidate"].sum()),
        "retained_excluded_candidate_rows": int(step09_candidates["retained_excluded_candidate"].sum()),
        "by_quality_tier": step09_candidates["quality_tier"].value_counts(dropna=False).to_dict(),
        "by_network_scale": step09_candidates.groupby(["network_type", "scale"], observed=True).size().to_dict(),
    }
    log["qc_summary"] = {
        "city_count": int(city_base["city_id"].nunique()),
        "city_network_rows": int(quality_check.shape[0]),
        "all_four_scales_present_rows": int(quality_check["all_four_scales_present"].sum()),
        "anomaly_rows": int((quality_check["anomaly_notes"].astype(str) != "").sum()),
        "excluded_candidate_main_violations": int(quality_check["excluded_candidate_main_violation"].sum()),
    }
    log["run_finished_at"] = now_iso()
    log["run_seconds"] = round(time.time() - start_time, 3)

    outputs = {
        "scale_signatures_long": signature_long,
        "scale_signatures_matrix": signature_matrix,
        "scale_transfer_features": transfer_features,
        "city_profiles_base": city_profile,
        "step09_morphotype_candidate_units": step09_candidates,
        "scale_signatures_summary_stats": description_stats,
        "scale_signatures_quality_checks": quality_check,
    }

    print(f"[{now_iso()}] Writing outputs to {out_dir}...")
    write_outputs(out_dir, outputs, log)
    print(f"[{now_iso()}] Step08 complete in {log['run_seconds']} seconds.")
    for key, df in outputs.items():
        print(f"  {key}: {df.shape[0]} rows x {df.shape[1]} columns")


if __name__ == "__main__":
    main()
