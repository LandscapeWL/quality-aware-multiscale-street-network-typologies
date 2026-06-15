#!/usr/bin/env python3
"""Step09: cluster local street-network morphotypes from Step08 outputs.

The fitting sample is strictly Step08 ``step09_main_candidate == True``.
Sensitivity cities, China pressure-test cities, walk-network rows, and retained
excluded candidates are projected onto the fitted model for review and
downstream comparison.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture


STEP_NAME = "09_cluster_morphotypes"
UPSTREAM_STEP = "08_generate_scale_signatures_city_profiles"

RAW_FEATURE_COLUMNS = [
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
Z_FEATURE_COLUMNS = [f"{feature}_z" for feature in RAW_FEATURE_COLUMNS]
Z_TO_RAW = dict(zip(Z_FEATURE_COLUMNS, RAW_FEATURE_COLUMNS))
RAW_TO_Z = dict(zip(RAW_FEATURE_COLUMNS, Z_FEATURE_COLUMNS))

REQUIRED_CANDIDATE_COLUMNS = [
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
    "valid_area_ratio",
    "edge_unit",
    "center_lon",
    "center_lat",
    "population_weight",
    "population_sum",
    "valid_metric",
    "valid_metric_quality",
    "local_metric_validity_score",
    "local_quality_flag",
    "use_in_local_clustering",
    "step09_main_candidate",
    "step09_sensitivity_candidate",
    "retained_excluded_candidate",
    "metric_missing_count",
    "all_clustering_metrics_present",
] + RAW_FEATURE_COLUMNS + Z_FEATURE_COLUMNS

OUTPUT_FILES = {
    "unit_parquet": "local_morphotypes.parquet",
    "unit_csv": "local_morphotypes.csv",
    "city_parquet": "city_morphotype_profiles.parquet",
    "city_csv": "city_morphotype_profiles.csv",
    "center_parquet": "morphotype_centers.parquet",
    "center_csv": "morphotype_centers.csv",
    "contrib_csv": "cluster_feature_contributions.csv",
    "diagnostics_csv": "clustering_model_diagnostics.csv",
    "feature_qc_csv": "feature_quality_checks.csv",
    "handoff_json": "step08_to_step09_handoff_report.json",
    "handoff_md": "step08_to_step09_handoff_report.md",
    "repro_json": "reproducibility_status_manifest.json",
    "run_log": "09_run_log.json",
    "model": "morphotype_clustering_model.pkl",
    "readme": "README.md",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root directory.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Step08 output directory. Defaults to data/08_generate_scale_signatures_city_profiles under --root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Step09 output directory. Defaults to data/09_cluster_morphotypes under --root.",
    )
    parser.add_argument("--k-min", type=int, default=4, help="Minimum GMM component count to evaluate.")
    parser.add_argument("--k-max", type=int, default=10, help="Maximum GMM component count to evaluate.")
    parser.add_argument(
        "--final-k",
        default="auto",
        help="Final GMM component count, or 'auto' to choose the lowest-BIC candidate.",
    )
    parser.add_argument(
        "--balance-max-per-city-scale",
        type=int,
        default=600,
        help="Maximum main-candidate rows sampled from each city/network/scale group for PCA and GMM fitting.",
    )
    parser.add_argument(
        "--eval-sample-size",
        type=int,
        default=20_000,
        help="Maximum balanced main rows sampled for candidate-k diagnostics.",
    )
    parser.add_argument(
        "--ward-sample-size",
        type=int,
        default=3_000,
        help="Maximum balanced main rows sampled for optional Ward diagnostics.",
    )
    parser.add_argument(
        "--clip-quantile",
        type=float,
        default=0.005,
        help="Two-sided feature clipping quantile learned from the balanced main training sample.",
    )
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for sampling and model fitting.")
    parser.add_argument(
        "--min-cluster-share",
        type=float,
        default=0.01,
        help="Minimum diagnostic sample share allowed when auto-selecting final k.",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [json_ready(v) for v in obj.tolist()]
    return scalar(obj)


def finite_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def dependency_versions() -> dict[str, str]:
    import importlib.metadata as md

    versions = {"python": sys.version.split()[0]}
    for package in ["pandas", "numpy", "scikit-learn", "pyarrow", "joblib"]:
        try:
            versions[package] = md.version(package)
        except md.PackageNotFoundError:
            versions[package] = "not_installed"
    versions["hdbscan"] = "installed" if importlib.util.find_spec("hdbscan") else "not_installed"
    return versions


def file_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    meta: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
    }
    if path.suffix == ".parquet":
        parquet = pq.ParquetFile(path)
        schema = parquet.schema_arrow
        meta.update(
            {
                "format": "parquet",
                "row_count": int(parquet.metadata.num_rows),
                "column_count": len(schema.names),
                "columns": list(schema.names),
            }
        )
    elif path.suffix == ".csv":
        header = pd.read_csv(path, nrows=0)
        meta.update({"format": "csv", "column_count": int(header.shape[1]), "columns": list(header.columns)})
    elif path.suffix == ".json":
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            keys = list(obj.keys()) if isinstance(obj, dict) else []
        except json.JSONDecodeError:
            keys = []
        meta.update({"format": "json", "top_level_keys": keys})
    return meta


def collect_input_metadata(input_dir: Path) -> dict[str, Any]:
    paths = [
        input_dir / "step09_morphotype_candidate_units.parquet",
        input_dir / "scale_signatures_matrix.parquet",
        input_dir / "city_profiles_base.parquet",
        input_dir / "scale_transfer_features.parquet",
        input_dir / "scale_signatures_quality_checks.csv",
        input_dir / "scale_signatures_long.parquet",
        input_dir / "scale_signatures_summary_stats.csv",
        input_dir / "08_run_log.json",
    ]
    return {path.name: file_metadata(path) for path in paths if path.exists()}


def input_integrity_record(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "path": meta.get("path"),
            "exists": meta.get("exists", True),
            "size_bytes": meta.get("size_bytes"),
            "mtime": meta.get("mtime"),
        }
        for name, meta in sorted(metadata.items())
    }


def compare_input_integrity(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_record = input_integrity_record(before)
    after_record = input_integrity_record(after)
    file_names = sorted(set(before_record) | set(after_record))
    files: dict[str, Any] = {}
    changed_files: list[str] = []
    missing_after: list[str] = []
    new_after: list[str] = []
    for name in file_names:
        before_meta = before_record.get(name)
        after_meta = after_record.get(name)
        if before_meta is None:
            same = False
            new_after.append(name)
        elif after_meta is None:
            same = False
            missing_after.append(name)
        else:
            same = (
                before_meta.get("exists") == after_meta.get("exists")
                and before_meta.get("size_bytes") == after_meta.get("size_bytes")
                and before_meta.get("mtime") == after_meta.get("mtime")
            )
        if not same:
            changed_files.append(name)
        files[name] = {"unchanged": same, "before": before_meta, "after": after_meta}
    return {
        "checked_at": now_iso(),
        "comparison": "size_bytes_and_mtime_before_read_vs_after_step09_write",
        "all_unchanged": len(changed_files) == 0,
        "changed_files": changed_files,
        "missing_after": missing_after,
        "new_after": new_after,
        "files": files,
    }


def read_step08_anomalies(input_dir: Path) -> dict[str, Any]:
    path = input_dir / "scale_signatures_quality_checks.csv"
    if not path.exists():
        return {"path": str(path), "status": "missing", "anomaly_rows": []}
    qc = pd.read_csv(path)
    if "anomaly_notes" not in qc.columns:
        return {"path": str(path), "status": "no_anomaly_notes_column", "anomaly_rows": []}
    anomaly_rows = qc[qc["anomaly_notes"].fillna("").astype(str).ne("")].copy()
    keep = [
        col
        for col in [
            "city_id",
            "city_name_en",
            "network_type",
            "quality_tier",
            "anomaly_notes",
            "walk_core_5km_invalid_dhaka_istanbul_flag",
            "excluded_candidate_main_violation",
        ]
        if col in anomaly_rows.columns
    ]
    return {
        "path": str(path),
        "status": "ok",
        "anomaly_count": int(anomaly_rows.shape[0]),
        "anomaly_rows": anomaly_rows[keep].to_dict(orient="records"),
    }


def add_check(checks: list[dict[str, Any]], name: str, status: str, details: Any, severity: str = "error") -> None:
    checks.append({"check": name, "status": status, "severity": severity, "details": details})


def shared_ucdb_city_flags(city_base: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "city_id",
        "city_name_en",
        "ucdb_id",
        "ucdb_name_main",
        "duplicate_ucdb_id_with",
        "match_note",
    ]
    existing = [col for col in cols if col in city_base.columns]
    out = city_base[existing].drop_duplicates("city_id").copy()
    if "ucdb_id" not in out.columns:
        out["shared_ucdb_boundary_flag"] = False
        out["shared_ucdb_id"] = pd.NA
        out["shared_ucdb_name"] = ""
        out["shared_ucdb_boundary_group"] = ""
        out["shared_ucdb_boundary_note"] = ""
        return out[
            [
                "city_id",
                "shared_ucdb_boundary_flag",
                "shared_ucdb_id",
                "shared_ucdb_name",
                "shared_ucdb_boundary_group",
                "shared_ucdb_boundary_note",
            ]
        ]

    dup_ucdb = out["ucdb_id"].notna() & out["ucdb_id"].duplicated(keep=False)
    sg_pair = out["city_name_en"].isin(["Shenzhen", "Guangzhou"]) & out["ucdb_id"].eq(10933) & dup_ucdb
    out["shared_ucdb_boundary_flag"] = sg_pair
    out["shared_ucdb_id"] = np.where(sg_pair, out["ucdb_id"], pd.NA)
    out["shared_ucdb_name"] = np.where(sg_pair, out.get("ucdb_name_main", ""), "")
    out["shared_ucdb_boundary_group"] = np.where(sg_pair, "UCDB_10933_Shenzhen_Guangzhou", "")
    out["shared_ucdb_boundary_note"] = np.where(
        sg_pair,
        "Shenzhen and Guangzhou are kept as separate city_id values but share UCDB urban-centre boundary 10933.",
        "",
    )
    return out[
        [
            "city_id",
            "shared_ucdb_boundary_flag",
            "shared_ucdb_id",
            "shared_ucdb_name",
            "shared_ucdb_boundary_group",
            "shared_ucdb_boundary_note",
        ]
    ]


def validate_handoff(
    candidates: pd.DataFrame,
    city_base: pd.DataFrame,
    input_metadata: dict[str, Any],
    input_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    missing_columns = [col for col in REQUIRED_CANDIDATE_COLUMNS if col not in candidates.columns]
    add_check(
        checks,
        "required_candidate_columns",
        "pass" if not missing_columns else "fail",
        {"missing_columns": missing_columns, "required_count": len(REQUIRED_CANDIDATE_COLUMNS)},
    )
    if missing_columns:
        raise ValueError(f"Missing required Step09 candidate columns: {missing_columns}")

    expected_inputs = [
        "step09_morphotype_candidate_units.parquet",
        "scale_signatures_matrix.parquet",
        "city_profiles_base.parquet",
        "scale_transfer_features.parquet",
        "scale_signatures_quality_checks.csv",
    ]
    missing_inputs = [name for name in expected_inputs if name not in input_metadata]
    add_check(
        checks,
        "required_step08_inputs_present",
        "pass" if not missing_inputs else "fail",
        {"missing_inputs": missing_inputs, "expected_inputs": expected_inputs},
    )
    if missing_inputs:
        raise FileNotFoundError(f"Missing Step08 required/auxiliary inputs: {missing_inputs}")

    duplicate_keys = int(candidates.duplicated(["city_id", "unit_id", "network_type", "scale"]).sum())
    add_check(
        checks,
        "local_morphotype_primary_key_unique",
        "pass" if duplicate_keys == 0 else "fail",
        {"duplicate_rows": duplicate_keys, "key": ["city_id", "unit_id", "network_type", "scale"]},
    )
    if duplicate_keys:
        raise ValueError("Step09 candidate unit keys are not unique.")

    main_mask = candidates["step09_main_candidate"].fillna(False).astype(bool)
    main_count = int(main_mask.sum())
    add_check(
        checks,
        "main_candidate_count",
        "pass" if main_count >= 1_000 else "fail",
        {"main_candidate_rows": main_count},
    )
    if main_count < 1_000:
        raise ValueError(f"Too few main candidates for clustering: {main_count}")

    main_sensitivity = int((main_mask & candidates["quality_tier"].ne("core")).sum())
    main_china = int((main_mask & candidates["sample_group"].eq("china_pressure_test")).sum())
    main_excluded = int((main_mask & candidates["retained_excluded_candidate"]).sum())
    add_check(
        checks,
        "main_training_scope_strict",
        "pass" if main_sensitivity == 0 and main_china == 0 and main_excluded == 0 else "fail",
        {
            "non_core_main_rows": main_sensitivity,
            "china_pressure_main_rows": main_china,
            "retained_excluded_main_rows": main_excluded,
        },
    )
    if main_sensitivity or main_china or main_excluded:
        raise ValueError("Main training sample contains sensitivity, China pressure-test, or excluded rows.")

    feature_missing = candidates[Z_FEATURE_COLUMNS].isna().sum().sort_values(ascending=False)
    add_check(
        checks,
        "step08_z_feature_missing_values",
        "warn" if int(feature_missing.sum()) else "pass",
        {"total_missing": int(feature_missing.sum()), "by_z_feature": feature_missing.to_dict()},
        severity="warning",
    )

    city_ids_in_candidates = set(candidates["city_id"].dropna())
    city_ids_in_base = set(city_base["city_id"].dropna())
    missing_city_meta = sorted(city_ids_in_candidates - city_ids_in_base)
    add_check(
        checks,
        "candidate_cities_have_city_profile",
        "pass" if not missing_city_meta else "fail",
        {"missing_city_ids": missing_city_meta},
    )
    if missing_city_meta:
        raise ValueError(f"Candidate city_ids missing from city profile: {missing_city_meta[:10]}")

    shared = shared_ucdb_city_flags(city_base)
    sg = shared[shared["shared_ucdb_boundary_flag"]]
    detected = sorted(city_base.merge(sg[["city_id"]], on="city_id")["city_name_en"].dropna().unique())
    add_check(
        checks,
        "shenzhen_guangzhou_shared_ucdb_10933_detected",
        "pass" if {"Guangzhou", "Shenzhen"}.issubset(set(detected)) else "warn",
        {"detected_cities": detected, "expected_shared_ucdb_id": 10933},
        severity="warning",
    )

    add_check(
        checks,
        "step08_outputs_not_overwritten_by_step09",
        "pass" if input_dir.resolve() != output_dir.resolve() else "fail",
        {"input_dir": str(input_dir), "output_dir": str(output_dir)},
    )
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("Step09 output_dir must not be the Step08 input_dir.")

    status_counts = pd.Series([check["status"] for check in checks]).value_counts().to_dict()
    return {
        "generated_at": now_iso(),
        "upstream_step": UPSTREAM_STEP,
        "input_metadata": input_metadata,
        "step08_input_integrity": {},
        "candidate_rows": int(candidates.shape[0]),
        "candidate_columns": int(candidates.shape[1]),
        "city_rows": int(city_base.shape[0]),
        "checks": checks,
        "status_counts": status_counts,
        "overall_status": "fail" if any(check["status"] == "fail" for check in checks) else "pass",
    }


def make_balanced_main_indices(candidates: pd.DataFrame, max_per_group: int, seed: int) -> np.ndarray:
    main = candidates[candidates["step09_main_candidate"]].copy()
    if main.empty:
        raise ValueError("No step09_main_candidate rows available.")
    rng = np.random.default_rng(seed)
    sampled: list[np.ndarray] = []
    for _, group in main.groupby(["city_id", "network_type", "scale"], observed=True, sort=True):
        idx = group.index.to_numpy()
        if len(idx) > max_per_group:
            idx = rng.choice(idx, size=max_per_group, replace=False)
        sampled.append(idx)
    balanced = np.concatenate(sampled)
    rng.shuffle(balanced)
    return balanced


def feature_qc_table(candidates: pd.DataFrame, balanced_idx: np.ndarray) -> tuple[pd.DataFrame, list[str], list[str]]:
    main_mask = candidates["step09_main_candidate"].fillna(False).astype(bool)
    rows: list[dict[str, Any]] = []
    selected: list[str] = []
    removed: list[str] = []
    for z_col in Z_FEATURE_COLUMNS:
        raw_col = Z_TO_RAW[z_col]
        all_values = finite_numeric(candidates[z_col])
        main_values = finite_numeric(candidates.loc[main_mask, z_col])
        balanced_values = finite_numeric(candidates.loc[balanced_idx, z_col])
        nonnull_main = int(main_values.notna().sum())
        nunique_main = int(main_values.nunique(dropna=True))
        if nonnull_main == 0:
            selected_for_model = False
            reason = "all_nan_in_step09_main_candidate_z_feature"
        elif nunique_main <= 1:
            selected_for_model = False
            reason = "constant_in_step09_main_candidate_z_feature"
        else:
            selected_for_model = True
            reason = ""
            selected.append(z_col)
        if not selected_for_model:
            removed.append(z_col)
        rows.append(
            {
                "raw_feature": raw_col,
                "z_feature": z_col,
                "selected_for_model": selected_for_model,
                "removed_reason": reason,
                "missing_rows_all": int(all_values.isna().sum()),
                "missing_rate_all": float(all_values.isna().mean()),
                "missing_rows_main_candidate": int(main_values.isna().sum()),
                "missing_rate_main_candidate": float(main_values.isna().mean()),
                "nonnull_rows_main_candidate": nonnull_main,
                "nunique_main_candidate": nunique_main,
                "balanced_training_rows": int(len(balanced_values)),
                "balanced_missing_rows": int(balanced_values.isna().sum()),
                "balanced_median_z": float(balanced_values.median()) if balanced_values.notna().any() else np.nan,
                "balanced_p01_z": float(balanced_values.quantile(0.01)) if balanced_values.notna().any() else np.nan,
                "balanced_p99_z": float(balanced_values.quantile(0.99)) if balanced_values.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows), selected, removed


def preprocess_z_features(
    candidates: pd.DataFrame,
    balanced_idx: np.ndarray,
    selected_z_features: list[str],
    clip_quantile: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], SimpleImputer]:
    if not 0 <= clip_quantile < 0.5:
        raise ValueError("--clip-quantile must be in [0, 0.5).")
    raw_all = candidates[selected_z_features].apply(finite_numeric).to_numpy(dtype=float)
    raw_train = candidates.loc[balanced_idx, selected_z_features].apply(finite_numeric).to_numpy(dtype=float)

    imputer = SimpleImputer(strategy="median")
    train_imputed = imputer.fit_transform(raw_train)
    all_imputed = imputer.transform(raw_all)

    lower = np.nanquantile(train_imputed, clip_quantile, axis=0) if clip_quantile > 0 else np.full(len(selected_z_features), -np.inf)
    upper = (
        np.nanquantile(train_imputed, 1.0 - clip_quantile, axis=0)
        if clip_quantile > 0
        else np.full(len(selected_z_features), np.inf)
    )
    bad = ~np.isfinite(lower) | ~np.isfinite(upper) | (lower >= upper)
    lower[bad] = -np.inf
    upper[bad] = np.inf

    train_clipped = np.clip(train_imputed, lower, upper)
    all_clipped = np.clip(all_imputed, lower, upper)
    clip_table = [
        {
            "z_feature": feature,
            "raw_feature": Z_TO_RAW[feature],
            "clip_lower": float(lo) if np.isfinite(lo) else None,
            "clip_upper": float(hi) if np.isfinite(hi) else None,
            "impute_median": float(median),
        }
        for feature, lo, hi, median in zip(selected_z_features, lower, upper, imputer.statistics_)
    ]
    metadata = {
        "feature_source": "Step08 *_z columns",
        "imputation": "median fitted on balanced step09_main_candidate sample only",
        "standardization": "no Step09 refit; Step08 z-scores are used directly",
        "clipping": "per-feature quantile thresholds fitted on balanced step09_main_candidate sample only",
        "clip_quantile": clip_quantile,
        "selected_z_features": selected_z_features,
        "selected_raw_features": [Z_TO_RAW[col] for col in selected_z_features],
        "clip_table": clip_table,
    }
    return train_clipped, all_clipped, metadata, imputer


def fit_pca(train_matrix: np.ndarray, all_matrix: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, PCA, dict[str, Any]]:
    max_components = min(8, train_matrix.shape[1], train_matrix.shape[0])
    min_components = min(3, max_components)
    full = PCA(n_components=max_components, random_state=seed)
    full.fit(train_matrix)
    cumulative = np.cumsum(full.explained_variance_ratio_)
    needed = int(np.searchsorted(cumulative, 0.85) + 1)
    n_components = min(max(min_components, needed), max_components)
    pca = PCA(n_components=n_components, random_state=seed)
    train_pca = pca.fit_transform(train_matrix)
    all_pca = pca.transform(all_matrix)
    info = {
        "fit_sample": "balanced step09_main_candidate sample",
        "n_components": int(n_components),
        "min_components": int(min_components),
        "max_components": int(max_components),
        "target_explained_variance": 0.85,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "target_met": bool(np.sum(pca.explained_variance_ratio_) >= 0.85),
        "umap_status": "not_run; PCA coordinates are used as 2D fallback",
    }
    return train_pca, all_pca, pca, info


def sample_matrix(matrix: np.ndarray, sample_size: int, seed: int) -> np.ndarray:
    if matrix.shape[0] <= sample_size:
        return matrix
    rng = np.random.default_rng(seed)
    idx = rng.choice(matrix.shape[0], size=sample_size, replace=False)
    return matrix[idx]


def model_metrics(matrix: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, float]:
    unique = np.unique(labels)
    if len(unique) < 2:
        return {"silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan}
    return {
        "silhouette": float(silhouette_score(matrix, labels, sample_size=min(10_000, matrix.shape[0]), random_state=seed)),
        "calinski_harabasz": float(calinski_harabasz_score(matrix, labels)),
        "davies_bouldin": float(davies_bouldin_score(matrix, labels)),
    }


def evaluate_models(
    train_pca: np.ndarray,
    k_min: int,
    k_max: int,
    eval_sample_size: int,
    ward_sample_size: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if k_min < 2 or k_max < k_min:
        raise ValueError("K range must satisfy 2 <= k_min <= k_max.")
    eval_matrix = sample_matrix(train_pca, eval_sample_size, seed)
    rows: list[dict[str, Any]] = []
    for k in range(k_min, k_max + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=seed,
            n_init=3,
            max_iter=500,
            reg_covar=1e-6,
        )
        labels = gmm.fit_predict(eval_matrix)
        counts = np.bincount(labels, minlength=k)
        metrics = model_metrics(eval_matrix, labels, seed)
        rows.append(
            {
                "algorithm": "gmm_full",
                "k": k,
                "status": "ok",
                "eval_rows": int(eval_matrix.shape[0]),
                **metrics,
                "aic": float(gmm.aic(eval_matrix)),
                "bic": float(gmm.bic(eval_matrix)),
                "min_cluster_share": float(counts.min() / counts.sum()),
                "max_cluster_share": float(counts.max() / counts.sum()),
                "cluster_sizes": json.dumps(counts.astype(int).tolist()),
                "not_run_reason": "",
            }
        )

    ward_matrix = sample_matrix(train_pca, ward_sample_size, seed + 17)
    ward_status = {"status": "ok", "eval_rows": int(ward_matrix.shape[0])}
    for k in range(k_min, k_max + 1):
        ward = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = ward.fit_predict(ward_matrix)
        counts = np.bincount(labels, minlength=k)
        metrics = model_metrics(ward_matrix, labels, seed)
        rows.append(
            {
                "algorithm": "ward",
                "k": k,
                "status": "ok",
                "eval_rows": int(ward_matrix.shape[0]),
                **metrics,
                "aic": np.nan,
                "bic": np.nan,
                "min_cluster_share": float(counts.min() / counts.sum()),
                "max_cluster_share": float(counts.max() / counts.sum()),
                "cluster_sizes": json.dumps(counts.astype(int).tolist()),
                "not_run_reason": "",
            }
        )

    hdbscan_status: dict[str, Any]
    if importlib.util.find_spec("hdbscan") is None:
        hdbscan_status = {"status": "not_run", "not_run_reason": "hdbscan package is not installed"}
        rows.append(
            {
                "algorithm": "hdbscan",
                "k": np.nan,
                "status": "not_run",
                "eval_rows": int(eval_matrix.shape[0]),
                "silhouette": np.nan,
                "calinski_harabasz": np.nan,
                "davies_bouldin": np.nan,
                "aic": np.nan,
                "bic": np.nan,
                "min_cluster_share": np.nan,
                "max_cluster_share": np.nan,
                "cluster_sizes": "",
                "not_run_reason": "hdbscan package is not installed",
            }
        )
    else:
        hdbscan_status = {"status": "available_not_run", "not_run_reason": "kept optional for reproducible baseline"}
        rows.append(
            {
                "algorithm": "hdbscan",
                "k": np.nan,
                "status": "available_not_run",
                "eval_rows": int(eval_matrix.shape[0]),
                "silhouette": np.nan,
                "calinski_harabasz": np.nan,
                "davies_bouldin": np.nan,
                "aic": np.nan,
                "bic": np.nan,
                "min_cluster_share": np.nan,
                "max_cluster_share": np.nan,
                "cluster_sizes": "",
                "not_run_reason": "kept optional for reproducible baseline",
            }
        )

    notes = {
        "gmm_status": "ok",
        "ward_status": ward_status,
        "hdbscan_status": hdbscan_status,
        "selection_rule": "auto chooses gmm_full with lowest BIC among candidates meeting min_cluster_share",
    }
    return pd.DataFrame(rows), notes


def choose_final_k(diagnostics: pd.DataFrame, final_k: str, min_cluster_share: float) -> int:
    if final_k != "auto":
        k = int(final_k)
        if k <= 1:
            raise ValueError("--final-k must be 'auto' or an integer greater than 1.")
        return k
    gmm = diagnostics[diagnostics["algorithm"].eq("gmm_full") & diagnostics["status"].eq("ok")].copy()
    eligible = gmm[gmm["min_cluster_share"].ge(min_cluster_share)].copy()
    if eligible.empty:
        eligible = gmm
    eligible = eligible.sort_values(["bic", "silhouette"], ascending=[True, False])
    return int(eligible.iloc[0]["k"])


def fit_final_gmm(train_pca: np.ndarray, k: int, seed: int) -> GaussianMixture:
    model = GaussianMixture(
        n_components=k,
        covariance_type="full",
        random_state=seed,
        n_init=5,
        max_iter=700,
        reg_covar=1e-6,
    )
    model.fit(train_pca)
    return model


def mean_available(values: list[float]) -> float:
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if arr.size == 0:
        return np.nan
    return float(arr.mean())


def descriptor_scores(center_z: dict[str, float]) -> dict[str, float]:
    density = mean_available(
        [
            center_z.get("edge_density_km_per_km2_z", np.nan),
            center_z.get("node_density_per_km2_z", np.nan),
            center_z.get("intersection_density_per_km2_z", np.nan),
        ]
    )
    segment = mean_available(
        [
            center_z.get("segment_length_mean_z", np.nan),
            center_z.get("segment_length_median_z", np.nan),
            center_z.get("segment_length_p90_z", np.nan),
        ]
    )
    four_way = center_z.get("four_way_share_z", np.nan)
    dead_end = center_z.get("dead_end_share_z", np.nan)
    orientation_order = center_z.get("orientation_order_z", np.nan)
    grid = mean_available([four_way, orientation_order, -dead_end if np.isfinite(dead_end) else np.nan])
    fine_grain = mean_available([density, -segment if np.isfinite(segment) else np.nan])
    culdesac = mean_available([dead_end, -four_way if np.isfinite(four_way) else np.nan])
    hierarchy = mean_available(
        [
            center_z.get("road_hierarchy_entropy_z", np.nan),
            center_z.get("betweenness_gini_z", np.nan),
            center_z.get("segment_length_p90_z", np.nan),
            -center_z.get("intersection_density_per_km2_z", np.nan)
            if np.isfinite(center_z.get("intersection_density_per_km2_z", np.nan))
            else np.nan,
        ]
    )
    circuity = mean_available([center_z.get("edge_circuity_mean_z", np.nan), center_z.get("edge_circuity_median_z", np.nan)])
    return {
        "density_score": density,
        "segment_length_score": segment,
        "grid_score": grid,
        "fine_grain_score": fine_grain,
        "culdesac_score": culdesac,
        "hierarchy_score": hierarchy,
        "circuity_score": circuity,
    }


def band(value: float, high: float = 0.45, low: float = -0.45) -> str:
    if not np.isfinite(value):
        return "mixed"
    if value >= high:
        return "high"
    if value <= low:
        return "low"
    return "medium"


def morphotype_name(scores: dict[str, float]) -> str:
    density_label = {
        "high": "High-density ",
        "medium": "Medium-density ",
        "low": "Low-density ",
        "mixed": "Mixed-density ",
    }[band(scores["density_score"])]
    candidates = [
        ("grid_score", "ordered grid"),
        ("fine_grain_score", "fine-grained blocks"),
        ("culdesac_score", "cul-de-sac network"),
        ("hierarchy_score", "hierarchical arterials"),
        ("circuity_score", "curvilinear network"),
    ]
    best_key, best_label = max(candidates, key=lambda item: scores.get(item[0], -np.inf))
    if not np.isfinite(scores.get(best_key, np.nan)) or scores.get(best_key, 0.0) < 0.25:
        best_label = "mixed fabric"
    return f"{density_label}{best_label}"


def build_cluster_tables(
    candidates: pd.DataFrame,
    balanced_idx: np.ndarray,
    train_matrix_z: np.ndarray,
    train_pca: np.ndarray,
    selected_z_features: list[str],
    model: GaussianMixture,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str]]:
    train_rows = candidates.loc[balanced_idx].copy()
    labels = model.predict(train_pca)
    train_rows["cluster_raw_label"] = labels
    z_df = pd.DataFrame(train_matrix_z, columns=selected_z_features, index=train_rows.index)
    z_df["cluster_raw_label"] = labels
    pca_df = pd.DataFrame(train_pca, columns=[f"pca_model_{i + 1}" for i in range(train_pca.shape[1])], index=train_rows.index)
    pca_df["cluster_raw_label"] = labels

    center_z = z_df.groupby("cluster_raw_label", observed=True)[selected_z_features].mean()
    raw_centers = train_rows.groupby("cluster_raw_label", observed=True)[RAW_FEATURE_COLUMNS].median(numeric_only=True)
    pca_centers = pca_df.groupby("cluster_raw_label", observed=True).mean()
    counts = train_rows["cluster_raw_label"].value_counts().sort_index()
    total = int(counts.sum())

    rows: list[dict[str, Any]] = []
    for raw_label in sorted(center_z.index):
        center_dict = center_z.loc[raw_label].to_dict()
        scores = descriptor_scores(center_dict)
        row = {
            "cluster_raw_label": int(raw_label),
            "n_balanced_training_units": int(counts.loc[raw_label]),
            "balanced_training_share": float(counts.loc[raw_label] / total),
            **scores,
            "morphotype_name": morphotype_name(scores),
        }
        for raw_feature in RAW_FEATURE_COLUMNS:
            row[f"{raw_feature}__median_raw"] = (
                float(raw_centers.loc[raw_label, raw_feature]) if raw_feature in raw_centers.columns else np.nan
            )
        for z_feature in selected_z_features:
            row[f"{z_feature}__center"] = float(center_z.loc[raw_label, z_feature])
        for pca_col in pca_centers.columns:
            row[pca_col] = float(pca_centers.loc[raw_label, pca_col])
        rows.append(row)

    center_df = pd.DataFrame(rows).sort_values(
        ["density_score", "grid_score", "fine_grain_score", "n_balanced_training_units"],
        ascending=[False, False, False, False],
    )
    center_df = center_df.reset_index(drop=True)
    center_df["morphotype"] = [f"MT{i:02d}" for i in range(1, len(center_df) + 1)]
    center_df["morphotype_label"] = center_df["morphotype"] + "_" + center_df["morphotype_name"]
    raw_to_mtype = center_df.set_index("cluster_raw_label")["morphotype"].to_dict()

    global_center = z_df[selected_z_features].mean()
    contrib_rows: list[dict[str, Any]] = []
    for _, center in center_df.iterrows():
        raw_label = int(center["cluster_raw_label"])
        diff = center_z.loc[raw_label, selected_z_features] - global_center
        ranked = diff.abs().sort_values(ascending=False)
        for rank, z_feature in enumerate(ranked.index, start=1):
            contrib_rows.append(
                {
                    "morphotype": center["morphotype"],
                    "morphotype_name": center["morphotype_name"],
                    "cluster_raw_label": raw_label,
                    "raw_feature": Z_TO_RAW[z_feature],
                    "z_feature": z_feature,
                    "center_z": float(center_z.loc[raw_label, z_feature]),
                    "balanced_global_mean_z": float(global_center.loc[z_feature]),
                    "contribution_from_global": float(diff.loc[z_feature]),
                    "abs_contribution_rank": rank,
                    "direction": "high" if diff.loc[z_feature] >= 0 else "low",
                }
            )
    return center_df, pd.DataFrame(contrib_rows), raw_to_mtype


def normalized_entropy(counts: np.ndarray) -> float:
    counts = counts.astype(float)
    total = counts.sum()
    if total <= 0 or counts.size <= 1:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum() / math.log(counts.size))


def scale_stability(unit_results: pd.DataFrame, morphotypes: list[str]) -> pd.Series:
    profile_rows = unit_results[unit_results["main_profile_eligible"]].copy()
    if profile_rows.empty:
        return pd.Series(np.nan, index=unit_results.index)
    distribution = (
        profile_rows.groupby(["city_id", "network_type", "scale", "morphotype"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    distribution["share"] = distribution["n"] / distribution.groupby(["city_id", "network_type", "scale"], observed=True)[
        "n"
    ].transform("sum")
    vectors: dict[tuple[str, str, str], np.ndarray] = {}
    for key, group in distribution.groupby(["city_id", "network_type", "scale"], observed=True):
        vectors[tuple(key)] = group.set_index("morphotype")["share"].reindex(morphotypes, fill_value=0.0).to_numpy(float)

    similarities: dict[tuple[str, str], float] = {}
    for city_id, network_type in profile_rows[["city_id", "network_type"]].drop_duplicates().itertuples(index=False, name=None):
        v1 = vectors.get((city_id, network_type, "hex_1km"))
        v2 = vectors.get((city_id, network_type, "hex_2km"))
        if v1 is None or v2 is None:
            similarities[(city_id, network_type)] = np.nan
        else:
            similarities[(city_id, network_type)] = float(np.clip(1.0 - 0.5 * np.abs(v1 - v2).sum(), 0.0, 1.0))
    return pd.Series(
        [similarities.get((row.city_id, row.network_type), np.nan) for row in unit_results[["city_id", "network_type"]].itertuples()],
        index=unit_results.index,
    )


def weighted_geometric_confidence(
    cluster_probability: pd.Series,
    quality_component: pd.Series,
    scale_component: pd.Series,
    local_quality_component: pd.Series,
) -> pd.Series:
    eps = 1e-6
    components = [
        (cluster_probability.clip(0, 1).fillna(0.0), 0.50),
        (quality_component.clip(0, 1).fillna(0.0), 0.25),
        (scale_component.clip(0, 1).fillna(0.5), 0.15),
        (local_quality_component.clip(0, 1).fillna(0.5), 0.10),
    ]
    log_score = sum(weight * np.log(series + eps) for series, weight in components)
    return pd.Series(np.exp(log_score), index=cluster_probability.index).clip(0, 1)


def assign_units(
    candidates: pd.DataFrame,
    all_pca: np.ndarray,
    model: GaussianMixture,
    center_df: pd.DataFrame,
    raw_to_mtype: dict[int, str],
    shared_flags: pd.DataFrame,
) -> pd.DataFrame:
    probs = model.predict_proba(all_pca)
    labels = probs.argmax(axis=1)
    cluster_probability = probs[np.arange(probs.shape[0]), labels]
    sorted_probs = np.sort(probs, axis=1)
    probability_margin = sorted_probs[:, -1] - sorted_probs[:, -2] if probs.shape[1] > 1 else np.ones(probs.shape[0])

    type_name_map = center_df.set_index("morphotype")["morphotype_name"].to_dict()
    raw_to_name = {raw: type_name_map[mtype] for raw, mtype in raw_to_mtype.items()}
    unit = candidates.copy()
    unit["cluster_raw_label"] = labels.astype(int)
    unit["morphotype"] = unit["cluster_raw_label"].map(raw_to_mtype)
    unit["morphotype_name"] = unit["cluster_raw_label"].map(raw_to_name)
    unit["cluster_probability"] = cluster_probability
    unit["cluster_probability_margin"] = probability_margin
    unit["pca_1"] = all_pca[:, 0] if all_pca.shape[1] >= 1 else np.nan
    unit["pca_2"] = all_pca[:, 1] if all_pca.shape[1] >= 2 else np.nan
    unit["umap_1"] = np.nan
    unit["umap_2"] = np.nan

    unit["analysis_scope"] = np.select(
        [
            unit["retained_excluded_candidate"].fillna(False).astype(bool),
            unit["step09_main_candidate"].fillna(False).astype(bool),
            unit["sample_group"].eq("china_pressure_test"),
            unit["step09_sensitivity_candidate"].fillna(False).astype(bool),
        ],
        ["excluded_review", "main_fit", "external_projection_china", "sensitivity_projection"],
        default="external_projection_other",
    )
    unit["main_profile_eligible"] = ~unit["analysis_scope"].eq("excluded_review")
    unit = unit.merge(shared_flags, on="city_id", how="left", validate="many_to_one")
    unit["shared_ucdb_boundary_flag"] = unit["shared_ucdb_boundary_flag"].fillna(False).astype(bool)
    for col in ["shared_ucdb_boundary_group", "shared_ucdb_boundary_note", "shared_ucdb_name"]:
        unit[col] = unit[col].fillna("")

    morphotypes = center_df["morphotype"].tolist()
    unit["scale_stability"] = scale_stability(unit, morphotypes)
    unit["quality_component"] = finite_numeric(unit["quality_confidence_component"]).fillna(
        finite_numeric(unit["quality_score"]) / 100.0
    )
    unit["quality_component"] = unit["quality_component"].clip(0, 1)
    unit["local_quality_component"] = (finite_numeric(unit["local_metric_validity_score"]) / 100.0).clip(0, 1)
    unit["scale_stability_component"] = finite_numeric(unit["scale_stability"]).clip(0, 1).fillna(0.5)
    unit["type_confidence_uncapped"] = weighted_geometric_confidence(
        unit["cluster_probability"],
        unit["quality_component"],
        unit["scale_stability_component"],
        unit["local_quality_component"],
    )
    unit["excluded_confidence_cap"] = np.where(unit["analysis_scope"].eq("excluded_review"), 0.25, 1.0)
    unit["type_confidence"] = np.minimum(unit["type_confidence_uncapped"], unit["excluded_confidence_cap"]).clip(0, 1)
    unit["confidence_capped_flag"] = unit["type_confidence"].lt(unit["type_confidence_uncapped"] - 1e-12)

    unit["low_quality_unit_flag"] = (
        unit["analysis_scope"].eq("excluded_review")
        | ~unit["valid_metric_quality"].fillna(False).astype(bool)
        | finite_numeric(unit["valid_area_ratio"]).lt(0.5).fillna(False)
        | finite_numeric(unit["local_metric_validity_score"]).lt(60).fillna(False)
    )
    unit["low_confidence_flag"] = unit["type_confidence"].lt(0.45)
    unit["low_scale_stability_flag"] = unit["scale_stability_component"].lt(0.35)
    unit["ambiguous_cluster_flag"] = unit["cluster_probability_margin"].lt(0.10)
    unit["anomaly_unit_flag"] = (
        unit["low_quality_unit_flag"]
        | unit["low_confidence_flag"]
        | unit["low_scale_stability_flag"]
        | unit["ambiguous_cluster_flag"]
    )

    reason_cols = {
        "low_quality_or_excluded": "low_quality_unit_flag",
        "low_type_confidence": "low_confidence_flag",
        "low_scale_stability": "low_scale_stability_flag",
        "ambiguous_cluster_probability": "ambiguous_cluster_flag",
    }
    reasons: list[str] = []
    for _, row in unit.iterrows():
        reasons.append(";".join(label for label, col in reason_cols.items() if bool(row[col])))
    unit["anomaly_reason"] = reasons

    preferred = [
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
        "analysis_scope",
        "main_profile_eligible",
        "step09_main_candidate",
        "step09_sensitivity_candidate",
        "retained_excluded_candidate",
        "cluster_raw_label",
        "morphotype",
        "morphotype_name",
        "cluster_probability",
        "cluster_probability_margin",
        "quality_component",
        "local_quality_component",
        "scale_stability",
        "scale_stability_component",
        "type_confidence_uncapped",
        "excluded_confidence_cap",
        "type_confidence",
        "confidence_capped_flag",
        "pca_1",
        "pca_2",
        "umap_1",
        "umap_2",
        "low_quality_unit_flag",
        "low_confidence_flag",
        "low_scale_stability_flag",
        "ambiguous_cluster_flag",
        "anomaly_unit_flag",
        "anomaly_reason",
        "shared_ucdb_boundary_flag",
        "shared_ucdb_id",
        "shared_ucdb_name",
        "shared_ucdb_boundary_group",
        "shared_ucdb_boundary_note",
        "metric_missing_count",
        "all_clustering_metrics_present",
    ]
    final_cols = [col for col in preferred + RAW_FEATURE_COLUMNS + Z_FEATURE_COLUMNS if col in unit.columns]
    return unit[final_cols]


def build_city_profiles(unit_results: pd.DataFrame, center_df: pd.DataFrame) -> pd.DataFrame:
    profile_units = unit_results[unit_results["main_profile_eligible"]].copy()
    morphotypes = center_df["morphotype"].tolist()
    name_map = center_df.set_index("morphotype")["morphotype_name"].to_dict()
    rows: list[dict[str, Any]] = []
    group_cols = ["city_id", "network_type", "scale"]
    meta_cols = [
        "city_name_en",
        "country",
        "iso3",
        "region",
        "sample_group",
        "quality_tier",
        "shared_ucdb_boundary_flag",
        "shared_ucdb_id",
        "shared_ucdb_name",
        "shared_ucdb_boundary_group",
        "shared_ucdb_boundary_note",
    ]
    for key, group in profile_units.groupby(group_cols, observed=True, sort=True):
        first = group.iloc[0]
        counts = group["morphotype"].value_counts().reindex(morphotypes, fill_value=0)
        area = group.groupby("morphotype", observed=True)["unit_area_km2"].sum().reindex(morphotypes, fill_value=0.0)
        pop = group.groupby("morphotype", observed=True)["population_sum"].sum().reindex(morphotypes, fill_value=0.0)
        n_total = int(counts.sum())
        area_total = float(area.sum())
        pop_total = float(pop.sum())
        unit_shares = counts / n_total if n_total else counts.astype(float)
        area_shares = area / area_total if area_total > 0 else area * np.nan
        pop_shares = pop / pop_total if pop_total > 0 else pop * np.nan
        dominant = str(unit_shares.idxmax()) if n_total else ""

        row = {col: first.get(col) for col in meta_cols}
        row.update({"city_id": key[0], "network_type": key[1], "scale": key[2]})
        row.update(
            {
                "profile_scope": "non_excluded_candidates",
                "contains_main_fit_units": bool(group["analysis_scope"].eq("main_fit").any()),
                "contains_external_projection_units": bool((~group["analysis_scope"].eq("main_fit")).any()),
                "n_units": n_total,
                "n_main_fit_units": int(group["analysis_scope"].eq("main_fit").sum()),
                "n_external_projection_units": int((~group["analysis_scope"].eq("main_fit")).sum()),
                "mean_type_confidence": float(group["type_confidence"].mean()),
                "median_type_confidence": float(group["type_confidence"].median()),
                "mean_cluster_probability": float(group["cluster_probability"].mean()),
                "scale_stability": float(group["scale_stability"].median()),
                "low_quality_unit_share": float(group["low_quality_unit_flag"].mean()),
                "anomaly_unit_share": float(group["anomaly_unit_flag"].mean()),
                "dominant_morphotype": dominant,
                "dominant_morphotype_name": name_map.get(dominant, ""),
                "dominant_unit_share": float(unit_shares.max()) if n_total else np.nan,
                "dominant_area_share": float(area_shares.max()) if area_total > 0 else np.nan,
                "dominant_population_share": float(pop_shares.max()) if pop_total > 0 else np.nan,
                "morphotype_entropy": normalized_entropy(counts.to_numpy()),
                "type_unit_share_sum": float(unit_shares.sum()),
            }
        )
        for morphotype in morphotypes:
            row[f"unit_count__{morphotype}"] = int(counts.loc[morphotype])
            row[f"unit_share__{morphotype}"] = float(unit_shares.loc[morphotype])
            row[f"area_share__{morphotype}"] = float(area_shares.loc[morphotype]) if area_total > 0 else np.nan
            row[f"population_share__{morphotype}"] = float(pop_shares.loc[morphotype]) if pop_total > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def markdown_handoff(report: dict[str, Any]) -> str:
    lines = [
        "# 08 to 09 handoff acceptance",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Upstream step: `{report['upstream_step']}`",
        f"- Candidate rows: `{report['candidate_rows']}`",
        f"- City rows: `{report['city_rows']}`",
        f"- Overall status: `{report['overall_status']}`",
        "",
        "## Checks",
        "",
        "| Check | Status | Severity | Details |",
        "|---|---:|---:|---|",
    ]
    for check in report["checks"]:
        details = json.dumps(json_ready(check["details"]), ensure_ascii=False)
        if len(details) > 500:
            details = details[:497] + "..."
        lines.append(f"| {check['check']} | {check['status']} | {check['severity']} | `{details}` |")
    lines.extend(["", "## Inputs", ""])
    for name, meta in report["input_metadata"].items():
        lines.append(
            f"- `{name}`: rows={meta.get('row_count', 'NA')}, columns={meta.get('column_count', 'NA')}, "
            f"size={meta.get('size_bytes')}, mtime={meta.get('mtime')}"
        )
    integrity = report.get("step08_input_integrity") or {}
    if integrity:
        lines.extend(["", "## Step08 Input Integrity", ""])
        lines.append(f"- Comparison: `{integrity.get('comparison')}`")
        lines.append(f"- All unchanged: `{integrity.get('all_unchanged')}`")
        lines.append(f"- Changed files: `{integrity.get('changed_files')}`")
    lines.append("")
    return "\n".join(lines)


def write_readme(output_dir: Path, log: dict[str, Any]) -> None:
    run_command = log.get(
        "run_command",
        f"python code_upload/09_cluster_morphotypes/step09_cluster_morphotypes.py --root {log.get('root', '.')}",
    )
    lines = [
        "# Step09 Cluster Morphotypes",
        "",
        "This directory contains the reproducible Step09 morphotype clustering outputs.",
        "",
        f"Project root used for this run: `{log.get('root', '')}`",
        "",
        "## Run command",
        "",
        "```bash",
        run_command,
        "```",
        "",
        "## Key rules",
        "",
        "- Main model fitting uses only `step09_main_candidate == True` from Step08.",
        "- PCA, GMM candidate-k evaluation, final GMM fitting, and morphotype naming use a balanced main-candidate sample.",
        "- Step08 `*_z` columns are the feature source; Step09 only imputes missing values and learns clipping thresholds from the balanced main sample.",
        "- Sensitivity rows, China pressure-test rows, walk rows, and retained excluded rows are external projections.",
        "- `retained_excluded_candidate` rows use `analysis_scope='excluded_review'` and are excluded from the city profile table.",
        "",
        "## Outputs",
        "",
    ]
    for key, path in log.get("output_paths", {}).items():
        lines.append(f"- `{key}`: `{path}`")
    lines.append("")
    (output_dir / OUTPUT_FILES["readme"]).write_text("\n".join(lines), encoding="utf-8")


def validate_outputs(
    unit_results: pd.DataFrame,
    city_profiles: pd.DataFrame,
    step08_integrity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    duplicate_keys = int(unit_results.duplicated(["city_id", "unit_id", "network_type", "scale"]).sum())
    if "type_unit_share_sum" in city_profiles.columns and not city_profiles.empty:
        share_delta = (city_profiles["type_unit_share_sum"] - 1.0).abs()
        max_share_delta = float(share_delta.max())
        bad_share_rows = int(share_delta.gt(1e-6).sum())
    else:
        max_share_delta = np.nan
        bad_share_rows = 0
    min_external = (
        int(city_profiles["n_external_projection_units"].min())
        if "n_external_projection_units" in city_profiles.columns and not city_profiles.empty
        else None
    )
    max_external = (
        int(city_profiles["n_external_projection_units"].max())
        if "n_external_projection_units" in city_profiles.columns and not city_profiles.empty
        else None
    )
    negative_external_rows = (
        int(city_profiles["n_external_projection_units"].lt(0).sum())
        if "n_external_projection_units" in city_profiles.columns
        else 0
    )
    return {
        "local_morphotype_duplicate_primary_keys": duplicate_keys,
        "city_profile_rows": int(city_profiles.shape[0]),
        "city_profile_max_abs_type_share_sum_delta": max_share_delta,
        "city_profile_bad_type_share_sum_rows": bad_share_rows,
        "city_profile_n_external_projection_units_min": min_external,
        "city_profile_n_external_projection_units_max": max_external,
        "city_profile_n_external_projection_units_negative_rows": negative_external_rows,
        "step08_outputs_overwritten": None
        if step08_integrity is None
        else not bool(step08_integrity.get("all_unchanged", False)),
        "step08_input_integrity_all_unchanged": None
        if step08_integrity is None
        else bool(step08_integrity.get("all_unchanged", False)),
        "step08_input_integrity_changed_files": []
        if step08_integrity is None
        else step08_integrity.get("changed_files", []),
    }


def write_outputs(
    output_dir: Path,
    unit_results: pd.DataFrame,
    city_profiles: pd.DataFrame,
    center_df: pd.DataFrame,
    contrib_df: pd.DataFrame,
    diagnostics: pd.DataFrame,
    feature_qc: pd.DataFrame,
    handoff: dict[str, Any],
    repro: dict[str, Any],
    log: dict[str, Any],
    model_bundle: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {key: output_dir / filename for key, filename in OUTPUT_FILES.items()}
    unit_results.to_parquet(paths["unit_parquet"], index=False)
    unit_results.to_csv(paths["unit_csv"], index=False, encoding="utf-8-sig")
    city_profiles.to_parquet(paths["city_parquet"], index=False)
    city_profiles.to_csv(paths["city_csv"], index=False, encoding="utf-8-sig")
    center_df.to_parquet(paths["center_parquet"], index=False)
    center_df.to_csv(paths["center_csv"], index=False, encoding="utf-8-sig")
    contrib_df.to_csv(paths["contrib_csv"], index=False, encoding="utf-8-sig")
    diagnostics.to_csv(paths["diagnostics_csv"], index=False, encoding="utf-8-sig")
    feature_qc.to_csv(paths["feature_qc_csv"], index=False, encoding="utf-8-sig")
    paths["handoff_json"].write_text(json.dumps(json_ready(handoff), ensure_ascii=False, indent=2), encoding="utf-8")
    paths["handoff_md"].write_text(markdown_handoff(handoff), encoding="utf-8")
    paths["repro_json"].write_text(json.dumps(json_ready(repro), ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump(model_bundle, paths["model"])
    log["output_paths"] = {key: str(path) for key, path in paths.items()}
    log["output_shapes"] = {
        "local_morphotypes": {"rows": int(unit_results.shape[0]), "columns": int(unit_results.shape[1])},
        "city_morphotype_profiles": {"rows": int(city_profiles.shape[0]), "columns": int(city_profiles.shape[1])},
        "morphotype_centers": {"rows": int(center_df.shape[0]), "columns": int(center_df.shape[1])},
        "cluster_feature_contributions": {"rows": int(contrib_df.shape[0]), "columns": int(contrib_df.shape[1])},
        "clustering_model_diagnostics": {"rows": int(diagnostics.shape[0]), "columns": int(diagnostics.shape[1])},
        "feature_quality_checks": {"rows": int(feature_qc.shape[0]), "columns": int(feature_qc.shape[1])},
    }
    write_readme(output_dir, log)
    paths["run_log"].write_text(json.dumps(json_ready(log), ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    input_dir = (args.input_dir or root / "data" / UPSTREAM_STEP).resolve()
    output_dir = (args.output_dir or root / "data" / STEP_NAME).resolve()
    start = time.time()

    candidate_path = input_dir / "step09_morphotype_candidate_units.parquet"
    city_base_path = input_dir / "city_profiles_base.parquet"
    if not candidate_path.exists():
        raise FileNotFoundError(f"Missing main Step09 input: {candidate_path}")
    if not city_base_path.exists():
        raise FileNotFoundError(f"Missing auxiliary city profile input: {city_base_path}")

    relative_run_command = "python code_upload/09_cluster_morphotypes/step09_cluster_morphotypes.py --root <PROJECT_ROOT>"
    resolved_run_command = f"python code_upload/09_cluster_morphotypes/step09_cluster_morphotypes.py --root {root}"
    log: dict[str, Any] = {
        "step": STEP_NAME,
        "script_path": str(Path(__file__).resolve()),
        "root": str(root),
        "relative_run_command": relative_run_command,
        "run_command": relative_run_command,
        "resolved_run_command_for_this_run": resolved_run_command,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "run_started_at": now_iso(),
        "parameters": {
            "k_min": args.k_min,
            "k_max": args.k_max,
            "final_k": args.final_k,
            "balance_max_per_city_scale": args.balance_max_per_city_scale,
            "eval_sample_size": args.eval_sample_size,
            "ward_sample_size": args.ward_sample_size,
            "clip_quantile": args.clip_quantile,
            "random_seed": args.random_seed,
            "min_cluster_share": args.min_cluster_share,
            "fit_sample_rule": "step09_main_candidate == True only",
            "projection_rule": "all rows retained in Step08 step09_morphotype_candidate_units.parquet are projected",
        },
        "dependency_versions": dependency_versions(),
    }

    input_metadata = collect_input_metadata(input_dir)
    print(f"[{now_iso()}] Reading Step08 outputs...")
    candidates = pd.read_parquet(candidate_path)
    city_base = pd.read_parquet(city_base_path)
    step08_anomalies = read_step08_anomalies(input_dir)

    print(f"[{now_iso()}] Validating 08 -> 09 handoff...")
    handoff = validate_handoff(candidates, city_base, input_metadata, input_dir, output_dir)

    print(f"[{now_iso()}] Building balanced main training sample...")
    balanced_idx = make_balanced_main_indices(candidates, args.balance_max_per_city_scale, args.random_seed)
    feature_qc, selected_z_features, removed_z_features = feature_qc_table(candidates, balanced_idx)
    if not selected_z_features:
        raise ValueError("No usable Step08 z features remain after feature QC.")

    print(f"[{now_iso()}] Using Step08 z features with main-sample imputation and clipping...")
    train_z, all_z, preprocess_meta, imputer = preprocess_z_features(
        candidates,
        balanced_idx,
        selected_z_features,
        args.clip_quantile,
    )

    print(f"[{now_iso()}] Fitting PCA on balanced main sample...")
    train_pca, all_pca, pca_model, pca_info = fit_pca(train_z, all_z, args.random_seed)

    print(f"[{now_iso()}] Evaluating GMM k candidates and optional diagnostics...")
    diagnostics, optional_model_notes = evaluate_models(
        train_pca,
        args.k_min,
        args.k_max,
        args.eval_sample_size,
        args.ward_sample_size,
        args.random_seed,
    )
    chosen_k = choose_final_k(diagnostics, args.final_k, args.min_cluster_share)
    diagnostics["chosen_final_model"] = diagnostics["algorithm"].eq("gmm_full") & diagnostics["k"].eq(chosen_k)

    print(f"[{now_iso()}] Fitting final GMM on balanced main sample, k={chosen_k}...")
    gmm = fit_final_gmm(train_pca, chosen_k, args.random_seed)
    center_df, contrib_df, raw_to_mtype = build_cluster_tables(
        candidates,
        balanced_idx,
        train_z,
        train_pca,
        selected_z_features,
        gmm,
    )

    print(f"[{now_iso()}] Projecting all Step08 candidate units...")
    shared_flags = shared_ucdb_city_flags(city_base)
    unit_results = assign_units(candidates, all_pca, gmm, center_df, raw_to_mtype, shared_flags)
    city_profiles = build_city_profiles(unit_results, center_df)
    output_validation = validate_outputs(unit_results, city_profiles)

    shared_sg = (
        unit_results[unit_results["shared_ucdb_boundary_flag"]][
            ["city_id", "city_name_en", "shared_ucdb_id", "shared_ucdb_boundary_group"]
        ]
        .drop_duplicates()
        .to_dict(orient="records")
    )
    log.update(
        {
            "handoff_status": handoff["overall_status"],
            "step08_anomaly_cities": step08_anomalies,
            "balanced_training_sample": {
                "rows": int(len(balanced_idx)),
                "source_main_candidate_rows": int(candidates["step09_main_candidate"].sum()),
                "city_count": int(candidates.loc[balanced_idx, "city_id"].nunique()),
                "network_scale_counts": candidates.loc[balanced_idx]
                .groupby(["network_type", "scale"], observed=True)
                .size()
                .to_dict(),
                "sampling_rule": f"up to {args.balance_max_per_city_scale} rows per city/network/scale among main candidates",
            },
            "feature_selection": {
                "feature_source": "Step08 z columns",
                "selected_z_features": selected_z_features,
                "selected_raw_features": [Z_TO_RAW[col] for col in selected_z_features],
                "removed_z_features": removed_z_features,
                "removed_reasons": feature_qc.loc[~feature_qc["selected_for_model"], ["z_feature", "removed_reason"]].to_dict(
                    orient="records"
                ),
            },
            "preprocessing": preprocess_meta,
            "pca": pca_info,
            "model_diagnostics": optional_model_notes,
            "chosen_final_model": {
                "algorithm": "gmm_full",
                "k": int(chosen_k),
                "selection_rule": optional_model_notes["selection_rule"] if args.final_k == "auto" else "manual final_k",
            },
            "cluster_summary": {
                "morphotypes": center_df[
                    ["morphotype", "morphotype_name", "cluster_raw_label", "n_balanced_training_units", "balanced_training_share"]
                ].to_dict(orient="records"),
                "unit_assignment_counts": unit_results["morphotype"].value_counts().sort_index().to_dict(),
                "analysis_scope_counts": unit_results["analysis_scope"].value_counts().to_dict(),
                "mean_type_confidence": float(unit_results["type_confidence"].mean()),
                "anomaly_unit_share": float(unit_results["anomaly_unit_flag"].mean()),
                "excluded_review_rows": int(unit_results["analysis_scope"].eq("excluded_review").sum()),
                "shared_ucdb_boundary_unit_rows": int(unit_results["shared_ucdb_boundary_flag"].sum()),
                "shenzhen_guangzhou_shared_ucdb_handling": shared_sg,
            },
            "output_validation": output_validation,
        }
    )

    repro = {
        "generated_at": now_iso(),
        "script_path": str(Path(__file__).resolve()),
        "relative_run_command": relative_run_command,
        "run_command": relative_run_command,
        "resolved_run_command_for_this_run": resolved_run_command,
        "root": str(root),
        "parameters": log["parameters"],
        "dependency_versions": log["dependency_versions"],
        "input_metadata": input_metadata,
        "selected_z_features": selected_z_features,
        "removed_z_features": removed_z_features,
        "balanced_training_sample": log["balanced_training_sample"],
        "pca": pca_info,
        "chosen_final_model": log["chosen_final_model"],
    }

    model_bundle = {
        "step": STEP_NAME,
        "created_at": now_iso(),
        "raw_feature_columns": RAW_FEATURE_COLUMNS,
        "z_feature_columns": Z_FEATURE_COLUMNS,
        "selected_z_features": selected_z_features,
        "removed_z_features": removed_z_features,
        "imputer": imputer,
        "pca": pca_model,
        "cluster_model": gmm,
        "cluster_center_table": center_df,
        "raw_label_to_morphotype": raw_to_mtype,
        "parameters": log["parameters"],
    }
    model_bundle["imputer_statistics"] = [entry["impute_median"] for entry in preprocess_meta["clip_table"]]
    model_bundle["clip_table"] = preprocess_meta["clip_table"]

    log["run_finished_at"] = now_iso()
    log["run_seconds_before_writing"] = round(time.time() - start, 3)

    print(f"[{now_iso()}] Writing Step09 outputs to {output_dir}...")
    write_outputs(
        output_dir,
        unit_results,
        city_profiles,
        center_df,
        contrib_df,
        diagnostics,
        feature_qc,
        handoff,
        repro,
        log,
        model_bundle,
    )
    post_write_input_metadata = collect_input_metadata(input_dir)
    step08_integrity = compare_input_integrity(input_metadata, post_write_input_metadata)
    handoff["step08_input_integrity"] = step08_integrity
    repro["step08_input_integrity"] = step08_integrity
    repro["input_metadata_after_step09_write"] = post_write_input_metadata
    log["step08_input_integrity"] = step08_integrity
    log["input_metadata_after_step09_write"] = post_write_input_metadata
    log["output_validation"] = validate_outputs(unit_results, city_profiles, step08_integrity)

    final_paths = {key: output_dir / filename for key, filename in OUTPUT_FILES.items()}
    final_paths["handoff_json"].write_text(
        json.dumps(json_ready(handoff), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    final_paths["handoff_md"].write_text(markdown_handoff(handoff), encoding="utf-8")
    final_paths["repro_json"].write_text(
        json.dumps(json_ready(repro), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_readme(output_dir, log)
    log["run_seconds_total"] = round(time.time() - start, 3)
    (output_dir / OUTPUT_FILES["run_log"]).write_text(json.dumps(json_ready(log), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{now_iso()}] Step09 complete in {log['run_seconds_total']} seconds.")
    print(f"  local_morphotypes: {unit_results.shape[0]} rows x {unit_results.shape[1]} columns")
    print(f"  city_morphotype_profiles: {city_profiles.shape[0]} rows x {city_profiles.shape[1]} columns")
    print(f"  GMM morphotypes: {chosen_k}")


if __name__ == "__main__":
    main()
