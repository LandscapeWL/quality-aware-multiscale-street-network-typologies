#!/usr/bin/env python3
"""Step11: statistical models and robustness checks for morphotype validation.

This step reads Step09 morphotype outputs and Step10 accessibility / detour
validation metrics.  It does not rerun or modify upstream Step09/Step10
outputs.  The outputs are descriptive tables, model coefficient tables,
robustness checks, figure/table-ready data, QC, and reproducibility records.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import json
import math
import platform
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    import statsmodels.formula.api as smf

    HAVE_STATSMODELS = True
except Exception:  # pragma: no cover - exercised only on stripped environments.
    smf = None
    HAVE_STATSMODELS = False


STEP_NAME = "11_statistical_models_robustness_checks"
STEP09_NAME = "09_cluster_morphotypes"
STEP10_NAME = "10_compute_accessibility_detour_validation_metrics"
STEP07_NAME = "07_generate_quality_scores_type_confidence"
STEP03_NAME = "03_download_population_built_environment_data"
STEP01_NAME = "01_city_boundaries_and_sample_inventory"

MAIN_SCALE = "hex_1km"
MAIN_ANALYSIS_SCOPE = "main_fit"
MAIN_NETWORK = "drive"
MORPHOTYPES = [f"MT{i:02d}" for i in range(1, 11)]
FACILITY_CATEGORIES = ["grocery", "healthcare", "park", "school", "transit"]
DEFAULT_CONFIDENCE_THRESHOLD = 0.70
DEFAULT_SMALL_GROUP_MIN_UNITS = 3

OUTPUT_FILES = {
    "input_report_json": "11_input_acceptance_report.json",
    "input_report_md": "11_input_acceptance_report.md",
    "input_field_check": "11_input_field_acceptance.csv",
    "unit_model_table": "unit_model_table.parquet",
    "morphotype_city_table": "morphotype_city_table.parquet",
    "city_model_table": "city_model_table.parquet",
    "model_table_a_csv": "model_table_A_OD.csv",
    "model_table_a_parquet": "model_table_A_OD.parquet",
    "model_table_b_csv": "model_table_B_facility_accessibility.csv",
    "model_table_b_parquet": "model_table_B_facility_accessibility.parquet",
    "model_table_c_csv": "model_table_C_inequality.csv",
    "model_table_c_parquet": "model_table_C_inequality.parquet",
    "morphotype_descriptive_stats": "morphotype_descriptive_stats.csv",
    "circuity_by_morphotype": "circuity_by_morphotype.csv",
    "facility_access_by_morphotype": "facility_access_by_morphotype.csv",
    "inequality_by_city_category": "inequality_by_city_category.csv",
    "confidence_by_morphotype": "confidence_by_morphotype.csv",
    "model_a_csv": "model_A_od_circuity_results.csv",
    "model_a_md": "model_A_od_circuity_results.md",
    "model_b_csv": "model_B_accessibility_results.csv",
    "model_b_md": "model_B_accessibility_results.md",
    "model_c_csv": "model_C_inequality_results.csv",
    "model_c_md": "model_C_inequality_results.md",
    "robust_high_conf": "robustness_high_confidence.csv",
    "robust_unweighted": "robustness_unweighted_vs_weighted.csv",
    "robust_small": "robustness_remove_small_groups.csv",
    "robust_md": "robustness_summary.md",
    "figure_circuity": "figure_data_morphotype_circuity.csv",
    "figure_access": "figure_data_accessibility_by_type.csv",
    "figure_inequality": "figure_data_inequality.csv",
    "table_model": "table_model_results.csv",
    "table_robust": "table_robustness_results.csv",
    "coef_csv": "model_coefficients.csv",
    "coef_xlsx": "model_coefficients.xlsx",
    "marginal_effects": "model_marginal_effects.csv",
    "model_xlsx": "validation_model_results.xlsx",
    "robust_xlsx": "robustness_results.xlsx",
    "robust_summary_table": "robustness_summary_table.csv",
    "summary_md": "11_result_interpretation_summary.md",
    "summary_json": "11_result_interpretation_summary.json",
    "model_explanation_md": "model_explanation_summary.md",
    "run_log": "11_run_log.json",
    "qc": "11_quality_checks.csv",
    "readme": "README.md",
    "repro": "reproducibility_status_manifest.json",
}


@dataclass
class StepPaths:
    root: Path
    output_dir: Path
    step09_dir: Path
    step10_dir: Path
    step07_dir: Path
    step03_dir: Path
    step01_city_dir: Path

    @property
    def step09_units(self) -> Path:
        return self.step09_dir / "local_morphotypes.parquet"

    @property
    def step09_city_profiles(self) -> Path:
        return self.step09_dir / "city_profiles.parquet"

    @property
    def step10_od(self) -> Path:
        return self.step10_dir / "OD_detour_metrics.parquet"

    @property
    def step10_accessibility(self) -> Path:
        return self.step10_dir / "facility_accessibility_metrics.parquet"

    @property
    def step10_inequality(self) -> Path:
        return self.step10_dir / "accessibility_inequality_metrics.parquet"

    @property
    def step10_validation_summary(self) -> Path:
        return self.step10_dir / "morphotype_validation_summary.csv"

    @property
    def step07_city_quality(self) -> Path:
        return self.step07_dir / "city_quality_scores.parquet"

    @property
    def step07_local_quality(self) -> Path:
        return self.step07_dir / "local_quality_flags.parquet"

    @property
    def step03_city_env(self) -> Path:
        return self.step03_dir / "city_population_built_environment_metrics.parquet"

    @property
    def step01_city_master(self) -> Path:
        return self.step01_city_dir / "city_master.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Output directory. Defaults to data/{STEP_NAME} under --root.",
    )
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--small-group-min-units", type=int, default=DEFAULT_SMALL_GROUP_MIN_UNITS)
    parser.add_argument(
        "--snap-tail-quantile",
        type=float,
        default=0.95,
        help="Quantile used to flag long walk snapping distances. Rows are not deleted.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Accepted for explicit reproducibility. Step11 outputs are regenerated in-place.",
    )
    return parser.parse_args()


def get_paths(root: Path, output_dir: Path | None) -> StepPaths:
    root = root.resolve()
    return StepPaths(
        root=root,
        output_dir=(output_dir or root / "data" / STEP_NAME).resolve(),
        step09_dir=root / "data" / STEP09_NAME,
        step10_dir=root / "data" / STEP10_NAME,
        step07_dir=root / "data" / STEP07_NAME,
        step03_dir=root / "data" / STEP03_NAME,
        step01_city_dir=root / "data" / STEP01_NAME / "city_sample",
    )


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def scalar(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
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
        return json_ready(obj.tolist())
    return scalar(obj)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def write_excel(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        used: set[str] = set()
        for raw_name, df in sheets.items():
            name = raw_name[:31].replace("/", "_").replace("\\", "_")
            base = name
            i = 2
            while name in used:
                suffix = f"_{i}"
                name = f"{base[:31 - len(suffix)]}{suffix}"
                i += 1
            used.add(name)
            df.to_excel(writer, sheet_name=name, index=False)


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_metadata(path: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return meta
    stat = path.stat()
    meta.update(
        {
            "size_bytes": int(stat.st_size),
            "mtime": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
            "sha256": file_sha256(path),
        }
    )
    try:
        if path.suffix == ".parquet":
            parquet = pq.ParquetFile(path)
            meta.update(
                {
                    "format": "parquet",
                    "row_count": int(parquet.metadata.num_rows),
                    "column_count": len(parquet.schema_arrow.names),
                    "columns": list(parquet.schema_arrow.names),
                }
            )
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
            meta.update(
                {
                    "format": "csv",
                    "row_count": int(len(df)),
                    "column_count": int(df.shape[1]),
                    "columns": list(df.columns),
                }
            )
        elif path.suffix == ".json":
            with path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            meta.update({"format": "json", "top_level_keys": list(obj) if isinstance(obj, dict) else []})
    except Exception as exc:
        meta["metadata_error"] = repr(exc)
    return meta


def package_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for package in ["pandas", "numpy", "pyarrow", "statsmodels", "scipy", "sklearn", "openpyxl"]:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def git_state(root: Path) -> dict[str, Any]:
    def run_git(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=root,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
            )
        except Exception:
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "status_short": run_git(["status", "--short"]),
        "note": "None values mean the project root is not a git repository or git was unavailable.",
    }


def read_parquet_selected(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if columns is None:
        return pd.read_parquet(path)
    available = set(pq.ParquetFile(path).schema_arrow.names)
    keep = [c for c in columns if c in available]
    return pd.read_parquet(path, columns=keep)


def read_optional_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def numeric(series: pd.Series, default: float = np.nan) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if not np.isnan(default):
        out = out.fillna(default)
    return out


def weighted_mean(values: pd.Series, weights: pd.Series | None = None) -> float:
    v = pd.to_numeric(values, errors="coerce")
    if weights is None:
        return float(v.mean()) if v.notna().any() else np.nan
    w = pd.to_numeric(weights, errors="coerce")
    mask = v.notna() & w.notna() & np.isfinite(v) & np.isfinite(w) & (w > 0)
    if mask.any() and float(w[mask].sum()) > 0:
        return float(np.average(v[mask], weights=w[mask]))
    return float(v.mean()) if v.notna().any() else np.nan


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    out = numerator.astype(float) / denominator.replace({0: np.nan}).astype(float)
    return out.replace([np.inf, -np.inf], np.nan)


def entropy_from_shares(shares: pd.DataFrame) -> pd.Series:
    arr = shares.fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    total = arr.sum(axis=1, keepdims=True)
    probs = np.divide(arr, total, out=np.zeros_like(arr), where=total > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(probs > 0, np.log(probs), 0.0)
    raw = -(probs * logp).sum(axis=1)
    denom = math.log(max(arr.shape[1], 2))
    return pd.Series(raw / denom, index=shares.index)


def add_log_column(df: pd.DataFrame, source: str, target: str) -> pd.DataFrame:
    if source in df.columns:
        df[target] = np.log1p(pd.to_numeric(df[source], errors="coerce").clip(lower=0))
    else:
        df[target] = np.nan
    return df


def load_inputs(paths: StepPaths) -> dict[str, pd.DataFrame]:
    required = [
        paths.step09_units,
        paths.step09_city_profiles,
        paths.step10_od,
        paths.step10_accessibility,
        paths.step10_inequality,
        paths.step10_validation_summary,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required Step11 inputs: {missing}")

    units_cols = [
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
        "population_sum",
        "built_share",
        "analysis_scope",
        "step09_main_candidate",
        "morphotype",
        "morphotype_name",
        "cluster_probability",
        "cluster_probability_margin",
        "type_confidence",
        "low_confidence_flag",
        "low_quality_unit_flag",
        "local_metric_validity_score",
        "local_quality_flag",
        "anomaly_unit_flag",
        "edge_density_km_per_km2",
        "node_density_per_km2",
        "intersection_density_per_km2",
        "average_node_degree",
        "four_way_share",
        "dead_end_share",
        "segment_length_mean",
        "edge_circuity_mean",
        "orientation_entropy",
        "orientation_order",
        "betweenness_gini",
        "road_hierarchy_entropy",
    ]
    inputs = {
        "units": read_parquet_selected(paths.step09_units, units_cols),
        "city_profiles": pd.read_parquet(paths.step09_city_profiles),
        "od": pd.read_parquet(paths.step10_od),
        "accessibility": pd.read_parquet(paths.step10_accessibility),
        "inequality": pd.read_parquet(paths.step10_inequality),
        "validation_summary": pd.read_csv(paths.step10_validation_summary),
        "city_quality": read_optional_parquet(paths.step07_city_quality),
        "local_quality": read_optional_parquet(paths.step07_local_quality),
        "city_env": read_optional_parquet(paths.step03_city_env),
        "city_master": read_optional_parquet(paths.step01_city_master),
    }
    return inputs


def filter_main_tables(inputs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame | list[str]]:
    units = inputs["units"]
    main_units = units[
        units["network_type"].eq(MAIN_NETWORK)
        & units["scale"].eq(MAIN_SCALE)
        & units["analysis_scope"].eq(MAIN_ANALYSIS_SCOPE)
    ].copy()
    main_units["main_analysis_included"] = True

    acc = inputs["accessibility"]
    acc_main = acc[
        acc["scale"].eq(MAIN_SCALE)
        & acc["analysis_scope"].eq(MAIN_ANALYSIS_SCOPE)
        & acc["main_analysis_included"].fillna(False).astype(bool)
    ].copy()

    od = inputs["od"]
    od_main = od[
        od["scale"].eq(MAIN_SCALE)
        & od["origin_analysis_scope"].eq(MAIN_ANALYSIS_SCOPE)
        & od["destination_analysis_scope"].eq(MAIN_ANALYSIS_SCOPE)
        & od["origin_main_analysis_included"].fillna(False).astype(bool)
        & od["destination_main_analysis_included"].fillna(False).astype(bool)
    ].copy()

    ineq = inputs["inequality"]
    ineq_main = ineq[ineq["scale"].eq(MAIN_SCALE)].copy()

    city_sets = [
        set(main_units["city_id"].dropna().astype(str)),
        set(acc_main["city_id"].dropna().astype(str)),
        set(od_main["city_id"].dropna().astype(str)),
        set(ineq_main.loc[ineq_main["group_level"].eq("city"), "city_id"].dropna().astype(str)),
    ]
    main_city_ids = sorted(set.intersection(*city_sets))
    main_units = main_units[main_units["city_id"].isin(main_city_ids)].copy()
    acc_main = acc_main[acc_main["city_id"].isin(main_city_ids)].copy()
    od_main = od_main[od_main["city_id"].isin(main_city_ids)].copy()
    ineq_main = ineq_main[ineq_main["city_id"].isin(main_city_ids) | ineq_main["city_id"].isna()].copy()

    return {
        "main_units": main_units,
        "acc_main": acc_main,
        "od_main": od_main,
        "ineq_main": ineq_main,
        "main_city_ids": main_city_ids,
    }


def build_city_controls(main_units: pd.DataFrame, inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base_cols = ["city_id", "city_name_en", "country", "iso3", "region", "sample_group", "quality_tier"]
    base = main_units[[c for c in base_cols if c in main_units.columns]].drop_duplicates("city_id").copy()

    quality_cols = [
        "city_id",
        "network_integrity_score",
        "historical_maturity_score",
        "poi_completeness_score",
        "population_built_support_score",
        "local_coverage_score",
        "quality_score",
        "quality_weight",
        "quality_confidence_component",
        "review_flag",
        "review_reason",
    ]
    env_cols = [
        "city_id",
        "morphology_prior",
        "ucdb_area_km2",
        "ucdb_pop_2025",
        "city_area_km2_calc",
        "ghsl_pop_density_2020_100m_per_km2",
        "ghsl_pop_density_2020_1000m_per_km2",
        "worldpop_pop_density_2020_per_km2",
        "ghsl_built_share_2020_100m",
        "ghsl_built_share_2020_1000m",
        "worldpop_to_ghsl_pop_2020_100m_ratio",
        "ghsl_pop_2020_100m_to_ucdb_pop_2025_ratio",
    ]
    master_cols = [
        "city_id",
        "geofabrik_extract_id",
        "geofabrik_extract_name",
        "ucdb_match_status",
        "match_status",
        "match_method",
        "match_score",
        "manual_rule_applied",
        "duplicate_ucdb_id_with",
    ]
    for name, cols in [
        ("city_quality", quality_cols),
        ("city_env", env_cols),
        ("city_master", master_cols),
    ]:
        right = inputs.get(name, pd.DataFrame())
        if right.empty or "city_id" not in right.columns:
            continue
        keep = [c for c in cols if c in right.columns and (c == "city_id" or c not in base.columns)]
        if len(keep) <= 1:
            continue
        base = base.merge(right[keep].drop_duplicates("city_id"), on="city_id", how="left")

    # Fallback quality_score from Step09 units if Step07 is unavailable.
    if "quality_score" not in base.columns and "quality_score" in main_units.columns:
        q = (
            main_units.groupby("city_id", as_index=False)
            .agg(quality_score=("quality_score", "mean"), quality_weight=("quality_weight", "mean"))
            .copy()
        )
        base = base.merge(q, on="city_id", how="left")

    base = add_log_column(base, "ucdb_pop_2025", "log_ucdb_pop_2025")
    base = add_log_column(base, "ucdb_area_km2", "log_ucdb_area_km2")
    base = add_log_column(base, "city_area_km2_calc", "log_city_area_km2")
    base = add_log_column(base, "ghsl_pop_density_2020_100m_per_km2", "log_ghsl_pop_density_2020_100m")
    return base


def prepare_unit_model_table(
    main_units: pd.DataFrame,
    acc_main: pd.DataFrame,
    city_controls: pd.DataFrame,
    confidence_threshold: float,
    snap_tail_quantile: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    acc = acc_main.copy()
    acc["access_15min_num"] = acc["access_15min"].astype(float)
    snap = pd.to_numeric(acc.get("walk_snap_distance_m", pd.Series(dtype=float)), errors="coerce")
    snap_threshold = float(snap.quantile(snap_tail_quantile)) if snap.notna().any() else np.nan
    acc["long_snap_distance_flag"] = snap > snap_threshold if np.isfinite(snap_threshold) else False

    unit_cols = [
        "city_id",
        "scale",
        "unit_id",
        "quality_score",
        "quality_weight",
        "quality_confidence_component",
        "cluster_probability",
        "cluster_probability_margin",
        "local_metric_validity_score",
        "local_quality_flag",
        "built_share",
        "edge_density_km_per_km2",
        "node_density_per_km2",
        "intersection_density_per_km2",
        "average_node_degree",
        "four_way_share",
        "dead_end_share",
        "segment_length_mean",
        "edge_circuity_mean",
        "orientation_entropy",
        "orientation_order",
        "betweenness_gini",
        "road_hierarchy_entropy",
    ]
    unit_covars = main_units[[c for c in unit_cols if c in main_units.columns]].drop_duplicates(["city_id", "scale", "unit_id"])
    unit_covars["step09_join_key_present"] = True
    join_keys = [c for c in ["city_id", "scale", "unit_id"] if c in acc.columns and c in unit_covars.columns]
    unit_table = acc.merge(unit_covars, on=join_keys, how="left", suffixes=("", "_step09"), validate="many_to_one")
    unit_table["step09_join_matched"] = unit_table["step09_join_key_present"].fillna(False).astype(bool)
    unit_table = unit_table.drop(columns=["step09_join_key_present"])

    city_control_keep = [
        c
        for c in [
            "city_id",
            "network_integrity_score",
            "historical_maturity_score",
            "poi_completeness_score",
            "population_built_support_score",
            "local_coverage_score",
            "log_ucdb_pop_2025",
            "log_city_area_km2",
            "ghsl_built_share_2020_100m",
            "log_ghsl_pop_density_2020_100m",
        ]
        if c in city_controls.columns and (c == "city_id" or c not in unit_table.columns)
    ]
    if len(city_control_keep) > 1:
        unit_table = unit_table.merge(city_controls[city_control_keep], on="city_id", how="left")

    type_counts = (
        main_units.groupby(["city_id", "morphotype"], as_index=False)
        .agg(morphotype_city_unit_count=("unit_id", "nunique"))
        .copy()
    )
    unit_table = unit_table.merge(type_counts, on=["city_id", "morphotype"], how="left")
    unit_table["high_confidence_flag"] = pd.to_numeric(unit_table["type_confidence"], errors="coerce") >= confidence_threshold
    unit_table["population_weight_for_model"] = pd.to_numeric(
        unit_table.get("population_sum", unit_table.get("population_weight_raw")), errors="coerce"
    )
    unit_table["population_weight_for_model"] = unit_table["population_weight_for_model"].fillna(
        pd.to_numeric(unit_table.get("population_weight_raw", 1.0), errors="coerce")
    )
    snap_info = {
        "snap_tail_quantile": snap_tail_quantile,
        "snap_distance_threshold_m": snap_threshold,
        "long_snap_distance_row_count": int(unit_table["long_snap_distance_flag"].sum()),
        "long_snap_distance_row_share": float(unit_table["long_snap_distance_flag"].mean()),
        "step10_facility_step09_join_match_share": float(unit_table["step09_join_matched"].mean()),
        "step10_facility_step09_join_unmatched_rows": int((~unit_table["step09_join_matched"]).sum()),
        "step10_facility_step09_join_key": join_keys,
        "policy": "Long snapping distances are flagged for diagnostics and robustness notes; no rows are hard-deleted.",
    }
    return unit_table, snap_info


def build_city_composition(main_units: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["city_id", "morphotype"]
    agg = (
        main_units.groupby(group_cols, as_index=False)
        .agg(
            unit_count=("unit_id", "nunique"),
            population_sum=("population_sum", "sum"),
            area_sum_km2=("unit_area_km2", "sum"),
            mean_type_confidence=("type_confidence", "mean"),
            median_type_confidence=("type_confidence", "median"),
        )
        .copy()
    )
    unit_pivot = agg.pivot(index="city_id", columns="morphotype", values="unit_count").fillna(0)
    pop_pivot = agg.pivot(index="city_id", columns="morphotype", values="population_sum").fillna(0.0)
    area_pivot = agg.pivot(index="city_id", columns="morphotype", values="area_sum_km2").fillna(0.0)
    for mt in MORPHOTYPES:
        if mt not in unit_pivot.columns:
            unit_pivot[mt] = 0
        if mt not in pop_pivot.columns:
            pop_pivot[mt] = 0.0
        if mt not in area_pivot.columns:
            area_pivot[mt] = 0.0
    unit_pivot = unit_pivot[MORPHOTYPES]
    pop_pivot = pop_pivot[MORPHOTYPES]
    area_pivot = area_pivot[MORPHOTYPES]

    out = pd.DataFrame(index=unit_pivot.index)
    out["city_unit_count"] = unit_pivot.sum(axis=1)
    out["city_population_sum"] = pop_pivot.sum(axis=1)
    out["city_area_sum_km2"] = area_pivot.sum(axis=1)
    unit_shares = unit_pivot.div(unit_pivot.sum(axis=1).replace({0: np.nan}), axis=0).fillna(0.0)
    pop_shares = pop_pivot.div(pop_pivot.sum(axis=1).replace({0: np.nan}), axis=0).fillna(0.0)
    area_shares = area_pivot.div(area_pivot.sum(axis=1).replace({0: np.nan}), axis=0).fillna(0.0)

    for mt in MORPHOTYPES:
        out[f"unit_count__{mt}"] = unit_pivot[mt].astype(int)
        out[f"unit_share__{mt}"] = unit_shares[mt]
        out[f"population_share__{mt}"] = pop_shares[mt]
        out[f"area_share__{mt}"] = area_shares[mt]

    out["morphotype_entropy"] = entropy_from_shares(pop_shares)
    out["unit_morphotype_entropy"] = entropy_from_shares(unit_shares)
    out["dominant_morphotype"] = pop_shares.idxmax(axis=1)
    out["dominant_population_share"] = pop_shares.max(axis=1)
    out["dominant_unit_share"] = unit_shares.lookup(unit_shares.index, out["dominant_morphotype"]) if hasattr(unit_shares, "lookup") else [
        unit_shares.loc[idx, mt] for idx, mt in out["dominant_morphotype"].items()
    ]
    name_map = (
        main_units[["morphotype", "morphotype_name"]]
        .dropna()
        .drop_duplicates("morphotype")
        .set_index("morphotype")["morphotype_name"]
        .to_dict()
    )
    out["dominant_morphotype_name"] = out["dominant_morphotype"].map(name_map)
    out = out.reset_index()
    return out


def aggregate_access_city_type_category(unit_model_table: pd.DataFrame) -> pd.DataFrame:
    df = unit_model_table.copy()
    df["access_15min_num"] = pd.to_numeric(df["access_15min_num"], errors="coerce")
    df["population_weight_for_model"] = pd.to_numeric(df["population_weight_for_model"], errors="coerce").fillna(0.0)
    df["weighted_access_num"] = df["access_15min_num"] * df["population_weight_for_model"]
    df["weighted_accessibility_score_num"] = (
        pd.to_numeric(df["accessibility_score"], errors="coerce") * df["population_weight_for_model"]
    )
    df["no_access_num"] = (1.0 - df["access_15min_num"]) * df["population_weight_for_model"]
    df["long_snap_num"] = df["long_snap_distance_flag"].astype(float) * df["population_weight_for_model"]
    keys = [
        "city_id",
        "city_name_en",
        "country",
        "iso3",
        "region",
        "sample_group",
        "quality_tier",
        "scale",
        "morphotype",
        "morphotype_name",
        "category",
    ]
    keys = [c for c in keys if c in df.columns]
    agg = (
        df.groupby(keys, dropna=False, as_index=False)
        .agg(
            unit_count=("unit_id", "nunique"),
            population_sum=("population_sum", "sum"),
            population_weight_sum=("population_weight_for_model", "sum"),
            weighted_access_num=("weighted_access_num", "sum"),
            weighted_accessibility_score_num=("weighted_accessibility_score_num", "sum"),
            no_access_num=("no_access_num", "sum"),
            long_snap_num=("long_snap_num", "sum"),
            unweighted_access_share=("access_15min_num", "mean"),
            median_distance_m=("network_distance_to_nearest_m", "median"),
            p90_distance_m=("network_distance_to_nearest_m", lambda s: pd.to_numeric(s, errors="coerce").quantile(0.9)),
            mean_type_confidence=("type_confidence", "mean"),
            median_type_confidence=("type_confidence", "median"),
            low_confidence_unit_share=("low_confidence_flag", "mean"),
            low_quality_unit_share=("low_quality_unit_flag", "mean"),
            quality_score=("quality_score", "mean"),
            quality_weight=("quality_weight", "mean"),
        )
        .copy()
    )
    agg["population_weighted_access_share"] = safe_divide(agg["weighted_access_num"], agg["population_weight_sum"])
    agg["population_weighted_accessibility_score"] = safe_divide(
        agg["weighted_accessibility_score_num"], agg["population_weight_sum"]
    )
    agg["no_access_population_share"] = safe_divide(agg["no_access_num"], agg["population_weight_sum"])
    agg["long_snap_share"] = safe_divide(agg["long_snap_num"], agg["population_weight_sum"])
    agg["log_unit_count"] = np.log1p(agg["unit_count"])
    return agg.drop(
        columns=[
            "weighted_access_num",
            "weighted_accessibility_score_num",
            "no_access_num",
            "long_snap_num",
        ]
    )


def build_morphotype_city_table(
    main_units: pd.DataFrame,
    od_main: pd.DataFrame,
    access_city_type_category: pd.DataFrame,
    city_controls: pd.DataFrame,
    small_group_min_units: int,
) -> pd.DataFrame:
    group_cols = [
        "city_id",
        "city_name_en",
        "country",
        "iso3",
        "region",
        "sample_group",
        "quality_tier",
        "scale",
        "morphotype",
        "morphotype_name",
    ]
    group_cols = [c for c in group_cols if c in main_units.columns]
    unit_metrics = [
        "built_share",
        "edge_density_km_per_km2",
        "node_density_per_km2",
        "intersection_density_per_km2",
        "average_node_degree",
        "four_way_share",
        "dead_end_share",
        "segment_length_mean",
        "edge_circuity_mean",
        "orientation_entropy",
        "orientation_order",
        "betweenness_gini",
        "road_hierarchy_entropy",
    ]
    agg_spec: dict[str, tuple[str, str]] = {
        "unit_count": ("unit_id", "nunique"),
        "population_sum": ("population_sum", "sum"),
        "area_sum_km2": ("unit_area_km2", "sum"),
        "mean_type_confidence": ("type_confidence", "mean"),
        "median_type_confidence": ("type_confidence", "median"),
        "mean_cluster_probability": ("cluster_probability", "mean"),
        "low_confidence_unit_share": ("low_confidence_flag", "mean"),
        "low_quality_unit_share": ("low_quality_unit_flag", "mean"),
        "quality_score": ("quality_score", "mean"),
        "quality_weight": ("quality_weight", "mean"),
        "local_metric_validity_score": ("local_metric_validity_score", "mean"),
    }
    for col in unit_metrics:
        if col in main_units.columns:
            agg_spec[f"mean_{col}"] = (col, "mean")
    mt_city = main_units.groupby(group_cols, dropna=False, as_index=False).agg(**agg_spec).copy()
    city_totals = (
        main_units.groupby("city_id", as_index=False)
        .agg(city_unit_count=("unit_id", "nunique"), city_population_sum=("population_sum", "sum"))
        .copy()
    )
    mt_city = mt_city.merge(city_totals, on="city_id", how="left")
    mt_city["population_share_in_city"] = safe_divide(mt_city["population_sum"], mt_city["city_population_sum"])
    mt_city["unit_share_in_city"] = safe_divide(mt_city["unit_count"], mt_city["city_unit_count"])
    mt_city["log_unit_count"] = np.log1p(mt_city["unit_count"])
    mt_city["small_group_flag"] = mt_city["unit_count"] < small_group_min_units

    od_type = od_main[od_main["group_level"].eq("origin_morphotype")].copy()
    if not od_type.empty:
        od_type = od_type.rename(
            columns={
                "origin_morphotype": "morphotype",
                "origin_morphotype_name": "morphotype_name_od",
            }
        )
        od_cols = [
            "city_id",
            "morphotype",
            "od_pair_count",
            "od_pair_weight_sum",
            "route_found_count",
            "route_found_weight_sum",
            "route_found_share",
            "route_found_weighted_share",
            "od_circuity_mean",
            "od_circuity_p50",
            "od_circuity_p90",
            "od_circuity_weighted_mean",
            "od_circuity_weighted_p50",
            "od_circuity_weighted_p90",
            "detour_ratio_mean",
            "detour_ratio_median",
            "detour_ratio_p90",
        ]
        mt_city = mt_city.merge(od_type[[c for c in od_cols if c in od_type.columns]], on=["city_id", "morphotype"], how="left")

    access_summary = (
        access_city_type_category.groupby(["city_id", "morphotype"], as_index=False)
        .agg(
            access_category_count=("category", "nunique"),
            mean_access_share=("population_weighted_access_share", "mean"),
            min_access_share=("population_weighted_access_share", "min"),
            mean_accessibility_score=("population_weighted_accessibility_score", "mean"),
            mean_long_snap_share=("long_snap_share", "mean"),
            access_population_sum=("population_sum", "sum"),
        )
        .copy()
    )
    access_pivot = access_city_type_category.pivot_table(
        index=["city_id", "morphotype"],
        columns="category",
        values="population_weighted_access_share",
        aggfunc="first",
    )
    access_pivot.columns = [f"access_share__{c}" for c in access_pivot.columns]
    access_pivot = access_pivot.reset_index()
    mt_city = mt_city.merge(access_summary, on=["city_id", "morphotype"], how="left")
    mt_city = mt_city.merge(access_pivot, on=["city_id", "morphotype"], how="left")

    control_cols = [
        c
        for c in [
            "city_id",
            "network_integrity_score",
            "historical_maturity_score",
            "poi_completeness_score",
            "population_built_support_score",
            "local_coverage_score",
            "log_ucdb_pop_2025",
            "log_city_area_km2",
            "ghsl_built_share_2020_100m",
            "log_ghsl_pop_density_2020_100m",
            "morphology_prior",
        ]
        if c in city_controls.columns and (c == "city_id" or c not in mt_city.columns)
    ]
    if len(control_cols) > 1:
        mt_city = mt_city.merge(city_controls[control_cols], on="city_id", how="left")
    return mt_city


def build_city_model_table(
    ineq_main: pd.DataFrame,
    main_units: pd.DataFrame,
    city_controls: pd.DataFrame,
) -> pd.DataFrame:
    composition = build_city_composition(main_units)
    city_rows = ineq_main[ineq_main["group_level"].eq("city")].copy()
    table = city_rows.merge(composition, on="city_id", how="left")
    control_cols = [c for c in city_controls.columns if c == "city_id" or c not in table.columns]
    table = table.merge(city_controls[control_cols], on="city_id", how="left")
    table["city_category_weight"] = pd.to_numeric(table["population_sum"], errors="coerce").fillna(
        pd.to_numeric(table.get("city_population_sum", 1.0), errors="coerce")
    )
    table = add_log_column(table, "population_sum", "log_city_category_population")
    return table


def prepare_model_a_table(morphotype_city_table: pd.DataFrame) -> pd.DataFrame:
    table = morphotype_city_table[morphotype_city_table["od_pair_count"].notna()].copy()
    table["origin_morphotype"] = table["morphotype"].astype(str)
    table["origin_morphotype_name"] = table["morphotype_name"]
    table["log_od_circuity_weighted_mean"] = np.log(
        pd.to_numeric(table["od_circuity_weighted_mean"], errors="coerce").clip(lower=1e-9)
    )
    if "route_found_weight_sum" in table.columns and table["route_found_weight_sum"].notna().any():
        table["od_model_weight"] = pd.to_numeric(table["route_found_weight_sum"], errors="coerce")
        table["od_model_weight_source"] = "route_found_weight_sum"
    else:
        table["od_model_weight"] = pd.to_numeric(table["od_pair_weight_sum"], errors="coerce")
        table["od_model_weight_source"] = "od_pair_weight_sum"
    table["od_pair_count_ge_30"] = pd.to_numeric(table["od_pair_count"], errors="coerce") >= 30
    table["od_pair_count_ge_100"] = pd.to_numeric(table["od_pair_count"], errors="coerce") >= 100
    return table


def prepare_model_b_table(unit_model_table: pd.DataFrame) -> pd.DataFrame:
    table = unit_model_table.copy()
    table["access_15min_num"] = pd.to_numeric(table["access_15min_num"], errors="coerce")
    table["accessibility_score"] = pd.to_numeric(table["accessibility_score"], errors="coerce")
    table["facility_model_weight"] = pd.to_numeric(table.get("population_sum", np.nan), errors="coerce")
    fallback_weight = pd.to_numeric(table.get("population_weight", table.get("population_weight_raw", 1.0)), errors="coerce")
    table["facility_model_weight"] = table["facility_model_weight"].where(table["facility_model_weight"] > 0, fallback_weight)
    table["facility_model_weight_source"] = np.where(
        pd.to_numeric(table.get("population_sum", np.nan), errors="coerce") > 0,
        "population_sum",
        "population_weight_or_raw",
    )
    return table


def minimal_control_terms(df: pd.DataFrame) -> list[str]:
    terms: list[str] = []
    for preferred_group in [
        ["log_ucdb_pop_2025"],
        ["log_ucdb_area_km2", "log_city_area_km2", "log_ghsl_pop_density_2020_100m"],
        ["ghsl_built_share_2020_100m"],
        ["quality_score"],
    ]:
        chosen = next((col for col in preferred_group if col in df.columns and df[col].notna().any()), None)
        if chosen:
            terms.append(chosen)
    return terms


def build_inequality_model_table(
    ineq_main: pd.DataFrame,
    city_controls: pd.DataFrame,
    morphotype_city_table: pd.DataFrame,
) -> pd.DataFrame:
    table = ineq_main[ineq_main["group_level"].eq("city_morphotype")].copy()
    control_cols = [c for c in city_controls.columns if c == "city_id" or c not in table.columns]
    table = table.merge(city_controls[control_cols], on="city_id", how="left")
    type_covars = morphotype_city_table[
        [
            c
            for c in [
                "city_id",
                "morphotype",
                "mean_type_confidence",
                "median_type_confidence",
                "low_confidence_unit_share",
                "small_group_flag",
                "population_share_in_city",
            ]
            if c in morphotype_city_table.columns
        ]
    ].drop_duplicates(["city_id", "morphotype"])
    table = table.merge(type_covars, on=["city_id", "morphotype"], how="left")
    table["inequality_model_weight"] = pd.to_numeric(table["population_sum"], errors="coerce")
    table["small_group_lt20"] = pd.to_numeric(table["unit_count"], errors="coerce") < 20
    table["small_group_lt30"] = pd.to_numeric(table["unit_count"], errors="coerce") < 30
    outcomes = ["population_weighted_access_share", "accessibility_gini", "no_access_population_share"]
    table["model_c_any_outcome_nan"] = table[outcomes].isna().any(axis=1)
    table["model_c_main_included"] = (~table["small_group_lt20"]) & (~table["model_c_any_outcome_nan"])
    return table


def weighted_summary_by_group(
    df: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
    weight_col: str,
    prefix: str = "",
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_cols, keys))
        record["row_count"] = int(len(group))
        if weight_col in group.columns:
            record[f"{prefix}weight_sum"] = float(pd.to_numeric(group[weight_col], errors="coerce").sum())
        for col in value_cols:
            if col not in group.columns:
                continue
            record[f"{prefix}{col}_weighted_mean"] = weighted_mean(group[col], group[weight_col] if weight_col in group.columns else None)
            record[f"{prefix}{col}_mean"] = float(pd.to_numeric(group[col], errors="coerce").mean())
            record[f"{prefix}{col}_median"] = float(pd.to_numeric(group[col], errors="coerce").median())
            record[f"{prefix}{col}_p10"] = float(pd.to_numeric(group[col], errors="coerce").quantile(0.10))
            record[f"{prefix}{col}_p90"] = float(pd.to_numeric(group[col], errors="coerce").quantile(0.90))
        records.append(record)
    return pd.DataFrame(records)


def build_descriptive_outputs(
    main_units: pd.DataFrame,
    od_main: pd.DataFrame,
    unit_model_table: pd.DataFrame,
    access_city_type_category: pd.DataFrame,
    city_model_table: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    type_cols = ["morphotype", "morphotype_name"]
    desc = (
        main_units.groupby(type_cols, as_index=False)
        .agg(
            city_count=("city_id", "nunique"),
            unit_count=("unit_id", "nunique"),
            population_sum=("population_sum", "sum"),
            area_sum_km2=("unit_area_km2", "sum"),
            mean_type_confidence=("type_confidence", "mean"),
            median_type_confidence=("type_confidence", "median"),
            low_confidence_unit_share=("low_confidence_flag", "mean"),
            low_quality_unit_share=("low_quality_unit_flag", "mean"),
            mean_quality_score=("quality_score", "mean"),
            mean_edge_density_km_per_km2=("edge_density_km_per_km2", "mean"),
            mean_intersection_density_per_km2=("intersection_density_per_km2", "mean"),
            mean_dead_end_share=("dead_end_share", "mean"),
            mean_edge_circuity_mean=("edge_circuity_mean", "mean"),
            mean_orientation_order=("orientation_order", "mean"),
        )
        .copy()
    )
    desc["population_share_total"] = safe_divide(desc["population_sum"], pd.Series(desc["population_sum"].sum(), index=desc.index))
    desc["unit_share_total"] = safe_divide(desc["unit_count"], pd.Series(desc["unit_count"].sum(), index=desc.index))

    od_type = od_main[od_main["group_level"].eq("origin_morphotype")].rename(
        columns={"origin_morphotype": "morphotype", "origin_morphotype_name": "morphotype_name"}
    )
    circuity = weighted_summary_by_group(
        od_type,
        ["morphotype", "morphotype_name"],
        [
            "route_found_weighted_share",
            "od_circuity_weighted_mean",
            "od_circuity_weighted_p50",
            "od_circuity_weighted_p90",
            "od_circuity_mean",
            "detour_ratio_median",
        ],
        "od_pair_weight_sum",
    )
    if not od_type.empty:
        od_counts = (
            od_type.groupby(["morphotype", "morphotype_name"], as_index=False)
            .agg(city_group_count=("city_id", "nunique"), od_pair_count=("od_pair_count", "sum"))
            .copy()
        )
        circuity = circuity.merge(od_counts, on=["morphotype", "morphotype_name"], how="left")

    facility_access = (
        access_city_type_category.groupby(["morphotype", "morphotype_name", "category"], as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "city_group_count": g["city_id"].nunique(),
                    "unit_count": g["unit_count"].sum(),
                    "population_sum": g["population_sum"].sum(),
                    "population_weighted_access_share": weighted_mean(g["population_weighted_access_share"], g["population_sum"]),
                    "unweighted_access_share": g["unweighted_access_share"].mean(),
                    "population_weighted_accessibility_score": weighted_mean(g["population_weighted_accessibility_score"], g["population_sum"]),
                    "median_distance_m": weighted_mean(g["median_distance_m"], g["population_sum"]),
                    "p90_distance_m": weighted_mean(g["p90_distance_m"], g["population_sum"]),
                    "long_snap_share": weighted_mean(g["long_snap_share"], g["population_sum"]),
                    "mean_type_confidence": weighted_mean(g["mean_type_confidence"], g["population_sum"]),
                }
            )
        )
        .reset_index(drop=True)
    )

    inequality = city_model_table.copy()
    confidence = (
        unit_model_table.drop_duplicates(["city_id", "unit_id"])
        .groupby(["morphotype", "morphotype_name"], as_index=False)
        .agg(
            unit_count=("unit_id", "nunique"),
            city_count=("city_id", "nunique"),
            confidence_mean=("type_confidence", "mean"),
            confidence_median=("type_confidence", "median"),
            confidence_p10=("type_confidence", lambda s: pd.to_numeric(s, errors="coerce").quantile(0.10)),
            confidence_p90=("type_confidence", lambda s: pd.to_numeric(s, errors="coerce").quantile(0.90)),
            high_confidence_share=("high_confidence_flag", "mean"),
            low_confidence_share=("low_confidence_flag", "mean"),
            low_quality_share=("low_quality_unit_flag", "mean"),
        )
        .copy()
    )
    return {
        "morphotype_descriptive_stats": desc.sort_values("morphotype"),
        "circuity_by_morphotype": circuity.sort_values("morphotype"),
        "facility_access_by_morphotype": facility_access.sort_values(["category", "morphotype"]),
        "inequality_by_city_category": inequality.sort_values(["city_id", "category"]),
        "confidence_by_morphotype": confidence.sort_values("morphotype"),
    }


def choose_reference(values: pd.Series, preferred: str = "MT01") -> str:
    present = sorted(v for v in values.dropna().astype(str).unique() if v)
    if preferred in present:
        return preferred
    return present[0] if present else preferred


def coefficient_kind(term: str) -> str:
    if term == "Intercept":
        return "intercept"
    if "morphotype" in term:
        return "morphotype"
    if "category" in term:
        return "facility_category"
    if "city_id" in term or "region" in term:
        return "fixed_effect"
    return "control"


def error_model_result(
    model_id: str,
    model_family: str,
    outcome: str,
    formula: str,
    status: str,
    message: str,
    scenario: str,
    n_input: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    row = {
        "model_id": model_id,
        "model_family": model_family,
        "scenario": scenario,
        "outcome": outcome,
        "term": "__model_error__",
        "term_kind": "error",
        "coef": np.nan,
        "std_err": np.nan,
        "t_value": np.nan,
        "p_value": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "nobs": 0,
        "n_input": n_input,
        "dropped_n": n_input,
        "r_squared": np.nan,
        "adj_r_squared": np.nan,
        "aic": np.nan,
        "bic": np.nan,
        "df_resid": np.nan,
        "cov_type": "none",
        "weight_col": None,
        "cluster_col": None,
        "formula": formula,
        "status": status,
        "message": message,
    }
    return pd.DataFrame([row]), row.copy()


def fit_formula_model(
    data: pd.DataFrame,
    formula: str,
    outcome: str,
    model_id: str,
    model_family: str,
    scenario: str = "main",
    weight_col: str | None = None,
    cluster_col: str | None = "city_id",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    n_input = int(len(data))
    if not HAVE_STATSMODELS:
        return error_model_result(
            model_id,
            model_family,
            outcome,
            formula,
            "failed",
            "statsmodels is not installed; no fallback was used in this run.",
            scenario,
            n_input,
        )
    model_df = data.copy().replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    if outcome not in model_df.columns:
        return error_model_result(model_id, model_family, outcome, formula, "failed", "outcome column missing", scenario, n_input)
    model_df[outcome] = pd.to_numeric(model_df[outcome], errors="coerce")
    model_df = model_df[model_df[outcome].notna()].copy()
    if weight_col is not None:
        if weight_col not in model_df.columns:
            return error_model_result(model_id, model_family, outcome, formula, "failed", "weight column missing", scenario, n_input)
        model_df[weight_col] = pd.to_numeric(model_df[weight_col], errors="coerce")
        model_df = model_df[model_df[weight_col].notna() & np.isfinite(model_df[weight_col]) & (model_df[weight_col] > 0)].copy()
    if len(model_df) < 5:
        return error_model_result(
            model_id,
            model_family,
            outcome,
            formula,
            "failed",
            "fewer than five usable rows after missing/weight filtering",
            scenario,
            n_input,
        )

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if weight_col is not None:
                fit = smf.wls(formula=formula, data=model_df, weights=model_df[weight_col], missing="drop").fit()
            else:
                fit = smf.ols(formula=formula, data=model_df, missing="drop").fit()

            cov_type_used = "nonrobust"
            result = fit
            cov_message = ""
            if cluster_col is not None and cluster_col in model_df.columns:
                try:
                    used_index = fit.model.data.row_labels
                    groups = model_df.loc[used_index, cluster_col]
                    if groups.nunique(dropna=True) >= 2:
                        result = fit.get_robustcov_results(cov_type="cluster", groups=groups)
                        cov_type_used = "cluster"
                    else:
                        result = fit.get_robustcov_results(cov_type="HC3")
                        cov_type_used = "HC3"
                except Exception as exc:
                    result = fit.get_robustcov_results(cov_type="HC3")
                    cov_type_used = "HC3"
                    cov_message = f"cluster covariance failed; HC3 fallback used: {exc!r}"
            else:
                result = fit.get_robustcov_results(cov_type="HC3")
                cov_type_used = "HC3"

        names = list(result.model.exog_names)
        params = np.asarray(result.params)
        bse = np.asarray(result.bse)
        tvalues = np.asarray(result.tvalues)
        pvalues = np.asarray(result.pvalues)
        ci = np.asarray(result.conf_int())
        nobs = int(result.nobs)
        warn_text = "; ".join(str(w.message) for w in caught[:5])
        if cov_message:
            warn_text = f"{warn_text}; {cov_message}" if warn_text else cov_message
        rows = []
        for i, term in enumerate(names):
            rows.append(
                {
                    "model_id": model_id,
                    "model_family": model_family,
                    "scenario": scenario,
                    "outcome": outcome,
                    "term": term,
                    "term_kind": coefficient_kind(term),
                    "coef": float(params[i]),
                    "std_err": float(bse[i]),
                    "t_value": float(tvalues[i]),
                    "p_value": float(pvalues[i]),
                    "ci_low": float(ci[i, 0]),
                    "ci_high": float(ci[i, 1]),
                    "nobs": nobs,
                    "n_input": n_input,
                    "dropped_n": int(n_input - nobs),
                    "r_squared": float(getattr(result, "rsquared", np.nan)),
                    "adj_r_squared": float(getattr(result, "rsquared_adj", np.nan)),
                    "aic": float(getattr(result, "aic", np.nan)),
                    "bic": float(getattr(result, "bic", np.nan)),
                    "df_resid": float(getattr(result, "df_resid", np.nan)),
                    "cov_type": cov_type_used,
                    "weight_col": weight_col,
                    "cluster_col": cluster_col,
                    "formula": formula,
                    "status": "ok",
                    "message": warn_text,
                }
            )
        info = {
            "model_id": model_id,
            "model_family": model_family,
            "scenario": scenario,
            "outcome": outcome,
            "formula": formula,
            "weight_col": weight_col,
            "cluster_col": cluster_col,
            "cov_type": cov_type_used,
            "n_input": n_input,
            "nobs": nobs,
            "dropped_n": int(n_input - nobs),
            "r_squared": float(getattr(result, "rsquared", np.nan)),
            "adj_r_squared": float(getattr(result, "rsquared_adj", np.nan)),
            "df_resid": float(getattr(result, "df_resid", np.nan)),
            "status": "ok",
            "message": warn_text,
            "converged": True,
        }
        return pd.DataFrame(rows), info
    except Exception as exc:
        return error_model_result(model_id, model_family, outcome, formula, "failed", repr(exc), scenario, n_input)


def run_model_a(
    model_a_table: pd.DataFrame,
    scenario: str = "main",
    weighted: bool = True,
    min_od_pair_count: int | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    df = model_a_table.copy()
    if min_od_pair_count is not None:
        df = df[pd.to_numeric(df["od_pair_count"], errors="coerce") >= min_od_pair_count].copy()
    df["origin_morphotype"] = df["origin_morphotype"].astype(str)
    ref = choose_reference(df["origin_morphotype"])
    formula_base = f"C(origin_morphotype, Treatment(reference='{ref}')) + C(city_id)"
    outcomes = [
        "log_od_circuity_weighted_mean",
        "od_circuity_weighted_mean",
        "od_circuity_weighted_p50",
        "od_circuity_weighted_p90",
        "route_found_weighted_share",
    ]
    result_frames: list[pd.DataFrame] = []
    infos: list[dict[str, Any]] = []
    for outcome in outcomes:
        formula = f"{outcome} ~ {formula_base}"
        weight_col = "od_model_weight" if weighted else None
        coefs, info = fit_formula_model(
            df,
            formula,
            outcome=outcome,
            model_id=f"A_{outcome}" if min_od_pair_count is None else f"A_{outcome}_odpair_ge_{min_od_pair_count}",
            model_family="A_OD_circuity",
            scenario=scenario,
            weight_col=weight_col,
            cluster_col="city_id",
        )
        result_frames.append(coefs)
        infos.append(info)
    return pd.concat(result_frames, ignore_index=True), infos


def run_model_b(model_b_table: pd.DataFrame, scenario: str = "main", weighted: bool = True) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    df = model_b_table.copy()
    df["morphotype"] = df["morphotype"].astype(str)
    df["category"] = df["category"].astype(str)
    ref = choose_reference(df["morphotype"])
    weight_col = "facility_model_weight" if weighted else None
    result_frames: list[pd.DataFrame] = []
    infos: list[dict[str, Any]] = []

    for outcome in ["access_15min_num", "accessibility_score"]:
        pooled_formula = (
            f"{outcome} ~ "
            f"C(morphotype, Treatment(reference='{ref}')) * C(category) "
            "+ C(city_id) + type_confidence"
        )
        coefs, info = fit_formula_model(
            df,
            pooled_formula,
            outcome=outcome,
            model_id=f"B_{outcome}_pooled_interaction",
            model_family="B_facility_accessibility",
            scenario=scenario,
            weight_col=weight_col,
            cluster_col="city_id",
        )
        result_frames.append(coefs)
        infos.append(info)

    for category in sorted(df["category"].dropna().unique()):
        sub = df[df["category"].eq(category)].copy()
        ref_sub = choose_reference(sub["morphotype"], preferred=ref)
        formula = (
            "access_15min_num ~ "
            f"C(morphotype, Treatment(reference='{ref_sub}')) "
            "+ C(city_id) + type_confidence"
        )
        coefs, info = fit_formula_model(
            sub,
            formula,
            outcome="access_15min_num",
            model_id=f"B_access_{category}",
            model_family="B_facility_accessibility",
            scenario=scenario,
            weight_col=weight_col,
            cluster_col="city_id",
        )
        result_frames.append(coefs)
        infos.append(info)
    return pd.concat(result_frames, ignore_index=True), infos


def run_model_c(
    model_c_table: pd.DataFrame,
    scenario: str = "main",
    weighted: bool = True,
    min_unit_count: int = 20,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    df = model_c_table.copy()
    df = df[pd.to_numeric(df["unit_count"], errors="coerce") >= min_unit_count].copy()
    df["category"] = df["category"].astype(str)
    df["morphotype"] = df["morphotype"].astype(str)
    ref = choose_reference(df["morphotype"])
    controls = minimal_control_terms(df)
    rhs = " + ".join([f"C(morphotype, Treatment(reference='{ref}')) * C(category)", *controls])
    outcomes = ["accessibility_gini", "no_access_population_share", "population_weighted_access_share"]
    weight_col = "inequality_model_weight" if weighted else None
    result_frames: list[pd.DataFrame] = []
    infos: list[dict[str, Any]] = []
    for outcome in outcomes:
        formula = f"{outcome} ~ {rhs}"
        coefs, info = fit_formula_model(
            df,
            formula,
            outcome=outcome,
            model_id=f"C_{outcome}_morphotype_category_min{min_unit_count}",
            model_family="C_accessibility_inequality",
            scenario=scenario,
            weight_col=weight_col,
            cluster_col="city_id",
        )
        result_frames.append(coefs)
        infos.append(info)
    return pd.concat(result_frames, ignore_index=True), infos


def key_terms(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        df["term_kind"].isin(["morphotype", "control"])
        & df["term"].ne("Intercept")
        & df["status"].eq("ok")
    ].copy()


def compare_weighted_unweighted(weighted_df: pd.DataFrame, unweighted_df: pd.DataFrame) -> pd.DataFrame:
    lhs = key_terms(weighted_df).copy()
    rhs = key_terms(unweighted_df).copy()
    cols = ["model_family", "model_id", "outcome", "term"]
    merged = lhs[cols + ["coef", "std_err", "p_value", "nobs"]].merge(
        rhs[cols + ["coef", "std_err", "p_value", "nobs"]],
        on=cols,
        how="outer",
        suffixes=("_weighted", "_unweighted"),
    )
    merged["coef_difference_unweighted_minus_weighted"] = merged["coef_unweighted"] - merged["coef_weighted"]
    merged["same_sign_flag"] = np.sign(merged["coef_weighted"]) == np.sign(merged["coef_unweighted"])
    return merged


def run_robustness(
    main_units: pd.DataFrame,
    model_a_table: pd.DataFrame,
    model_b_table: pd.DataFrame,
    model_c_table: pd.DataFrame,
    confidence_threshold: float,
    small_group_min_units: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    infos: list[dict[str, Any]] = []

    high_a = model_a_table[
        pd.to_numeric(model_a_table["mean_type_confidence"], errors="coerce") >= confidence_threshold
    ].copy()
    high_b = model_b_table[
        pd.to_numeric(model_b_table["type_confidence"], errors="coerce") >= confidence_threshold
    ].copy()
    high_c = model_c_table[
        pd.to_numeric(model_c_table.get("mean_type_confidence", np.nan), errors="coerce") >= confidence_threshold
    ].copy()

    high_frames: list[pd.DataFrame] = []
    for runner, data in [
        (run_model_a, high_a),
        (run_model_b, high_b),
        (run_model_c, high_c),
    ]:
        coefs, info = runner(data, scenario="high_confidence", weighted=True)
        high_frames.append(coefs)
        infos.extend(info)
    high_conf = pd.concat(high_frames, ignore_index=True)

    weighted_frames: list[pd.DataFrame] = []
    unweighted_frames: list[pd.DataFrame] = []
    for runner, data in [
        (run_model_a, model_a_table),
        (run_model_b, model_b_table),
        (run_model_c, model_c_table),
    ]:
        w, info_w = runner(data, scenario="weighted_reference", weighted=True)
        u, info_u = runner(data, scenario="unweighted", weighted=False)
        weighted_frames.append(w)
        unweighted_frames.append(u)
        infos.extend(info_w)
        infos.extend(info_u)
    weighted_ref = pd.concat(weighted_frames, ignore_index=True)
    unweighted = pd.concat(unweighted_frames, ignore_index=True)
    unweighted_vs_weighted = compare_weighted_unweighted(weighted_ref, unweighted)

    small_a = model_a_table[
        pd.to_numeric(model_a_table["od_pair_count"], errors="coerce").ge(30)
    ].copy()
    small_a_100 = model_a_table[
        pd.to_numeric(model_a_table["od_pair_count"], errors="coerce").ge(100)
    ].copy()
    small_b = model_b_table[
        pd.to_numeric(model_b_table.get("morphotype_city_unit_count", 999), errors="coerce") >= small_group_min_units
    ].copy()
    small_c = model_c_table[
        pd.to_numeric(model_c_table["unit_count"], errors="coerce").ge(30)
    ].dropna(
        subset=["accessibility_gini", "no_access_population_share", "population_weighted_access_share"]
    ).copy()
    small_frames: list[pd.DataFrame] = []
    coefs, info = run_model_a(small_a, scenario="od_pair_count_ge_30", weighted=True, min_od_pair_count=30)
    small_frames.append(coefs)
    infos.extend(info)
    coefs, info = run_model_a(small_a_100, scenario="od_pair_count_ge_100", weighted=True, min_od_pair_count=100)
    small_frames.append(coefs)
    infos.extend(info)
    coefs, info = run_model_b(small_b, scenario="remove_small_groups", weighted=True)
    small_frames.append(coefs)
    infos.extend(info)
    coefs, info = run_model_c(small_c, scenario="unit_count_ge_30", weighted=True, min_unit_count=30)
    small_frames.append(coefs)
    infos.extend(info)
    remove_small = pd.concat(small_frames, ignore_index=True)
    combined = pd.concat(
        [
            high_conf.assign(robustness_file="robustness_high_confidence"),
            unweighted.assign(robustness_file="robustness_unweighted"),
            remove_small.assign(robustness_file="robustness_remove_small_groups"),
        ],
        ignore_index=True,
    )
    return high_conf, unweighted_vs_weighted, remove_small, combined, infos


def null_report(df: pd.DataFrame, cols: list[str]) -> list[dict[str, Any]]:
    records = []
    n = len(df)
    for col in cols:
        if col in df.columns:
            missing = int(df[col].isna().sum())
            records.append({"column": col, "missing_count": missing, "missing_share": missing / n if n else None})
    return records


def collect_input_report(
    paths: StepPaths,
    inputs: dict[str, pd.DataFrame],
    main_tables: dict[str, pd.DataFrame | list[str]],
    small_group_min_units: int,
) -> dict[str, Any]:
    main_units = main_tables["main_units"]
    acc_main = main_tables["acc_main"]
    od_main = main_tables["od_main"]
    ineq_main = main_tables["ineq_main"]
    main_city_ids = main_tables["main_city_ids"]
    assert isinstance(main_units, pd.DataFrame)
    assert isinstance(acc_main, pd.DataFrame)
    assert isinstance(od_main, pd.DataFrame)
    assert isinstance(ineq_main, pd.DataFrame)

    od_type = od_main[od_main["group_level"].eq("origin_morphotype")].copy()
    acc_groups = (
        acc_main.groupby(["city_id", "morphotype", "category"], as_index=False)
        .agg(unit_count=("unit_id", "nunique"))
        .copy()
    )
    ineq_city_type = ineq_main[ineq_main["group_level"].eq("city_morphotype")].copy()
    report = {
        "generated_at": now_iso(),
        "step": STEP_NAME,
        "analysis_contract": {
            "main_filter": {
                "units": "network_type == 'drive' AND scale == 'hex_1km' AND analysis_scope == 'main_fit'",
                "step10_accessibility": "main_analysis_included == True AND analysis_scope == 'main_fit' AND scale == 'hex_1km'",
                "step10_od": "origin/destination main_analysis_included == True AND origin/destination analysis_scope == 'main_fit' AND scale == 'hex_1km'",
                "step10_inequality": "scale == 'hex_1km', city-level rows for model C",
            },
            "main_city_expected": 48,
            "morphotype_expected": 10,
            "small_group_min_units": small_group_min_units,
        },
        "input_paths": {
            "step09_units": file_metadata(paths.step09_units),
            "step09_city_profiles": file_metadata(paths.step09_city_profiles),
            "step10_od": file_metadata(paths.step10_od),
            "step10_accessibility": file_metadata(paths.step10_accessibility),
            "step10_inequality": file_metadata(paths.step10_inequality),
            "step10_validation_summary": file_metadata(paths.step10_validation_summary),
            "step07_city_quality": file_metadata(paths.step07_city_quality),
            "step07_local_quality": file_metadata(paths.step07_local_quality),
            "step03_city_env": file_metadata(paths.step03_city_env),
            "step01_city_master": file_metadata(paths.step01_city_master),
        },
        "main_scope": {
            "main_city_count": len(main_city_ids),
            "main_city_ids": main_city_ids,
            "main_unit_rows": int(len(main_units)),
            "main_unit_cities": int(main_units["city_id"].nunique()),
            "main_unit_morphotypes": int(main_units["morphotype"].nunique()),
            "step10_accessibility_rows": int(len(acc_main)),
            "step10_accessibility_units": int(acc_main["unit_id"].nunique()),
            "step10_accessibility_cities": int(acc_main["city_id"].nunique()),
            "step10_od_rows": int(len(od_main)),
            "step10_od_cities": int(od_main["city_id"].nunique()),
            "step10_inequality_rows": int(len(ineq_main)),
            "step10_inequality_city_rows": int(ineq_main["group_level"].eq("city").sum()),
        },
        "step10_nan_and_small_groups": {
            "od_nulls": null_report(
                od_main,
                [
                    "od_circuity_weighted_mean",
                    "od_circuity_weighted_p50",
                    "od_circuity_weighted_p90",
                    "route_found_weighted_share",
                    "od_pair_count",
                ],
            ),
            "od_small_groups": {
                "threshold": small_group_min_units,
                "rows": int((pd.to_numeric(od_type.get("od_pair_count", pd.Series(dtype=float)), errors="coerce") < small_group_min_units).sum()),
                "examples": od_type.loc[
                    pd.to_numeric(od_type.get("od_pair_count", pd.Series(dtype=float)), errors="coerce") < small_group_min_units,
                    ["city_id", "origin_morphotype", "od_pair_count"],
                ]
                .head(20)
                .to_dict("records"),
            },
            "accessibility_nulls": null_report(
                acc_main,
                ["access_15min", "accessibility_score", "network_distance_to_nearest_m", "walk_snap_distance_m"],
            ),
            "accessibility_small_groups": {
                "threshold": small_group_min_units,
                "rows": int((acc_groups["unit_count"] < small_group_min_units).sum()),
                "examples": acc_groups.loc[acc_groups["unit_count"] < small_group_min_units]
                .head(20)
                .to_dict("records"),
            },
            "inequality_nulls": null_report(
                ineq_main,
                ["accessibility_gini", "no_access_population_share", "population_weighted_access_share", "unit_count"],
            ),
            "inequality_small_groups": {
                "threshold": small_group_min_units,
                "rows": int(
                    (
                        (ineq_city_type["unit_count"] < small_group_min_units)
                        if "unit_count" in ineq_city_type.columns
                        else pd.Series(False, index=ineq_city_type.index)
                    ).sum()
                ),
                "examples": ineq_city_type.loc[
                    (
                        (ineq_city_type["unit_count"] < small_group_min_units)
                        if "unit_count" in ineq_city_type.columns
                        else pd.Series(False, index=ineq_city_type.index)
                    ),
                    ["city_id", "morphotype", "category", "unit_count"],
                ]
                .head(20)
                .to_dict("records"),
            },
            "inequality_city_morphotype_lt20_or_nan": {
                "rows": int(
                    (
                        (pd.to_numeric(ineq_city_type.get("unit_count", pd.Series(dtype=float)), errors="coerce") < 20)
                        | ineq_city_type[
                            ["population_weighted_access_share", "accessibility_gini", "no_access_population_share"]
                        ].isna().any(axis=1)
                    ).sum()
                ),
                "examples": ineq_city_type.loc[
                    (
                        (pd.to_numeric(ineq_city_type.get("unit_count", pd.Series(dtype=float)), errors="coerce") < 20)
                        | ineq_city_type[
                            ["population_weighted_access_share", "accessibility_gini", "no_access_population_share"]
                        ].isna().any(axis=1)
                    ),
                    [
                        "city_id",
                        "city_name_en",
                        "morphotype",
                        "category",
                        "unit_count",
                        "population_weighted_access_share",
                        "accessibility_gini",
                        "no_access_population_share",
                    ],
                ]
                .head(30)
                .to_dict("records"),
            },
            "inequality_city_morphotype_lt30_or_nan": {
                "rows": int(
                    (
                        (pd.to_numeric(ineq_city_type.get("unit_count", pd.Series(dtype=float)), errors="coerce") < 30)
                        | ineq_city_type[
                            ["population_weighted_access_share", "accessibility_gini", "no_access_population_share"]
                        ].isna().any(axis=1)
                    ).sum()
                ),
            },
        },
    }
    report["status"] = (
        "pass"
        if report["main_scope"]["main_city_count"] == 48 and report["main_scope"]["main_unit_morphotypes"] == 10
        else "warn"
    )
    return report


def build_input_field_check(
    paths: StepPaths,
    inputs: dict[str, pd.DataFrame],
    main_tables: dict[str, pd.DataFrame | list[str]],
    unit_model_table: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "step09_units": [
            "city_id",
            "network_type",
            "scale",
            "analysis_scope",
            "unit_id",
            "morphotype",
            "type_confidence",
        ],
        "step09_city_profiles": ["city_id", "network_type", "scale", "dominant_morphotype", "morphotype_entropy"],
        "step10_od": [
            "city_id",
            "scale",
            "group_level",
            "origin_morphotype",
            "od_circuity_weighted_mean",
            "route_found_weight_sum",
            "od_pair_weight_sum",
        ],
        "step10_accessibility": [
            "city_id",
            "scale",
            "unit_id",
            "main_analysis_included",
            "analysis_scope",
            "morphotype",
            "category",
            "access_15min",
            "accessibility_score",
        ],
        "step10_inequality": [
            "city_id",
            "scale",
            "group_level",
            "morphotype",
            "category",
            "unit_count",
            "population_weighted_access_share",
            "accessibility_gini",
            "no_access_population_share",
        ],
    }
    path_map = {
        "step09_units": paths.step09_units,
        "step09_city_profiles": paths.step09_city_profiles,
        "step10_od": paths.step10_od,
        "step10_accessibility": paths.step10_accessibility,
        "step10_inequality": paths.step10_inequality,
        "step07_city_quality": paths.step07_city_quality,
        "step03_city_env": paths.step03_city_env,
        "step01_city_master": paths.step01_city_master,
    }
    records: list[dict[str, Any]] = []
    for name, path in path_map.items():
        df = inputs.get(
            {
                "step09_units": "units",
                "step09_city_profiles": "city_profiles",
                "step10_od": "od",
                "step10_accessibility": "accessibility",
                "step10_inequality": "inequality",
                "step07_city_quality": "city_quality",
                "step03_city_env": "city_env",
                "step01_city_master": "city_master",
            }[name],
            pd.DataFrame(),
        )
        missing = [c for c in required_columns.get(name, []) if c not in df.columns]
        records.append(
            {
                "input_name": name,
                "path": str(path),
                "exists": path.exists(),
                "row_count": int(len(df)) if isinstance(df, pd.DataFrame) else 0,
                "column_count": int(df.shape[1]) if isinstance(df, pd.DataFrame) else 0,
                "required_columns_missing": ";".join(missing),
                "status": "pass" if path.exists() and not missing else ("warn" if path.exists() else "fail"),
            }
        )
    main_units = main_tables["main_units"]
    assert isinstance(main_units, pd.DataFrame)
    records.append(
        {
            "input_name": "main_units_filter",
            "path": str(paths.step09_units),
            "exists": True,
            "row_count": int(len(main_units)),
            "column_count": int(main_units.shape[1]),
            "required_columns_missing": "",
            "status": "pass" if len(main_units) == 74861 and main_units["city_id"].nunique() == 48 else "warn",
            "detail": f"rows={len(main_units)}, cities={main_units['city_id'].nunique()}, morphotypes={main_units['morphotype'].nunique()}",
        }
    )
    match_share = float(unit_model_table["step09_join_matched"].mean()) if "step09_join_matched" in unit_model_table.columns else np.nan
    records.append(
        {
            "input_name": "step10_facility_join_step09_main_units",
            "path": f"{paths.step10_accessibility} + {paths.step09_units}",
            "exists": True,
            "row_count": int(len(unit_model_table)),
            "column_count": int(unit_model_table.shape[1]),
            "required_columns_missing": "",
            "status": "pass" if math.isclose(match_share, 1.0, rel_tol=0, abs_tol=1e-12) else "fail",
            "detail": f"join_key=city_id+scale+unit_id; match_share={match_share:.12f}",
        }
    )
    return pd.DataFrame(records)


def write_input_report(paths: StepPaths, report: dict[str, Any]) -> None:
    write_json(paths.output_dir / OUTPUT_FILES["input_report_json"], report)
    small = report["step10_nan_and_small_groups"]
    md = [
        f"# {STEP_NAME} input acceptance report",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Acceptance status: {report['status']}",
        f"- Main analysis city count: {report['main_scope']['main_city_count']}",
        f"- Main analysis unit count: {report['main_scope']['main_unit_rows']}",
        f"- morphotype count: {report['main_scope']['main_unit_morphotypes']}",
        f"- Step10 facility accessibility row count: {report['main_scope']['step10_accessibility_rows']}",
        f"- Step10 OD metric row count: {report['main_scope']['step10_od_rows']}",
        f"- Step10 inequality metric row count: {report['main_scope']['step10_inequality_rows']}",
        "",
        "## Step10 NaN and Very Small Groups",
        "",
        f"- OD very small group rows: {small['od_small_groups']['rows']} (threshold < {small['od_small_groups']['threshold']})",
        f"- Facility accessibility very small group rows: {small['accessibility_small_groups']['rows']} (city x morphotype x category)",
        f"- Inequality city_morphotype very small group rows: {small['inequality_small_groups']['rows']}",
        f"- Inequality city_morphotype rows with unit_count<20 or key outcome NaN: {small['inequality_city_morphotype_lt20_or_nan']['rows']}",
        f"- Inequality city_morphotype rows with unit_count<30 or key outcome NaN: {small['inequality_city_morphotype_lt30_or_nan']['rows']}",
        "",
        "See the JSON file for detailed fields, mtime, size, sha256, and missingness ratios.",
        "",
    ]
    (paths.output_dir / OUTPUT_FILES["input_report_md"]).write_text("\n".join(md), encoding="utf-8")


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No records to display_"
    show = df.head(max_rows).copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    headers = list(show.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in headers) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Showing the first {max_rows} rows out of {len(df)} total._")
    return "\n".join(lines)


def write_model_markdown(path: Path, title: str, results: pd.DataFrame, infos: list[dict[str, Any]]) -> None:
    info_df = pd.DataFrame(infos)
    focus = results[
        results["status"].eq("ok")
        & results["term_kind"].isin(["morphotype", "control"])
        & results["term"].ne("Intercept")
    ].copy()
    focus = focus.sort_values(["model_id", "term_kind", "p_value"]).head(30)
    lines = [
        f"# {title}",
        "",
        "## Model Specifications",
        "",
        markdown_table(
            info_df[
                [
                    c
                    for c in [
                        "model_id",
                        "scenario",
                        "outcome",
                        "nobs",
                        "r_squared",
                        "adj_r_squared",
                        "cov_type",
                        "weight_col",
                        "status",
                    ]
                    if c in info_df.columns
                ]
            ],
            max_rows=40,
        ),
        "",
        "## Key Coefficients",
        "",
        markdown_table(
            focus[
                [
                    "model_id",
                    "outcome",
                    "term",
                    "coef",
                    "std_err",
                    "p_value",
                    "ci_low",
                    "ci_high",
                    "nobs",
                ]
            ],
            max_rows=30,
        ),
        "",
        "Note: city-cluster standard errors are reported by default; if clustered covariance fails, the model falls back to HC3 and records that in the message field.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def dominant_inequality_summary(city_model_table: pd.DataFrame) -> pd.DataFrame:
    return (
        city_model_table.groupby(["dominant_morphotype", "dominant_morphotype_name", "category"], dropna=False, as_index=False)
        .agg(
            city_count=("city_id", "nunique"),
            mean_accessibility_gini=("accessibility_gini", "mean"),
            mean_no_access_population_share=("no_access_population_share", "mean"),
            mean_population_weighted_access_share=("population_weighted_access_share", "mean"),
            mean_morphotype_entropy=("morphotype_entropy", "mean"),
        )
        .copy()
    )


def build_marginal_effects_table(
    model_a_table: pd.DataFrame,
    model_b_table: pd.DataFrame,
    model_c_table: pd.DataFrame,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    a = (
        model_a_table.groupby(["origin_morphotype", "origin_morphotype_name"], as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "source_model": "A_OD_circuity",
                    "category": "all",
                    "outcome": "od_circuity_weighted_mean",
                    "marginal_mean": weighted_mean(g["od_circuity_weighted_mean"], g["od_model_weight"]),
                    "unweighted_mean": pd.to_numeric(g["od_circuity_weighted_mean"], errors="coerce").mean(),
                    "n": len(g),
                    "weight_sum": pd.to_numeric(g["od_model_weight"], errors="coerce").sum(),
                }
            )
        )
        .reset_index(drop=True)
        .rename(columns={"origin_morphotype": "morphotype", "origin_morphotype_name": "morphotype_name"})
    )
    records.append(a)
    for outcome in ["access_15min_num", "accessibility_score"]:
        b = (
            model_b_table.groupby(["morphotype", "morphotype_name", "category"], as_index=False)
            .apply(
                lambda g, outcome=outcome: pd.Series(
                    {
                        "source_model": "B_facility_accessibility",
                        "outcome": outcome,
                        "marginal_mean": weighted_mean(g[outcome], g["facility_model_weight"]),
                        "unweighted_mean": pd.to_numeric(g[outcome], errors="coerce").mean(),
                        "n": len(g),
                        "weight_sum": pd.to_numeric(g["facility_model_weight"], errors="coerce").sum(),
                    }
                )
            )
            .reset_index(drop=True)
        )
        records.append(b)
    c_base = model_c_table[model_c_table["model_c_main_included"].fillna(False)].copy()
    for outcome in ["population_weighted_access_share", "accessibility_gini", "no_access_population_share"]:
        c = (
            c_base.groupby(["morphotype", "morphotype_name", "category"], as_index=False)
            .apply(
                lambda g, outcome=outcome: pd.Series(
                    {
                        "source_model": "C_accessibility_inequality",
                        "outcome": outcome,
                        "marginal_mean": weighted_mean(g[outcome], g["inequality_model_weight"]),
                        "unweighted_mean": pd.to_numeric(g[outcome], errors="coerce").mean(),
                        "n": len(g),
                        "weight_sum": pd.to_numeric(g["inequality_model_weight"], errors="coerce").sum(),
                    }
                )
            )
            .reset_index(drop=True)
        )
        records.append(c)
    return pd.concat(records, ignore_index=True, sort=False)


def build_robustness_summary_table(
    high_conf: pd.DataFrame,
    unweighted_vs_weighted: pd.DataFrame,
    remove_small: pd.DataFrame,
    model_infos: list[dict[str, Any]],
) -> pd.DataFrame:
    info_df = pd.DataFrame(model_infos)
    rows = [
        {
            "scenario": "high_confidence",
            "coefficient_rows": len(high_conf),
            "model_count": int(high_conf["model_id"].nunique()) if "model_id" in high_conf.columns else 0,
            "ok_rows": int(high_conf["status"].eq("ok").sum()) if "status" in high_conf.columns else 0,
            "note": f"type_confidence >= {DEFAULT_CONFIDENCE_THRESHOLD}",
        },
        {
            "scenario": "unweighted_vs_weighted",
            "coefficient_rows": len(unweighted_vs_weighted),
            "model_count": int(unweighted_vs_weighted["model_id"].nunique()) if "model_id" in unweighted_vs_weighted.columns else 0,
            "ok_rows": int(unweighted_vs_weighted["same_sign_flag"].notna().sum()) if "same_sign_flag" in unweighted_vs_weighted.columns else 0,
            "note": "main population/OD weights compared with unweighted models",
        },
        {
            "scenario": "remove_small_groups",
            "coefficient_rows": len(remove_small),
            "model_count": int(remove_small["model_id"].nunique()) if "model_id" in remove_small.columns else 0,
            "ok_rows": int(remove_small["status"].eq("ok").sum()) if "status" in remove_small.columns else 0,
            "note": "OD pair count >=30/100, Model C unit_count >=30, Model B city-type unit_count >=3",
        },
    ]
    if not info_df.empty:
        for scenario, g in info_df.groupby("scenario", dropna=False):
            rows.append(
                {
                    "scenario": f"model_info__{scenario}",
                    "coefficient_rows": np.nan,
                    "model_count": int(g["model_id"].nunique()) if "model_id" in g.columns else len(g),
                    "ok_rows": int(g["status"].eq("ok").sum()) if "status" in g.columns else np.nan,
                    "note": f"nobs range {g['nobs'].min() if 'nobs' in g else np.nan} - {g['nobs'].max() if 'nobs' in g else np.nan}",
                }
            )
    return pd.DataFrame(rows)


def build_result_summary(
    descriptive: dict[str, pd.DataFrame],
    city_model_table: pd.DataFrame,
    model_results: pd.DataFrame,
    robustness_unweighted: pd.DataFrame,
    snap_info: dict[str, Any],
) -> dict[str, Any]:
    circ = descriptive["circuity_by_morphotype"].copy()
    circ_col = "od_circuity_weighted_mean_weighted_mean"
    if circ_col not in circ.columns:
        circ_col = next((c for c in circ.columns if "od_circuity_weighted_mean" in c), "")
    access = descriptive["facility_access_by_morphotype"].copy()
    access_all = (
        access.groupby(["morphotype", "morphotype_name"], as_index=False)
        .agg(mean_population_weighted_access_share=("population_weighted_access_share", "mean"))
        .copy()
    )
    high_circ = circ.sort_values(circ_col, ascending=False).head(3) if circ_col else pd.DataFrame()
    low_circ = circ.sort_values(circ_col, ascending=True).head(3) if circ_col else pd.DataFrame()
    high_access = access_all.sort_values("mean_population_weighted_access_share", ascending=False).head(3)
    low_access = access_all.sort_values("mean_population_weighted_access_share", ascending=True).head(3)

    dom_ineq = dominant_inequality_summary(city_model_table)
    dom_overall = (
        dom_ineq.groupby(["dominant_morphotype", "dominant_morphotype_name"], as_index=False)
        .agg(
            mean_accessibility_gini=("mean_accessibility_gini", "mean"),
            mean_no_access_population_share=("mean_no_access_population_share", "mean"),
            mean_population_weighted_access_share=("mean_population_weighted_access_share", "mean"),
        )
        .copy()
    )
    model_c_type_terms = model_results[
        model_results["term_kind"].eq("morphotype")
        & model_results["model_family"].eq("C_accessibility_inequality")
        & model_results["status"].eq("ok")
    ][["model_id", "outcome", "term", "coef", "p_value", "ci_low", "ci_high"]]

    stable = None
    if not robustness_unweighted.empty and "same_sign_flag" in robustness_unweighted.columns:
        stable = float(robustness_unweighted["same_sign_flag"].dropna().mean())
    return {
        "generated_at": now_iso(),
        "circuity_high_morphotypes": high_circ[
            [c for c in ["morphotype", "morphotype_name", circ_col] if c in high_circ.columns]
        ].to_dict("records"),
        "circuity_low_morphotypes": low_circ[
            [c for c in ["morphotype", "morphotype_name", circ_col] if c in low_circ.columns]
        ].to_dict("records"),
        "accessibility_high_morphotypes": high_access.to_dict("records"),
        "accessibility_low_morphotypes": low_access.to_dict("records"),
        "inequality_by_dominant_morphotype": dom_overall.sort_values("mean_accessibility_gini", ascending=False).to_dict("records"),
        "model_c_morphotype_terms": model_c_type_terms.to_dict("records"),
        "robustness": {
            "weighted_vs_unweighted_same_sign_share": stable,
            "high_confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
            "snap_tail_policy": snap_info,
        },
        "caution": [
            "Models are explanatory associations, not causal effects.",
            "City fixed effects in Models A/B absorb city-level quality differences; Model C uses facility-category terms plus the minimum city-level controls.",
            "Facility completeness and Step10 route failures can affect estimated accessibility and circuity.",
            "Long walk-network snapping distances are flagged and summarized, not hard-deleted.",
            "Small morphotype-city groups are retained in the main analysis and removed only in robustness checks.",
        ],
    }


def write_result_summary(path_md: Path, path_json: Path, summary: dict[str, Any]) -> None:
    write_json(path_json, summary)

    def bullet_records(records: list[dict[str, Any]], value_key: str | None = None) -> list[str]:
        lines = []
        for rec in records:
            name = rec.get("morphotype_name") or rec.get("dominant_morphotype_name") or rec.get("morphotype") or rec.get("dominant_morphotype")
            code = rec.get("morphotype") or rec.get("dominant_morphotype")
            if value_key and value_key in rec:
                lines.append(f"- {code} {name}: {rec[value_key]:.4f}")
            else:
                lines.append(f"- {code} {name}")
        return lines or ["- No available records"]

    circ_high = summary["circuity_high_morphotypes"]
    circ_low = summary["circuity_low_morphotypes"]
    circ_key = next((k for rec in circ_high for k in rec if k.startswith("od_circuity")), None)
    lines = [
        "# Step11 Result Interpretation Summary",
        "",
        "## OD Detour",
        "",
        "Morphotypes with higher detour:",
        *bullet_records(circ_high, circ_key),
        "",
        "Morphotypes with lower detour:",
        *bullet_records(circ_low, circ_key),
        "",
        "## Facility Accessibility",
        "",
        "Morphotypes with better average accessibility:",
        *bullet_records(summary["accessibility_high_morphotypes"], "mean_population_weighted_access_share"),
        "",
        "Morphotypes with weaker average accessibility:",
        *bullet_records(summary["accessibility_low_morphotypes"], "mean_population_weighted_access_share"),
        "",
        "## Accessibility Inequality",
        "",
        "Model C uses the city_morphotype x facility category level and filters rows where unit_count < 20 or a key dependent variable is NaN; morphotype and category interaction terms are shown in the model table. City-level dominant morphotype and entropy remain in `city_model_table.parquet` as descriptive context.",
        "",
        "## Robustness",
        "",
        f"- Share of key coefficients with the same sign in weighted and unweighted models: {summary['robustness']['weighted_vs_unweighted_same_sign_share']}",
        f"- High-confidence threshold: {summary['robustness']['high_confidence_threshold']}",
        f"- Snapping long-tail policy: {summary['robustness']['snap_tail_policy']['policy']}",
        "",
        "## Interpretation Cautions",
        "",
        *[f"- {item}" for item in summary["caution"]],
        "",
    ]
    path_md.write_text("\n".join(lines), encoding="utf-8")


def write_robustness_summary(
    path: Path,
    high_conf: pd.DataFrame,
    unweighted_vs_weighted: pd.DataFrame,
    remove_small: pd.DataFrame,
    snap_info: dict[str, Any],
) -> None:
    same_sign = (
        float(unweighted_vs_weighted["same_sign_flag"].dropna().mean())
        if "same_sign_flag" in unweighted_vs_weighted.columns and unweighted_vs_weighted["same_sign_flag"].notna().any()
        else np.nan
    )
    lines = [
        "# Step11 Robustness Analysis Summary",
        "",
        "## Executed Scenarios",
        "",
        f"- High-confidence subsample: type_confidence >= {DEFAULT_CONFIDENCE_THRESHOLD} or city-morphotype mean confidence reaches that threshold.",
        "- Weighted vs unweighted: main models use population/OD weights; diagnostic models do not use weights.",
        f"- Very small group filter: morphotype-city groups with unit_count < {DEFAULT_SMALL_GROUP_MIN_UNITS} or very small OD pair counts are removed in robustness checks.",
        "- Snapping long tail: flagged only, not hard-deleted.",
        "",
        "## Results Overview",
        "",
        f"- High-confidence model coefficient rows: {len(high_conf)}",
        f"- Unweighted comparison rows: {len(unweighted_vs_weighted)}",
        f"- Small-group filtered model coefficient rows: {len(remove_small)}",
        f"- Share of key coefficients with the same sign in weighted/unweighted models: {same_sign}",
        f"- Snapping long-tail threshold: {snap_info.get('snap_distance_threshold_m')} m",
        f"- Snapping long-tail row share: {snap_info.get('long_snap_distance_row_share')}",
        "",
        "Detailed coefficients are in `robustness_high_confidence.csv`, `robustness_unweighted_vs_weighted.csv`, `robustness_remove_small_groups.csv`, and `robustness_results.xlsx`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_qc(
    paths: StepPaths,
    input_report: dict[str, Any],
    model_results: pd.DataFrame,
    output_paths: list[Path],
    unit_model_table: pd.DataFrame | None = None,
    model_c_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    records = []

    def add(check: str, status: str, value: Any, expected: Any, detail: str = "") -> None:
        records.append({"check": check, "status": status, "value": value, "expected": expected, "detail": detail})

    required_inputs = [
        paths.step09_units,
        paths.step09_city_profiles,
        paths.step10_od,
        paths.step10_accessibility,
        paths.step10_inequality,
        paths.step10_validation_summary,
    ]
    add("required_inputs_exist", "pass" if all(p.exists() for p in required_inputs) else "fail", all(p.exists() for p in required_inputs), True)
    add(
        "main_city_count",
        "pass" if input_report["main_scope"]["main_city_count"] == 48 else "warn",
        input_report["main_scope"]["main_city_count"],
        48,
    )
    add(
        "morphotype_count",
        "pass" if input_report["main_scope"]["main_unit_morphotypes"] == 10 else "warn",
        input_report["main_scope"]["main_unit_morphotypes"],
        10,
    )
    failed_models = int(model_results["status"].ne("ok").sum()) if not model_results.empty and "status" in model_results.columns else 0
    add("model_fit_failures", "pass" if failed_models == 0 else "warn", failed_models, 0)
    pending_self_written = {OUTPUT_FILES["run_log"], OUTPUT_FILES["qc"], OUTPUT_FILES["repro"]}
    check_paths = [p for p in output_paths if p.name not in pending_self_written]
    missing_outputs = [p.name for p in check_paths if not p.exists()]
    add("expected_outputs_exist", "pass" if not missing_outputs else "fail", len(check_paths) - len(missing_outputs), len(check_paths), ",".join(missing_outputs))
    od_small = input_report["step10_nan_and_small_groups"]["od_small_groups"]["rows"]
    access_small = input_report["step10_nan_and_small_groups"]["accessibility_small_groups"]["rows"]
    ineq_small = input_report["step10_nan_and_small_groups"]["inequality_small_groups"]["rows"]
    add("step10_small_groups_present", "warn" if (od_small + access_small + ineq_small) > 0 else "pass", od_small + access_small + ineq_small, 0)
    if unit_model_table is not None and "step09_join_matched" in unit_model_table.columns:
        match_share = float(unit_model_table["step09_join_matched"].mean())
        add("step10_facility_join_step09", "pass" if math.isclose(match_share, 1.0, abs_tol=1e-12) else "fail", match_share, 1.0)
    if model_c_table is not None and "model_c_main_included" in model_c_table.columns:
        excluded = int((~model_c_table["model_c_main_included"].fillna(False)).sum())
        add("model_c_small_nan_marked", "pass", excluded, ">=0", "unit_count<20 or key outcome NaN excluded from Model C main fit")
    return pd.DataFrame(records)


def output_file_paths(paths: StepPaths) -> list[Path]:
    return [paths.output_dir / name for name in OUTPUT_FILES.values()]


def write_readme(paths: StepPaths, args: argparse.Namespace) -> None:
    lines = [
        f"# {STEP_NAME}",
        "",
        "This directory is generated by `code_upload/11_statistical_models_robustness_checks/step11_model_robustness.py`.",
        "",
        "## Main Inputs",
        "",
        "- Step09 `local_morphotypes.parquet` and `city_profiles.parquet`",
        "- Step10 `OD_detour_metrics.parquet`, `facility_accessibility_metrics.parquet`, and `accessibility_inequality_metrics.parquet`",
        "- Step07 city quality, Step03 population/built-environment, and Step01 city_master controls are joined when available.",
        "",
        "## Main Filters",
        "",
        "- Step09 main units: `analysis_scope == 'main_fit'` + `network_type == 'drive'` + `scale == 'hex_1km'`; expected 74,861 rows and 48 cities.",
        "- Step10 facility accessibility: `main_analysis_included == True` + `analysis_scope == 'main_fit'` + `scale == 'hex_1km'`.",
        "- Step10 OD main model uses only `group_level == 'origin_morphotype'`.",
        "- Step10 inequality main model uses `group_level == 'city_morphotype'`; the main model filters rows with `unit_count < 20` or a key outcome equal to NaN.",
        "- Step10 facility table and Step09 main units are verified for 100% matching on `city_id + scale + unit_id`.",
        "",
        "## Model Definitions",
        "",
        "- Model A: `log(od_circuity_weighted_mean)` / `od_circuity_weighted_mean` etc. ~ `C(origin_morphotype)+C(city_id)`, with `route_found_weight_sum` weights and fallback to `od_pair_weight_sum` when missing.",
        "- Model B: unit-facility long table, `access_15min` and `accessibility_score` ~ `C(morphotype)*C(category)+C(city_id)+type_confidence`, with population weights.",
        "- Model C: city_morphotype-category table, three inequality/accessibility outcomes ~ `C(morphotype)*C(category)+minimum city controls`.",
        "- Minimum controls prioritize `log(ucdb_pop_2025)`, `log(ucdb_area_km2)`, `ghsl_built_share_2020_100m`, and `quality_score`; missing fields are recorded as fallbacks in the log. `quality_tier` and `quality_weight` are not used as valid explanatory variables.",
        "",
        "## Run Command",
        "",
        f"```bash\npython code_upload/11_statistical_models_robustness_checks/step11_model_robustness.py --root {args.root}\n```",
        "",
        "## Outputs",
        "",
        "Model tables, descriptive statistics, robustness tables, paper figure data, Excel summaries, QC, run logs, and the reproducibility status manifest are written to this directory.",
        "",
    ]
    (paths.output_dir / OUTPUT_FILES["readme"]).write_text("\n".join(lines), encoding="utf-8")


def write_repro(
    paths: StepPaths,
    args: argparse.Namespace,
    script_path: Path,
    input_report: dict[str, Any],
    model_infos: list[dict[str, Any]],
    output_paths: list[Path],
) -> None:
    repro = {
        "generated_at": now_iso(),
        "step": STEP_NAME,
        "command": " ".join(sys.argv),
        "args": vars(args),
        "script_path": str(script_path),
        "script_sha256": file_sha256(script_path),
        "package_versions": package_versions(),
        "git_state": git_state(paths.root),
        "input_report_status": input_report.get("status"),
        "main_scope": input_report.get("main_scope"),
        "model_formulas": [
            {
                "model_id": info.get("model_id"),
                "scenario": info.get("scenario"),
                "outcome": info.get("outcome"),
                "formula": info.get("formula"),
                "weight_col": info.get("weight_col"),
                "cluster_col": info.get("cluster_col"),
                "nobs": info.get("nobs"),
                "status": info.get("status"),
                "converged": info.get("converged"),
            }
            for info in model_infos
        ],
        "nan_policy": "Model-specific missing outcomes, non-finite predictors generated by formulas, and non-positive weights are dropped by statsmodels/explicit filters. Main data rows are not altered upstream.",
        "small_group_policy": f"Main analysis retains small groups; robustness removes unit_count < {args.small_group_min_units}.",
        "output_metadata": {p.name: file_metadata(p) for p in output_paths if p.exists()},
    }
    write_json(paths.output_dir / OUTPUT_FILES["repro"], repro)


def main() -> None:
    start = time.time()
    args = parse_args()
    paths = get_paths(args.root, args.output_dir)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()

    inputs = load_inputs(paths)
    main_tables = filter_main_tables(inputs)
    input_report = collect_input_report(paths, inputs, main_tables, args.small_group_min_units)
    write_input_report(paths, input_report)

    main_units = main_tables["main_units"]
    acc_main = main_tables["acc_main"]
    od_main = main_tables["od_main"]
    ineq_main = main_tables["ineq_main"]
    assert isinstance(main_units, pd.DataFrame)
    assert isinstance(acc_main, pd.DataFrame)
    assert isinstance(od_main, pd.DataFrame)
    assert isinstance(ineq_main, pd.DataFrame)

    city_controls = build_city_controls(main_units, inputs)
    unit_model_table, snap_info = prepare_unit_model_table(
        main_units,
        acc_main,
        city_controls,
        confidence_threshold=args.confidence_threshold,
        snap_tail_quantile=args.snap_tail_quantile,
    )
    access_city_type_category = aggregate_access_city_type_category(unit_model_table)
    morphotype_city_table = build_morphotype_city_table(
        main_units,
        od_main,
        access_city_type_category,
        city_controls,
        args.small_group_min_units,
    )
    city_model_table = build_city_model_table(ineq_main, main_units, city_controls)
    model_a_table = prepare_model_a_table(morphotype_city_table)
    model_b_table = prepare_model_b_table(unit_model_table)
    model_c_table = build_inequality_model_table(ineq_main, city_controls, morphotype_city_table)

    write_parquet(unit_model_table, paths.output_dir / OUTPUT_FILES["unit_model_table"])
    write_parquet(morphotype_city_table, paths.output_dir / OUTPUT_FILES["morphotype_city_table"])
    write_parquet(city_model_table, paths.output_dir / OUTPUT_FILES["city_model_table"])
    write_csv(model_a_table, paths.output_dir / OUTPUT_FILES["model_table_a_csv"])
    write_parquet(model_a_table, paths.output_dir / OUTPUT_FILES["model_table_a_parquet"])
    write_csv(model_b_table, paths.output_dir / OUTPUT_FILES["model_table_b_csv"])
    write_parquet(model_b_table, paths.output_dir / OUTPUT_FILES["model_table_b_parquet"])
    write_csv(model_c_table, paths.output_dir / OUTPUT_FILES["model_table_c_csv"])
    write_parquet(model_c_table, paths.output_dir / OUTPUT_FILES["model_table_c_parquet"])
    input_field_check = build_input_field_check(paths, inputs, main_tables, unit_model_table)
    write_csv(input_field_check, paths.output_dir / OUTPUT_FILES["input_field_check"])

    descriptive = build_descriptive_outputs(main_units, od_main, unit_model_table, access_city_type_category, city_model_table)
    for key, df in descriptive.items():
        write_csv(df, paths.output_dir / OUTPUT_FILES[key])
    write_csv(descriptive["circuity_by_morphotype"], paths.output_dir / OUTPUT_FILES["figure_circuity"])
    write_csv(descriptive["facility_access_by_morphotype"], paths.output_dir / OUTPUT_FILES["figure_access"])
    write_csv(city_model_table, paths.output_dir / OUTPUT_FILES["figure_inequality"])

    model_a, info_a = run_model_a(model_a_table, scenario="main", weighted=True)
    model_b, info_b = run_model_b(model_b_table, scenario="main", weighted=True)
    model_c, info_c = run_model_c(model_c_table, scenario="main_unit_count_ge_20", weighted=True, min_unit_count=20)
    model_results = pd.concat([model_a, model_b, model_c], ignore_index=True)
    model_infos = [*info_a, *info_b, *info_c]

    write_csv(model_a, paths.output_dir / OUTPUT_FILES["model_a_csv"])
    write_csv(model_b, paths.output_dir / OUTPUT_FILES["model_b_csv"])
    write_csv(model_c, paths.output_dir / OUTPUT_FILES["model_c_csv"])
    write_model_markdown(paths.output_dir / OUTPUT_FILES["model_a_md"], "Model A: OD Detour", model_a, info_a)
    write_model_markdown(paths.output_dir / OUTPUT_FILES["model_b_md"], "Model B: Facility Accessibility", model_b, info_b)
    write_model_markdown(paths.output_dir / OUTPUT_FILES["model_c_md"], "Model C: Accessibility Inequality", model_c, info_c)

    high_conf, unweighted_vs_weighted, remove_small, robust_combined, robust_infos = run_robustness(
        main_units,
        model_a_table,
        model_b_table,
        model_c_table,
        args.confidence_threshold,
        args.small_group_min_units,
    )
    model_infos.extend(robust_infos)
    write_csv(high_conf, paths.output_dir / OUTPUT_FILES["robust_high_conf"])
    write_csv(unweighted_vs_weighted, paths.output_dir / OUTPUT_FILES["robust_unweighted"])
    write_csv(remove_small, paths.output_dir / OUTPUT_FILES["robust_small"])
    write_csv(robust_combined, paths.output_dir / OUTPUT_FILES["table_robust"])
    robustness_summary_table = build_robustness_summary_table(high_conf, unweighted_vs_weighted, remove_small, robust_infos)
    write_csv(robustness_summary_table, paths.output_dir / OUTPUT_FILES["robust_summary_table"])
    write_robustness_summary(
        paths.output_dir / OUTPUT_FILES["robust_md"],
        high_conf,
        unweighted_vs_weighted,
        remove_small,
        snap_info,
    )

    write_csv(model_results, paths.output_dir / OUTPUT_FILES["table_model"])
    write_csv(model_results, paths.output_dir / OUTPUT_FILES["coef_csv"])
    marginal_effects = build_marginal_effects_table(model_a_table, model_b_table, model_c_table)
    write_csv(marginal_effects, paths.output_dir / OUTPUT_FILES["marginal_effects"])
    write_excel(
        paths.output_dir / OUTPUT_FILES["coef_xlsx"],
        {
            "all_coefficients": model_results,
            "model_A": model_a,
            "model_B": model_b,
            "model_C": model_c,
        },
    )
    write_excel(
        paths.output_dir / OUTPUT_FILES["model_xlsx"],
        {
            "model_A_od": model_a,
            "model_B_access": model_b,
            "model_C_inequality": model_c,
            "morphotype_desc": descriptive["morphotype_descriptive_stats"],
            "circuity_by_type": descriptive["circuity_by_morphotype"],
            "access_by_type": descriptive["facility_access_by_morphotype"],
            "marginal_effects": marginal_effects,
            "model_table_A": model_a_table,
            "model_table_C": model_c_table,
        },
    )
    write_excel(
        paths.output_dir / OUTPUT_FILES["robust_xlsx"],
        {
            "high_confidence": high_conf,
            "unweighted_vs_weighted": unweighted_vs_weighted,
            "remove_small_groups": remove_small,
            "summary": robustness_summary_table,
            "combined": robust_combined,
        },
    )

    summary = build_result_summary(descriptive, city_model_table, model_results, unweighted_vs_weighted, snap_info)
    write_result_summary(
        paths.output_dir / OUTPUT_FILES["summary_md"],
        paths.output_dir / OUTPUT_FILES["summary_json"],
        summary,
    )
    (paths.output_dir / OUTPUT_FILES["model_explanation_md"]).write_text(
        (paths.output_dir / OUTPUT_FILES["summary_md"]).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    write_readme(paths, args)

    output_paths = output_file_paths(paths)
    qc = build_qc(paths, input_report, model_results, output_paths, unit_model_table=unit_model_table, model_c_table=model_c_table)
    write_csv(qc, paths.output_dir / OUTPUT_FILES["qc"])

    run_log = {
        "generated_at": now_iso(),
        "step": STEP_NAME,
        "command": " ".join(sys.argv),
        "args": vars(args),
        "duration_seconds": time.time() - start,
        "script_path": str(script_path),
        "script_sha256": file_sha256(script_path),
        "package_versions": package_versions(),
        "git_state": git_state(paths.root),
        "input_report": input_report,
        "snap_info": snap_info,
        "model_infos": model_infos,
        "qc_status_counts": qc["status"].value_counts(dropna=False).to_dict(),
        "output_metadata": {p.name: file_metadata(p) for p in output_paths if p.exists()},
    }
    write_json(paths.output_dir / OUTPUT_FILES["run_log"], run_log)
    write_repro(paths, args, script_path, input_report, model_infos, output_paths)

    print(json.dumps({"status": "ok", "output_dir": str(paths.output_dir), "duration_seconds": time.time() - start}, ensure_ascii=False))


if __name__ == "__main__":
    main()
