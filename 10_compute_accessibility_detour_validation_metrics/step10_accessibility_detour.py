#!/usr/bin/env python3
"""Step10: accessibility and detour validation for Step09 morphotypes.

This step validates the real-world explanatory value of Step09 morphotypes
without re-clustering.  The main analysis uses Step09 ``analysis_scope ==
"main_fit"``, drive, hex_1km units; walk-network 1200 m access is joined back
to the drive morphotype units.
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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from shapely import wkb

try:
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra as scipy_dijkstra
except Exception:  # pragma: no cover - fallback is intentionally slow.
    cKDTree = None
    csr_matrix = None
    scipy_dijkstra = None


STEP_NAME = "10_compute_accessibility_detour_validation_metrics"
STEP09_NAME = "09_cluster_morphotypes"
STEP05_NAME = "05_build_multiscale_spatial_units"
STEP04_NAME = "04_download_facilities_quality_history_data"
STEP02_NAME = "02_download_road_network_data"

MAIN_SCALE = "hex_1km"
MAIN_NETWORK = "drive"
WALK_NETWORK = "walk"
ACCESS_THRESHOLD_M = 1200.0
DEFAULT_RANDOM_SEED = 20260610
FACILITY_CATEGORIES = ["grocery", "school", "healthcare", "park", "transit"]

OUTPUT_FILES = {
    "input_report_json": "10_input_acceptance_report.json",
    "input_report_md": "10_input_acceptance_report.md",
    "facility_table": "facility_classification_table.parquet",
    "facility_qc": "facility_classification_counts_QC.csv",
    "od_samples": "OD_detour_samples.parquet",
    "od_metrics": "OD_detour_metrics.parquet",
    "accessibility": "facility_accessibility_metrics.parquet",
    "inequality": "accessibility_inequality_metrics.parquet",
    "summary": "morphotype_validation_summary.csv",
    "effects": "morphotype_validation_effects.csv",
    "low_confidence": "low_confidence_sensitivity.csv",
    "qc": "10_quality_checks.csv",
    "run_log": "10_run_log.json",
    "readme": "README.md",
    "repro": "reproducibility_status_manifest.json",
}


@dataclass
class StepPaths:
    root: Path
    output_dir: Path
    step09_dir: Path
    step05_dir: Path
    step04_dir: Path
    step02_dir: Path

    @property
    def step09_units(self) -> Path:
        return self.step09_dir / "local_morphotypes.parquet"

    @property
    def step09_profiles(self) -> Path:
        return self.step09_dir / "city_profiles.parquet"

    @property
    def step05_index(self) -> Path:
        return self.step05_dir / "spatial_units_index.parquet"

    @property
    def step05_grid_1km(self) -> Path:
        return self.step05_dir / "grid_1km.gpkg"

    @property
    def osm_facility_dir(self) -> Path:
        return self.step04_dir / "OSM_facilities" / "extract_parquet"

    @property
    def overture_city_dir(self) -> Path:
        return self.step04_dir / "OverturePlaces" / "city_parquet"

    @property
    def graphml_dir(self) -> Path:
        return self.step02_dir / "city_road_networks_graphml"

    @property
    def gpkg_dir(self) -> Path:
        return self.step02_dir / "city_road_networks_gpkg"


@dataclass
class NetworkContext:
    city_id: str
    network_type: str
    graph: nx.Graph | nx.DiGraph
    node_ids: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    tree: Any
    x_scale: float
    y_scale: float
    source_path: Path

    def nearest(self, lon_values: Iterable[float], lat_values: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
        lon_arr = np.asarray(list(lon_values), dtype=float)
        lat_arr = np.asarray(list(lat_values), dtype=float)
        valid = np.isfinite(lon_arr) & np.isfinite(lat_arr)
        node_out = np.full(lon_arr.shape, None, dtype=object)
        dist_out = np.full(lon_arr.shape, np.nan, dtype=float)
        if not valid.any() or len(self.node_ids) == 0:
            return node_out, dist_out
        query_xy = np.column_stack([lon_arr[valid] * self.x_scale, lat_arr[valid] * self.y_scale])
        if self.tree is not None:
            dist, idx = self.tree.query(query_xy, k=1)
        else:
            node_xy = np.column_stack([self.lon * self.x_scale, self.lat * self.y_scale])
            diff = query_xy[:, None, :] - node_xy[None, :, :]
            dist_matrix = np.sqrt(np.sum(diff * diff, axis=2))
            idx = np.argmin(dist_matrix, axis=1)
            dist = dist_matrix[np.arange(dist_matrix.shape[0]), idx]
        node_out[valid] = self.node_ids[idx]
        dist_out[valid] = dist
        return node_out, dist_out


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
    parser.add_argument(
        "--mode",
        choices=["pilot", "main", "inputs-only"],
        default="pilot",
        help="pilot runs three representative core cities unless --pilot-cities is set; main runs all core cities.",
    )
    parser.add_argument(
        "--pilot-cities",
        nargs="*",
        default=None,
        help="Optional explicit city_id list for pilot mode.",
    )
    parser.add_argument(
        "--limit-cities",
        type=int,
        default=None,
        help="Optional maximum number of cities after mode/pilot selection, useful for resumable smoke checks.",
    )
    parser.add_argument(
        "--max-od-pairs-per-city",
        type=int,
        default=None,
        help="Optional hard cap for OD pairs per city after origin x destination sampling.",
    )
    parser.add_argument(
        "--max-origin-units-per-city",
        type=int,
        default=None,
        help="Origin units sampled per city. Defaults: 300 in pilot, all eligible origins in main.",
    )
    parser.add_argument(
        "--od-destinations-per-origin",
        type=int,
        default=None,
        help="Destinations sampled per origin. Defaults: 4 in pilot, 8 in main.",
    )
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--access-threshold-m", type=float, default=ACCESS_THRESHOLD_M)
    parser.add_argument("--min-straight-m", type=float, default=300.0)
    parser.add_argument(
        "--facility-sources",
        choices=["osm", "osm_overture"],
        default="osm_overture",
        help="OSM is always primary; Overture can be added as supplemental coverage.",
    )
    parser.add_argument(
        "--graph-source",
        choices=["auto", "gpkg", "graphml"],
        default="auto",
        help="auto prefers GPKG nodes/edges and falls back to GraphML.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing Step10 outputs.")
    return parser.parse_args()


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
    path.write_text(json.dumps(json_ready(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def write_dataframe_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def file_metadata(path: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return meta
    stat = path.stat()
    meta.update(
        {
            "size_bytes": int(stat.st_size),
            "mtime": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
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
            header = pd.read_csv(path, nrows=0)
            meta.update({"format": "csv", "column_count": int(header.shape[1]), "columns": list(header.columns)})
        elif path.suffix == ".gpkg":
            try:
                import fiona

                meta.update({"format": "gpkg", "layers": list(fiona.listlayers(path))})
            except Exception as exc:
                meta.update({"format": "gpkg", "layers_error": repr(exc)})
        elif path.suffix == ".json":
            with path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            meta.update({"format": "json", "top_level_keys": list(obj) if isinstance(obj, dict) else []})
    except Exception as exc:
        meta["metadata_error"] = repr(exc)
    return meta


def get_paths(root: Path, output_dir: Path | None) -> StepPaths:
    root = root.resolve()
    return StepPaths(
        root=root,
        output_dir=(output_dir or root / "data" / STEP_NAME).resolve(),
        step09_dir=root / "data" / STEP09_NAME,
        step05_dir=root / "data" / STEP05_NAME,
        step04_dir=root / "data" / STEP04_NAME,
        step02_dir=root / "data" / STEP02_NAME,
    )


def package_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for package in ["pandas", "geopandas", "networkx", "numpy", "pyarrow", "shapely", "scipy"]:
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
    }


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_city_seed(city_id: str, random_seed: int) -> int:
    """Derive a cross-process stable uint32 seed for city-level OD sampling."""
    payload = f"{int(random_seed)}::{city_id}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % (2**32)


def load_step09_drive_hex_units(paths: StepPaths) -> pd.DataFrame:
    required = [
        "city_id",
        "city_name_en",
        "country",
        "iso3",
        "region",
        "sample_group",
        "quality_tier",
        "quality_score",
        "network_type",
        "scale",
        "analysis_scope",
        "unit_id",
        "unit_area_km2",
        "valid_area_ratio",
        "edge_unit",
        "center_lon",
        "center_lat",
        "population_weight",
        "population_sum",
        "step09_main_candidate",
        "step09_sensitivity_candidate",
        "retained_excluded_candidate",
        "morphotype",
        "morphotype_name",
        "cluster_probability",
        "type_confidence",
        "low_confidence_flag",
        "low_quality_unit_flag",
        "shared_ucdb_boundary_flag",
        "shared_ucdb_id",
        "shared_ucdb_name",
        "shared_ucdb_boundary_group",
        "shared_ucdb_boundary_note",
    ]
    schema_names = set(pq.ParquetFile(paths.step09_units).schema_arrow.names)
    cols = [col for col in required if col in schema_names]
    df = pd.read_parquet(paths.step09_units, columns=cols)
    return df[(df["network_type"] == MAIN_NETWORK) & (df["scale"] == MAIN_SCALE)].copy()


def filter_main_analysis_units(drive_hex_units: pd.DataFrame) -> pd.DataFrame:
    """Step10 main analysis contract from the design handoff."""
    main = drive_hex_units[drive_hex_units["analysis_scope"] == "main_fit"].copy()
    main["main_analysis_included"] = True
    return main


def choose_pilot_cities(drive_hex_units: pd.DataFrame, pilot_cities: list[str] | None) -> list[str]:
    available = set(drive_hex_units["city_id"].dropna().astype(str))
    if pilot_cities:
        missing = [city_id for city_id in pilot_cities if city_id not in available]
        if missing:
            raise ValueError(f"Pilot city_id not present in Step09 drive hex_1km units: {missing}")
        return list(dict.fromkeys(pilot_cities))
    # Mandatory design-pilot set: one core city plus excluded/shared-boundary
    # review cities.  Non-main cities are marked as extension rows and never
    # counted as main conclusions.
    preferred = ["main_030", "main_070", "chn_003", "chn_004"]
    return [city_id for city_id in preferred if city_id in available]


def selected_city_ids(args: argparse.Namespace, drive_hex_units: pd.DataFrame, main_units: pd.DataFrame) -> list[str]:
    if args.mode == "inputs-only":
        cities = choose_pilot_cities(drive_hex_units, args.pilot_cities)
    elif args.mode == "pilot":
        cities = choose_pilot_cities(drive_hex_units, args.pilot_cities)
    else:
        counts = main_units.groupby("city_id").size().sort_values()
        cities = list(counts.index.astype(str))
    if args.limit_cities is not None:
        cities = cities[: args.limit_cities]
    return cities


def collect_input_report(paths: StepPaths, all_main_units: pd.DataFrame, run_city_ids: list[str]) -> dict[str, Any]:
    target_set = set(run_city_ids)
    graph_records: list[dict[str, Any]] = []
    for city_id in sorted(target_set):
        for network_type in [MAIN_NETWORK, WALK_NETWORK]:
            graph_records.append(
                {
                    "city_id": city_id,
                    "network_type": network_type,
                    "graphml_exists": (paths.graphml_dir / network_type / f"{city_id}_{network_type}.graphml").exists(),
                    "gpkg_exists": (paths.gpkg_dir / network_type / f"{city_id}_{network_type}.gpkg").exists(),
                }
            )
    all_step09 = pd.read_parquet(
        paths.step09_units,
        columns=[
            "city_id",
            "city_name_en",
            "quality_tier",
            "network_type",
            "scale",
            "step09_main_candidate",
            "shared_ucdb_boundary_flag",
            "shared_ucdb_id",
            "shared_ucdb_name",
            "shared_ucdb_boundary_group",
            "shared_ucdb_boundary_note",
        ],
    )
    dhaka_or_excluded = all_step09[
        all_step09["quality_tier"].astype(str).str.contains("excluded", case=False, na=False)
        | all_step09["city_name_en"].astype(str).str.contains("Dhaka", case=False, na=False)
    ]
    shared_review = all_step09[all_step09["city_id"].isin(["chn_003", "chn_004"])].drop_duplicates(
        [
            "city_id",
            "city_name_en",
            "quality_tier",
            "shared_ucdb_boundary_flag",
            "shared_ucdb_id",
            "shared_ucdb_name",
            "shared_ucdb_boundary_group",
            "shared_ucdb_boundary_note",
        ]
    )
    osm_files = sorted(paths.osm_facility_dir.glob("*_facilities.parquet"))
    overture_files = sorted(paths.overture_city_dir.glob("*.parquet"))
    report = {
        "generated_at": now_iso(),
        "step": STEP_NAME,
        "analysis_contract": {
            "main_filter": "analysis_scope == 'main_fit' AND network_type == 'drive' AND scale == 'hex_1km'",
            "main_scale": MAIN_SCALE,
            "morphotype_source": "Step09 drive main model result",
            "od_network": "drive length",
            "access_network": "walk length",
            "access_threshold_m": ACCESS_THRESHOLD_M,
            "excluded_review_policy": "Dhaka/excluded_review excluded from main conclusions but recorded here.",
            "shared_ucdb_policy": "chn_003 Shenzhen and chn_004 Guangzhou shared UCDB boundary flags retained; not interpreted as independent boundaries.",
        },
        "paths": {
            "step09_units": file_metadata(paths.step09_units),
            "step09_profiles": file_metadata(paths.step09_profiles),
            "step05_index": file_metadata(paths.step05_index),
            "step05_grid_1km": file_metadata(paths.step05_grid_1km),
            "osm_facility_dir": {"path": str(paths.osm_facility_dir), "exists": paths.osm_facility_dir.exists(), "file_count": len(osm_files)},
            "overture_city_dir": {
                "path": str(paths.overture_city_dir),
                "exists": paths.overture_city_dir.exists(),
                "file_count": len(overture_files),
            },
        },
        "main_scope": {
            "all_drive_hex_1km_main_fit_rows": int(len(all_main_units)),
            "all_drive_hex_1km_main_fit_cities": int(all_main_units["city_id"].nunique()),
            "all_drive_hex_1km_main_fit_morphotypes": int(all_main_units["morphotype"].nunique()),
            "design_expected_rows": 74861,
            "design_expected_cities": 48,
            "design_expected_morphotypes": 10,
            "run_city_ids": run_city_ids,
            "run_city_count": len(run_city_ids),
            "run_main_fit_rows": int(all_main_units[all_main_units["city_id"].isin(target_set)].shape[0]),
            "run_drive_hex_rows_including_extensions": int(
                all_step09[
                    (all_step09["city_id"].isin(target_set))
                    & (all_step09["network_type"] == MAIN_NETWORK)
                    & (all_step09["scale"] == MAIN_SCALE)
                ].shape[0]
            ),
        },
        "graph_availability_for_run_scope": graph_records,
        "excluded_or_dhaka_record": {
            "row_count": int(len(dhaka_or_excluded)),
            "cities": dhaka_or_excluded[["city_id", "city_name_en", "quality_tier"]]
            .drop_duplicates()
            .sort_values(["city_id", "quality_tier"])
            .to_dict("records"),
        },
        "shenzhen_guangzhou_shared_ucdb_record": shared_review.sort_values("city_id").to_dict("records"),
    }
    missing_required = []
    for label, path in [
        ("step09_units", paths.step09_units),
        ("step09_profiles", paths.step09_profiles),
        ("step05_index", paths.step05_index),
        ("step05_grid_1km", paths.step05_grid_1km),
    ]:
        if not path.exists():
            missing_required.append(label)
    missing_graphs = [rec for rec in graph_records if not rec["graphml_exists"] and not rec["gpkg_exists"]]
    report["status"] = "pass" if not missing_required and not missing_graphs and len(osm_files) > 0 else "warn"
    report["missing_required_inputs"] = missing_required
    report["missing_graph_inputs_for_run_scope"] = missing_graphs
    return report


def write_input_report(paths: StepPaths, report: dict[str, Any]) -> None:
    json_path = paths.output_dir / OUTPUT_FILES["input_report_json"]
    md_path = paths.output_dir / OUTPUT_FILES["input_report_md"]
    write_json(json_path, report)
    missing_graphs = report.get("missing_graph_inputs_for_run_scope", [])
    excluded_cities = report.get("excluded_or_dhaka_record", {}).get("cities", [])
    shared_record = report.get("shenzhen_guangzhou_shared_ucdb_record", [])
    md_lines = [
        f"# {STEP_NAME} input acceptance report",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Acceptance status: {report['status']}",
        f"- Main analysis scope: analysis_scope == main_fit + drive + {MAIN_SCALE}",
        f"- City count in this run: {report['main_scope']['run_city_count']}",
        f"- main_fit unit count in this run: {report['main_scope']['run_main_fit_rows']}",
        f"- Drive hex unit count including extension records in this run: {report['main_scope']['run_drive_hex_rows_including_extensions']}",
        f"- OSM facility parquet file count: {report['paths']['osm_facility_dir']['file_count']}",
        f"- Overture city parquet file count: {report['paths']['overture_city_dir']['file_count']}",
        "",
        "## Special Sample Records",
        "",
        f"- Dhaka/excluded record cities: {excluded_cities}",
        f"- Shenzhen/Guangzhou shared UCDB boundary records: {shared_record}",
        "",
        "## Missing Items",
        "",
        f"- Missing required upstream inputs: {report['missing_required_inputs']}",
        f"- Missing city road networks for this run: {missing_graphs}",
        "",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")


def select_run_units(drive_hex_units: pd.DataFrame, main_units: pd.DataFrame, city_ids: list[str], mode: str) -> pd.DataFrame:
    if mode == "main":
        units = main_units[main_units["city_id"].isin(city_ids)].copy()
    else:
        units = drive_hex_units[drive_hex_units["city_id"].isin(city_ids)].copy()
        units["main_analysis_included"] = units["analysis_scope"].eq("main_fit")
    units["main_analysis_included"] = units["main_analysis_included"].fillna(False).astype(bool)
    units["pilot_extension_flag"] = ~units["main_analysis_included"]
    return units


def load_units_with_geometry(paths: StepPaths, run_units: pd.DataFrame) -> pd.DataFrame:
    units = run_units.copy()
    grid = gpd.read_file(paths.step05_grid_1km, layer="hex_1km")
    if grid.crs is not None and str(grid.crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
        grid = grid.to_crs("EPSG:4326")
    grid = grid[grid["unit_id"].isin(set(units["unit_id"]))]
    point = grid.geometry.representative_point()
    grid_points = pd.DataFrame(
        {
            "unit_id": grid["unit_id"].astype(str).to_numpy(),
            "unit_center_lon": point.x.to_numpy(dtype=float),
            "unit_center_lat": point.y.to_numpy(dtype=float),
            "unit_geometry_area_deg2": np.full(len(grid), np.nan, dtype=float),
        }
    )
    merged = units.merge(grid_points, on="unit_id", how="left", validate="one_to_one")
    missing = int(merged["unit_center_lon"].isna().sum())
    if missing:
        raise RuntimeError(f"{missing} selected Step09 units could not be joined to Step05 hex geometry.")
    merged["population_sum"] = pd.to_numeric(merged["population_sum"], errors="coerce").fillna(0.0)
    merged["population_for_weight"] = merged["population_sum"].clip(lower=0.0)
    merged["population_weight_raw"] = pd.to_numeric(merged.get("population_weight"), errors="coerce").fillna(0.0).clip(lower=0.0)
    merged["raw_center_lon_from_step09"] = pd.to_numeric(merged.get("center_lon"), errors="coerce")
    merged["raw_center_lat_from_step09"] = pd.to_numeric(merged.get("center_lat"), errors="coerce")
    return merged


def category_from_osm(primary_key: Any, primary_value: Any, tags_json: Any = None) -> list[str]:
    key = str(primary_key or "").strip().lower()
    value = str(primary_value or "").strip().lower()
    categories: set[str] = set()
    if key == "shop" and value in {
        "supermarket",
        "convenience",
        "greengrocer",
        "grocery",
        "general",
        "department_store",
        "mall",
        "bakery",
        "butcher",
        "seafood",
        "deli",
        "farm",
    }:
        categories.add("grocery")
    if key == "amenity" and value in {"marketplace"}:
        categories.add("grocery")
    if key == "amenity" and value in {"school", "kindergarten", "college", "university", "childcare", "music_school", "language_school"}:
        categories.add("school")
    if key == "amenity" and value in {"hospital", "clinic", "doctors", "dentist", "pharmacy", "nursing_home"}:
        categories.add("healthcare")
    if key == "healthcare" and value not in {"", "no"}:
        categories.add("healthcare")
    if key in {"leisure", "boundary", "landuse"} and value in {
        "park",
        "garden",
        "playground",
        "recreation_ground",
        "nature_reserve",
        "protected_area",
        "village_green",
        "common",
    }:
        categories.add("park")
    if key in {"public_transport"} and value in {"station", "platform", "stop_position", "stop_area"}:
        categories.add("transit")
    if key == "highway" and value in {"bus_stop", "platform"}:
        categories.add("transit")
    if key == "railway" and value in {"station", "halt", "tram_stop", "subway_entrance"}:
        categories.add("transit")
    if key == "amenity" and value in {"bus_station", "ferry_terminal"}:
        categories.add("transit")

    # Fallback for extracts whose primary tag is not the category-bearing tag.
    if tags_json and len(categories) == 0:
        try:
            tags = json.loads(tags_json) if isinstance(tags_json, str) else {}
        except Exception:
            tags = {}
        for tag_key, tag_value in tags.items():
            categories.update(category_from_osm(tag_key, tag_value, None))
    return sorted(categories)


def flatten_texts(value: Any) -> list[str]:
    texts: list[str] = []
    if value is None:
        return texts
    if isinstance(value, str):
        return [value.lower()]
    if isinstance(value, dict):
        for sub_value in value.values():
            texts.extend(flatten_texts(sub_value))
    elif isinstance(value, (list, tuple, set, np.ndarray)):
        for item in value:
            texts.extend(flatten_texts(item))
    else:
        try:
            if not pd.isna(value):
                texts.append(str(value).lower())
        except Exception:
            texts.append(str(value).lower())
    return texts


def category_from_overture(row: pd.Series) -> list[str]:
    text = " ".join(
        flatten_texts(row.get("categories"))
        + flatten_texts(row.get("basic_category"))
        + flatten_texts(row.get("taxonomy"))
    )
    categories: set[str] = set()
    if any(token in text for token in ["supermarket", "grocery", "convenience_store", "market", "food_store", "bakery"]):
        categories.add("grocery")
    if any(token in text for token in ["school", "kindergarten", "university", "college", "education"]):
        categories.add("school")
    if any(token in text for token in ["health", "hospital", "clinic", "doctor", "dentist", "pharmacy", "medical"]):
        categories.add("healthcare")
    if any(token in text for token in ["park", "playground", "garden", "recreation", "nature_reserve"]):
        categories.add("park")
    if any(token in text for token in ["transit", "bus_station", "train_station", "metro", "subway", "transportation", "airport", "ferry"]):
        categories.add("transit")
    return sorted(categories)


def wkb_to_point_xy(geom_wkb: Any) -> tuple[float, float]:
    if geom_wkb is None:
        return (math.nan, math.nan)
    try:
        geom = wkb.loads(bytes(geom_wkb))
    except Exception:
        return (math.nan, math.nan)
    if geom.is_empty:
        return (math.nan, math.nan)
    if geom.geom_type == "Point":
        point = geom
    else:
        point = geom.representative_point()
    return (float(point.x), float(point.y))


def classify_osm_facilities(paths: StepPaths, city_ids: list[str]) -> pd.DataFrame:
    target = set(city_ids)
    records: list[dict[str, Any]] = []
    columns = [
        "city_id",
        "city_name_en",
        "iso3",
        "geofabrik_extract_id",
        "osm_type",
        "osm_id",
        "primary_key",
        "primary_value",
        "name",
        "name_en",
        "all_tags_json",
        "geometry",
    ]
    for parquet_path in sorted(paths.osm_facility_dir.glob("*_facilities.parquet")):
        schema_names = set(pq.ParquetFile(parquet_path).schema_arrow.names)
        read_cols = [col for col in columns if col in schema_names]
        if "city_id" not in read_cols or "geometry" not in read_cols:
            continue
        df = pd.read_parquet(parquet_path, columns=read_cols)
        df = df[df["city_id"].isin(target)]
        if df.empty:
            continue
        for row in df.itertuples(index=False):
            row_dict = row._asdict()
            categories = category_from_osm(row_dict.get("primary_key"), row_dict.get("primary_value"), row_dict.get("all_tags_json"))
            if not categories:
                continue
            lon, lat = wkb_to_point_xy(row_dict.get("geometry"))
            if not np.isfinite(lon) or not np.isfinite(lat):
                continue
            for category in categories:
                records.append(
                    {
                        "facility_id": f"osm:{row_dict.get('osm_type')}:{row_dict.get('osm_id')}:{category}",
                        "source": "osm",
                        "source_role": "primary",
                        "city_id": row_dict.get("city_id"),
                        "city_name_en": row_dict.get("city_name_en"),
                        "iso3": row_dict.get("iso3"),
                        "category": category,
                        "raw_key": row_dict.get("primary_key"),
                        "raw_value": row_dict.get("primary_value"),
                        "name": row_dict.get("name"),
                        "name_en": row_dict.get("name_en"),
                        "lon": lon,
                        "lat": lat,
                    }
                )
    return pd.DataFrame.from_records(records)


def classify_overture_facilities(paths: StepPaths, city_ids: list[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    columns = [
        "id",
        "geometry",
        "categories",
        "basic_category",
        "taxonomy",
        "confidence",
        "names",
        "operating_status",
        "city_id",
        "city_name_en",
    ]
    for city_id in city_ids:
        matches = sorted(paths.overture_city_dir.glob(f"{city_id}_*.parquet"))
        for parquet_path in matches:
            schema_names = set(pq.ParquetFile(parquet_path).schema_arrow.names)
            read_cols = [col for col in columns if col in schema_names]
            if "city_id" not in read_cols or "geometry" not in read_cols:
                continue
            df = pd.read_parquet(parquet_path, columns=read_cols)
            if "operating_status" in df.columns:
                df = df[~df["operating_status"].astype(str).str.contains("closed", case=False, na=False)]
            for _, row in df.iterrows():
                categories = category_from_overture(row)
                if not categories:
                    continue
                lon, lat = wkb_to_point_xy(row.get("geometry"))
                if not np.isfinite(lon) or not np.isfinite(lat):
                    continue
                name_obj = row.get("names")
                name = None
                if isinstance(name_obj, dict):
                    name = name_obj.get("primary")
                for category in categories:
                    records.append(
                        {
                            "facility_id": f"overture:{row.get('id')}:{category}",
                            "source": "overture",
                            "source_role": "supplement",
                            "city_id": row.get("city_id"),
                            "city_name_en": row.get("city_name_en"),
                            "iso3": None,
                            "category": category,
                            "raw_key": "overture_category",
                            "raw_value": row.get("basic_category"),
                            "name": name,
                            "name_en": name,
                            "lon": lon,
                            "lat": lat,
                            "confidence": row.get("confidence"),
                        }
                    )
    return pd.DataFrame.from_records(records)


def classify_facilities(paths: StepPaths, city_ids: list[str], facility_sources: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    osm = classify_osm_facilities(paths, city_ids)
    frames = [osm]
    if facility_sources == "osm_overture":
        overture = classify_overture_facilities(paths, city_ids)
        frames.append(overture)
    facilities = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if facilities.empty:
        empty = pd.DataFrame(
            columns=[
                "facility_id",
                "source",
                "source_role",
                "city_id",
                "city_name_en",
                "iso3",
                "category",
                "raw_key",
                "raw_value",
                "name",
                "lon",
                "lat",
                "used_for_access",
            ]
        )
        return empty, pd.DataFrame()
    facilities = facilities[facilities["category"].isin(FACILITY_CATEGORIES)].copy()
    raw_counts = (
        facilities.groupby(["city_id", "city_name_en", "category", "source"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    raw_counts["count_type"] = "raw_classified"
    facilities["lon_round_30m"] = (pd.to_numeric(facilities["lon"], errors="coerce") / 0.0003).round().astype("Int64")
    facilities["lat_round_30m"] = (pd.to_numeric(facilities["lat"], errors="coerce") / 0.0003).round().astype("Int64")
    facilities["source_priority"] = facilities["source"].map({"osm": 0, "overture": 1}).fillna(9)
    facilities = facilities.sort_values(["city_id", "category", "lon_round_30m", "lat_round_30m", "source_priority"])
    facilities["dedup_group_raw_count"] = facilities.groupby(
        ["city_id", "category", "lon_round_30m", "lat_round_30m"], dropna=False
    )["facility_id"].transform("size")
    facilities = facilities.drop_duplicates(["city_id", "category", "lon_round_30m", "lat_round_30m"], keep="first")
    facilities["used_for_access"] = True
    facilities = facilities.drop(columns=["source_priority"], errors="ignore")
    dedup_counts = (
        facilities.groupby(["city_id", "city_name_en", "category", "source"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    dedup_counts["count_type"] = "deduplicated_for_access"
    facility_qc = pd.concat([raw_counts, dedup_counts], ignore_index=True, sort=False)
    return facilities.reset_index(drop=True), facility_qc


def resolve_graph_path(paths: StepPaths, city_id: str, network_type: str, graph_source: str) -> Path:
    gpkg = paths.gpkg_dir / network_type / f"{city_id}_{network_type}.gpkg"
    graphml = paths.graphml_dir / network_type / f"{city_id}_{network_type}.graphml"
    if graph_source == "gpkg":
        return gpkg
    if graph_source == "graphml":
        return graphml
    return gpkg if gpkg.exists() else graphml


def graph_from_gpkg(path: Path, network_type: str) -> tuple[nx.Graph | nx.DiGraph, pd.DataFrame]:
    nodes = gpd.read_file(path, layer="nodes", ignore_geometry=True)
    edges = gpd.read_file(path, layer="edges", ignore_geometry=True)
    raw_node_count = int(len(nodes))
    raw_edge_count = int(len(edges))
    nodes = nodes[["osmid", "x", "y"]].copy()
    nodes["osmid"] = nodes["osmid"].astype(object)
    nodes["x"] = pd.to_numeric(nodes["x"], errors="coerce")
    nodes["y"] = pd.to_numeric(nodes["y"], errors="coerce")
    nodes = nodes[np.isfinite(nodes["x"]) & np.isfinite(nodes["y"])].drop_duplicates("osmid")
    valid_edge_count = 0
    nonpositive_edge_count = 0
    graph: nx.Graph | nx.DiGraph = nx.Graph() if network_type == WALK_NETWORK else nx.DiGraph()
    graph.add_nodes_from((row.osmid, {"x": float(row.x), "y": float(row.y)}) for row in nodes.itertuples(index=False))
    for row in edges[["u", "v", "length"]].itertuples(index=False):
        length = float(row.length) if pd.notna(row.length) else math.nan
        if not np.isfinite(length) or length <= 0:
            nonpositive_edge_count += 1
            continue
        valid_edge_count += 1
        u = row.u
        v = row.v
        existing = graph.get_edge_data(u, v)
        if existing is None or length < float(existing.get("length", math.inf)):
            graph.add_edge(u, v, length=length)
    graph.graph["raw_node_count"] = raw_node_count
    graph.graph["valid_xy_node_count"] = int(len(nodes))
    graph.graph["raw_edge_count"] = raw_edge_count
    graph.graph["positive_length_edge_records"] = int(valid_edge_count)
    graph.graph["nonpositive_or_missing_length_edge_records"] = int(nonpositive_edge_count)
    return graph, nodes


def graph_from_graphml(path: Path, network_type: str) -> tuple[nx.Graph | nx.DiGraph, pd.DataFrame]:
    raw = nx.read_graphml(path)
    graph: nx.Graph | nx.DiGraph = nx.Graph() if network_type == WALK_NETWORK else nx.DiGraph()
    node_records: list[dict[str, Any]] = []
    for node_id, attrs in raw.nodes(data=True):
        try:
            lon = float(attrs.get("x"))
            lat = float(attrs.get("y"))
        except Exception:
            continue
        graph.add_node(node_id, x=lon, y=lat)
        node_records.append({"osmid": node_id, "x": lon, "y": lat})
    edge_iter = raw.edges(data=True, keys=True) if raw.is_multigraph() else raw.edges(data=True)
    valid_edge_count = 0
    nonpositive_edge_count = 0
    for edge in edge_iter:
        if raw.is_multigraph():
            u, v, _key, attrs = edge
        else:
            u, v, attrs = edge
        try:
            length = float(attrs.get("length"))
        except Exception:
            nonpositive_edge_count += 1
            continue
        if not np.isfinite(length) or length <= 0:
            nonpositive_edge_count += 1
            continue
        valid_edge_count += 1
        existing = graph.get_edge_data(u, v)
        if existing is None or length < float(existing.get("length", math.inf)):
            graph.add_edge(u, v, length=length)
    graph.graph["raw_node_count"] = int(raw.number_of_nodes())
    graph.graph["valid_xy_node_count"] = int(len(node_records))
    graph.graph["raw_edge_count"] = int(raw.number_of_edges())
    graph.graph["positive_length_edge_records"] = int(valid_edge_count)
    graph.graph["nonpositive_or_missing_length_edge_records"] = int(nonpositive_edge_count)
    return graph, pd.DataFrame.from_records(node_records)


def load_network(paths: StepPaths, city_id: str, network_type: str, graph_source: str) -> NetworkContext:
    path = resolve_graph_path(paths, city_id, network_type, graph_source)
    if not path.exists():
        raise FileNotFoundError(f"Missing {network_type} graph for {city_id}: {path}")
    if path.suffix == ".gpkg":
        graph, nodes = graph_from_gpkg(path, network_type)
    else:
        graph, nodes = graph_from_graphml(path, network_type)
    node_ids = nodes["osmid"].to_numpy(dtype=object)
    lon = pd.to_numeric(nodes["x"], errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(nodes["y"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(lon) & np.isfinite(lat)
    node_ids = node_ids[valid]
    lon = lon[valid]
    lat = lat[valid]
    mean_lat = float(np.nanmean(lat)) if len(lat) else 0.0
    x_scale = 111_320.0 * max(0.05, math.cos(math.radians(mean_lat)))
    y_scale = 110_540.0
    tree = None
    if cKDTree is not None and len(node_ids) > 0:
        tree = cKDTree(np.column_stack([lon * x_scale, lat * y_scale]))
    return NetworkContext(
        city_id=city_id,
        network_type=network_type,
        graph=graph,
        node_ids=node_ids,
        lon=lon,
        lat=lat,
        tree=tree,
        x_scale=x_scale,
        y_scale=y_scale,
        source_path=path,
    )


def haversine_m(lon1: Any, lat1: Any, lon2: Any, lat2: Any) -> np.ndarray:
    lon1_arr = np.asarray(lon1, dtype=float)
    lat1_arr = np.asarray(lat1, dtype=float)
    lon2_arr = np.asarray(lon2, dtype=float)
    lat2_arr = np.asarray(lat2, dtype=float)
    radius = 6_371_008.8
    phi1 = np.radians(lat1_arr)
    phi2 = np.radians(lat2_arr)
    dphi = np.radians(lat2_arr - lat1_arr)
    dlambda = np.radians(lon2_arr - lon1_arr)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return radius * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def snap_units(city_units: pd.DataFrame, drive_ctx: NetworkContext, walk_ctx: NetworkContext) -> pd.DataFrame:
    snapped = city_units.copy()
    drive_nodes, drive_dist = drive_ctx.nearest(snapped["unit_center_lon"], snapped["unit_center_lat"])
    walk_nodes, walk_dist = walk_ctx.nearest(snapped["unit_center_lon"], snapped["unit_center_lat"])
    snapped["drive_node"] = drive_nodes
    snapped["drive_snap_distance_m"] = drive_dist
    snapped["walk_node"] = walk_nodes
    snapped["walk_snap_distance_m"] = walk_dist
    return snapped


def snap_facilities(city_facilities: pd.DataFrame, walk_ctx: NetworkContext) -> pd.DataFrame:
    if city_facilities.empty:
        out = city_facilities.copy()
        out["walk_node"] = pd.Series(dtype=object)
        out["walk_snap_distance_m"] = pd.Series(dtype=float)
        return out
    out = city_facilities.copy()
    nodes, dist = walk_ctx.nearest(out["lon"], out["lat"])
    out["walk_node"] = nodes
    out["walk_snap_distance_m"] = dist
    return out


def build_sparse_router(graph: nx.Graph | nx.DiGraph) -> tuple[dict[Any, int], Any] | tuple[None, None]:
    if csr_matrix is None or scipy_dijkstra is None:
        return None, None
    node_list = list(graph.nodes())
    node_to_pos = {node: idx for idx, node in enumerate(node_list)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for u, v, attrs in graph.edges(data=True):
        length = attrs.get("length")
        try:
            length_f = float(length)
        except Exception:
            continue
        if not np.isfinite(length_f) or length_f <= 0:
            continue
        if u not in node_to_pos or v not in node_to_pos:
            continue
        rows.append(node_to_pos[u])
        cols.append(node_to_pos[v])
        data.append(length_f)
    if not data:
        return None, None
    matrix = csr_matrix((np.asarray(data, dtype=float), (np.asarray(rows), np.asarray(cols))), shape=(len(node_list), len(node_list)))
    return node_to_pos, matrix


def sample_od_pairs(
    city_units: pd.DataFrame,
    drive_ctx: NetworkContext,
    max_origin_units: int | None,
    destinations_per_origin: int,
    max_pairs: int | None,
    random_seed: int,
    min_straight_m: float,
) -> pd.DataFrame:
    city_id = str(city_units["city_id"].iloc[0])
    city_seed = stable_city_seed(city_id, random_seed)
    rng = np.random.default_rng(city_seed)
    candidates = city_units[city_units["drive_node"].notna()].copy().reset_index(drop=True)
    if len(candidates) < 2:
        return pd.DataFrame()
    weights_raw = candidates["population_weight_raw"].to_numpy(dtype=float)
    if not np.isfinite(weights_raw).all() or weights_raw.sum() <= 0:
        weights_raw = candidates["population_for_weight"].to_numpy(dtype=float)
    if not np.isfinite(weights_raw).all() or weights_raw.sum() <= 0:
        weights_raw = np.ones(len(candidates), dtype=float)
    weights_norm = weights_raw / weights_raw.sum()
    candidates["population_weight_raw_for_od"] = weights_raw
    candidates["population_weight_norm_for_od"] = weights_norm
    if max_origin_units is None or max_origin_units >= len(candidates):
        origin_indices = np.arange(len(candidates))
    else:
        origin_indices = rng.choice(len(candidates), size=max_origin_units, replace=False, p=weights_norm)
    records: list[dict[str, Any]] = []
    target_pairs = int(len(origin_indices) * destinations_per_origin)
    if max_pairs is not None:
        target_pairs = min(target_pairs, int(max_pairs))
    max_attempts_per_origin = max(destinations_per_origin * 80, 200)
    node_to_pos, sparse_matrix = build_sparse_router(drive_ctx.graph)
    use_sparse_router = node_to_pos is not None and sparse_matrix is not None
    for oi in origin_indices:
        if len(records) >= target_pairs:
            break
        origin = candidates.iloc[int(oi)]
        origin_destinations: list[tuple[pd.Series, float]] = []
        attempts = 0
        while (
            len(origin_destinations) < destinations_per_origin
            and attempts < max_attempts_per_origin
            and len(records) + len(origin_destinations) < target_pairs
        ):
            attempts += 1
            di = int(rng.choice(len(candidates), replace=True, p=weights_norm))
            if oi == di:
                continue
            dest = candidates.iloc[int(di)]
            straight_m = float(
                haversine_m(origin.unit_center_lon, origin.unit_center_lat, dest.unit_center_lon, dest.unit_center_lat)
            )
            if not np.isfinite(straight_m) or straight_m < min_straight_m:
                continue
            origin_destinations.append((dest, straight_m))

        sparse_distances = None
        if use_sparse_router and origin.drive_node in node_to_pos:
            try:
                sparse_distances = scipy_dijkstra(
                    sparse_matrix,
                    directed=drive_ctx.graph.is_directed(),
                    indices=node_to_pos[origin.drive_node],
                    return_predecessors=False,
                )
            except Exception:
                sparse_distances = None

        for dest, straight_m in origin_destinations:
            route_found = False
            failure_reason = None
            route_m = math.nan
            if origin.drive_node == dest.drive_node:
                failure_reason = "same_snapped_drive_node"
            else:
                if sparse_distances is not None and dest.drive_node in node_to_pos:
                    route_m = float(sparse_distances[node_to_pos[dest.drive_node]])
                    route_found = np.isfinite(route_m)
                    if not route_found:
                        failure_reason = "no_path"
                else:
                    try:
                        route_m = float(
                            nx.shortest_path_length(drive_ctx.graph, origin.drive_node, dest.drive_node, weight="length")
                        )
                        route_found = np.isfinite(route_m)
                    except nx.NetworkXNoPath:
                        failure_reason = "no_path"
                    except nx.NodeNotFound:
                        failure_reason = "node_not_found"
                    except Exception as exc:
                        failure_reason = f"routing_error:{type(exc).__name__}"
            detour_ratio = route_m / straight_m if route_found and straight_m > 0 else math.nan
            if route_found:
                failure_reason = None
            records.append(
                {
                    "city_id": city_id,
                    "city_name_en": origin.city_name_en,
                    "scale": MAIN_SCALE,
                    "origin_unit_id": origin.unit_id,
                    "destination_unit_id": dest.unit_id,
                    "origin_morphotype": origin.morphotype,
                    "origin_morphotype_name": origin.morphotype_name,
                    "destination_morphotype": dest.morphotype,
                    "destination_morphotype_name": dest.morphotype_name,
                    "origin_population_sum": float(origin.population_sum),
                    "destination_population_sum": float(dest.population_sum),
                    "origin_population_weight_raw": float(origin.population_weight_raw_for_od),
                    "destination_population_weight_raw": float(dest.population_weight_raw_for_od),
                    "origin_population_weight_norm": float(origin.population_weight_norm_for_od),
                    "destination_population_weight_norm": float(dest.population_weight_norm_for_od),
                    "od_pair_weight_raw": float(origin.population_weight_norm_for_od * dest.population_weight_norm_for_od),
                    "city_random_seed": int(city_seed),
                    "origin_type_confidence": float(origin.type_confidence) if pd.notna(origin.type_confidence) else math.nan,
                    "destination_type_confidence": float(dest.type_confidence) if pd.notna(dest.type_confidence) else math.nan,
                    "origin_low_confidence_flag": bool(origin.low_confidence_flag),
                    "destination_low_confidence_flag": bool(dest.low_confidence_flag),
                    "origin_analysis_scope": origin.analysis_scope,
                    "destination_analysis_scope": dest.analysis_scope,
                    "origin_main_analysis_included": bool(origin.main_analysis_included),
                    "destination_main_analysis_included": bool(dest.main_analysis_included),
                    "origin_shared_ucdb_boundary_group": origin.shared_ucdb_boundary_group,
                    "destination_shared_ucdb_boundary_group": dest.shared_ucdb_boundary_group,
                    "origin_center_lon": float(origin.unit_center_lon),
                    "origin_center_lat": float(origin.unit_center_lat),
                    "destination_center_lon": float(dest.unit_center_lon),
                    "destination_center_lat": float(dest.unit_center_lat),
                    "origin_drive_node": origin.drive_node,
                    "destination_drive_node": dest.drive_node,
                    "straight_distance_m": straight_m,
                    "drive_route_length_m": route_m,
                    "detour_ratio": detour_ratio,
                    "route_found": bool(route_found),
                    "failure_reason": failure_reason,
                }
            )
    od = pd.DataFrame.from_records(records)
    if not od.empty:
        od = normalize_od_pair_weights(od)
    return od


def normalize_od_pair_weights(od_samples: pd.DataFrame) -> pd.DataFrame:
    """Normalize OD pair population-product weights within each city/scale sample."""
    if od_samples.empty:
        return od_samples.copy()
    od = od_samples.copy()
    raw = pd.to_numeric(od.get("od_pair_weight_raw"), errors="coerce").fillna(0.0).clip(lower=0.0)
    od["od_pair_weight_raw"] = raw
    group_sum = raw.groupby([od["city_id"], od["scale"]], dropna=False).transform("sum")
    group_count = raw.groupby([od["city_id"], od["scale"]], dropna=False).transform("size")
    od["od_pair_weight"] = np.where(group_sum > 0, raw / group_sum, 1.0 / group_count)
    od["od_pair_weight_scope"] = "city_scale_sample_normalized_origin_destination_population_product"
    return od


def weighted_quantile(values: pd.Series | np.ndarray, weights: pd.Series | np.ndarray, quantiles: list[float]) -> list[float]:
    values_arr = np.asarray(values, dtype=float)
    weights_arr = np.asarray(weights, dtype=float)
    mask = np.isfinite(values_arr) & np.isfinite(weights_arr) & (weights_arr >= 0)
    values_arr = values_arr[mask]
    weights_arr = weights_arr[mask]
    if len(values_arr) == 0:
        return [math.nan for _ in quantiles]
    if weights_arr.sum() <= 0:
        weights_arr = np.ones(len(values_arr), dtype=float)
    sorter = np.argsort(values_arr)
    values_arr = values_arr[sorter]
    weights_arr = weights_arr[sorter]
    cum = np.cumsum(weights_arr) - 0.5 * weights_arr
    cum = cum / weights_arr.sum()
    return [float(np.interp(q, cum, values_arr)) for q in quantiles]


def weighted_mean(values: pd.Series | np.ndarray, weights: pd.Series | np.ndarray) -> float:
    values_arr = np.asarray(values, dtype=float)
    weights_arr = np.asarray(weights, dtype=float)
    mask = np.isfinite(values_arr) & np.isfinite(weights_arr) & (weights_arr >= 0)
    values_arr = values_arr[mask]
    weights_arr = weights_arr[mask]
    if len(values_arr) == 0:
        return math.nan
    if weights_arr.sum() <= 0:
        return float(np.mean(values_arr))
    return float(np.sum(values_arr * weights_arr) / np.sum(weights_arr))


def weighted_gini(values: pd.Series | np.ndarray, weights: pd.Series | np.ndarray) -> float:
    values_arr = np.asarray(values, dtype=float)
    weights_arr = np.asarray(weights, dtype=float)
    mask = np.isfinite(values_arr) & np.isfinite(weights_arr) & (weights_arr >= 0)
    values_arr = values_arr[mask]
    weights_arr = weights_arr[mask]
    if len(values_arr) == 0 or weights_arr.sum() <= 0:
        return math.nan
    if np.allclose(values_arr, 0):
        return 0.0
    sorter = np.argsort(values_arr)
    values_arr = values_arr[sorter]
    weights_arr = weights_arr[sorter]
    cumw = np.cumsum(weights_arr)
    cumxw = np.cumsum(values_arr * weights_arr)
    total_xw = cumxw[-1]
    if total_xw <= 0:
        return 0.0
    lorenz_x = np.insert(cumw / cumw[-1], 0, 0)
    lorenz_y = np.insert(cumxw / total_xw, 0, 0)
    area = np.trapz(lorenz_y, lorenz_x)
    return float(1 - 2 * area)


def compute_accessibility(
    city_units: pd.DataFrame,
    city_facilities: pd.DataFrame,
    walk_ctx: NetworkContext,
    threshold_m: float,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    snapped_facilities = city_facilities[city_facilities["used_for_access"] == True].copy() if not city_facilities.empty else city_facilities
    for category in FACILITY_CATEGORIES:
        cat_fac = snapped_facilities[snapped_facilities["category"] == category] if not snapped_facilities.empty else snapped_facilities
        source_counts = cat_fac["source"].value_counts().to_dict() if not cat_fac.empty and "source" in cat_fac else {}
        facility_nodes = [node for node in cat_fac.get("walk_node", pd.Series(dtype=object)).dropna().unique().tolist() if node in walk_ctx.graph]
        if facility_nodes:
            try:
                distances = nx.multi_source_dijkstra_path_length(walk_ctx.graph, facility_nodes, cutoff=threshold_m, weight="length")
            except Exception:
                distances = {}
        else:
            distances = {}
        out = city_units[
            [
                "city_id",
                "city_name_en",
                "country",
                "iso3",
                "region",
                "sample_group",
                "quality_tier",
                "analysis_scope",
                "main_analysis_included",
                "pilot_extension_flag",
                "scale",
                "unit_id",
                "unit_area_km2",
                "population_sum",
                "population_weight",
                "population_weight_raw",
                "morphotype",
                "morphotype_name",
                "type_confidence",
                "low_confidence_flag",
                "low_quality_unit_flag",
                "shared_ucdb_boundary_flag",
                "shared_ucdb_id",
                "shared_ucdb_name",
                "shared_ucdb_boundary_group",
                "shared_ucdb_boundary_note",
                "unit_center_lon",
                "unit_center_lat",
                "walk_node",
                "walk_snap_distance_m",
            ]
        ].copy()
        out["category"] = category
        out["facility_count_total"] = int(len(cat_fac))
        out["facility_count_osm"] = int(source_counts.get("osm", 0))
        out["facility_count_overture"] = int(source_counts.get("overture", 0))
        out["access_threshold_m"] = float(threshold_m)
        out["network_distance_to_nearest_m"] = out["walk_node"].map(distances).astype(float)
        out["access_15min"] = out["network_distance_to_nearest_m"].le(threshold_m).fillna(False)
        out["accessibility_score"] = (1.0 - out["network_distance_to_nearest_m"] / threshold_m).clip(lower=0.0, upper=1.0)
        out.loc[out["network_distance_to_nearest_m"].isna(), "accessibility_score"] = 0.0
        records.append(out)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def aggregate_access_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for keys, grp in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(group_cols, keys))
        weights = pd.to_numeric(grp["population_sum"], errors="coerce").fillna(0.0).clip(lower=0.0)
        pop_sum = float(weights.sum())
        access = grp["access_15min"].astype(float)
        score = pd.to_numeric(grp["accessibility_score"], errors="coerce").fillna(0.0)
        dist = pd.to_numeric(grp["network_distance_to_nearest_m"], errors="coerce")
        rec.update(
            {
                "population_weighted_access_share": float((access * weights).sum() / pop_sum) if pop_sum > 0 else float(access.mean()),
                "accessibility_gini": weighted_gini(score, weights if pop_sum > 0 else np.ones(len(grp))),
                "no_access_population_share": float(((1 - access) * weights).sum() / pop_sum) if pop_sum > 0 else float((1 - access).mean()),
                "unit_count": int(grp["unit_id"].nunique()),
                "population_sum": pop_sum,
                "facility_count_total": int(grp["facility_count_total"].max()) if "facility_count_total" in grp else 0,
                "facility_count_osm": int(grp["facility_count_osm"].max()) if "facility_count_osm" in grp else 0,
                "facility_count_overture": int(grp["facility_count_overture"].max()) if "facility_count_overture" in grp else 0,
            }
        )
        p10, p50, p90 = weighted_quantile(
            dist.fillna(float("inf")).replace([np.inf, -np.inf], np.nan),
            weights if pop_sum > 0 else np.ones(len(grp)),
            [0.1, 0.5, 0.9],
        )
        rec.update({"p10": p10, "p50": p50, "p90": p90})
        records.append(rec)
    return pd.DataFrame.from_records(records)


def compute_inequality(accessibility: pd.DataFrame) -> pd.DataFrame:
    city = aggregate_access_group(accessibility, ["city_id", "city_name_en", "scale", "category"])
    city["group_level"] = "city"
    morph = aggregate_access_group(accessibility, ["city_id", "city_name_en", "scale", "category", "morphotype", "morphotype_name"])
    morph["group_level"] = "city_morphotype"
    global_morph = aggregate_access_group(accessibility, ["scale", "category", "morphotype", "morphotype_name"])
    global_morph["group_level"] = "morphotype_all_cities"
    return pd.concat([city, morph, global_morph], ignore_index=True, sort=False)


def compute_od_metrics(od_samples: pd.DataFrame) -> pd.DataFrame:
    if od_samples.empty:
        return pd.DataFrame()
    od_samples = normalize_od_pair_weights(od_samples)
    records: list[dict[str, Any]] = []
    group_specs = [
        ("city", ["city_id", "city_name_en", "scale"]),
        ("origin_morphotype", ["city_id", "city_name_en", "scale", "origin_morphotype", "origin_morphotype_name"]),
        (
            "origin_destination_morphotype",
            [
                "city_id",
                "city_name_en",
                "scale",
                "origin_morphotype",
                "origin_morphotype_name",
                "destination_morphotype",
                "destination_morphotype_name",
            ],
        ),
    ]
    for group_level, cols in group_specs:
        for keys, grp in od_samples.groupby(cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            rec = dict(zip(cols, keys))
            found = grp[grp["route_found"] == True].copy()  # noqa: E712
            detour = pd.to_numeric(found["detour_ratio"], errors="coerce")
            grp_weights = pd.to_numeric(grp.get("od_pair_weight"), errors="coerce").fillna(0.0).clip(lower=0.0)
            found_weights = pd.to_numeric(found.get("od_pair_weight"), errors="coerce").fillna(0.0).clip(lower=0.0)
            weight_sum = float(grp_weights.sum())
            found_weight_sum = float(found_weights.sum())
            weighted_p50, weighted_p90 = weighted_quantile(detour, found_weights, [0.5, 0.9])
            origin_analysis_scopes = sorted(str(v) for v in grp.get("origin_analysis_scope", pd.Series(dtype=object)).dropna().unique())
            destination_analysis_scopes = sorted(str(v) for v in grp.get("destination_analysis_scope", pd.Series(dtype=object)).dropna().unique())
            origin_main_values = grp.get("origin_main_analysis_included", pd.Series(dtype=object)).dropna().unique()
            destination_main_values = grp.get("destination_main_analysis_included", pd.Series(dtype=object)).dropna().unique()
            rec.update(
                {
                    "group_level": group_level,
                    "origin_analysis_scope": "|".join(origin_analysis_scopes),
                    "destination_analysis_scope": "|".join(destination_analysis_scopes),
                    "origin_main_analysis_included": bool(origin_main_values[0]) if len(origin_main_values) == 1 else None,
                    "destination_main_analysis_included": bool(destination_main_values[0]) if len(destination_main_values) == 1 else None,
                    "od_pair_count": int(len(grp)),
                    "od_pair_weight_sum": weight_sum,
                    "route_found_count": int(len(found)),
                    "route_found_weight_sum": found_weight_sum,
                    "route_found_share": float(len(found) / len(grp)) if len(grp) else math.nan,
                    "route_found_weighted_share": found_weight_sum / weight_sum if weight_sum > 0 else math.nan,
                    "od_circuity_mean": float(detour.mean()) if len(found) else math.nan,
                    "od_circuity_p50": float(detour.median()) if len(found) else math.nan,
                    "od_circuity_p90": float(detour.quantile(0.9)) if len(found) else math.nan,
                    "od_circuity_weighted_mean": weighted_mean(detour, found_weights) if len(found) else math.nan,
                    "od_circuity_weighted_p50": weighted_p50 if len(found) else math.nan,
                    "od_circuity_weighted_p90": weighted_p90 if len(found) else math.nan,
                    "detour_ratio_mean": float(detour.mean()) if len(found) else math.nan,
                    "detour_ratio_median": float(detour.median()) if len(found) else math.nan,
                    "detour_ratio_p90": float(detour.quantile(0.9)) if len(found) else math.nan,
                    "straight_distance_m_median": float(pd.to_numeric(found["straight_distance_m"], errors="coerce").median()) if len(found) else math.nan,
                    "drive_route_length_m_median": float(pd.to_numeric(found["drive_route_length_m"], errors="coerce").median()) if len(found) else math.nan,
                    "failure_reasons": json.dumps(grp["failure_reason"].dropna().value_counts().to_dict(), ensure_ascii=False),
                }
            )
            records.append(rec)
    return pd.DataFrame.from_records(records)


def compute_validation_summary(accessibility: pd.DataFrame, od_metrics: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    group_cols = ["main_analysis_included", "analysis_scope", "morphotype", "morphotype_name"]
    for keys, grp in accessibility.groupby(group_cols, dropna=False):
        main_analysis_included, analysis_scope, morphotype, morphotype_name = keys
        rec = {
            "main_analysis_included": bool(main_analysis_included),
            "analysis_scope": analysis_scope,
            "morphotype": morphotype,
            "morphotype_name": morphotype_name,
            "city_count": int(grp["city_id"].nunique()),
            "unit_count": int(grp["unit_id"].nunique()),
            "population_sum": float(grp.drop_duplicates("unit_id")["population_sum"].sum()),
            "mean_type_confidence": float(pd.to_numeric(grp.drop_duplicates("unit_id")["type_confidence"], errors="coerce").mean()),
            "low_confidence_unit_share": float(grp.drop_duplicates("unit_id")["low_confidence_flag"].astype(float).mean()),
        }
        for category, cat_grp in grp.groupby("category"):
            weights = pd.to_numeric(cat_grp["population_sum"], errors="coerce").fillna(0.0).clip(lower=0.0)
            pop_sum = weights.sum()
            access = cat_grp["access_15min"].astype(float)
            rec[f"access_share__{category}"] = float((access * weights).sum() / pop_sum) if pop_sum > 0 else float(access.mean())
            rec[f"median_distance_m__{category}"] = float(pd.to_numeric(cat_grp["network_distance_to_nearest_m"], errors="coerce").median())
        od_group = od_metrics[
            (od_metrics.get("group_level") == "origin_morphotype")
            & (od_metrics.get("origin_morphotype") == morphotype)
            & (od_metrics.get("origin_analysis_scope") == str(analysis_scope))
            & (od_metrics.get("origin_main_analysis_included") == bool(main_analysis_included))
        ]
        rec["od_city_group_count"] = int(len(od_group))
        rec["od_route_found_share_mean"] = float(pd.to_numeric(od_group.get("route_found_share"), errors="coerce").mean()) if len(od_group) else math.nan
        rec["od_route_found_weighted_share_mean"] = (
            float(pd.to_numeric(od_group.get("route_found_weighted_share"), errors="coerce").mean()) if len(od_group) else math.nan
        )
        rec["od_circuity_mean_mean"] = float(pd.to_numeric(od_group.get("od_circuity_mean"), errors="coerce").mean()) if len(od_group) else math.nan
        rec["od_circuity_p50_mean"] = float(pd.to_numeric(od_group.get("od_circuity_p50"), errors="coerce").mean()) if len(od_group) else math.nan
        rec["od_circuity_weighted_mean_mean"] = (
            float(pd.to_numeric(od_group.get("od_circuity_weighted_mean"), errors="coerce").mean()) if len(od_group) else math.nan
        )
        rec["od_circuity_weighted_p50_mean"] = (
            float(pd.to_numeric(od_group.get("od_circuity_weighted_p50"), errors="coerce").mean()) if len(od_group) else math.nan
        )
        rec["od_circuity_weighted_p90_mean"] = (
            float(pd.to_numeric(od_group.get("od_circuity_weighted_p90"), errors="coerce").mean()) if len(od_group) else math.nan
        )
        rec["od_detour_ratio_median_mean"] = rec["od_circuity_p50_mean"]
        records.append(rec)
    return pd.DataFrame.from_records(records).sort_values(["main_analysis_included", "analysis_scope", "morphotype"])


def compute_validation_effects(accessibility: pd.DataFrame, od_metrics: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    city_access = aggregate_access_group(accessibility, ["city_id", "category", "main_analysis_included", "analysis_scope"])
    morph_access = aggregate_access_group(
        accessibility,
        ["city_id", "category", "main_analysis_included", "analysis_scope", "morphotype", "morphotype_name"],
    )
    merged = morph_access.merge(
        city_access[["city_id", "category", "main_analysis_included", "analysis_scope", "population_weighted_access_share"]],
        on=["city_id", "category", "main_analysis_included", "analysis_scope"],
        how="left",
        suffixes=("_morphotype", "_city"),
    )
    access_group_cols = ["main_analysis_included", "analysis_scope", "morphotype", "morphotype_name", "category"]
    for keys, grp in merged.groupby(access_group_cols, dropna=False):
        main_analysis_included, analysis_scope, morphotype, morphotype_name, category = keys
        weights = pd.to_numeric(grp["population_sum"], errors="coerce").fillna(0.0).clip(lower=0.0)
        effect = grp["population_weighted_access_share_morphotype"] - grp["population_weighted_access_share_city"]
        records.append(
            {
                "domain": "accessibility",
                "metric": "population_weighted_access_share_city_centered",
                "main_analysis_included": bool(main_analysis_included),
                "analysis_scope": analysis_scope,
                "category": category,
                "morphotype": morphotype,
                "morphotype_name": morphotype_name,
                "effect_weighted_mean": float((effect * weights).sum() / weights.sum()) if weights.sum() > 0 else float(effect.mean()),
                "effect_median": float(effect.median()),
                "city_group_count": int(len(grp)),
            }
        )
    if not od_metrics.empty:
        city_cols = [
            "city_id",
            "origin_analysis_scope",
            "origin_main_analysis_included",
            "od_circuity_p50",
            "od_circuity_weighted_p50",
        ]
        city_od = od_metrics[od_metrics["group_level"] == "city"][city_cols].rename(
            columns={
                "od_circuity_p50": "city_od_circuity_p50",
                "od_circuity_weighted_p50": "city_od_circuity_weighted_p50",
            }
        )
        morph_od = od_metrics[od_metrics["group_level"] == "origin_morphotype"].merge(
            city_od,
            on=["city_id", "origin_analysis_scope", "origin_main_analysis_included"],
            how="left",
        )
        morph_od["effect_unweighted_p50"] = morph_od["od_circuity_p50"] - morph_od["city_od_circuity_p50"]
        morph_od["effect_weighted_p50"] = morph_od["od_circuity_weighted_p50"] - morph_od["city_od_circuity_weighted_p50"]
        od_group_cols = ["origin_main_analysis_included", "origin_analysis_scope", "origin_morphotype", "origin_morphotype_name"]
        for keys, grp in morph_od.groupby(od_group_cols, dropna=False):
            main_analysis_included, analysis_scope, morphotype, morphotype_name = keys
            weights = pd.to_numeric(grp["route_found_weight_sum"], errors="coerce").fillna(0.0).clip(lower=0.0)
            for metric_name, effect_col in [
                ("od_circuity_weighted_p50_city_centered", "effect_weighted_p50"),
                ("od_circuity_p50_city_centered_unweighted_diagnostic", "effect_unweighted_p50"),
            ]:
                effect = pd.to_numeric(grp[effect_col], errors="coerce")
                records.append(
                    {
                        "domain": "detour",
                        "metric": metric_name,
                        "main_analysis_included": bool(main_analysis_included),
                        "analysis_scope": analysis_scope,
                        "category": "od_drive",
                        "morphotype": morphotype,
                        "morphotype_name": morphotype_name,
                        "effect_weighted_mean": weighted_mean(effect, weights),
                        "effect_median": float(effect.median()) if effect.notna().any() else math.nan,
                        "city_group_count": int(len(grp)),
                    }
                )
    return pd.DataFrame.from_records(records)


def compute_low_confidence_sensitivity(accessibility: pd.DataFrame, od_samples: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    unit_base = accessibility.drop_duplicates(["unit_id"])[["unit_id", "type_confidence", "low_confidence_flag"]]
    thresholds = [
        ("all_units", None),
        ("exclude_low_confidence_flag", "flag"),
        ("type_confidence_ge_0_60", 0.60),
        ("type_confidence_ge_0_70", 0.70),
    ]
    baseline = None
    for label, threshold in thresholds:
        if threshold is None:
            subset = accessibility
        elif threshold == "flag":
            subset = accessibility[accessibility["low_confidence_flag"] == False]  # noqa: E712
        else:
            subset = accessibility[pd.to_numeric(accessibility["type_confidence"], errors="coerce") >= float(threshold)]
        agg = aggregate_access_group(subset, ["category", "morphotype", "morphotype_name"])
        agg["sensitivity_filter"] = label
        if label == "all_units":
            baseline = agg[["category", "morphotype", "population_weighted_access_share"]].rename(
                columns={"population_weighted_access_share": "baseline_access_share"}
            )
        if baseline is not None:
            agg = agg.merge(baseline, on=["category", "morphotype"], how="left")
            agg["delta_vs_all_units"] = agg["population_weighted_access_share"] - agg["baseline_access_share"]
        records.extend(agg.to_dict("records"))
    access_sensitivity = pd.DataFrame.from_records(records)
    access_sensitivity["domain"] = "accessibility"

    detour_records: list[dict[str, Any]] = []
    if not od_samples.empty:
        od = od_samples.merge(unit_base.rename(columns={"unit_id": "origin_unit_id"}), on="origin_unit_id", how="left")
        baseline_detour = None
        for label, threshold in thresholds:
            if threshold is None:
                subset = od
            elif threshold == "flag":
                subset = od[od["low_confidence_flag"] == False]  # noqa: E712
            else:
                subset = od[pd.to_numeric(od["type_confidence"], errors="coerce") >= float(threshold)]
            found = subset[subset["route_found"] == True]  # noqa: E712
            agg = (
                found.groupby(["origin_morphotype", "origin_morphotype_name"], dropna=False)
                .agg(
                    route_found_count=("route_found", "size"),
                    detour_ratio_median=("detour_ratio", "median"),
                    detour_ratio_mean=("detour_ratio", "mean"),
                )
                .reset_index()
                .rename(columns={"origin_morphotype": "morphotype", "origin_morphotype_name": "morphotype_name"})
            )
            agg["sensitivity_filter"] = label
            if label == "all_units":
                baseline_detour = agg[["morphotype", "detour_ratio_median"]].rename(
                    columns={"detour_ratio_median": "baseline_detour_ratio_median"}
                )
            if baseline_detour is not None:
                agg = agg.merge(baseline_detour, on="morphotype", how="left")
                agg["delta_vs_all_units"] = agg["detour_ratio_median"] - agg["baseline_detour_ratio_median"]
            agg["domain"] = "detour"
            detour_records.extend(agg.to_dict("records"))
    detour_sensitivity = pd.DataFrame.from_records(detour_records)
    return pd.concat([access_sensitivity, detour_sensitivity], ignore_index=True, sort=False)


def q_summary(values: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce")
    return {
        "mean": float(numeric.mean()) if numeric.notna().any() else math.nan,
        "median": float(numeric.median()) if numeric.notna().any() else math.nan,
        "p90": float(numeric.quantile(0.9)) if numeric.notna().any() else math.nan,
        "max": float(numeric.max()) if numeric.notna().any() else math.nan,
    }


def run_city(
    paths: StepPaths,
    city_units: pd.DataFrame,
    facilities: pd.DataFrame,
    args: argparse.Namespace,
    max_origin_units: int | None,
    destinations_per_origin: int,
    max_od_pairs: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    city_id = str(city_units["city_id"].iloc[0])
    t0 = time.time()
    city_log: dict[str, Any] = {
        "city_id": city_id,
        "city_name_en": str(city_units["city_name_en"].iloc[0]),
        "unit_count": int(len(city_units)),
        "random_seed_base": int(args.random_seed),
        "city_random_seed": int(stable_city_seed(city_id, args.random_seed)),
        "started_at": now_iso(),
    }
    drive_ctx = load_network(paths, city_id, MAIN_NETWORK, args.graph_source)
    walk_ctx = load_network(paths, city_id, WALK_NETWORK, args.graph_source)
    city_log["drive_graph"] = {
        "path": str(drive_ctx.source_path),
        "nodes": int(drive_ctx.graph.number_of_nodes()),
        "edges": int(drive_ctx.graph.number_of_edges()),
        "raw_node_count": drive_ctx.graph.graph.get("raw_node_count"),
        "valid_xy_node_count": drive_ctx.graph.graph.get("valid_xy_node_count"),
        "raw_edge_count": drive_ctx.graph.graph.get("raw_edge_count"),
        "positive_length_edge_records": drive_ctx.graph.graph.get("positive_length_edge_records"),
        "nonpositive_or_missing_length_edge_records": drive_ctx.graph.graph.get("nonpositive_or_missing_length_edge_records"),
    }
    city_log["walk_graph"] = {
        "path": str(walk_ctx.source_path),
        "nodes": int(walk_ctx.graph.number_of_nodes()),
        "edges": int(walk_ctx.graph.number_of_edges()),
        "raw_node_count": walk_ctx.graph.graph.get("raw_node_count"),
        "valid_xy_node_count": walk_ctx.graph.graph.get("valid_xy_node_count"),
        "raw_edge_count": walk_ctx.graph.graph.get("raw_edge_count"),
        "positive_length_edge_records": walk_ctx.graph.graph.get("positive_length_edge_records"),
        "nonpositive_or_missing_length_edge_records": walk_ctx.graph.graph.get("nonpositive_or_missing_length_edge_records"),
    }
    snapped_units = snap_units(city_units, drive_ctx, walk_ctx)
    city_facilities = facilities[facilities["city_id"] == city_id].copy()
    snapped_facilities = snap_facilities(city_facilities, walk_ctx)
    od_samples = sample_od_pairs(
        snapped_units,
        drive_ctx,
        max_origin_units=max_origin_units,
        destinations_per_origin=destinations_per_origin,
        max_pairs=max_od_pairs,
        random_seed=args.random_seed,
        min_straight_m=args.min_straight_m,
    )
    accessibility = compute_accessibility(snapped_units, snapped_facilities, walk_ctx, args.access_threshold_m)
    city_log["unit_snap_quality"] = {
        "drive_snap_distance_m": q_summary(snapped_units["drive_snap_distance_m"]),
        "walk_snap_distance_m": q_summary(snapped_units["walk_snap_distance_m"]),
    }
    city_log["facility_count_by_category_source"] = (
        snapped_facilities.groupby(["category", "source"], dropna=False).size().reset_index(name="count").to_dict("records")
        if not snapped_facilities.empty
        else []
    )
    city_log["facility_snapped_node_count_by_category_source"] = (
        snapped_facilities[snapped_facilities["walk_node"].notna()]
        .groupby(["category", "source"], dropna=False)
        .agg(snapped_facility_count=("facility_id", "size"), unique_walk_node_count=("walk_node", "nunique"))
        .reset_index()
        .to_dict("records")
        if not snapped_facilities.empty
        else []
    )
    city_log["facility_snap_quality"] = q_summary(snapped_facilities.get("walk_snap_distance_m", pd.Series(dtype=float)))
    city_log["od_pair_count"] = int(len(od_samples))
    eligible_origin_count = int(snapped_units["drive_node"].notna().sum())
    planned_origin_count = eligible_origin_count if max_origin_units is None else min(int(max_origin_units), eligible_origin_count)
    city_log["od_target_pair_count"] = int(
        min(
            planned_origin_count * destinations_per_origin,
            max_od_pairs if max_od_pairs is not None else 10**18,
        )
    )
    city_log["od_eligible_origin_count"] = eligible_origin_count
    city_log["od_planned_origin_count"] = planned_origin_count
    city_log["od_reached_share"] = float(len(od_samples) / city_log["od_target_pair_count"]) if city_log["od_target_pair_count"] else math.nan
    city_log["od_route_found_share"] = (
        float(od_samples["route_found"].mean()) if not od_samples.empty and "route_found" in od_samples else math.nan
    )
    city_log["od_route_failure_share"] = 1.0 - city_log["od_route_found_share"] if np.isfinite(city_log["od_route_found_share"]) else math.nan
    city_log["accessibility_rows"] = int(len(accessibility))
    city_log["unit_center_unique_coordinate_count"] = int(
        snapped_units[["unit_center_lon", "unit_center_lat"]].round(8).drop_duplicates().shape[0]
    )
    city_log["elapsed_seconds"] = round(time.time() - t0, 3)
    city_log["completed_at"] = now_iso()
    return od_samples, accessibility, city_log


def build_qc(
    paths: StepPaths,
    run_log: dict[str, Any],
    facilities: pd.DataFrame,
    od_samples: pd.DataFrame,
    accessibility: pd.DataFrame,
    inequality: pd.DataFrame,
    input_report: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(check_name: str, status: str, value: Any, threshold: Any = None, details: Any = None) -> None:
        rows.append(
            {
                "check_name": check_name,
                "status": status,
                "value": json.dumps(json_ready(value), ensure_ascii=False) if isinstance(value, (dict, list)) else value,
                "threshold": threshold,
                "details": json.dumps(json_ready(details), ensure_ascii=False) if isinstance(details, (dict, list)) else details,
            }
        )

    for key, filename in OUTPUT_FILES.items():
        if key in {"qc", "run_log", "readme", "repro"}:
            continue
        path = paths.output_dir / filename
        add(f"output_exists__{filename}", "pass" if path.exists() else "fail", path.exists(), "exists")
    add("input_acceptance_status", "pass" if input_report.get("status") == "pass" else "warn", input_report.get("status"))
    expected_seeds = {city_id: int(stable_city_seed(city_id, run_log.get("random_seed"))) for city_id in run_log.get("selected_city_ids", [])}
    add(
        "stable_per_city_random_seeds_recorded",
        "pass" if run_log.get("per_city_random_seeds") == expected_seeds else "fail",
        run_log.get("per_city_random_seeds"),
        expected_seeds,
    )
    add(
        "main_analysis_reference_rows",
        "pass" if run_log.get("main_analysis_reference_rows") == 74861 else "fail",
        run_log.get("main_analysis_reference_rows"),
        74861,
    )
    add(
        "main_analysis_reference_cities",
        "pass" if run_log.get("main_analysis_reference_cities") == 48 else "fail",
        run_log.get("main_analysis_reference_cities"),
        48,
    )
    add(
        "main_analysis_reference_morphotypes",
        "pass" if run_log.get("main_analysis_reference_morphotypes") == 10 else "fail",
        run_log.get("main_analysis_reference_morphotypes"),
        10,
    )
    add(
        "unique_city_unit_network_scale_key",
        "pass" if run_log.get("selected_duplicate_city_unit_network_scale_keys") == 0 else "fail",
        run_log.get("selected_duplicate_city_unit_network_scale_keys"),
        0,
    )
    add("facility_rows", "pass" if len(facilities) > 0 else "fail", int(len(facilities)), ">0")
    category_counts = facilities["category"].value_counts().reindex(FACILITY_CATEGORIES, fill_value=0).to_dict() if not facilities.empty else {}
    add("facility_category_coverage", "pass" if all(v > 0 for v in category_counts.values()) else "warn", category_counts, "all categories >0")
    add("od_sample_rows", "pass" if len(od_samples) > 0 else "fail", int(len(od_samples)), ">0")
    if not od_samples.empty and "od_pair_weight" in od_samples:
        weight_sums = (
            od_samples.groupby(["city_id", "scale"], dropna=False)["od_pair_weight"]
            .sum()
            .round(10)
            .to_dict()
        )
        max_weight_sum_error = max((abs(float(v) - 1.0) for v in weight_sums.values()), default=math.nan)
        add("od_pair_weight_city_scale_sums_to_one", "pass" if max_weight_sum_error <= 1e-8 else "fail", weight_sums, "sum == 1 per city/scale")
    else:
        add("od_pair_weight_present", "fail", False, "column exists")
    route_share = float(od_samples["route_found"].mean()) if not od_samples.empty else math.nan
    add("od_route_found_share", "pass" if np.isfinite(route_share) and route_share >= 0.70 else "warn", route_share, ">=0.70")
    expected_access_rows = int(run_log.get("selected_unit_count", 0)) * len(FACILITY_CATEGORIES)
    add("accessibility_row_count", "pass" if len(accessibility) == expected_access_rows else "fail", int(len(accessibility)), expected_access_rows)
    add("inequality_rows", "pass" if len(inequality) > 0 else "fail", int(len(inequality)), ">0")
    missing_morph = int(accessibility["morphotype"].isna().sum()) if not accessibility.empty else 0
    add("accessibility_missing_morphotype_rows", "pass" if missing_morph == 0 else "fail", missing_morph, 0)
    city_logs = run_log.get("city_logs", [])
    drive_snap_medians = [c.get("unit_snap_quality", {}).get("drive_snap_distance_m", {}).get("median") for c in city_logs]
    walk_snap_medians = [c.get("unit_snap_quality", {}).get("walk_snap_distance_m", {}).get("median") for c in city_logs]
    drive_snap_max = [c.get("unit_snap_quality", {}).get("drive_snap_distance_m", {}).get("max") for c in city_logs]
    walk_snap_max = [c.get("unit_snap_quality", {}).get("walk_snap_distance_m", {}).get("max") for c in city_logs]
    add("drive_snap_median_max_m", "pass" if pd.Series(drive_snap_medians).dropna().max() <= 500 else "warn", pd.Series(drive_snap_medians).dropna().max(), "<=500m")
    add("walk_snap_median_max_m", "pass" if pd.Series(walk_snap_medians).dropna().max() <= 500 else "warn", pd.Series(walk_snap_medians).dropna().max(), "<=500m")
    add("drive_snap_max_m_by_city", "pass", drive_snap_max, "record")
    add("walk_snap_max_m_by_city", "pass", walk_snap_max, "record")
    center_unique_counts = [c.get("unit_center_unique_coordinate_count") for c in city_logs]
    add(
        "hex_unit_centers_not_single_coordinate",
        "pass" if pd.Series(center_unique_counts).dropna().min() > 1 else "fail",
        center_unique_counts,
        ">1 unique center per city",
    )
    graph_valid = True
    graph_records = []
    for city_log in city_logs:
        for graph_key in ["drive_graph", "walk_graph"]:
            graph = city_log.get(graph_key, {})
            ok = (
                (graph.get("valid_xy_node_count") or 0) > 0
                and (graph.get("positive_length_edge_records") or 0) > 0
                and (graph.get("nodes") or 0) > 0
                and (graph.get("edges") or 0) > 0
            )
            graph_valid = graph_valid and ok
            graph_records.append({"city_id": city_log.get("city_id"), "graph": graph_key, "ok": ok, **graph})
    add("graph_nodes_xy_and_positive_lengths", "pass" if graph_valid else "fail", graph_records, "all graphs valid")
    dhaka_main_rows = (
        accessibility[
            accessibility["city_name_en"].astype(str).str.contains("Dhaka", case=False, na=False)
            & (accessibility["main_analysis_included"] == True)  # noqa: E712
        ]
        if not accessibility.empty and "main_analysis_included" in accessibility
        else pd.DataFrame()
    )
    add("dhaka_not_marked_main_analysis", "pass" if len(dhaka_main_rows) == 0 else "fail", len(dhaka_main_rows), 0)
    shared_record = input_report.get("shenzhen_guangzhou_shared_ucdb_record", [])
    add("shared_ucdb_record_retained", "pass" if len(shared_record) > 0 else "warn", shared_record, "chn_003/chn_004 recorded if present upstream")
    selected_shared = set(run_log.get("selected_city_ids", [])) & {"chn_003", "chn_004"}
    if selected_shared:
        output_shared = set(accessibility[accessibility["city_id"].isin(selected_shared)]["city_id"].unique()) if not accessibility.empty else set()
        add("selected_shared_ucdb_cities_output_separately", "pass" if output_shared == selected_shared else "fail", sorted(output_shared), sorted(selected_shared))
    return pd.DataFrame.from_records(rows)


def write_readme(paths: StepPaths, run_log: dict[str, Any]) -> None:
    lines = [
        f"# {STEP_NAME}",
        "",
        "This directory is generated by `code_upload/10_compute_accessibility_detour_validation_metrics/step10_accessibility_detour.py`.",
        "",
        "## Main Analysis Conventions",
        "",
        "- Main analysis: Step09 `analysis_scope == \"main_fit\"`, drive, `hex_1km`.",
        "- Main scale: `hex_1km`.",
        "- morphotype: Step09 drive main model results; clustering is not rerun.",
        "- OD detour: drive road-network `length`.",
        "- OD main conclusions prioritize population-weighted circuity metrics: `od_pair_weight = origin_population_weight_norm * destination_population_weight_norm`, normalized within the city/scale sample; unweighted metrics are diagnostic only.",
        "- 15-minute facility accessibility: walk road-network `length`, threshold 1200 m.",
        "- Dhaka/excluded_review is excluded from main conclusions; the Shenzhen/Guangzhou shared UCDB boundary is retained only as a flag.",
        "",
        "## Output Files",
        "",
        "- `10_input_acceptance_report.json/md`: upstream input, special sample, and road-network availability acceptance checks.",
        "- `facility_classification_table.parquet`: OSM primary facility classification with Overture as supplemental coverage.",
        "- `facility_classification_counts_QC.csv`: raw, deduplicated, and snapped-to-walk-node facility counts by city and category.",
        "- `OD_detour_samples.parquet`: sampled OD pairs and drive detour ratios.",
        "- `OD_detour_metrics.parquet`: city and morphotype OD detour aggregates.",
        "- `facility_accessibility_metrics.parquet`: unit x five facility category 1200 m walk-network accessibility.",
        "- `accessibility_inequality_metrics.parquet`: city, within-city morphotype, and global morphotype accessibility inequality metrics.",
        "- `morphotype_validation_summary.csv`: morphotype explanatory-power summary.",
        "- `morphotype_validation_effects.csv`: city-centered effect table.",
        "- `low_confidence_sensitivity.csv`: low-confidence sensitivity results.",
        "- `10_quality_checks.csv`, `10_run_log.json`, `reproducibility_status_manifest.json`: QC, run log, and reproducibility information.",
        "",
        "## This Run",
        "",
        f"- Mode: `{run_log.get('mode')}`",
        f"- Cities: {run_log.get('selected_city_ids')}",
        f"- OD origin cap: {run_log.get('max_origin_units_per_city')}",
        f"- OD destinations per origin: {run_log.get('od_destinations_per_origin')}",
        f"- OD hard cap per city: {run_log.get('max_od_pairs_per_city')}",
        f"- Random seed: {run_log.get('random_seed')}",
        f"- Stable per-city seeds: {run_log.get('per_city_random_seeds')}",
        f"- Run command: `{run_log.get('command')}`",
        "",
    ]
    (paths.output_dir / OUTPUT_FILES["readme"]).write_text("\n".join(lines), encoding="utf-8")


def write_repro(paths: StepPaths, args: argparse.Namespace, run_log: dict[str, Any]) -> None:
    script_path = Path(__file__).resolve()
    script_stat = script_path.stat()
    repro = {
        "generated_at": now_iso(),
        "step": STEP_NAME,
        "script": {
            "path": str(script_path),
            "size_bytes": int(script_stat.st_size),
            "mtime": datetime.fromtimestamp(script_stat.st_mtime).astimezone().isoformat(timespec="seconds"),
            "sha256": file_sha256(script_path),
        },
        "command": run_log.get("command"),
        "parameters": vars(args),
        "package_versions": package_versions(),
        "git_state": git_state(paths.root),
        "output_files": {name: file_metadata(paths.output_dir / filename) for name, filename in OUTPUT_FILES.items()},
        "random_seed": args.random_seed,
        "per_city_random_seeds": run_log.get("per_city_random_seeds"),
        "resumability_note": "City-level loop is deterministic for a fixed city list and seed; rerun with --mode main and --overwrite to regenerate fixed summary outputs.",
    }
    write_json(paths.output_dir / OUTPUT_FILES["repro"], repro)


def run_step10(args: argparse.Namespace) -> dict[str, Any]:
    paths = get_paths(args.root, args.output_dir)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.overwrite:
        existing = [paths.output_dir / filename for filename in OUTPUT_FILES.values() if (paths.output_dir / filename).exists()]
        if existing:
            raise FileExistsError(
                "Step10 outputs already exist. Pass --overwrite to regenerate: "
                + ", ".join(path.name for path in existing[:5])
                + (" ..." if len(existing) > 5 else "")
            )

    started_at = now_iso()
    t0 = time.time()
    drive_hex_units = load_step09_drive_hex_units(paths)
    all_main_units = filter_main_analysis_units(drive_hex_units)
    city_ids = selected_city_ids(args, drive_hex_units, all_main_units)
    if not city_ids:
        raise RuntimeError("No city selected for Step10.")
    max_od_pairs = args.max_od_pairs_per_city
    max_origin_units = args.max_origin_units_per_city
    if max_origin_units is None and args.mode != "main":
        max_origin_units = 300
    destinations_per_origin = args.od_destinations_per_origin
    if destinations_per_origin is None:
        destinations_per_origin = 8 if args.mode == "main" else 4

    input_report = collect_input_report(paths, all_main_units, city_ids)
    write_input_report(paths, input_report)
    if args.mode == "inputs-only":
        run_log = {
            "step": STEP_NAME,
            "mode": args.mode,
            "started_at": started_at,
            "completed_at": now_iso(),
            "selected_city_ids": city_ids,
            "selected_unit_count": 0,
            "random_seed": args.random_seed,
            "command": " ".join(sys.argv),
            "note": "inputs-only mode wrote input acceptance reports only.",
        }
        write_json(paths.output_dir / OUTPUT_FILES["run_log"], run_log)
        write_repro(paths, args, run_log)
        return run_log

    run_units = select_run_units(drive_hex_units, all_main_units, city_ids, args.mode)
    selected_units = load_units_with_geometry(paths, run_units)
    facilities, facility_qc = classify_facilities(paths, city_ids, args.facility_sources)
    write_dataframe_parquet(facilities, paths.output_dir / OUTPUT_FILES["facility_table"])

    city_logs: list[dict[str, Any]] = []
    od_frames: list[pd.DataFrame] = []
    access_frames: list[pd.DataFrame] = []
    for city_id in city_ids:
        city_units = selected_units[selected_units["city_id"] == city_id].copy()
        try:
            od_samples, accessibility, city_log = run_city(
                paths,
                city_units,
                facilities,
                args,
                max_origin_units=max_origin_units,
                destinations_per_origin=destinations_per_origin,
                max_od_pairs=max_od_pairs,
            )
            od_frames.append(od_samples)
            access_frames.append(accessibility)
            city_logs.append(city_log)
        except Exception as exc:
            city_logs.append(
                {
                    "city_id": city_id,
                    "city_name_en": str(city_units["city_name_en"].iloc[0]) if not city_units.empty else None,
                    "unit_count": int(len(city_units)),
                    "error": repr(exc),
                    "completed_at": now_iso(),
                }
            )
            raise

    od_samples_all = pd.concat(od_frames, ignore_index=True, sort=False) if od_frames else pd.DataFrame()
    od_samples_all = normalize_od_pair_weights(od_samples_all)
    accessibility_all = pd.concat(access_frames, ignore_index=True, sort=False) if access_frames else pd.DataFrame()
    snapped_count_records: list[dict[str, Any]] = []
    for city_log in city_logs:
        for rec in city_log.get("facility_snapped_node_count_by_category_source", []):
            snapped_count_records.append(
                {
                    "city_id": city_log.get("city_id"),
                    "city_name_en": city_log.get("city_name_en"),
                    "category": rec.get("category"),
                    "source": rec.get("source"),
                    "count": rec.get("snapped_facility_count"),
                    "count_type": "snapped_to_walk_node",
                    "unique_walk_node_count": rec.get("unique_walk_node_count"),
                }
            )
    facility_qc_out = pd.concat(
        [facility_qc, pd.DataFrame.from_records(snapped_count_records)],
        ignore_index=True,
        sort=False,
    )
    od_metrics = compute_od_metrics(od_samples_all)
    inequality = compute_inequality(accessibility_all)
    summary = compute_validation_summary(accessibility_all, od_metrics)
    effects = compute_validation_effects(accessibility_all, od_metrics)
    low_confidence = compute_low_confidence_sensitivity(accessibility_all, od_samples_all)

    write_dataframe_parquet(od_samples_all, paths.output_dir / OUTPUT_FILES["od_samples"])
    write_dataframe_parquet(od_metrics, paths.output_dir / OUTPUT_FILES["od_metrics"])
    write_dataframe_parquet(accessibility_all, paths.output_dir / OUTPUT_FILES["accessibility"])
    write_dataframe_parquet(inequality, paths.output_dir / OUTPUT_FILES["inequality"])
    facility_qc_out.to_csv(paths.output_dir / OUTPUT_FILES["facility_qc"], index=False)
    summary.to_csv(paths.output_dir / OUTPUT_FILES["summary"], index=False)
    effects.to_csv(paths.output_dir / OUTPUT_FILES["effects"], index=False)
    low_confidence.to_csv(paths.output_dir / OUTPUT_FILES["low_confidence"], index=False)

    completed_at = now_iso()
    run_log = {
        "step": STEP_NAME,
        "mode": args.mode,
        "started_at": started_at,
        "completed_at": completed_at,
        "elapsed_seconds": round(time.time() - t0, 3),
        "selected_city_ids": city_ids,
        "selected_city_count": len(city_ids),
        "selected_unit_count": int(len(selected_units)),
        "selected_duplicate_city_unit_network_scale_keys": int(
            selected_units.duplicated(["city_id", "unit_id", "network_type", "scale"]).sum()
        ),
        "main_analysis_reference_rows": int(len(all_main_units)),
        "main_analysis_reference_cities": int(all_main_units["city_id"].nunique()),
        "main_analysis_reference_morphotypes": int(all_main_units["morphotype"].nunique()),
        "max_origin_units_per_city": None if max_origin_units is None else int(max_origin_units),
        "od_destinations_per_origin": int(destinations_per_origin),
        "max_od_pairs_per_city": None if max_od_pairs is None else int(max_od_pairs),
        "random_seed": int(args.random_seed),
        "per_city_random_seeds": {city_id: int(stable_city_seed(city_id, args.random_seed)) for city_id in city_ids},
        "od_weighting_policy": (
            "OD main circuity metrics use od_pair_weight, defined as "
            "origin_population_weight_norm * destination_population_weight_norm normalized within each city_id/scale sample; "
            "unweighted metrics are retained as diagnostics."
        ),
        "facility_sources": args.facility_sources,
        "graph_source": args.graph_source,
        "access_threshold_m": float(args.access_threshold_m),
        "min_straight_m": float(args.min_straight_m),
        "facility_rows": int(len(facilities)),
        "facility_qc_counts": facility_qc.to_dict("records"),
        "od_sample_rows": int(len(od_samples_all)),
        "accessibility_rows": int(len(accessibility_all)),
        "inequality_rows": int(len(inequality)),
        "command": " ".join(sys.argv),
        "city_logs": city_logs,
    }
    qc = build_qc(paths, run_log, facilities, od_samples_all, accessibility_all, inequality, input_report)
    qc.to_csv(paths.output_dir / OUTPUT_FILES["qc"], index=False)
    write_json(paths.output_dir / OUTPUT_FILES["run_log"], run_log)
    write_readme(paths, run_log)
    write_repro(paths, args, run_log)
    return run_log


def main() -> None:
    args = parse_args()
    run_log = run_step10(args)
    print(
        json.dumps(
            {
                "step": STEP_NAME,
                "mode": run_log.get("mode"),
                "selected_city_count": run_log.get("selected_city_count", len(run_log.get("selected_city_ids", []))),
                "selected_city_ids": run_log.get("selected_city_ids"),
                "elapsed_seconds": run_log.get("elapsed_seconds"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
