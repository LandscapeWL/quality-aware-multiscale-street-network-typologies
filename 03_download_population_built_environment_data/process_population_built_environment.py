from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import reduce
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
STEP_DIR = ROOT / "data/03_download_population_built_environment_data"
CITY_DIR = ROOT / "data/01_city_boundaries_and_sample_list/city_sample"
CITY_TABLE_CSV = CITY_DIR / "city_master.csv"
CITY_TABLE_PARQUET = CITY_DIR / "city_master.parquet"
CITY_BOUNDARY_GPKG = CITY_DIR / "city_boundaries.gpkg"

GHSL_POP_DIR = STEP_DIR / "GHSL_population"
GHSL_BUILT_DIR = STEP_DIR / "GHSL_built_surface"
WORLDPOP_DIR = STEP_DIR / "WorldPop" / "2020"
WSF_DIR = STEP_DIR / "WSF"

INVENTORY_TABLE = STEP_DIR / "raster_data_inventory.csv"
PROCESSING_PLAN_TABLE = STEP_DIR / "population_built_environment_processing_plan.csv"
METADATA_TABLE = STEP_DIR / "raster_metadata_sample.csv"
RUN_LOG = STEP_DIR / "population_built_environment_quality_checks.json"
CITY_METRICS_PARQUET = STEP_DIR / "city_population_built_environment_metrics.parquet"
CITY_METRICS_CSV = STEP_DIR / "city_population_built_environment_metrics.csv"
ZONAL_DIAGNOSTICS_TABLE = STEP_DIR / "city_population_built_environment_zonal_stats_diagnostics.csv"
ZONAL_RUN_LOG = STEP_DIR / "city_population_built_environment_zonal_stats_run_log.json"


@dataclass(frozen=True)
class ExpectedFile:
    dataset: str
    layer: str
    year: int
    resolution_m: int | None
    path: Path
    note: str


EXPECTED_GHSL_FILES = [
    ExpectedFile(
        dataset="GHSL",
        layer="population",
        year=2020,
        resolution_m=100,
        path=GHSL_POP_DIR / "GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0.zip",
        note="Primary analysis population raster",
    ),
    ExpectedFile(
        dataset="GHSL",
        layer="population",
        year=2020,
        resolution_m=1000,
        path=GHSL_POP_DIR / "GHS_POP_E2020_GLOBE_R2023A_54009_1000_V1_0.zip",
        note="Quick-check and sensitivity fallback population raster",
    ),
    ExpectedFile(
        dataset="GHSL",
        layer="built_surface",
        year=2020,
        resolution_m=100,
        path=GHSL_BUILT_DIR / "GHS_BUILT_S_E2020_GLOBE_R2023A_54009_100_V1_0.zip",
        note="Primary analysis built-surface raster",
    ),
    ExpectedFile(
        dataset="GHSL",
        layer="built_surface",
        year=2020,
        resolution_m=1000,
        path=GHSL_BUILT_DIR / "GHS_BUILT_S_E2020_GLOBE_R2023A_54009_1000_V1_0.zip",
        note="Quick-check and sensitivity fallback built-surface raster",
    ),
]


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def read_city_table() -> pd.DataFrame:
    if CITY_TABLE_CSV.exists():
        return pd.read_csv(CITY_TABLE_CSV)
    if CITY_TABLE_PARQUET.exists():
        return pd.read_parquet(CITY_TABLE_PARQUET)
    raise FileNotFoundError(f"Missing city table: {CITY_TABLE_CSV} or {CITY_TABLE_PARQUET}")


def list_zip_rasters(path: Path) -> list[str]:
    if not path.exists():
        return []
    with zipfile.ZipFile(path) as zf:
        return sorted(
            name
            for name in zf.namelist()
            if name.lower().endswith((".tif", ".tiff")) and not name.endswith("/")
        )


def build_local_inventory() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in EXPECTED_GHSL_FILES:
        rasters = list_zip_rasters(spec.path)
        rows.append(
            {
                "dataset": spec.dataset,
                "layer": spec.layer,
                "year": spec.year,
                "resolution_m": spec.resolution_m,
                "iso3": "",
                "path": rel(spec.path),
                "exists": spec.path.exists(),
                "bytes": file_size(spec.path),
                "archive_raster_count": len(rasters),
                "archive_first_raster": rasters[0] if rasters else "",
                "status": "ready" if spec.path.exists() and rasters else "missing_or_invalid",
                "note": spec.note,
            }
        )

    for tif in sorted(WORLDPOP_DIR.glob("*/*_ppp_2020.tif")):
        iso3 = tif.parent.name.upper()
        rows.append(
            {
                "dataset": "WorldPop",
                "layer": "population",
                "year": 2020,
                "resolution_m": None,
                "iso3": iso3,
                "path": rel(tif),
                "exists": tif.exists(),
                "bytes": file_size(tif),
                "archive_raster_count": "",
                "archive_first_raster": "",
                "status": "ready" if tif.exists() else "missing",
                "note": "National population raster for GHSL population sensitivity checks or fallback use",
            }
        )

    return pd.DataFrame(rows)


def summarize_inventory(inventory: pd.DataFrame | None = None) -> pd.DataFrame:
    if inventory is None:
        inventory = build_local_inventory()
    if inventory.empty:
        return pd.DataFrame(columns=["dataset", "layer", "file_count", "ready_count", "total_gib"])
    grouped = (
        inventory.groupby(["dataset", "layer"], dropna=False)
        .agg(
            file_count=("path", "count"),
            ready_count=("status", lambda values: int((values == "ready").sum())),
            total_gib=("bytes", lambda values: round(float(values.sum()) / 1024**3, 3)),
        )
        .reset_index()
    )
    return grouped.sort_values(["dataset", "layer"]).reset_index(drop=True)


def expected_worldpop_iso3() -> set[str]:
    city_table = read_city_table()
    return {str(value).upper() for value in city_table["iso3"].dropna().unique()}


def available_worldpop_iso3() -> set[str]:
    return {path.parent.name.upper() for path in WORLDPOP_DIR.glob("*/*_ppp_2020.tif")}


def validate_worldpop_coverage() -> pd.DataFrame:
    expected = expected_worldpop_iso3()
    available = available_worldpop_iso3()
    rows = []
    for iso3 in sorted(expected | available):
        rows.append(
            {
                "iso3": iso3,
                "expected_by_city_table": iso3 in expected,
                "worldpop_tif_exists": iso3 in available,
                "status": "ready" if iso3 in expected and iso3 in available else "extra_or_missing",
                "path": rel(WORLDPOP_DIR / iso3 / f"{iso3.lower()}_ppp_2020.tif"),
            }
        )
    return pd.DataFrame(rows)


def build_city_processing_plan() -> pd.DataFrame:
    cities = read_city_table()
    coverage = validate_worldpop_coverage().set_index("iso3")
    rows: list[dict[str, Any]] = []
    ghsl_ready = all(row.path.exists() and list_zip_rasters(row.path) for row in EXPECTED_GHSL_FILES)
    for _, city in cities.iterrows():
        iso3 = str(city["iso3"]).upper()
        worldpop_path = WORLDPOP_DIR / iso3 / f"{iso3.lower()}_ppp_2020.tif"
        worldpop_ready = bool(coverage.loc[iso3, "worldpop_tif_exists"]) if iso3 in coverage.index else False
        rows.append(
            {
                "city_id": city["city_id"],
                "city_name_en": city["city_name_en"],
                "iso3": iso3,
                "region": city.get("region", ""),
                "sample_group": city.get("sample_group", ""),
                "city_boundary": rel(CITY_BOUNDARY_GPKG),
                "ghsl_pop_100m_zip": rel(EXPECTED_GHSL_FILES[0].path),
                "ghsl_built_100m_zip": rel(EXPECTED_GHSL_FILES[2].path),
                "worldpop_2020_tif": rel(worldpop_path),
                "status": "ready_zonal_stats" if ghsl_ready and worldpop_ready and CITY_BOUNDARY_GPKG.exists() else "blocked_missing_input",
                "dependency_reason": "" if ghsl_ready and worldpop_ready and CITY_BOUNDARY_GPKG.exists() else "Missing city boundary, GHSL zip, or WorldPop tif for this city's ISO3",
            }
        )
    return pd.DataFrame(rows)


def rasterio_module():
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("rasterio is required to read raster metadata and run zonal stats.") from exc
    return rasterio


def rasterio_zip_uri(zip_path: Path, inner_raster: str) -> str:
    return f"zip://{zip_path}!{inner_raster}"


def collect_raster_metadata(sample_worldpop: int = 5, include_ghsl: bool = True) -> pd.DataFrame:
    rasterio = rasterio_module()
    inventory = build_local_inventory()
    rows: list[dict[str, Any]] = []

    candidates = []
    if include_ghsl:
        for _, row in inventory[inventory["dataset"].eq("GHSL") & inventory["status"].eq("ready")].iterrows():
            candidates.append((row, rasterio_zip_uri(ROOT / row["path"], row["archive_first_raster"])))
    for _, row in inventory[inventory["dataset"].eq("WorldPop") & inventory["status"].eq("ready")].head(sample_worldpop).iterrows():
        candidates.append((row, str(ROOT / row["path"])))

    for row, raster_path in candidates:
        with rasterio.open(raster_path) as src:
            rows.append(
                {
                    "dataset": row["dataset"],
                    "layer": row["layer"],
                    "iso3": row["iso3"],
                    "source_path": row["path"],
                    "driver": src.driver,
                    "crs": str(src.crs),
                    "width": src.width,
                    "height": src.height,
                    "count": src.count,
                    "dtype": src.dtypes[0] if src.dtypes else "",
                    "nodata": src.nodata,
                    "bounds": json.dumps(tuple(round(v, 6) for v in src.bounds), ensure_ascii=False),
                    "transform": str(src.transform),
                }
            )
    return pd.DataFrame(rows)


def geopandas_module():
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise RuntimeError("geopandas is required to read city boundaries and run zonal stats.") from exc
    return gpd


def clean_geometry(geometry):
    from shapely.validation import make_valid

    if geometry is None or geometry.is_empty:
        return geometry
    return make_valid(geometry) if not geometry.is_valid else geometry


def load_city_boundaries():
    gpd = geopandas_module()
    if not CITY_BOUNDARY_GPKG.exists():
        raise FileNotFoundError(f"Missing city boundary file: {CITY_BOUNDARY_GPKG}")
    cities = gpd.read_file(CITY_BOUNDARY_GPKG)
    if cities.crs is None:
        cities = cities.set_crs("EPSG:4326")
    cities = cities[cities.geometry.notna() & ~cities.geometry.is_empty].copy()
    cities["geometry"] = cities.geometry.apply(clean_geometry)
    required = {"city_id", "city_name_en", "iso3", "geometry"}
    missing = sorted(required - set(cities.columns))
    if missing:
        raise ValueError(f"City boundaries are missing required fields: {missing}")
    return cities


def city_base_table(cities) -> pd.DataFrame:
    keep_cols = [
        "city_id",
        "city_name_en",
        "country",
        "iso3",
        "region",
        "sample_group",
        "morphology_prior",
        "ucdb_id",
        "ucdb_name_main",
        "ucdb_area_km2",
        "ucdb_pop_2025",
    ]
    available = [col for col in keep_cols if col in cities.columns]
    base = pd.DataFrame(cities.drop(columns="geometry", errors="ignore"))[available].copy()
    equal_area = cities.to_crs("EPSG:6933")
    base["city_area_km2_calc"] = equal_area.geometry.area.to_numpy() / 1_000_000
    return base


def ghsl_raster_sources(include_1000m: bool = True) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for spec in EXPECTED_GHSL_FILES:
        if spec.resolution_m == 1000 and not include_1000m:
            continue
        rasters = list_zip_rasters(spec.path)
        if not spec.path.exists() or not rasters:
            continue
        layer_slug = "pop" if spec.layer == "population" else "built_s"
        sources.append(
            {
                "dataset": spec.dataset,
                "layer": spec.layer,
                "year": spec.year,
                "resolution_m": spec.resolution_m,
                "prefix": f"ghsl_{layer_slug}_{spec.year}_{spec.resolution_m}m",
                "path": spec.path,
                "raster_path": rasterio_zip_uri(spec.path, rasters[0]),
            }
        )
    return sources


def array_stats(masked_array: np.ma.MaskedArray) -> dict[str, Any]:
    arr = np.ma.masked_invalid(masked_array.astype("float64", copy=False))
    valid = arr.compressed()
    if valid.size == 0:
        return {
            "sum": np.nan,
            "mean": np.nan,
            "min": np.nan,
            "max": np.nan,
            "valid_pixel_count": 0,
            "masked_pixel_count": int(arr.size),
        }
    return {
        "sum": float(valid.sum(dtype="float64")),
        "mean": float(valid.mean(dtype="float64")),
        "min": float(valid.min()),
        "max": float(valid.max()),
        "valid_pixel_count": int(valid.size),
        "masked_pixel_count": int(arr.size - valid.size),
    }


def zonal_stats_for_open_raster(src, cities, prefix: str, source_info: dict[str, Any], all_touched: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    import rasterio.mask
    from shapely.geometry import mapping

    projected = cities.to_crs(src.crs) if str(cities.crs) != str(src.crs) else cities
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    for _, city in projected.iterrows():
        city_id = city["city_id"]
        row: dict[str, Any] = {"city_id": city_id}
        diagnostic = {
            "city_id": city_id,
            "dataset": source_info["dataset"],
            "layer": source_info["layer"],
            "year": source_info["year"],
            "resolution_m": source_info.get("resolution_m"),
            "prefix": prefix,
            "source_path": source_info["source_path"],
            "status": "ok",
            "error": "",
        }
        try:
            data, _ = rasterio.mask.mask(
                src,
                [mapping(city.geometry)],
                crop=True,
                filled=False,
                indexes=1,
                all_touched=all_touched,
            )
            arr = np.ma.array(data, copy=False)
            if arr.ndim == 3:
                arr = arr[0]
            if src.nodata is not None:
                arr = np.ma.masked_equal(arr, src.nodata, copy=False)
            stats = array_stats(arr)
            for key, value in stats.items():
                row[f"{prefix}_{key}"] = value
                diagnostic[key] = value
        except ValueError as exc:
            row[f"{prefix}_sum"] = np.nan
            row[f"{prefix}_mean"] = np.nan
            row[f"{prefix}_min"] = np.nan
            row[f"{prefix}_max"] = np.nan
            row[f"{prefix}_valid_pixel_count"] = 0
            row[f"{prefix}_masked_pixel_count"] = 0
            diagnostic.update(status="no_overlap", error=str(exc), sum=np.nan, valid_pixel_count=0)
        rows.append(row)
        diagnostics.append(diagnostic)

    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def compute_ghsl_zonal_stats(cities, include_1000m: bool, all_touched: bool) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    rasterio = rasterio_module()
    stats_tables: list[pd.DataFrame] = []
    diagnostics: list[pd.DataFrame] = []
    for source in ghsl_raster_sources(include_1000m=include_1000m):
        source_info = {
            "dataset": source["dataset"],
            "layer": source["layer"],
            "year": source["year"],
            "resolution_m": source["resolution_m"],
            "source_path": rel(source["path"]),
        }
        with rasterio.open(source["raster_path"]) as src:
            stats, diag = zonal_stats_for_open_raster(
                src=src,
                cities=cities,
                prefix=source["prefix"],
                source_info=source_info,
                all_touched=all_touched,
            )
        stats_tables.append(stats)
        diagnostics.append(diag)
    return stats_tables, diagnostics


def compute_worldpop_zonal_stats(cities, all_touched: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    rasterio = rasterio_module()
    stats_tables: list[pd.DataFrame] = []
    diagnostics: list[pd.DataFrame] = []
    prefix = "worldpop_pop_2020"

    for iso3, group in cities.groupby(cities["iso3"].astype(str).str.upper(), sort=True):
        tif = WORLDPOP_DIR / iso3 / f"{iso3.lower()}_ppp_2020.tif"
        source_info = {
            "dataset": "WorldPop",
            "layer": "population",
            "year": 2020,
            "resolution_m": None,
            "source_path": rel(tif),
        }
        if not tif.exists():
            missing = pd.DataFrame(
                {
                    "city_id": group["city_id"].to_list(),
                    f"{prefix}_sum": np.nan,
                    f"{prefix}_mean": np.nan,
                    f"{prefix}_min": np.nan,
                    f"{prefix}_max": np.nan,
                    f"{prefix}_valid_pixel_count": 0,
                    f"{prefix}_masked_pixel_count": 0,
                }
            )
            diag = pd.DataFrame(
                {
                    "city_id": group["city_id"].to_list(),
                    "dataset": "WorldPop",
                    "layer": "population",
                    "year": 2020,
                    "resolution_m": None,
                    "prefix": prefix,
                    "source_path": rel(tif),
                    "status": "missing_raster",
                    "error": f"Missing WorldPop tif for ISO3={iso3}",
                    "sum": np.nan,
                    "valid_pixel_count": 0,
                }
            )
            stats_tables.append(missing)
            diagnostics.append(diag)
            continue
        with rasterio.open(tif) as src:
            stats, diag = zonal_stats_for_open_raster(
                src=src,
                cities=group,
                prefix=prefix,
                source_info=source_info,
                all_touched=all_touched,
            )
        stats_tables.append(stats)
        diagnostics.append(diag)

    stats_all = pd.concat(stats_tables, ignore_index=True) if stats_tables else pd.DataFrame(columns=["city_id"])
    diag_all = pd.concat(diagnostics, ignore_index=True) if diagnostics else pd.DataFrame(columns=["city_id"])
    return stats_all, diag_all


def add_derived_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    area = out["city_area_km2_calc"].replace({0: np.nan})
    if "ghsl_pop_2020_100m_sum" in out:
        out["ghsl_pop_density_2020_100m_per_km2"] = out["ghsl_pop_2020_100m_sum"] / area
    if "ghsl_pop_2020_1000m_sum" in out:
        out["ghsl_pop_density_2020_1000m_per_km2"] = out["ghsl_pop_2020_1000m_sum"] / area
    if "worldpop_pop_2020_sum" in out:
        out["worldpop_pop_density_2020_per_km2"] = out["worldpop_pop_2020_sum"] / area
    if "ghsl_built_s_2020_100m_sum" in out:
        out["ghsl_built_share_2020_100m"] = out["ghsl_built_s_2020_100m_sum"] / (area * 1_000_000)
    if "ghsl_built_s_2020_1000m_sum" in out:
        out["ghsl_built_share_2020_1000m"] = out["ghsl_built_s_2020_1000m_sum"] / (area * 1_000_000)
    if {"worldpop_pop_2020_sum", "ghsl_pop_2020_100m_sum"}.issubset(out.columns):
        denominator = out["ghsl_pop_2020_100m_sum"].replace({0: np.nan})
        out["worldpop_to_ghsl_pop_2020_100m_ratio"] = out["worldpop_pop_2020_sum"] / denominator
    if {"ghsl_pop_2020_100m_sum", "ucdb_pop_2025"}.issubset(out.columns):
        denominator = out["ucdb_pop_2025"].replace({0: np.nan})
        out["ghsl_pop_2020_100m_to_ucdb_pop_2025_ratio"] = out["ghsl_pop_2020_100m_sum"] / denominator
    return out


def compute_city_population_environment_metrics(include_1000m: bool = True, all_touched: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    cities = load_city_boundaries()
    base = city_base_table(cities)
    ghsl_stats, ghsl_diag = compute_ghsl_zonal_stats(cities, include_1000m=include_1000m, all_touched=all_touched)
    worldpop_stats, worldpop_diag = compute_worldpop_zonal_stats(cities, all_touched=all_touched)
    stats_tables = ghsl_stats + [worldpop_stats]
    metrics = reduce(lambda left, right: left.merge(right, on="city_id", how="left"), [base] + stats_tables)
    metrics = add_derived_metrics(metrics)
    diagnostics = pd.concat(ghsl_diag + [worldpop_diag], ignore_index=True)
    diagnostics["all_touched"] = all_touched
    return metrics, diagnostics


def write_zonal_stats_outputs(include_1000m: bool = True, all_touched: bool = False) -> dict[str, Any]:
    STEP_DIR.mkdir(parents=True, exist_ok=True)
    metrics, diagnostics = compute_city_population_environment_metrics(
        include_1000m=include_1000m,
        all_touched=all_touched,
    )
    metrics.to_parquet(CITY_METRICS_PARQUET, index=False)
    metrics.to_csv(CITY_METRICS_CSV, index=False)
    diagnostics.to_csv(ZONAL_DIAGNOSTICS_TABLE, index=False)
    log = {
        "generated_at": utc_now(),
        "city_count": int(len(metrics)),
        "diagnostic_rows": int(len(diagnostics)),
        "include_1000m": include_1000m,
        "all_touched": all_touched,
        "outputs": {
            "metrics_parquet": rel(CITY_METRICS_PARQUET),
            "metrics_csv": rel(CITY_METRICS_CSV),
            "diagnostics_csv": rel(ZONAL_DIAGNOSTICS_TABLE),
        },
        "diagnostic_status_counts": diagnostics["status"].value_counts(dropna=False).to_dict(),
    }
    ZONAL_RUN_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return log


def write_prepare_outputs() -> dict[str, str]:
    STEP_DIR.mkdir(parents=True, exist_ok=True)
    inventory = build_local_inventory()
    plan = build_city_processing_plan()
    inventory.to_csv(INVENTORY_TABLE, index=False)
    plan.to_csv(PROCESSING_PLAN_TABLE, index=False)
    log = {
        "generated_at": utc_now(),
        "inventory_rows": int(len(inventory)),
        "plan_rows": int(len(plan)),
        "inventory_summary": summarize_inventory(inventory).to_dict(orient="records"),
        "plan_status_counts": plan["status"].value_counts(dropna=False).to_dict(),
    }
    RUN_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "inventory_table": rel(INVENTORY_TABLE),
        "processing_plan_table": rel(PROCESSING_PLAN_TABLE),
        "run_log": rel(RUN_LOG),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check and process the downloaded Step 03 population and built-environment raster data.")
    parser.add_argument("--prepare-only", action="store_true", help="Write the local data inventory and city processing plan.")
    parser.add_argument("--run-zonal-stats", action="store_true", help="Run GHSL/WorldPop zonal stats for the 86 city boundaries.")
    parser.add_argument("--skip-1000m", action="store_true", help="Compute only the 100 m GHSL primary metrics and skip the 1000 m GHSL comparison metrics.")
    parser.add_argument("--all-touched", action="store_true", help="Run zonal stats with all_touched=True; by default only pixels whose centers fall inside the boundary are counted.")
    parser.add_argument("--metadata-sample", type=int, default=0, help="Read raster metadata for this many WorldPop samples.")
    args = parser.parse_args()

    inventory = build_local_inventory()
    plan = build_city_processing_plan()
    print(summarize_inventory(inventory).to_string(index=False))
    print(plan["status"].value_counts(dropna=False).to_string())
    if args.prepare_only:
        print(write_prepare_outputs())
    if args.run_zonal_stats:
        print(
            json.dumps(
                write_zonal_stats_outputs(
                    include_1000m=not args.skip_1000m,
                    all_touched=args.all_touched,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
    if args.metadata_sample:
        print(collect_raster_metadata(sample_worldpop=args.metadata_sample).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
