from __future__ import annotations

import json
import math
import os
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS, Transformer
from rasterio import windows
from rasterio.features import rasterize
from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon
from shapely.ops import transform, unary_union
from shapely.prepared import prep
from shapely.validation import make_valid


ROOT = Path(__file__).resolve().parents[2]

CODE_DIR = ROOT / "code_upload/05_build_multiscale_spatial_units"
OUTPUT_DIR = ROOT / "data/05_build_multiscale_spatial_units"

INPUTS = {
    "city_boundaries": ROOT / "data/01_city_boundaries_and_sample_inventory/city_sample/city_boundaries.gpkg",
    "city_master": ROOT / "data/01_city_boundaries_and_sample_inventory/city_sample/city_master.csv",
    "city_metrics": ROOT / "data/03_download_population_built_environment_data/city_population_built_environment_metrics.csv",
    "ghsl_pop_100m_zip": ROOT
    / "data/03_download_population_built_environment_data/GHSL_population/GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0.zip",
    "ghsl_built_100m_zip": ROOT
    / "data/03_download_population_built_environment_data/GHSL_built_area/GHS_BUILT_S_E2020_GLOBE_R2023A_54009_100_V1_0.zip",
}

OUTPUTS = {
    "repaired_boundaries": OUTPUT_DIR / "city_boundaries_step05_repaired.gpkg",
    "full_city": OUTPUT_DIR / "full_city_spatial_units.gpkg",
    "core_5km": OUTPUT_DIR / "core_5km_spatial_units.gpkg",
    "hex_1km": OUTPUT_DIR / "hex_1km_spatial_units.gpkg",
    "hex_2km": OUTPUT_DIR / "hex_2km_spatial_units.gpkg",
    "unit_index_parquet": OUTPUT_DIR / "spatial_units_index.parquet",
    "unit_index_csv": OUTPUT_DIR / "spatial_units_index.csv",
    "quality_checks": OUTPUT_DIR / "spatial_units_quality_checks.csv",
    "build_log": OUTPUT_DIR / "run_log.json",
}

SCALE_FILE_KEYS = {
    "full_city": "full_city",
    "core_5km": "core_5km",
    "hex_1km": "hex_1km",
    "hex_2km": "hex_2km",
}

CORE_RADIUS_M = 5_000.0
HEX_SPECS = {
    "hex_1km": 1_000.0,
    "hex_2km": 2_000.0,
}
GEOGRAPHIC_CRS = "EPSG:4326"
RASTER_PIXEL_AREA_M2 = 10_000.0
MIN_HEX_UNIT_AREA_M2 = 1.0

REQUIRED_UNIT_COLUMNS = [
    "unit_id",
    "city_id",
    "city_name_en",
    "scale",
    "sample_group",
    "unit_area_km2",
    "city_area_km2",
    "valid_area_ratio",
    "edge_unit",
    "center_method",
    "population_weight",
    "built_share",
]


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _as_float(value: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _polygonal(geom):
    if geom is None or geom.is_empty:
        return Polygon()
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    if isinstance(geom, GeometryCollection):
        parts = []
        for part in geom.geoms:
            poly = _polygonal(part)
            if poly is not None and not poly.is_empty:
                parts.append(poly)
        if not parts:
            return Polygon()
        return unary_union(parts)
    return geom


def _repair_geometry(geom):
    if geom is None or geom.is_empty:
        return Polygon(), "empty_input"
    if geom.is_valid:
        return _polygonal(geom), "not_repaired"

    candidate = _polygonal(make_valid(geom))
    method = "make_valid"
    if candidate.is_empty or not candidate.is_valid:
        candidate = _polygonal(geom.buffer(0))
        method = "buffer0"
    if candidate.is_empty or not candidate.is_valid:
        candidate = _polygonal(make_valid(candidate))
        method = f"{method}+make_valid"
    return candidate, method


def _local_aeqd_crs(lon: float, lat: float) -> CRS:
    return CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat:.10f} +lon_0={lon:.10f} "
        "+datum=WGS84 +units=m +no_defs +type=crs"
    )


def _projectors(local_crs: CRS):
    to_local = Transformer.from_crs(GEOGRAPHIC_CRS, local_crs, always_xy=True).transform
    to_wgs84 = Transformer.from_crs(local_crs, GEOGRAPHIC_CRS, always_xy=True).transform
    return to_local, to_wgs84


def _hexagon(cx: float, cy: float, side_m: float) -> Polygon:
    # Pointy-top hexagons; flat-to-flat width is sqrt(3) * side_m.
    coords = []
    for i in range(6):
        angle = math.radians(30 + i * 60)
        coords.append((cx + side_m * math.cos(angle), cy + side_m * math.sin(angle)))
    return Polygon(coords)


def _make_hex_cells(city_geom_local, flat_to_flat_m: float):
    side_m = flat_to_flat_m / math.sqrt(3)
    dx = flat_to_flat_m
    dy = 1.5 * side_m
    full_hex_area_km2 = (3 * math.sqrt(3) / 2 * side_m**2) / 1_000_000
    minx, miny, maxx, maxy = city_geom_local.bounds
    prepared_city = prep(city_geom_local)

    records = []
    row = 0
    y = miny - 2 * side_m
    while y <= maxy + 2 * side_m:
        x = minx - dx + (row % 2) * dx / 2
        while x <= maxx + dx:
            hex_geom = _hexagon(x, y, side_m)
            if prepared_city.intersects(hex_geom):
                clipped = _polygonal(hex_geom.intersection(city_geom_local))
                if (
                    clipped is not None
                    and not clipped.is_empty
                    and clipped.area >= MIN_HEX_UNIT_AREA_M2
                ):
                    unit_area_km2 = clipped.area / 1_000_000
                    valid_area_ratio = unit_area_km2 / full_hex_area_km2
                    records.append(
                        {
                            "geometry_local": clipped,
                            "unit_area_km2": unit_area_km2,
                            "valid_area_ratio": min(max(valid_area_ratio, 0.0), 1.0),
                            "edge_unit": valid_area_ratio < 0.995,
                            "hex_full_area_km2": full_hex_area_km2,
                        }
                    )
            x += dx
        y += dy
        row += 1
    return records


def _base_unit_record(
    city: pd.Series,
    *,
    unit_id: str,
    scale: str,
    unit_area_km2: float,
    city_area_km2: float,
    valid_area_ratio: float,
    edge_unit: bool,
    center_method: str,
    center_lon: float,
    center_lat: float,
    city_pop_total: float,
    default_built_share: float,
    grid_nominal_m: float | None = None,
    hex_full_area_km2: float | None = None,
    core_radius_m: float | None = None,
) -> dict[str, Any]:
    if scale == "full_city":
        population_sum = city_pop_total
        population_weight = 1.0 if city_pop_total > 0 else np.nan
        population_method = "city_total_ghsl_100m_from_step03"
    else:
        population_sum = np.nan
        population_weight = np.nan
        population_method = "pending_ghsl_100m_windowed_rasterization"

    return {
        "unit_id": unit_id,
        "city_id": city["city_id"],
        "city_name_en": city["city_name_en"],
        "country": city.get("country"),
        "iso3": city.get("iso3"),
        "region": city.get("region"),
        "sample_group": city.get("sample_group"),
        "scale": scale,
        "unit_area_km2": unit_area_km2,
        "city_area_km2": city_area_km2,
        "valid_area_ratio": valid_area_ratio,
        "edge_unit": bool(edge_unit),
        "center_method": center_method,
        "center_lon": center_lon,
        "center_lat": center_lat,
        "grid_nominal_m": grid_nominal_m,
        "hex_full_area_km2": hex_full_area_km2,
        "core_radius_m": core_radius_m,
        "population_sum": population_sum,
        "population_weight": population_weight,
        "population_weight_method": population_method,
        "built_share": default_built_share,
        "built_share_method": "city_level_ghsl_100m_from_step03",
        "ghsl_pop_2020_100m_city_sum": city_pop_total,
        "raster_pop_valid_pixels": np.nan,
        "raster_built_valid_pixels": np.nan,
        "geometry_repaired": bool(city.get("geometry_repaired", False)),
        "repair_method": city.get("repair_method", "not_repaired"),
    }


def _ordered_unit_columns(gdf: gpd.GeoDataFrame) -> list[str]:
    preferred = REQUIRED_UNIT_COLUMNS + [
        "country",
        "iso3",
        "region",
        "center_lon",
        "center_lat",
        "grid_nominal_m",
        "hex_full_area_km2",
        "core_radius_m",
        "population_sum",
        "population_weight_method",
        "built_share_method",
        "ghsl_pop_2020_100m_city_sum",
        "raster_pop_valid_pixels",
        "raster_built_valid_pixels",
        "geometry_repaired",
        "repair_method",
    ]
    columns = [c for c in preferred if c in gdf.columns]
    columns.extend(c for c in gdf.columns if c not in columns and c != "geometry")
    columns.append("geometry")
    return columns


def read_and_repair_boundaries() -> gpd.GeoDataFrame:
    boundaries = gpd.read_file(INPUTS["city_boundaries"])
    if boundaries.crs is None:
        boundaries = boundaries.set_crs(GEOGRAPHIC_CRS)
    else:
        boundaries = boundaries.to_crs(GEOGRAPHIC_CRS)

    repaired_geoms = []
    repair_methods = []
    valid_before = []
    valid_after = []
    for geom in boundaries.geometry:
        valid_before.append(bool(geom is not None and geom.is_valid))
        repaired, method = _repair_geometry(geom)
        repaired_geoms.append(repaired)
        repair_methods.append(method)
        valid_after.append(bool(repaired is not None and repaired.is_valid and not repaired.is_empty))

    repaired_boundaries = boundaries.copy()
    repaired_boundaries["geometry_valid_before"] = valid_before
    repaired_boundaries["geometry_valid_after"] = valid_after
    repaired_boundaries["geometry_repaired"] = [
        (not before) or method != "not_repaired"
        for before, method in zip(valid_before, repair_methods)
    ]
    repaired_boundaries["repair_method"] = repair_methods
    repaired_boundaries = repaired_boundaries.set_geometry(gpd.GeoSeries(repaired_geoms, crs=GEOGRAPHIC_CRS))

    if not repaired_boundaries.geometry.is_valid.all():
        bad = repaired_boundaries.loc[~repaired_boundaries.geometry.is_valid, ["city_id", "city_name_en"]]
        raise ValueError(f"Geometry repair left invalid cities: {bad.to_dict(orient='records')}")
    return repaired_boundaries


def build_unit_geometries(
    boundaries: gpd.GeoDataFrame, metrics: pd.DataFrame
) -> dict[str, gpd.GeoDataFrame]:
    metrics_by_city = metrics.set_index("city_id").to_dict(orient="index")
    unit_records: dict[str, list[dict[str, Any]]] = {scale: [] for scale in SCALE_FILE_KEYS}
    unit_geometries: dict[str, list[Any]] = {scale: [] for scale in SCALE_FILE_KEYS}

    for _, city in boundaries.sort_values("city_id").iterrows():
        city_id = city["city_id"]
        city_metrics = metrics_by_city.get(city_id, {})
        city_pop_total = _as_float(city_metrics.get("ghsl_pop_2020_100m_sum"), 0.0)
        default_built_share = _as_float(city_metrics.get("ghsl_built_share_2020_100m"), np.nan)

        representative = city.geometry.representative_point()
        local_crs = _local_aeqd_crs(representative.x, representative.y)
        to_local, to_wgs84 = _projectors(local_crs)
        city_geom_local = _polygonal(transform(to_local, city.geometry))
        if city_geom_local.is_empty:
            raise ValueError(f"Empty projected city geometry after repair: {city_id}")

        city_area_km2 = city_geom_local.area / 1_000_000
        center_local = city_geom_local.centroid
        center_wgs84 = transform(to_wgs84, center_local)
        center_lon = float(center_wgs84.x)
        center_lat = float(center_wgs84.y)
        center_method = "projected_geometry_centroid_aeqd"

        full_record = _base_unit_record(
            city,
            unit_id=f"{city_id}_full_city",
            scale="full_city",
            unit_area_km2=city_area_km2,
            city_area_km2=city_area_km2,
            valid_area_ratio=1.0,
            edge_unit=False,
            center_method="not_applicable",
            center_lon=center_lon,
            center_lat=center_lat,
            city_pop_total=city_pop_total,
            default_built_share=default_built_share,
        )
        unit_records["full_city"].append(full_record)
        unit_geometries["full_city"].append(transform(to_wgs84, city_geom_local))

        core_local = _polygonal(center_local.buffer(CORE_RADIUS_M).intersection(city_geom_local))
        core_area_km2 = core_local.area / 1_000_000
        core_record = _base_unit_record(
            city,
            unit_id=f"{city_id}_core_5km",
            scale="core_5km",
            unit_area_km2=core_area_km2,
            city_area_km2=city_area_km2,
            valid_area_ratio=core_area_km2 / city_area_km2 if city_area_km2 > 0 else np.nan,
            edge_unit=core_area_km2 < (math.pi * CORE_RADIUS_M**2 / 1_000_000) * 0.995,
            center_method=center_method,
            center_lon=center_lon,
            center_lat=center_lat,
            city_pop_total=city_pop_total,
            default_built_share=default_built_share,
            core_radius_m=CORE_RADIUS_M,
        )
        unit_records["core_5km"].append(core_record)
        unit_geometries["core_5km"].append(transform(to_wgs84, core_local))

        for scale, flat_to_flat_m in HEX_SPECS.items():
            hex_cells = _make_hex_cells(city_geom_local, flat_to_flat_m)
            for seq, cell in enumerate(hex_cells, start=1):
                record = _base_unit_record(
                    city,
                    unit_id=f"{city_id}_{scale}_{seq:05d}",
                    scale=scale,
                    unit_area_km2=cell["unit_area_km2"],
                    city_area_km2=city_area_km2,
                    valid_area_ratio=cell["valid_area_ratio"],
                    edge_unit=cell["edge_unit"],
                    center_method="not_applicable",
                    center_lon=center_lon,
                    center_lat=center_lat,
                    city_pop_total=city_pop_total,
                    default_built_share=default_built_share,
                    grid_nominal_m=flat_to_flat_m,
                    hex_full_area_km2=cell["hex_full_area_km2"],
                )
                unit_records[scale].append(record)
                unit_geometries[scale].append(transform(to_wgs84, cell["geometry_local"]))

    gdfs = {}
    for scale in SCALE_FILE_KEYS:
        gdf = gpd.GeoDataFrame(
            unit_records[scale],
            geometry=gpd.GeoSeries(unit_geometries[scale], crs=GEOGRAPHIC_CRS),
            crs=GEOGRAPHIC_CRS,
        )
        gdfs[scale] = gdf[_ordered_unit_columns(gdf)]
    return gdfs


def _zip_tif_uri(zip_path: Path) -> str:
    stem = zip_path.name.replace(".zip", ".tif")
    return f"zip://{zip_path}!{stem}"


def _bounded_window(src, bounds, pad_m: float = 250.0):
    left, bottom, right, top = bounds
    win = windows.from_bounds(
        left - pad_m,
        bottom - pad_m,
        right + pad_m,
        top + pad_m,
        transform=src.transform,
    )
    col_start = max(0, math.floor(win.col_off))
    row_start = max(0, math.floor(win.row_off))
    col_stop = min(src.width, math.ceil(win.col_off + win.width))
    row_stop = min(src.height, math.ceil(win.row_off + win.height))
    if col_stop <= col_start or row_stop <= row_start:
        return None
    return windows.Window(
        col_off=col_start,
        row_off=row_start,
        width=col_stop - col_start,
        height=row_stop - row_start,
    )


def _zonal_sum_for_units(
    units_city: gpd.GeoDataFrame,
    raster_crs: CRS,
    out_shape: tuple[int, int],
    out_transform,
    pop_values: np.ndarray,
    pop_valid: np.ndarray,
    built_values: np.ndarray,
    built_valid: np.ndarray,
) -> pd.DataFrame:
    units_projected = units_city.to_crs(raster_crs)
    shapes = []
    label_to_unit_id: dict[int, str] = {}
    for label, (idx, row) in enumerate(units_projected.iterrows(), start=1):
        if row.geometry is not None and not row.geometry.is_empty:
            shapes.append((row.geometry, label))
            label_to_unit_id[label] = units_city.loc[idx, "unit_id"]

    if not shapes:
        return pd.DataFrame(columns=["unit_id", "population_sum", "raster_pop_valid_pixels"])

    labels = rasterize(
        shapes,
        out_shape=out_shape,
        transform=out_transform,
        fill=0,
        dtype="int32",
        all_touched=False,
    )
    flat_labels = labels.reshape(-1)
    max_label = len(shapes) + 1

    flat_pop_valid = pop_valid.reshape(-1) & (flat_labels > 0)
    pop_sums = np.bincount(
        flat_labels[flat_pop_valid],
        weights=pop_values.reshape(-1)[flat_pop_valid],
        minlength=max_label + 1,
    )
    pop_counts = np.bincount(flat_labels[flat_pop_valid], minlength=max_label + 1)

    flat_built_valid = built_valid.reshape(-1) & (flat_labels > 0)
    built_sums = np.bincount(
        flat_labels[flat_built_valid],
        weights=built_values.reshape(-1)[flat_built_valid],
        minlength=max_label + 1,
    )
    built_counts = np.bincount(flat_labels[flat_built_valid], minlength=max_label + 1)

    rows = []
    for label, unit_id in label_to_unit_id.items():
        built_count = int(built_counts[label]) if label < len(built_counts) else 0
        if built_count > 0:
            built_share = float(built_sums[label] / (built_count * RASTER_PIXEL_AREA_M2))
            built_share = min(max(built_share, 0.0), 1.0)
        else:
            built_share = np.nan
        rows.append(
            {
                "unit_id": unit_id,
                "population_sum": float(pop_sums[label]) if label < len(pop_sums) else 0.0,
                "raster_pop_valid_pixels": int(pop_counts[label]) if label < len(pop_counts) else 0,
                "built_share": built_share,
                "raster_built_valid_pixels": built_count,
            }
        )
    return pd.DataFrame(rows)


def attach_raster_metrics(
    gdfs: dict[str, gpd.GeoDataFrame],
    boundaries: gpd.GeoDataFrame,
    metrics: pd.DataFrame,
) -> tuple[dict[str, gpd.GeoDataFrame], dict[str, Any]]:
    start = time.perf_counter()
    metrics_by_city = metrics.set_index("city_id").to_dict(orient="index")
    pop_uri = _zip_tif_uri(INPUTS["ghsl_pop_100m_zip"])
    built_uri = _zip_tif_uri(INPUTS["ghsl_built_100m_zip"])
    city_status: list[dict[str, Any]] = []
    scales_for_raster = ["core_5km", "hex_1km", "hex_2km"]

    max_seconds_env = os.environ.get("STEP05_MAX_RASTER_SECONDS", "").strip()
    max_seconds = float(max_seconds_env) if max_seconds_env else None

    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(pop_uri) as pop_src, rasterio.open(built_uri) as built_src:
            raster_crs = pop_src.crs
            if built_src.crs != raster_crs or built_src.transform != pop_src.transform:
                raise ValueError("GHSL population and built-up rasters are not grid-aligned")

            boundaries_raster = boundaries.to_crs(raster_crs)
            for pos, (idx, city) in enumerate(boundaries_raster.sort_values("city_id").iterrows(), start=1):
                elapsed = time.perf_counter() - start
                if max_seconds is not None and elapsed > max_seconds:
                    raise TimeoutError(
                        f"STEP05_MAX_RASTER_SECONDS={max_seconds} exceeded after {pos - 1} cities"
                    )

                city_id = city["city_id"]
                print(
                    f"[05] GHSL 100m raster aggregation {pos:02d}/"
                    f"{len(boundaries_raster)} {city_id}",
                    flush=True,
                )
                window = _bounded_window(pop_src, city.geometry.bounds)
                if window is None:
                    city_status.append({"city_id": city_id, "raster_status": "empty_window"})
                    continue

                pop_masked = pop_src.read(1, window=window, masked=True)
                built_masked = built_src.read(1, window=window, masked=True)
                pop_values = np.asarray(np.ma.filled(pop_masked, 0.0), dtype="float64")
                built_values = np.asarray(np.ma.filled(built_masked, 0.0), dtype="float64")
                pop_valid = ~np.ma.getmaskarray(pop_masked)
                built_valid = ~np.ma.getmaskarray(built_masked)
                win_transform = pop_src.window_transform(window)
                out_shape = pop_values.shape

                city_pop_total = _as_float(
                    metrics_by_city.get(city_id, {}).get("ghsl_pop_2020_100m_sum"), 0.0
                )
                scale_status = {}
                for scale in scales_for_raster:
                    units_city = gdfs[scale].loc[gdfs[scale]["city_id"] == city_id]
                    stats = _zonal_sum_for_units(
                        units_city,
                        raster_crs,
                        out_shape,
                        win_transform,
                        pop_values,
                        pop_valid,
                        built_values,
                        built_valid,
                    )
                    stats_by_unit = stats.set_index("unit_id")
                    scale_pop_sum = float(stats["population_sum"].sum()) if len(stats) else 0.0
                    scale_status[f"{scale}_population_sum"] = scale_pop_sum
                    scale_status[f"{scale}_unit_count"] = int(len(units_city))

                    if len(stats_by_unit):
                        target_idx = units_city.index
                        aligned = stats_by_unit.reindex(gdfs[scale].loc[target_idx, "unit_id"])
                        pop_sum = aligned["population_sum"].to_numpy(dtype="float64")
                        gdfs[scale].loc[target_idx, "population_sum"] = pop_sum
                        if city_pop_total > 0:
                            gdfs[scale].loc[target_idx, "population_weight"] = (
                                pop_sum / city_pop_total
                            )
                        gdfs[scale].loc[target_idx, "population_weight_method"] = (
                            "ghsl_100m_windowed_rasterization"
                        )
                        gdfs[scale].loc[target_idx, "raster_pop_valid_pixels"] = aligned[
                            "raster_pop_valid_pixels"
                        ].to_numpy()
                        gdfs[scale].loc[target_idx, "raster_built_valid_pixels"] = aligned[
                            "raster_built_valid_pixels"
                        ].to_numpy()

                        built_share = aligned["built_share"]
                        built_has_value = built_share.notna().to_numpy()
                        if built_has_value.any():
                            built_idx = target_idx[built_has_value]
                            gdfs[scale].loc[built_idx, "built_share"] = built_share.loc[
                                built_has_value
                            ].to_numpy(dtype="float64")
                            gdfs[scale].loc[built_idx, "built_share_method"] = (
                                "ghsl_100m_windowed_rasterization"
                            )

                city_status.append(
                    {
                        "city_id": city_id,
                        "raster_status": "ok",
                        "raster_window_rows": int(out_shape[0]),
                        "raster_window_cols": int(out_shape[1]),
                        **scale_status,
                    }
                )

    return gdfs, {
        "population_weight_method": "ghsl_100m_windowed_rasterization",
        "built_share_method": "ghsl_100m_windowed_rasterization_with_city_level_fallback",
        "city_status": city_status,
        "elapsed_seconds": round(time.perf_counter() - start, 3),
    }


def fill_area_proxy_for_missing(
    gdfs: dict[str, gpd.GeoDataFrame], metrics: pd.DataFrame, reason: str
) -> dict[str, gpd.GeoDataFrame]:
    metrics_by_city = metrics.set_index("city_id").to_dict(orient="index")
    for scale, gdf in gdfs.items():
        missing = gdf["population_weight"].isna() | gdf["population_sum"].isna()
        if not missing.any():
            continue
        for city_id, city_units in gdf.loc[missing].groupby("city_id"):
            city_metrics = metrics_by_city.get(city_id, {})
            city_pop_total = _as_float(city_metrics.get("ghsl_pop_2020_100m_sum"), 0.0)
            default_built_share = _as_float(city_metrics.get("ghsl_built_share_2020_100m"), np.nan)
            selector = (gdf["city_id"] == city_id) & missing
            if scale == "full_city":
                weights = np.ones(selector.sum())
            else:
                city_area = float(gdf.loc[selector, "city_area_km2"].iloc[0])
                weights = gdf.loc[selector, "unit_area_km2"].to_numpy(dtype="float64") / city_area
            gdfs[scale].loc[selector, "population_weight"] = weights
            gdfs[scale].loc[selector, "population_sum"] = weights * city_pop_total
            gdfs[scale].loc[selector, "population_weight_method"] = (
                f"area_proxy_from_city_total_pop:{reason}"
            )
            gdfs[scale].loc[selector, "built_share"] = gdf.loc[selector, "built_share"].fillna(
                default_built_share
            )
            gdfs[scale].loc[selector, "built_share_method"] = (
                "city_level_ghsl_100m_from_step03"
            )
    return gdfs


def build_quality_checks(
    gdfs: dict[str, gpd.GeoDataFrame],
    boundaries: gpd.GeoDataFrame,
    raster_summary: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    boundary_meta = boundaries.set_index("city_id")[
        ["city_name_en", "sample_group", "geometry_repaired", "repair_method"]
    ].to_dict(orient="index")

    for city_id in sorted(boundary_meta):
        row = {
            "city_id": city_id,
            "city_name_en": boundary_meta[city_id]["city_name_en"],
            "sample_group": boundary_meta[city_id]["sample_group"],
            "geometry_repaired": bool(boundary_meta[city_id]["geometry_repaired"]),
            "repair_method": boundary_meta[city_id]["repair_method"],
            "population_weight_method": raster_summary.get("population_weight_method"),
            "built_share_method": raster_summary.get("built_share_method"),
        }
        city_area = np.nan
        for scale in ["full_city", "core_5km", "hex_1km", "hex_2km"]:
            subset = gdfs[scale].loc[gdfs[scale]["city_id"] == city_id]
            prefix = {
                "full_city": "full",
                "core_5km": "core",
                "hex_1km": "hex1",
                "hex_2km": "hex2",
            }[scale]
            if len(subset):
                if scale == "full_city":
                    city_area = float(subset["city_area_km2"].iloc[0])
                row[f"{prefix}_count"] = int(len(subset))
                row[f"{prefix}_area_km2"] = float(subset["unit_area_km2"].sum())
                row[f"{prefix}_area_coverage_ratio"] = (
                    float(subset["unit_area_km2"].sum() / city_area) if city_area > 0 else np.nan
                )
                row[f"{prefix}_edge_unit_count"] = int(subset["edge_unit"].sum())
                row[f"{prefix}_invalid_geometry_count"] = int((~subset.geometry.is_valid).sum())
                row[f"{prefix}_population_weight_sum"] = float(
                    subset["population_weight"].fillna(0).sum()
                )
                row[f"{prefix}_population_sum"] = float(subset["population_sum"].fillna(0).sum())
                row[f"{prefix}_built_share_mean"] = float(subset["built_share"].mean())
            else:
                row[f"{prefix}_count"] = 0
                row[f"{prefix}_area_km2"] = 0.0
                row[f"{prefix}_area_coverage_ratio"] = np.nan
                row[f"{prefix}_edge_unit_count"] = 0
                row[f"{prefix}_invalid_geometry_count"] = 0
                row[f"{prefix}_population_weight_sum"] = 0.0
                row[f"{prefix}_population_sum"] = 0.0
                row[f"{prefix}_built_share_mean"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def write_outputs(
    boundaries: gpd.GeoDataFrame,
    gdfs: dict[str, gpd.GeoDataFrame],
    quality_checks: pd.DataFrame,
    log: dict[str, Any],
) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in OUTPUTS.values():
        if path.exists():
            path.unlink()

    boundaries.to_file(OUTPUTS["repaired_boundaries"], driver="GPKG", layer="city_boundaries")
    for scale, file_key in SCALE_FILE_KEYS.items():
        gdf = gdfs[scale][_ordered_unit_columns(gdfs[scale])]
        gdf.to_file(OUTPUTS[file_key], driver="GPKG", layer=scale)

    index = pd.concat(
        [gdfs[scale].drop(columns="geometry").copy() for scale in SCALE_FILE_KEYS],
        ignore_index=True,
    )
    index.to_parquet(OUTPUTS["unit_index_parquet"], index=False)
    index.to_csv(OUTPUTS["unit_index_csv"], index=False)
    quality_checks.to_csv(OUTPUTS["quality_checks"], index=False)
    OUTPUTS["build_log"].write_text(
        json.dumps(log, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return index


def validate_outputs(
    gdfs: dict[str, gpd.GeoDataFrame],
    boundaries: gpd.GeoDataFrame,
    index: pd.DataFrame,
    quality_checks: pd.DataFrame,
) -> dict[str, Any]:
    expected_cities = set(boundaries["city_id"])
    checks = {
        "files_exist": {name: path.exists() for name, path in OUTPUTS.items()},
        "city_count_repaired_boundaries": int(boundaries["city_id"].nunique()),
        "all_repaired_geometries_valid": bool(boundaries.geometry.is_valid.all()),
        "index_rows": int(len(index)),
        "index_city_count": int(index["city_id"].nunique()),
        "quality_check_rows": int(len(quality_checks)),
        "quality_check_city_count": int(quality_checks["city_id"].nunique()),
        "scale_rows": {scale: int(len(gdf)) for scale, gdf in gdfs.items()},
        "scale_city_counts": {scale: int(gdf["city_id"].nunique()) for scale, gdf in gdfs.items()},
        "scale_geometries_valid": {scale: bool(gdf.geometry.is_valid.all()) for scale, gdf in gdfs.items()},
        "missing_required_columns": {
            scale: [c for c in REQUIRED_UNIT_COLUMNS if c not in gdf.columns]
            for scale, gdf in gdfs.items()
        },
        "city_coverage_ok": {
            scale: set(gdf["city_id"]) == expected_cities for scale, gdf in gdfs.items()
        },
    }
    checks["ok"] = (
        all(checks["files_exist"].values())
        and checks["city_count_repaired_boundaries"] == 86
        and checks["index_city_count"] == 86
        and checks["quality_check_city_count"] == 86
        and checks["all_repaired_geometries_valid"]
        and all(checks["scale_geometries_valid"].values())
        and all(not missing for missing in checks["missing_required_columns"].values())
        and all(checks["city_coverage_ok"].values())
    )
    return checks


def run_step05() -> dict[str, Any]:
    run_start = time.perf_counter()
    warnings.filterwarnings("ignore", message=".*normalized/laundered.*")

    missing_inputs = [str(path) for path in INPUTS.values() if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(f"Missing required inputs: {missing_inputs}")

    boundaries = read_and_repair_boundaries()
    city_master = pd.read_csv(INPUTS["city_master"])
    metrics = pd.read_csv(INPUTS["city_metrics"])

    gdfs = build_unit_geometries(boundaries, metrics)

    raster_summary: dict[str, Any]
    raster_error = None
    try:
        gdfs, raster_summary = attach_raster_metrics(gdfs, boundaries, metrics)
    except Exception as exc:
        raster_error = repr(exc)
        raster_summary = {
            "population_weight_method": "area_proxy_from_city_total_pop",
            "built_share_method": "city_level_ghsl_100m_from_step03",
            "city_status": [],
            "elapsed_seconds": None,
            "raster_error": raster_error,
        }

    reason = "raster_error" if raster_error else "missing_unit_raster_pixels"
    gdfs = fill_area_proxy_for_missing(gdfs, metrics, reason=reason)
    if raster_error is None:
        has_proxy = any(
            gdf["population_weight_method"].astype(str).str.startswith("area_proxy").any()
            for gdf in gdfs.values()
        )
        if has_proxy:
            raster_summary["population_weight_method"] = (
                "mixed_ghsl_100m_windowed_rasterization_and_area_proxy_for_missing_pixels"
            )

    repaired_cities = boundaries.loc[
        boundaries["geometry_repaired"],
        ["city_id", "city_name_en", "repair_method"],
    ].to_dict(orient="records")

    quality_checks = build_quality_checks(gdfs, boundaries, raster_summary)

    log = {
        "step": "05_build_multiscale_spatial_units",
        "started_at": _now(),
        "input_paths": {name: str(path) for name, path in INPUTS.items()},
        "output_paths": {name: str(path) for name, path in OUTPUTS.items()},
        "parameters": {
            "core_radius_m": CORE_RADIUS_M,
            "hex_grid": {
                "shape": "pointy_top_regular_hexagon",
                "nominal_dimension": "flat_to_flat_width_m",
                "hex_1km_flat_to_flat_m": HEX_SPECS["hex_1km"],
                "hex_2km_flat_to_flat_m": HEX_SPECS["hex_2km"],
                "edge_unit_threshold_valid_area_ratio": 0.995,
                "min_hex_unit_area_m2": MIN_HEX_UNIT_AREA_M2,
            },
            "core_center_method": "projected_geometry_centroid_aeqd",
            "geometry_repair": "shapely.make_valid_then_polygonal_extraction_with_buffer0_fallback",
            "population_weight_denominator": "city ghsl_pop_2020_100m_sum from step03 metrics",
            "raster_pixel_assignment": "pixel_center_all_touched_false",
        },
        "input_row_counts": {
            "city_boundaries": int(len(boundaries)),
            "city_master": int(len(city_master)),
            "city_metrics": int(len(metrics)),
        },
        "unit_counts": {scale: int(len(gdf)) for scale, gdf in gdfs.items()},
        "city_counts": {scale: int(gdf["city_id"].nunique()) for scale, gdf in gdfs.items()},
        "repaired_geometry_cities": repaired_cities,
        "population_weight_method": raster_summary.get("population_weight_method"),
        "built_share_method": raster_summary.get("built_share_method"),
        "raster_summary": raster_summary,
        "raster_error": raster_error,
    }

    index = write_outputs(boundaries, gdfs, quality_checks, log)
    validation = validate_outputs(gdfs, boundaries, index, quality_checks)

    log["finished_at"] = _now()
    log["elapsed_seconds_total"] = round(time.perf_counter() - run_start, 3)
    log["validation"] = validation
    OUTPUTS["build_log"].write_text(
        json.dumps(log, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "outputs": {name: str(path) for name, path in OUTPUTS.items()},
        "unit_counts": log["unit_counts"],
        "city_counts": log["city_counts"],
        "repaired_geometry_cities": repaired_cities,
        "population_weight_method": log["population_weight_method"],
        "built_share_method": log["built_share_method"],
        "validation": validation,
        "elapsed_seconds_total": log["elapsed_seconds_total"],
    }


if __name__ == "__main__":
    result = run_step05()
    print(json.dumps(result, ensure_ascii=False, indent=2))
