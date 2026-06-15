#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import numbers
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import geopandas as gpd
import osmium
import pandas as pd
import requests
from overturemaps import core as overture
from shapely.geometry import LineString, Point, Polygon, mapping
from shapely.strtree import STRtree


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "04_download_facilities_quality_history_data"
CODE_ROOT = ROOT / "code_upload" / "04_download_facilities_quality_history_data"

CITY_BOUNDARY = ROOT / "data" / "01_city_boundaries_and_sample_list" / "city_sample" / "city_boundaries.gpkg"
CITY_MASTER = ROOT / "data" / "01_city_boundaries_and_sample_list" / "city_sample" / "city_master.csv"
PBF_MANIFEST = ROOT / "data" / "02_download_road_network_data" / "download_manifest" / "country_region_download_manifest.csv"
PBF_DIR = ROOT / "data" / "02_download_road_network_data" / "OSM_snapshots"

OVERTURE_DIR = DATA_ROOT / "OverturePlaces"
OSM_FACILITY_DIR = DATA_ROOT / "OSM_facilities"
OHSOME_DIR = DATA_ROOT / "ohsome"
OSM_HISTORY_DIR = DATA_ROOT / "OSM_history"
LOG_DIR = DATA_ROOT / "logs"
STATUS_DIR = DATA_ROOT / "status"

FACILITY_KEYS = [
    "amenity",
    "shop",
    "tourism",
    "leisure",
    "healthcare",
    "office",
    "craft",
    "public_transport",
    "emergency",
    "sport",
    "historic",
]

KEEP_TAGS = [
    "name",
    "name:en",
    "amenity",
    "shop",
    "tourism",
    "leisure",
    "healthcare",
    "office",
    "craft",
    "public_transport",
    "emergency",
    "sport",
    "historic",
    "opening_hours",
    "operator",
    "brand",
    "addr:housenumber",
    "addr:street",
    "addr:city",
]

OHSOME_ENDPOINT = "https://api.ohsome.org/v1/elements/count"
OHSOME_TIME = "2008-01-01/2026-01-01/P1Y"
OHSOME_INDICATORS = {
    "all_nodes": "type:node",
    "all_ways": "type:way",
    "road_ways": "type:way and highway=*",
    "building_ways": "type:way and building=*",
    "poi_nodes": "type:node and (amenity=* or shop=* or tourism=* or leisure=* or healthcare=* or office=* or craft=*)",
    "named_poi_nodes": "type:node and name=* and (amenity=* or shop=* or tourism=* or leisure=* or healthcare=* or office=* or craft=*)",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_dirs() -> None:
    for path in [
        DATA_ROOT,
        OVERTURE_DIR / "city_parquet",
        OSM_FACILITY_DIR / "extract_parquet",
        OHSOME_DIR / "raw",
        OSM_HISTORY_DIR,
        LOG_DIR,
        STATUS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-").lower()
    return text or "item"


def setup_logging(task: str) -> Path:
    ensure_dirs()
    log_path = LOG_DIR / f"{task}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logging.info("root=%s", ROOT)
    logging.info("data_root=%s", DATA_ROOT)
    return log_path


def read_status(path: Path, key: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    if key not in df.columns:
        return {}
    return {str(row[key]): row.to_dict() for _, row in df.iterrows()}


def write_status(path: Path, rows: dict[str, dict[str, Any]], key: str) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows.values())
    if key in df.columns:
        df = df.sort_values(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def load_cities(simplify_m: float | None = None) -> gpd.GeoDataFrame:
    cities = gpd.read_file(CITY_BOUNDARY)
    cities = cities[cities.geometry.notna()].copy()
    cities["geometry"] = cities.geometry.buffer(0)
    if cities.crs is None:
        cities = cities.set_crs("EPSG:4326")
    else:
        cities = cities.to_crs("EPSG:4326")
    if simplify_m and simplify_m > 0:
        work = cities.to_crs("EPSG:3857")
        work["geometry"] = work.geometry.simplify(simplify_m, preserve_topology=True).buffer(0)
        cities = work.to_crs("EPSG:4326")
    return cities


def write_run_manifest(task: str, extra: dict[str, Any] | None = None) -> None:
    ensure_dirs()
    payload = {
        "task": task,
        "started_or_updated_at_utc": utc_now(),
        "root": str(ROOT),
        "data_root": str(DATA_ROOT),
        "city_boundary": str(CITY_BOUNDARY),
    }
    if extra:
        payload.update(extra)
    with (STATUS_DIR / f"{task}_run_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def retry_sleep(attempt: int, base_seconds: float = 6.0, cap_seconds: float = 120.0) -> None:
    delay = min(cap_seconds, base_seconds * (2 ** max(0, attempt - 1)))
    delay = delay * (0.75 + 0.5 * min(1.0, (time.time() % 10) / 10.0))
    logging.info("sleeping %.1f seconds before retry", delay)
    time.sleep(delay)


def run_overture(max_workers: int = 2, force: bool = False) -> None:
    setup_logging("overture_places")
    write_run_manifest("overture_places", {"max_workers": max_workers})
    cities = load_cities()
    status_path = STATUS_DIR / "overture_places_status.csv"
    status = read_status(status_path, "city_id")
    lock = Lock()

    def update(city_id: str, row: dict[str, Any]) -> None:
        with lock:
            base = status.get(city_id, {"city_id": city_id})
            base.update(row)
            base["updated_at_utc"] = utc_now()
            status[city_id] = base
            write_status(status_path, status, "city_id")

    def process_city(row: pd.Series) -> None:
        city_id = str(row.city_id)
        city_name = str(row.city_name_en)
        out_path = OVERTURE_DIR / "city_parquet" / f"{city_id}_{slugify(city_name)}.parquet"
        prior = status.get(city_id, {})
        if not force and prior.get("status") == "success" and out_path.exists():
            logging.info("skip overture success city_id=%s", city_id)
            return
        bbox = tuple(float(x) for x in row.geometry.bounds)
        update(
            city_id,
            {
                "city_name_en": city_name,
                "status": "running",
                "output_path": str(out_path),
                "bbox": ",".join(f"{x:.7f}" for x in bbox),
            },
        )
        for attempt in range(1, 5):
            try:
                logging.info("overture start city_id=%s city=%s attempt=%s", city_id, city_name, attempt)
                places = overture.geodataframe(
                    "place",
                    bbox=bbox,
                    stac=True,
                    connect_timeout=30,
                    request_timeout=600,
                )
                if places.crs is None:
                    places = places.set_crs("EPSG:4326")
                else:
                    places = places.to_crs("EPSG:4326")
                places = places[places.geometry.notna()].copy()
                if len(places):
                    city_geom = row.geometry
                    places = places[places.geometry.intersects(city_geom)].copy()
                places["city_id"] = city_id
                places["city_name_en"] = city_name
                places["downloaded_at_utc"] = utc_now()
                if len(places):
                    places.to_parquet(out_path, index=False)
                elif out_path.exists():
                    out_path.unlink()
                update(
                    city_id,
                    {
                        "status": "success",
                        "record_count": int(len(places)),
                        "error": "",
                    },
                )
                logging.info("overture done city_id=%s records=%s", city_id, len(places))
                return
            except Exception as exc:
                logging.warning("overture failed city_id=%s attempt=%s error=%s", city_id, attempt, exc)
                update(
                    city_id,
                    {
                        "status": "retrying" if attempt < 4 else "failed",
                        "record_count": "",
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                if attempt < 4:
                    retry_sleep(attempt)
        logging.error("overture exhausted city_id=%s", city_id)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(process_city, row) for _, row in cities.iterrows()]
        for future in as_completed(futures):
            future.result()

    summarize_overture()


def summarize_overture() -> None:
    status_path = STATUS_DIR / "overture_places_status.csv"
    if status_path.exists():
        status = pd.read_csv(status_path)
        status.to_csv(OVERTURE_DIR / "overture_places_summary.csv", index=False)


def facility_tags(tags: Any) -> dict[str, str]:
    raw = {tag.k: tag.v for tag in tags}
    if not any(k in raw for k in FACILITY_KEYS):
        return {}
    kept = {k: raw[k] for k in KEEP_TAGS if k in raw}
    return kept


def primary_facility(tags: dict[str, str]) -> tuple[str, str]:
    for key in FACILITY_KEYS:
        if key in tags:
            return key, tags[key]
    return "", ""


class FacilityExtractor(osmium.SimpleHandler):
    def __init__(self, cities: gpd.GeoDataFrame):
        super().__init__()
        self.city_rows = cities.reset_index(drop=True)
        self.city_geoms = list(self.city_rows.geometry)
        self.tree = STRtree(self.city_geoms)
        self.geom_index = {id(geom): i for i, geom in enumerate(self.city_geoms)}
        self.records: list[dict[str, Any]] = []
        self.scanned_nodes = 0
        self.scanned_ways = 0
        self.facility_candidates = 0

    def _query_indices(self, point: Point) -> list[int]:
        raw = self.tree.query(point)
        indices: list[int] = []
        for item in raw:
            if isinstance(item, numbers.Integral):
                indices.append(int(item))
            else:
                found = self.geom_index.get(id(item))
                if found is not None:
                    indices.append(found)
        return indices

    def _matching_cities(self, point: Point) -> list[pd.Series]:
        matches: list[pd.Series] = []
        for idx in self._query_indices(point):
            geom = self.city_geoms[idx]
            if geom.covers(point):
                matches.append(self.city_rows.iloc[idx])
        return matches

    def _append_record(self, osm_type: str, osm_id: int, geom: Any, point: Point, tags: dict[str, str]) -> None:
        cities = self._matching_cities(point)
        if not cities:
            return
        primary_key, primary_value = primary_facility(tags)
        tag_json = json.dumps(tags, ensure_ascii=False, sort_keys=True)
        for city in cities:
            self.records.append(
                {
                    "city_id": str(city.city_id),
                    "city_name_en": str(city.city_name_en),
                    "iso3": str(city.iso3),
                    "geofabrik_extract_id": str(city.geofabrik_extract_id),
                    "osm_type": osm_type,
                    "osm_id": int(osm_id),
                    "primary_key": primary_key,
                    "primary_value": primary_value,
                    "name": tags.get("name", ""),
                    "name_en": tags.get("name:en", ""),
                    "all_tags_json": tag_json,
                    "geometry": geom,
                }
            )

    def node(self, node: Any) -> None:
        self.scanned_nodes += 1
        tags = facility_tags(node.tags)
        if not tags:
            return
        self.facility_candidates += 1
        try:
            point = Point(float(node.location.lon), float(node.location.lat))
        except Exception:
            return
        self._append_record("node", int(node.id), point, point, tags)

    def way(self, way: Any) -> None:
        self.scanned_ways += 1
        tags = facility_tags(way.tags)
        if not tags:
            return
        self.facility_candidates += 1
        try:
            coords = [(float(node.lon), float(node.lat)) for node in way.nodes]
        except Exception:
            return
        coords = [xy for xy in coords if all(math.isfinite(v) for v in xy)]
        if len(coords) < 2:
            return
        try:
            if way.is_closed() and len(coords) >= 4:
                geom = Polygon(coords)
                if not geom.is_valid:
                    geom = geom.buffer(0)
                if geom.is_empty:
                    return
                point = geom.representative_point()
            else:
                geom = LineString(coords)
                if geom.is_empty:
                    return
                point = geom.interpolate(0.5, normalized=True)
        except Exception:
            return
        self._append_record("way", int(way.id), geom, point, tags)


def run_osm_facilities(force: bool = False) -> None:
    setup_logging("osm_facilities")
    write_run_manifest("osm_facilities")
    cities = load_cities()
    manifest = pd.read_csv(PBF_MANIFEST)
    status_path = STATUS_DIR / "osm_facilities_status.csv"
    status = read_status(status_path, "extract_id")

    for _, extract in manifest.iterrows():
        extract_id = str(extract.extract_id)
        city_subset = cities[cities["geofabrik_extract_id"].astype(str).eq(extract_id)].copy()
        if city_subset.empty:
            continue
        pbf_path = PBF_DIR / f"{extract_id}-latest.osm.pbf"
        out_path = OSM_FACILITY_DIR / "extract_parquet" / f"{extract_id}_facilities.parquet"
        prior = status.get(extract_id, {})
        if not force and prior.get("status") == "success" and out_path.exists():
            logging.info("skip osm extract success extract_id=%s", extract_id)
            continue
        status[extract_id] = {
            **prior,
            "extract_id": extract_id,
            "extract_name": str(extract.extract_name),
            "city_count": int(len(city_subset)),
            "status": "running",
            "pbf_path": str(pbf_path),
            "output_path": str(out_path),
            "updated_at_utc": utc_now(),
        }
        write_status(status_path, status, "extract_id")
        if not pbf_path.exists():
            logging.error("missing pbf extract_id=%s path=%s", extract_id, pbf_path)
            status[extract_id].update({"status": "failed", "record_count": "", "error": "missing pbf"})
            write_status(status_path, status, "extract_id")
            continue
        try:
            logging.info("osm extract start extract_id=%s cities=%s", extract_id, len(city_subset))
            handler = FacilityExtractor(city_subset)
            pbf_size_gb = pbf_path.stat().st_size / (1024 ** 3)
            location_index = "sparse_file_array" if pbf_size_gb >= 2.0 else "flex_mem"
            logging.info(
                "osm extract location index extract_id=%s pbf_size_gb=%.2f idx=%s",
                extract_id,
                pbf_size_gb,
                location_index,
            )
            handler.apply_file(str(pbf_path), locations=True, idx=location_index)
            if handler.records:
                gdf = gpd.GeoDataFrame(handler.records, geometry="geometry", crs="EPSG:4326")
                gdf.to_parquet(out_path, index=False)
            elif out_path.exists():
                out_path.unlink()
            status[extract_id].update(
                {
                    "status": "success",
                    "record_count": int(len(handler.records)),
                    "facility_candidates": int(handler.facility_candidates),
                    "scanned_nodes": int(handler.scanned_nodes),
                    "scanned_ways": int(handler.scanned_ways),
                    "error": "",
                    "updated_at_utc": utc_now(),
                }
            )
            write_status(status_path, status, "extract_id")
            logging.info(
                "osm extract done extract_id=%s records=%s candidates=%s",
                extract_id,
                len(handler.records),
                handler.facility_candidates,
            )
        except Exception as exc:
            logging.error("osm extract failed extract_id=%s error=%s", extract_id, exc)
            logging.error(traceback.format_exc())
            status[extract_id].update(
                {
                    "status": "failed",
                    "record_count": "",
                    "error": f"{type(exc).__name__}: {exc}",
                    "updated_at_utc": utc_now(),
                }
            )
            write_status(status_path, status, "extract_id")

    summarize_osm_facilities()


def summarize_osm_facilities() -> None:
    status_path = STATUS_DIR / "osm_facilities_status.csv"
    if not status_path.exists():
        return
    status = pd.read_csv(status_path)
    status.to_csv(OSM_FACILITY_DIR / "osm_facilities_extract_summary.csv", index=False)

    frames = []
    for path in sorted((OSM_FACILITY_DIR / "extract_parquet").glob("*_facilities.parquet")):
        try:
            cols = ["city_id", "city_name_en", "primary_key", "primary_value", "osm_type"]
            frame = gpd.read_parquet(path, columns=cols)
            frames.append(frame)
        except Exception as exc:
            logging.warning("summary read failed path=%s error=%s", path, exc)
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        summary = (
            combined.groupby(["city_id", "city_name_en", "primary_key", "primary_value", "osm_type"], dropna=False)
            .size()
            .reset_index(name="feature_count")
            .sort_values(["city_id", "feature_count"], ascending=[True, False])
        )
        summary.to_csv(OSM_FACILITY_DIR / "osm_facilities_city_type_summary.csv", index=False)


def ohsome_payload(city: pd.Series, indicator: str, filter_query: str) -> dict[str, str]:
    feature = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "id": str(city.city_id),
                    "name": str(city.city_name_en),
                },
                "geometry": mapping(city.geometry),
            }
        ],
    }
    return {
        "bpolys": json.dumps(feature, ensure_ascii=False),
        "time": OHSOME_TIME,
        "filter": filter_query,
    }


def parse_ohsome_result(city: pd.Series, indicator: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in payload.get("result", []):
        rows.append(
            {
                "city_id": str(city.city_id),
                "city_name_en": str(city.city_name_en),
                "iso3": str(city.iso3),
                "indicator": indicator,
                "timestamp": item.get("timestamp", ""),
                "value": item.get("value", None),
            }
        )
    return rows


def run_ohsome(max_workers: int = 1, force: bool = False) -> None:
    setup_logging("ohsome_history")
    write_run_manifest("ohsome_history", {"time": OHSOME_TIME, "max_workers": max_workers})
    cities = load_cities(simplify_m=100)
    status_path = STATUS_DIR / "ohsome_history_status.csv"
    status = read_status(status_path, "task_id")
    result_rows: list[dict[str, Any]] = []
    lock = Lock()

    def update(task_id: str, row: dict[str, Any]) -> None:
        with lock:
            base = status.get(task_id, {"task_id": task_id})
            base.update(row)
            base["updated_at_utc"] = utc_now()
            status[task_id] = base
            write_status(status_path, status, "task_id")

    def process(city: pd.Series, indicator: str, filter_query: str) -> list[dict[str, Any]]:
        city_id = str(city.city_id)
        task_id = f"{city_id}__{indicator}"
        raw_path = OHSOME_DIR / "raw" / f"{task_id}.json"
        prior = status.get(task_id, {})
        if not force and prior.get("status") == "success" and raw_path.exists():
            try:
                payload = json.loads(raw_path.read_text(encoding="utf-8"))
                return parse_ohsome_result(city, indicator, payload)
            except Exception:
                pass
        update(
            task_id,
            {
                "city_id": city_id,
                "city_name_en": str(city.city_name_en),
                "indicator": indicator,
                "status": "running",
                "raw_path": str(raw_path),
                "time": OHSOME_TIME,
                "filter": filter_query,
            },
        )
        for attempt in range(1, 9):
            try:
                logging.info("ohsome start task=%s attempt=%s", task_id, attempt)
                response = requests.post(
                    OHSOME_ENDPOINT,
                    data=ohsome_payload(city, indicator, filter_query),
                    timeout=180,
                    headers={"User-Agent": "202606osm-step04/1.0"},
                )
                if response.status_code >= 500:
                    raise requests.HTTPError(f"{response.status_code}: {response.text[:500]}")
                if response.status_code >= 400:
                    raise requests.HTTPError(f"{response.status_code}: {response.text[:500]}")
                payload = response.json()
                raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                rows = parse_ohsome_result(city, indicator, payload)
                update(
                    task_id,
                    {
                        "status": "success",
                        "rows": int(len(rows)),
                        "error": "",
                    },
                )
                logging.info("ohsome done task=%s rows=%s", task_id, len(rows))
                time.sleep(0.5)
                return rows
            except Exception as exc:
                logging.warning("ohsome failed task=%s attempt=%s error=%s", task_id, attempt, exc)
                update(
                    task_id,
                    {
                        "status": "retrying" if attempt < 8 else "failed",
                        "rows": "",
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                if attempt < 8:
                    retry_sleep(attempt, base_seconds=10, cap_seconds=300)
        return []

    tasks = [(row, indicator, filter_query) for _, row in cities.iterrows() for indicator, filter_query in OHSOME_INDICATORS.items()]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(process, row, indicator, filter_query) for row, indicator, filter_query in tasks]
        for future in as_completed(futures):
            rows = future.result()
            with lock:
                result_rows.extend(rows)
                if result_rows:
                    out = pd.DataFrame(result_rows)
                    out.to_csv(OHSOME_DIR / "ohsome_quality_history_long.partial.csv", index=False)

    summarize_ohsome()


def summarize_ohsome() -> None:
    rows: list[dict[str, Any]] = []
    cities = load_cities(simplify_m=100)
    city_lookup = {str(row.city_id): row for _, row in cities.iterrows()}
    for raw_path in sorted((OHSOME_DIR / "raw").glob("*.json")):
        task = raw_path.stem
        if "__" not in task:
            continue
        city_id, indicator = task.split("__", 1)
        city = city_lookup.get(city_id)
        if city is None:
            continue
        try:
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            rows.extend(parse_ohsome_result(city, indicator, payload))
        except Exception as exc:
            logging.warning("ohsome summary read failed path=%s error=%s", raw_path, exc)
    if rows:
        out = pd.DataFrame(rows).sort_values(["city_id", "indicator", "timestamp"])
        out_path = OHSOME_DIR / "ohsome_quality_history_long.csv"
        out.to_csv(out_path, index=False)
        out.to_csv(OSM_HISTORY_DIR / "osm_history_annual_quality_long.csv", index=False)
        wide = out.pivot_table(index=["city_id", "city_name_en", "iso3", "timestamp"], columns="indicator", values="value", aggfunc="first").reset_index()
        wide.to_csv(OHSOME_DIR / "ohsome_quality_history_wide.csv", index=False)
        wide.to_csv(OSM_HISTORY_DIR / "osm_history_annual_quality_wide.csv", index=False)
    status_path = STATUS_DIR / "ohsome_history_status.csv"
    if status_path.exists():
        pd.read_csv(status_path).to_csv(OHSOME_DIR / "ohsome_history_status.csv", index=False)


def print_status() -> None:
    ensure_dirs()
    for name, path, key in [
        ("overture_places", STATUS_DIR / "overture_places_status.csv", "city_id"),
        ("osm_facilities", STATUS_DIR / "osm_facilities_status.csv", "extract_id"),
        ("ohsome_history", STATUS_DIR / "ohsome_history_status.csv", "task_id"),
    ]:
        print(f"\n{name}")
        if not path.exists():
            print("  no status file yet")
            continue
        df = pd.read_csv(path).fillna("")
        counts = df["status"].value_counts(dropna=False).to_dict() if "status" in df.columns else {}
        print(f"  rows={len(df)} status_counts={counts}")
        if "record_count" in df.columns:
            numeric = pd.to_numeric(df["record_count"], errors="coerce")
            print(f"  records={int(numeric.sum(skipna=True))}")
        if "rows" in df.columns:
            numeric = pd.to_numeric(df["rows"], errors="coerce")
            print(f"  result_rows={int(numeric.sum(skipna=True))}")
        failed = df[df.get("status", "").astype(str).eq("failed")] if "status" in df.columns else pd.DataFrame()
        if len(failed):
            cols = [c for c in [key, "city_id", "indicator", "error"] if c in failed.columns]
            print(failed[cols].head(10).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and extract Step 04 facility and quality-history datasets.")
    parser.add_argument(
        "--only",
        choices=["all", "overture", "osm_facilities", "ohsome", "status"],
        default="all",
        help="Task to run.",
    )
    parser.add_argument("--force", action="store_true", help="Re-run successful items.")
    parser.add_argument("--overture-workers", type=int, default=2)
    parser.add_argument("--ohsome-workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if args.only == "status":
        print_status()
        return
    if args.only in {"all", "overture"}:
        run_overture(max_workers=args.overture_workers, force=args.force)
    if args.only in {"all", "osm_facilities"}:
        run_osm_facilities(force=args.force)
    if args.only in {"all", "ohsome"}:
        run_ohsome(max_workers=args.ohsome_workers, force=args.force)


if __name__ == "__main__":
    main()
