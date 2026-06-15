from __future__ import annotations

import argparse
import csv
import json
import os
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import osmnx as ox
import osmium
from pyproj import Geod
from shapely.geometry import LineString
from shapely.prepared import prep


ROOT = Path(__file__).resolve().parents[2]
STEP_DIR = ROOT / "data/02_download_road_network_data"
CITY_BOUNDARIES = ROOT / "data/01_city_boundaries_and_sample_inventory/city_sample/city_boundaries.gpkg"
PBF_DIR = STEP_DIR / "OSM_snapshots"
TASK_TABLE = STEP_DIR / "road_network_build_task_table.csv"
STATUS_TABLE = STEP_DIR / "road_network_build_status_table.csv"
EVENT_LOG = STEP_DIR / "run_log/road_network_build_events.jsonl"
TASK_LOG_DIR = STEP_DIR / "run_log/task_logs"
GRAPHML_DIR = STEP_DIR / "city_road_network_graphml"
GPKG_DIR = STEP_DIR / "city_road_network_gpkg"
TMP_DIR = Path(os.environ.get("OSM_ROAD_BUILD_TMPDIR", ROOT / "tmp" / "osmium")).resolve()

NETWORK_TYPES = ("drive", "walk")
DEFAULT_BUFFER_METERS = float(os.environ.get("OSM_ROAD_BUILD_BUFFER_METERS", "500"))
MAX_WORKERS = int(os.environ.get("OSM_ROAD_BUILD_MAX_WORKERS", "1"))
LOCATION_INDEX = os.environ.get("OSM_ROAD_BUILD_LOCATION_INDEX", "sparse_file_array")

TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(TMP_DIR)
os.environ["TEMP"] = str(TMP_DIR)
os.environ["TMP"] = str(TMP_DIR)

DRIVE_EXCLUDED = {
    "abandoned",
    "bridleway",
    "bus_guideway",
    "construction",
    "corridor",
    "cycleway",
    "elevator",
    "escalator",
    "footway",
    "path",
    "pedestrian",
    "planned",
    "platform",
    "proposed",
    "raceway",
    "steps",
    "track",
    "via_ferrata",
}
WALK_EXCLUDED = {
    "abandoned",
    "bus_guideway",
    "construction",
    "motor",
    "motorway",
    "motorway_link",
    "planned",
    "proposed",
    "raceway",
    "trunk",
    "trunk_link",
    "via_ferrata",
}
NO_ACCESS = {"no", "private", "agricultural", "forestry", "delivery", "customers"}
WALK_YES = {"yes", "designated", "permissive", "official"}
ONEWAY_YES = {"yes", "true", "1"}


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def write_event(event: dict[str, Any]) -> None:
    EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with EVENT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def log_line(task_id: str, text: str) -> None:
    TASK_LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (TASK_LOG_DIR / f"{task_id}.log").open("a", encoding="utf-8") as f:
        f.write(f"[{utc_now()}] {text}\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def buffer_geometry_meters(geom, meters: float):
    if not meters:
        return geom
    series = gpd.GeoSeries([geom], crs="EPSG:4326")
    local_crs = series.estimate_utm_crs()
    if local_crs is None:
        local_crs = "EPSG:3857"
    return series.to_crs(local_crs).buffer(meters).to_crs("EPSG:4326").iloc[0]


def task_fields() -> list[str]:
    return [
        "task_id",
        "city_id",
        "city_name_en",
        "iso3",
        "geofabrik_extract_id",
        "network_type",
        "pbf_path",
        "output_graphml",
        "output_gpkg",
        "status",
        "dependency_reason",
        "buffer_meters",
        "created_at",
    ]


def status_fields() -> list[str]:
    return [
        "task_id",
        "city_id",
        "city_name_en",
        "network_type",
        "status",
        "started_at",
        "finished_at",
        "node_count",
        "edge_count",
        "output_graphml",
        "output_gpkg",
        "error",
    ]


def prepare_tasks(force: bool = True) -> list[dict[str, Any]]:
    cities = gpd.read_file(CITY_BOUNDARIES)
    rows: list[dict[str, Any]] = []
    created_at = utc_now()
    for _, city in cities.iterrows():
        city_id = city["city_id"]
        extract_id = city["geofabrik_extract_id"]
        pbf_path = PBF_DIR / f"{extract_id}-latest.osm.pbf"
        for network_type in NETWORK_TYPES:
            graphml = GRAPHML_DIR / network_type / f"{city_id}_{network_type}.graphml"
            gpkg = GPKG_DIR / network_type / f"{city_id}_{network_type}.gpkg"
            status = "ready_build" if pbf_path.exists() else "blocked_missing_pbf"
            reason = "" if pbf_path.exists() else f"Missing local PBF: {rel(pbf_path)}"
            rows.append(
                {
                    "task_id": f"{city_id}_{network_type}",
                    "city_id": city_id,
                    "city_name_en": city["city_name_en"],
                    "iso3": city["iso3"],
                    "geofabrik_extract_id": extract_id,
                    "network_type": network_type,
                    "pbf_path": rel(pbf_path),
                    "output_graphml": rel(graphml),
                    "output_gpkg": rel(gpkg),
                    "status": status,
                    "dependency_reason": reason,
                    "buffer_meters": str(DEFAULT_BUFFER_METERS),
                    "created_at": created_at,
                }
            )
    if force or not TASK_TABLE.exists():
        write_csv(TASK_TABLE, rows, task_fields())
    return rows


def load_status_map() -> dict[str, dict[str, str]]:
    return {row["task_id"]: row for row in read_csv(STATUS_TABLE)}


def write_status_map(status_by_id: dict[str, dict[str, Any]]) -> None:
    rows = [status_by_id[k] for k in sorted(status_by_id)]
    write_csv(STATUS_TABLE, rows, status_fields())


def first_tag_value(value: str | None) -> str:
    if not value:
        return ""
    return value.split(";")[0].strip().lower()


def allowed_way(tags: dict[str, str], network_type: str) -> bool:
    highway = first_tag_value(tags.get("highway"))
    if not highway or tags.get("area") == "yes":
        return False
    if network_type == "drive":
        if highway in DRIVE_EXCLUDED:
            return False
        for key in ("access", "vehicle", "motor_vehicle", "motorcar"):
            if first_tag_value(tags.get(key)) in NO_ACCESS:
                return False
        return True
    if highway in WALK_EXCLUDED:
        return False
    foot = first_tag_value(tags.get("foot"))
    access = first_tag_value(tags.get("access"))
    if foot in {"no", "private"}:
        return False
    if access in NO_ACCESS and foot not in WALK_YES:
        return False
    return True


def direction(tags: dict[str, str], network_type: str) -> str:
    if network_type != "drive":
        return "both"
    oneway = first_tag_value(tags.get("oneway"))
    junction = first_tag_value(tags.get("junction"))
    if oneway == "-1":
        return "reverse"
    if oneway in ONEWAY_YES or junction == "roundabout":
        return "forward"
    return "both"


class TaskGraphBuilder:
    def __init__(self, task: dict[str, str], polygon) -> None:
        self.task = task
        self.polygon = polygon
        self.prepared = prep(polygon)
        self.bounds = polygon.bounds
        self.network_type = task["network_type"]
        self.geod = Geod(ellps="WGS84")
        self.nodes: dict[int, dict[str, float]] = {}
        self.edges: list[tuple[int, int, dict[str, Any]]] = []
        self.ways_kept = 0

    def way_bbox_intersects(self, way_bounds: tuple[float, float, float, float]) -> bool:
        minx, miny, maxx, maxy = way_bounds
        poly_minx, poly_miny, poly_maxx, poly_maxy = self.bounds
        return not (maxx < poly_minx or minx > poly_maxx or maxy < poly_miny or miny > poly_maxy)

    def add_way_segments(
        self,
        way_id: int,
        tags: dict[str, str],
        highway: str,
        way_direction: str,
        node_ids: list[int],
        coords: list[tuple[float, float]],
    ) -> None:
        kept_for_way = False
        poly_minx, poly_miny, poly_maxx, poly_maxy = self.bounds
        for idx in range(len(coords) - 1):
            a = coords[idx]
            b = coords[idx + 1]
            seg_minx = min(a[0], b[0])
            seg_maxx = max(a[0], b[0])
            seg_miny = min(a[1], b[1])
            seg_maxy = max(a[1], b[1])
            if seg_maxx < poly_minx or seg_minx > poly_maxx or seg_maxy < poly_miny or seg_miny > poly_maxy:
                continue
            segment = LineString([a, b])
            if segment.is_empty or not self.prepared.intersects(segment):
                continue
            length = float(self.geod.line_length([a[0], b[0]], [a[1], b[1]]))
            if length <= 0:
                continue
            u = node_ids[idx]
            v = node_ids[idx + 1]
            self.nodes[u] = {"x": a[0], "y": a[1]}
            self.nodes[v] = {"x": b[0], "y": b[1]}
            base_attrs = {
                "osmid": str(way_id),
                "highway": highway,
                "name": tags.get("name", ""),
                "service": tags.get("service", ""),
                "bridge": tags.get("bridge", ""),
                "tunnel": tags.get("tunnel", ""),
                "maxspeed": tags.get("maxspeed", ""),
                "oneway": way_direction != "both",
                "length": length,
                "geometry": segment,
            }
            if way_direction in {"forward", "both"}:
                self.edges.append((u, v, dict(base_attrs)))
            if way_direction in {"reverse", "both"}:
                reverse_attrs = dict(base_attrs)
                reverse_attrs["geometry"] = LineString([b, a])
                self.edges.append((v, u, reverse_attrs))
            kept_for_way = True
        if kept_for_way:
            self.ways_kept += 1

    def to_graph(self, ways_seen: int) -> nx.MultiDiGraph:
        graph_attrs = {
            "city_id": self.task["city_id"],
            "city_name_en": self.task["city_name_en"],
            "network_type": self.task["network_type"],
            "pbf_path": self.task["pbf_path"],
            "buffer_meters": self.task.get("buffer_meters") or str(DEFAULT_BUFFER_METERS),
            "crs": "EPSG:4326",
            "simplified": False,
            "created_at": utc_now(),
            "ways_seen": ways_seen,
            "ways_kept": self.ways_kept,
        }
        graph = nx.MultiDiGraph(**graph_attrs)
        for node_id, attrs in self.nodes.items():
            graph.add_node(node_id, **attrs)
        for u, v, attrs in self.edges:
            if u in self.nodes and v in self.nodes:
                graph.add_edge(u, v, **attrs)
        return graph


class ExtractRoadCollector(osmium.SimpleHandler):
    def __init__(self, builders: list[TaskGraphBuilder]) -> None:
        super().__init__()
        self.builders_by_type: dict[str, list[TaskGraphBuilder]] = {network_type: [] for network_type in NETWORK_TYPES}
        for builder in builders:
            self.builders_by_type[builder.network_type].append(builder)
        self.ways_seen = 0

    def way(self, way) -> None:
        self.ways_seen += 1
        tags = {tag.k: tag.v for tag in way.tags}
        allowed_networks = [network_type for network_type in NETWORK_TYPES if allowed_way(tags, network_type)]
        if not allowed_networks:
            return
        node_ids: list[int] = []
        coords: list[tuple[float, float]] = []
        try:
            for node in way.nodes:
                if not node.location.valid():
                    return
                node_ids.append(int(node.ref))
                coords.append((float(node.location.lon), float(node.location.lat)))
        except Exception:
            return
        if len(coords) < 2:
            return

        way_bounds = (
            min(x for x, _ in coords),
            min(y for _, y in coords),
            max(x for x, _ in coords),
            max(y for _, y in coords),
        )
        highway = first_tag_value(tags.get("highway"))
        for network_type in allowed_networks:
            way_direction = direction(tags, network_type)
            for builder in self.builders_by_type[network_type]:
                if builder.way_bbox_intersects(way_bounds):
                    builder.add_way_segments(int(way.id), tags, highway, way_direction, node_ids, coords)


def keep_largest_weak_component(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    if graph.number_of_nodes() == 0:
        return graph
    components = nx.weakly_connected_components(graph)
    largest = max(components, key=len)
    return graph.subgraph(largest).copy()


def stringify_complex_values(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    for col in out.columns:
        if col == out.geometry.name:
            continue
        if out[col].map(lambda v: isinstance(v, (list, tuple, set, dict))).any():
            out[col] = out[col].map(
                lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, tuple, set, dict)) else v
            )
    return out


def save_graph_outputs(graph: nx.MultiDiGraph, graphml_path: Path, gpkg_path: Path) -> None:
    graphml_path.parent.mkdir(parents=True, exist_ok=True)
    gpkg_path.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(graph, filepath=graphml_path)
    if gpkg_path.exists():
        gpkg_path.unlink()
    nodes, edges = ox.graph_to_gdfs(graph)
    stringify_complex_values(nodes).to_file(gpkg_path, layer="nodes", driver="GPKG")
    stringify_complex_values(edges).to_file(gpkg_path, layer="edges", driver="GPKG")


def initial_result(task: dict[str, str], status: str = "running") -> dict[str, Any]:
    task_id = task["task_id"]
    return {
        "task_id": task_id,
        "city_id": task["city_id"],
        "city_name_en": task["city_name_en"],
        "network_type": task["network_type"],
        "status": status,
        "started_at": utc_now(),
        "finished_at": "",
        "node_count": "",
        "edge_count": "",
        "output_graphml": task["output_graphml"],
        "output_gpkg": task["output_gpkg"],
        "error": "",
    }


def finish_builder(task: dict[str, str], builder: TaskGraphBuilder, ways_seen: int, result: dict[str, Any]) -> dict[str, Any]:
    task_id = task["task_id"]
    graphml_path = ROOT / task["output_graphml"]
    gpkg_path = ROOT / task["output_gpkg"]
    try:
        graph = builder.to_graph(ways_seen=ways_seen)
        if graph.number_of_edges() == 0:
            raise RuntimeError("No road edges intersected the city boundary buffer")
        graph = keep_largest_weak_component(graph)
        try:
            graph = ox.simplify_graph(graph)
        except Exception as exc:
            log_line(task_id, f"simplification skipped: {type(exc).__name__}: {exc}")
        save_graph_outputs(graph, graphml_path, gpkg_path)
        result.update(
            status="complete",
            finished_at=utc_now(),
            node_count=str(graph.number_of_nodes()),
            edge_count=str(graph.number_of_edges()),
        )
        write_event({**result, "event": "finish"})
        log_line(task_id, f"COMPLETE nodes={result['node_count']} edges={result['edge_count']}")
        return result
    except Exception as exc:
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        result.update(status="failed", finished_at=utc_now(), error=error)
        write_event({**result, "event": "finish"})
        log_line(task_id, f"FAILED {error}")
        return result


def build_extract_group(extract_id: str, tasks: list[dict[str, str]], city_geometries: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    pending_tasks: list[dict[str, str]] = []
    for task in tasks:
        result = initial_result(task)
        graphml_path = ROOT / task["output_graphml"]
        gpkg_path = ROOT / task["output_gpkg"]
        if graphml_path.exists() and gpkg_path.exists():
            result.update(status="already_complete", finished_at=utc_now())
            results.append(result)
            write_event({**result, "event": "finish"})
            log_line(task["task_id"], "already complete")
        else:
            pending_tasks.append(task)

    if not pending_tasks:
        return results

    pbf_path = ROOT / pending_tasks[0]["pbf_path"]
    if not pbf_path.exists():
        for task in pending_tasks:
            result = initial_result(task)
            error = f"Missing PBF: {task['pbf_path']}"
            result.update(status="failed", finished_at=utc_now(), error=error)
            results.append(result)
            write_event({**result, "event": "finish"})
            log_line(task["task_id"], f"FAILED {error}")
        return results

    builders: list[TaskGraphBuilder] = []
    running_results: dict[str, dict[str, Any]] = {}
    for task in pending_tasks:
        result = initial_result(task)
        running_results[task["task_id"]] = result
        write_event({**result, "event": "start", "extract_id": extract_id})
        log_line(task["task_id"], f"START {task['city_name_en']} {task['network_type']} from {task['pbf_path']}")
        geom = city_geometries[task["city_id"]]
        buffered = buffer_geometry_meters(geom, float(task.get("buffer_meters") or DEFAULT_BUFFER_METERS))
        builders.append(TaskGraphBuilder(task, buffered))

    try:
        log_line(f"extract_{extract_id}", f"SCAN START tasks={len(pending_tasks)} pbf={rel(pbf_path)}")
        collector = ExtractRoadCollector(builders)
        collector.apply_file(str(pbf_path), locations=True, idx=LOCATION_INDEX)
        log_line(f"extract_{extract_id}", f"SCAN FINISH ways_seen={collector.ways_seen}")
        for builder in builders:
            task = builder.task
            results.append(finish_builder(task, builder, collector.ways_seen, running_results[task["task_id"]]))
        return results
    except Exception as exc:
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        log_line(f"extract_{extract_id}", f"SCAN FAILED {error}")
        for task in pending_tasks:
            result = running_results[task["task_id"]]
            result.update(status="failed", finished_at=utc_now(), error=error)
            results.append(result)
            write_event({**result, "event": "finish"})
            log_line(task["task_id"], f"FAILED {error}")
        return results


def runnable_tasks(limit: int | None = None, task_id: str | None = None, failed_only: bool = False) -> list[dict[str, str]]:
    rows = read_csv(TASK_TABLE) or prepare_tasks(force=True)
    ready = [row for row in rows if row["status"] == "ready_build"]
    if failed_only:
        failed_ids = {row["task_id"] for row in read_csv(STATUS_TABLE) if row.get("status") == "failed"}
        ready = [row for row in ready if row["task_id"] in failed_ids]
    if task_id:
        ready = [row for row in ready if row["task_id"] == task_id]
    if limit is not None:
        ready = ready[:limit]
    return ready


def run_builds(limit: int | None = None, task_id: str | None = None, failed_only: bool = False) -> int:
    tasks = runnable_tasks(limit=limit, task_id=task_id, failed_only=failed_only)
    cities = gpd.read_file(CITY_BOUNDARIES)[["city_id", "geometry"]]
    city_geometries = dict(zip(cities["city_id"], cities.geometry))
    status_by_id = load_status_map()
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for task in tasks:
        groups[task["geofabrik_extract_id"]].append(task)
    write_event(
        {
            "event": "manager_start",
            "time": utc_now(),
            "task_count": len(tasks),
            "extract_count": len(groups),
            "max_workers": MAX_WORKERS,
            "location_index": LOCATION_INDEX,
            "tmp_dir": str(TMP_DIR),
            "failed_only": failed_only,
        }
    )
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [
            pool.submit(build_extract_group, extract_id, extract_tasks, city_geometries)
            for extract_id, extract_tasks in groups.items()
        ]
        for future in as_completed(futures):
            results = future.result()
            for result in results:
                status_by_id[result["task_id"]] = result
            write_status_map(status_by_id)
    write_event({"event": "manager_finish", "time": utc_now(), "task_count": len(tasks), "extract_count": len(groups)})
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build city road networks from local OSM PBF files.")
    parser.add_argument("--prepare-only", action="store_true", help="Only write the road-network build task table.")
    parser.add_argument("--limit", type=int, default=None, help="Run at most N ready tasks.")
    parser.add_argument("--task-id", default=None, help="Run a single task_id from the task table.")
    parser.add_argument("--failed-only", action="store_true", help="Run only tasks marked failed in the status table.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = prepare_tasks(force=True)
    print(f"prepared {len(tasks)} road-network build tasks at {TASK_TABLE}")
    if args.prepare_only:
        return 0
    return run_builds(limit=args.limit, task_id=args.task_id, failed_only=args.failed_only)


if __name__ == "__main__":
    raise SystemExit(main())
