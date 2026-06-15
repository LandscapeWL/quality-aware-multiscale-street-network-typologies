#!/usr/bin/env python3
"""Step12: draw publication figures from stable Step09/10/11 outputs.

This step is deliberately figure-only.  It reads upstream stable outputs and
writes figure image files, figure source data, QC, lineage, logs, and README
records under ``data/12_draw_paper_figures``.  It does not generate Step13 tables,
appendices, variable dictionaries, method records, or final Excel packages.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import importlib.util
import json
import math
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Polygon, Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox

try:
    import geopandas as gpd

    HAVE_GEOPANDAS = True
except Exception:  # pragma: no cover - depends on local optional GIS stack.
    gpd = None
    HAVE_GEOPANDAS = False


STEP_NAME = "12_draw_paper_figures"
STEP01_NAME = "01_city_boundaries_and_sample_list"
STEP02_NAME = "02_download_road_network_data"
STEP06_NAME = "06_clean_road_network_and_compute_morphology_metrics"
STEP07_NAME = "07_generate_quality_scores_and_type_confidence"
STEP08_NAME = "08_generate_scale_signatures_and_city_profiles"
STEP09_NAME = "09_cluster_morphotypes"
STEP10_NAME = "10_compute_accessibility_and_detour_validation_metrics"
STEP11_NAME = "11_statistical_models_and_robustness_checks"
STEP13_NAME = "13_export_paper_tables_and_supplementary_materials"

MORPHOTYPES = [f"MT{i:02d}" for i in range(1, 11)]
FACILITY_CATEGORIES = ["grocery", "healthcare", "park", "school", "transit"]

MORPHOTYPE_LABELS_EN = {
    "MT01": "Dense fine-grain core",
    "MT02": "Dense ordered grid",
    "MT03": "Mid-density fine-grain",
    "MT04": "Mid-density cul-de-sac blocks",
    "MT05": "Mid-density long-grid corridors",
    "MT06": "Mid-density mixed fabric",
    "MT07": "Low-density curvilinear network",
    "MT08": "Low-density cul-de-sac fabric",
    "MT09": "Sparse peripheral cul-de-sac",
    "MT10": "Sparse curvilinear superblocks",
}

MORPHOTYPE_SHORT_LABELS = {
    mt: f"{mt}\n{label}" for mt, label in MORPHOTYPE_LABELS_EN.items()
}

CATEGORY_LABELS = {
    "grocery": "Grocery",
    "healthcare": "Healthcare",
    "park": "Park",
    "school": "School",
    "transit": "Transit",
}

SCALE_LABELS = {
    "core_5km": "Core 5 km",
    "hex_1km": "Hex 1 km",
    "hex_2km": "Hex 2 km",
    "full_city": "Full city",
}
SCALE_ORDER = {"core_5km": 0, "hex_1km": 1, "hex_2km": 2, "full_city": 3}

METRIC_LABELS = {
    "edge_density_km_per_km2": "Edge density",
    "intersection_density_per_km2": "Intersection density",
    "four_way_share": "Four-way share",
    "dead_end_share": "Dead-end share",
    "segment_length_median": "Median segment length",
    "edge_circuity_mean": "Edge circuity",
    "orientation_order": "Orientation order",
    "road_hierarchy_entropy": "Hierarchy entropy",
}
FIG4_METRICS = [
    "edge_density_km_per_km2",
    "intersection_density_per_km2",
    "four_way_share",
    "dead_end_share",
    "segment_length_median",
    "edge_circuity_mean",
]

SEQUENTIAL_CMAP = "viridis"
TOP_JOURNAL_DIVERGING_CMAP = "RdBu_r"
FIG5_HEATMAP_CMAP = TOP_JOURNAL_DIVERGING_CMAP
FIG5_STYLE_SEQUENTIAL_CMAP = "Fig5StyleSequentialBlueWhiteRed"
FIG5_STYLE_SEQUENTIAL_CMAP_SOURCE = f"sequential low-to-high adaptation of {FIG5_HEATMAP_CMAP}"
FIG5_STYLE_SEQUENTIAL_CMAP_COLORS = ("#2166AC", "#F7F7F7", "#B2182B")
FIG5_HEATMAP_VALUE_FONTSIZE = 6.7
FIG5_HEATMAP_TICK_FONTSIZE = 7.4
FIG5_HEATMAP_AXIS_LABEL_FONTSIZE = 7.8
FIG5_COLORBAR_LABEL_FONTSIZE = 7.3
FIG5_COLORBAR_TICK_FONTSIZE = 6.8
TOP_JOURNAL_SEQUENTIAL_CMAP = FIG5_STYLE_SEQUENTIAL_CMAP
FIG6_FIG7_HEATMAP_CMAP = FIG5_HEATMAP_CMAP
FIG7_HEATMAP_CMAP = FIG5_HEATMAP_CMAP
FIG7_COLOR_VARIABLE = "access_share_diff_pp"
FIG7_COLORBAR_LABEL_PREVIOUS = "Difference from category mean (percentage points)"
FIG7_COLORBAR_LABEL = "Difference from mean (pp)"
FIG7_COLORBAR_LABELPAD = 4
FIG7_SAVEFIG_PAD_INCHES = 0.18
FIG7_VECTOR_DRIFT_GUARD_NOTE = (
    "Fig8 SVG/PDF are non-target vector exports in the current Fig6C order-only fix. "
    "They are copied at Step12 start and restored after full redraw so Matplotlib "
    "timestamp/hashsalt metadata drift cannot advance during this guard run."
)
TYPE_CENTER_SCORE_LIMIT = 2.2
MORPHOTYPE_PALETTE_NAME = "Tableau 10 colorblind-friendly morphotype palette"
MORPHOTYPE_COLORS = {
    "MT01": "#4E79A7",
    "MT02": "#F28E2B",
    "MT03": "#59A14F",
    "MT04": "#E15759",
    "MT05": "#B07AA1",
    "MT06": "#76B7B2",
    "MT07": "#EDC948",
    "MT08": "#9C755F",
    "MT09": "#FF9DA7",
    "MT10": "#BAB0AC",
}
TYPE_COLORS = MORPHOTYPE_COLORS.copy()
QUALITY_TIER_COLORS_FOR_ANALYSIS = {
    "core": "#4E79A7",
    "sensitivity": "#F28E2B",
    "excluded_candidate": "#E15759",
    "review": "#B07AA1",
    "unknown": "#BAB0AC",
}
SEQUENTIAL_LINE_COLOR = "#31688E"
SEQUENTIAL_BAND_COLOR = "#35B779"
SEQUENTIAL_BOX_FACE_COLOR = "#E6F4D7"
FIG8_INTERVAL_PALETTE_NAME = "Fig8 neutral gray interval palette"
FIG8_INTERVAL_LINE_COLOR = "#3B4A6B"
FIG8_INTERVAL_RANGE_COLOR = "#B8B8B8"
FIG8_INTERVAL_CMAP = "Fig8NeutralGrayInterval"
FIG9_ZERO_REFERENCE_LINE_COLOR = "#CBD5E1"
FIG9_ZERO_REFERENCE_LINEWIDTH = 0.55
FIG9_ZERO_REFERENCE_ALPHA = 0.85
FIG8_GRAYSCALE_PALETTE_NAME = "Black-white-gray neutral boxplot palette"
FIG8_BOX_FACE_COLOR = "#D9D9D9"
FIG8_BOX_EDGE_COLOR = "#404040"
FIG8_MEDIAN_COLOR = "#000000"
FIG8_WHISKER_COLOR = "#595959"
FIG8_FLIER_FACE_COLOR = "#FFFFFF"
FIG8_GRID_COLOR = "#E6E6E6"
FIG8_NOTE_COLOR = "#4D4D4D"
FIG8_NOTE_Y = 0.018
FIG8_SUBPLOTS_BOTTOM = 0.30
REGION_COLORS = {
    "Africa": "#B56576",
    "Asia": "#2A9D8F",
    "China pressure test": "#D4A373",
    "Europe": "#457B9D",
    "Latin America": "#E76F51",
    "North America": "#6D597A",
    "Oceania": "#588157",
}
FIG2_QUALITY_TIER_COLORS = {
    "core": "#0072B2",
    "sensitivity": "#E69F00",
    "excluded_candidate": "#D55E00",
    "review": "#CC79A7",
    "unknown": "#94A3B8",
}
QUALITY_TIER_LABELS = {
    "core": "Core quality tier",
    "sensitivity": "Sensitivity tier",
    "excluded_candidate": "Excluded candidate",
    "review": "Review tier",
    "unknown": "Unknown tier",
}
SAMPLE_GROUP_MARKERS = {
    "main_80": "o",
    "china_pressure_test": "D",
    "unknown": "s",
}
SAMPLE_GROUP_LABELS = {
    "main_80": "Main sample",
    "china_pressure_test": "China pressure test",
    "unknown": "Unknown sample group",
}
FIG2_SIZE_VARIABLE = "ucdb_pop_2025"
FIG2_SIZE_VARIABLE_LABEL = "2025 UCDB population"
FIG2_POINT_SIZE_MIN = 10.0
FIG2_POINT_SIZE_MAX = 90.0
FIG2_ALIGNMENT_THRESHOLD = 1e-9
FIG2_CITY_NUMBER_RULE = "Sequential after Fig1 longitude/latitude filtering, preserving the current Fig1 source-data order."
FIG2_CITY_INDEX_COLUMNS = 4
FIG2_CITY_INDEX_INNER_PAD = 0.006
FIG2_CITY_LABEL_FONTSIZE = 5.4
FIG2_CITY_INDEX_TITLE_FONTSIZE = 8.1
FIG2_CITY_INDEX_ENTRY_FONTSIZE = 5.8
FIG2_MAP_BOTTOM_GAP_PREVIOUS = 0.080
FIG2_MAP_BOTTOM_GAP_CURRENT = 0.045
FIG2_LAYOUT_LEFT = 0.115
FIG2_LAYOUT_RIGHT = 0.965
FIG2_INDEX_Y = 0.030
FIG2_INDEX_H = 0.285
FIG2_BOTTOM_Y = 0.365
FIG2_BOTTOM_H = 0.190
FIG2_MAP_TOP = 0.950
FIG2_BOTTOM_HORIZONTAL_GAP = 0.055
FIG2_CITY_INDEX_RIGHT_NOTE_TEXT = "numbered in current Fig1 source-data order"
FIG3_QUALITY_SCORE_LABEL = "Quality score (0-100; higher = stronger OSM quality/confidence)"
FIG3_QUALITY_SCORE_DEFINITION = (
    "quality_score is a project-derived Step07 OSM data quality/confidence score, based on OSM road network "
    "integrity, historical maturity, POI completeness and local coverage, with population/built-up support; "
    "it is not an official OSM field and not a pure road-morphology score."
)
FIG3_QUALITY_TIER_COMMUNICATION_NOTE = (
    "Fig2 uses the score-by-tier panel plus thumbnail badges/strips to communicate quality tiers/groups; "
    "the world map is not drawn in Fig2 because Fig1 already shows the global sample map."
)
FIG3_NO_MAP_FIELDS_NOTE = (
    "Fig2 source data omit map-panel coordinate, marker-size/color and label-offset fields such as `map_lon`, "
    "`map_lat`, `point_size`, `point_color` and map label offsets."
)
FIG3_KEY_LABEL_BOTTOM_N = 6
FIG3_KEY_LABEL_TOP_N = 6
FIG3_SCORE_AXIS_PADDING = 5.0
FIG3_ROAD_THUMBNAIL_TARGET_COUNT = FIG3_KEY_LABEL_BOTTOM_N + FIG3_KEY_LABEL_TOP_N
FIG3_ROAD_THUMBNAIL_COLUMNS = 6
FIG3_ROAD_THUMBNAIL_FIXED_HALF_SIDE_KM = 5.0
FIG3_ROAD_THUMBNAIL_FIXED_SIDE_KM = FIG3_ROAD_THUMBNAIL_FIXED_HALF_SIDE_KM * 2.0
FIG3_ROAD_THUMBNAIL_BASE_HALF_SIDE_KM = FIG3_ROAD_THUMBNAIL_FIXED_HALF_SIDE_KM
FIG3_ROAD_THUMBNAIL_HALF_SIDE_KM_OPTIONS = (FIG3_ROAD_THUMBNAIL_FIXED_HALF_SIDE_KM,)
FIG3_ROAD_THUMBNAIL_SCALE_BAR_KM = 2.0
FIG3_ROAD_THUMBNAIL_RECENTER_MIN_IMPROVEMENT = 1.25
FIG3_ROAD_THUMBNAIL_MIN_EDGES = 300
FIG3_ROAD_THUMBNAIL_MAX_PLOTTED_EDGES = 1800
FIG3_ROAD_THUMBNAIL_COLUMN_GAP = 0.026
FIG3_ROAD_THUMBNAIL_ROW_GAP = 0.038
FIG3_ROAD_THUMBNAIL_ALIGNMENT_TOLERANCE = 5e-4
FIG3_VISUAL_ALIGNMENT_TOLERANCE = 0.003
FIG3_ROAD_THUMBNAIL_HEADER_OFFSET = 0.006
FIG3_ROAD_THUMBNAIL_BACKGROUND_COLOR = "#FFFFFF"
FIG3_ROAD_THUMBNAIL_ROAD_COLOR = "#000000"
FIG3_ROAD_THUMBNAIL_BORDER_COLOR = "#000000"
FIG3_ROAD_THUMBNAIL_GROUP_COLORS = {
    "low": "#1D4ED8",
    "high": "#B45309",
}
FIG3_ROAD_THUMBNAIL_SELECTION_RULE = (
    "Fig2 road thumbnails include the lowest six and highest six quality_score cities. Low ranks are assigned "
    "by ascending quality_score (Low 1 = lowest score; city_id breaks ties), and High ranks are assigned by "
    "descending quality_score (High 1 = highest score; city_id breaks ties). excluded_candidate status is "
    "recorded as an annotation without changing low/high rank order."
)
FIG3_QUALITY_COMPONENTS = [
    ("network_integrity_score", "Road network integrity"),
    ("historical_maturity_score", "Historical maturity"),
    ("poi_completeness_score", "POI completeness"),
    ("local_coverage_score", "Local coverage"),
    ("population_built_support_score", "Pop./built support"),
]
FIG3_QUALITY_COMPONENT_ABBREVIATIONS = {
    "network_integrity_score": "R",
    "historical_maturity_score": "H",
    "poi_completeness_score": "P",
    "local_coverage_score": "L",
    "population_built_support_score": "B",
}
FIG3_QUALITY_COMPONENT_STRIP_LEGEND = (
    "R road network integrity; H historical maturity; P POI completeness; "
    "L local coverage; B population/built support"
)
FIG3_SCORE_PANEL_PREVIOUS_HEIGHT = 0.255
FIG3_SCORE_PANEL_COMPACT_HEIGHT = 0.165
FIG3_COMPONENT_SUMMARY_PREVIOUS_BAR_HEIGHT = 0.038
FIG3_COMPONENT_SUMMARY_PREVIOUS_Y_SPAN = 0.380
FIG3_COMPONENT_SUMMARY_Y_TOP = 0.770
FIG3_COMPONENT_SUMMARY_Y_BOTTOM = 0.285
FIG3_COMPONENT_SUMMARY_BAR_HEIGHT = 0.060
FIG3_COMPONENT_SUMMARY_MIN_VERTICAL_GAP = 0.095
FIG3_COMPONENT_SUMMARY_AXIS_TICKS = (0.0, 50.0, 100.0)
FIG3_THUMBNAIL_PRIMARY_AREA_RATIO_MIN = 1.10
FIG3_COMPONENT_VISUAL_LEGEND_ITEMS = (
    ("p10_p90_distribution_band", "band", "p10-p90 band"),
    ("all_city_mean_dot", "dot", "All-city mean"),
    ("low_6_component_marker", "low_marker", "Low 6"),
    ("high_6_component_marker", "high_marker", "High 6"),
)
FIG3_COMPONENT_VISUAL_LEGEND_COMBINED_TEXTS = (
    "p10-90 band, mean, Low/High 6",
    "p10-p90 band, mean, Low/High 6",
    "Band=p10-90; dot=all-city mean; blue/orange markers=Low/High 6.",
    "Band=p10-p90; dot=all-city mean; blue/orange markers=Low/High 6.",
)
FIG3_ROAD_THUMBNAIL_VERTICAL_ANCHOR = "top"
FIG3_QUALITY_TO_THUMBNAIL_VERTICAL_GAP_MAX = 0.115
FIG3_QUALITY_TO_THUMBNAIL_VERTICAL_GAP_MIN_REDUCTION = 0.045
FIG3_QUALITY_THUMBNAIL_VISIBLE_GAP_MIN = 0.006

OUTPUT_FILES = {
    "input_report_json": "step12_input_acceptance_report.json",
    "input_report_md": "step12_input_acceptance_report.md",
    "qc": "step12_quality_checks.csv",
    "manifest": "figure_manifest.csv",
    "lineage": "figure_data_lineage.csv",
    "run_log": "run_log.json",
    "repro": "reproducibility_manifest.json",
    "readme": "README.md",
}

FIG2_CANONICAL_SLUG = "global_sample_cities_quality_tiers"
FIG4_CANONICAL_SLUG = "multiscale_network_signatures"
FIG6_RADAR_SLUG = "morphotype_radar_signatures"
FIG7_CANONICAL_SLUG = "scale_network_transition_matrix"
FIG8_CORE_CITY_SLUG = "core_city_morphotype_stability_and_performance"
FIG9_ACCESS_DETOUR_SLUG = "accessibility_detour_performance_by_morphotype"
FIG10_OBSERVED_GROUP_MEANS_SLUG = "observed_weighted_group_means"
FIG11_CITY_INEQUALITY_SLUG = "city_level_accessibility_inequality"
FIG12_MODEL_COEFFICIENT_SLUG = "model_coefficient_forest"
FIG6_PANEL_COMPOSITION = "composition"
FIG6_PANEL_TRANSITION = "transition"
FIG6_PANEL_PERFORMANCE = "performance_deviation"
FIG6_PANEL_C_SOURCE_NOTE = "Former access-performance deviation heatmap migrated to current Fig6C."
FIG6_MORPHOTYPE_AXIS_ORDER_TEXT = "MT01->MT10"
FIG6_PANEL_C_MORPHOTYPE_AXIS_ORDER_NOTE = (
    "Fig6C morphotype x-axis order is aligned with Fig6A/Fig6B as MT01->MT10; "
    "former access-heatmap morphotype_rank is retained only as source metric metadata."
)
FIG6_LAYOUT_GRID_SPEC = "GridSpec(3,2): vertically stacked A/B/C heatmaps with one narrow right colorbar per panel"
FIG6_LAYOUT_CHOICE_NOTE = (
    "Fig6 uses a vertical A/B/C stack to keep all heatmap rows legible: A composition, B transition, "
    "C former access-performance deviation heatmap. Each panel has its own right-side colorbar and "
    "panel-specific normalization."
)
FIG6_OVERLAP_CONTROL_NOTE = (
    "Separate GridSpec rows and dedicated right-side colorbar axes are used for A/B/C; hspace separates "
    "titles, tick labels and colorbars, and bbox_inches='tight' is used for PNG/SVG/PDF export."
)
FIG6_PANEL_A_NORM_VMIN = 0.0
FIG6_PANEL_A_NORM_VMAX_FLOOR = 25.0
FIG6_PANEL_A_NORM_VMAX_DATA_FACTOR = 1.05
FIG6_TOP_CELL_COUNT = 5
FIG6_REMOVED_SUPTITLE_TEXT = "Scale/network and performance heatmap overview"
FIG6_REMOVED_BOTTOM_NOTE_TEXT = (
    "A: n_units-weighted morphotype composition. B: target minus source share. "
    "C: former access-share deviation from facility-category mean. "
    f"Thin outlines mark top-{FIG6_TOP_CELL_COUNT} A/B cells."
)
FIG6_TITLE_FOOTNOTE_REMOVAL_NOTE = (
    "Fig6 rendered canvas removes the figure-level top title and bottom footnote/note; "
    "panel A/B/C titles, axis labels, tick labels, colorbar labels and in-cell heatmap labels remain."
)
FIG6_TOP_CELL_OUTLINE_LINEWIDTH = 1.1
FIG6_PANEL_A_TOP_CELL_OUTLINE_COLOR = "#111827"
FIG6_PANEL_B_POSITIVE_OUTLINE_COLOR = "#991B1B"
FIG6_PANEL_B_NEGATIVE_OUTLINE_COLOR = "#1D4ED8"
FIG12_HEATMAP_CMAP = "Fig7NeutralSequential"
FIG12_HEATMAP_CMAP_COLORS = ("#F8FAFC", "#CBD5E1", "#475569")
FIG12_SORTING_RULE = "Core drive cities sorted by hex_1km dominant_share descending, with city_name_en as the stable tie-breaker."
FIG12_DOMINANT_SHARE_PRIORITY = (
    "dominant_population_share; fallback to max population_share__MTxx; fallback to max unit_share__MTxx"
)
FIG8_FIGURE_HEIGHT_CM = 21.4
FIG8_LEFT_MARGIN = 0.285
FIG8_MAIN_BOTTOM = 0.168
FIG8_COLORBAR_Y = 0.130
FIG8_SHAPE_LEGEND_Y = 0.086
FIG8_MORPHOTYPE_LEGEND_Y = 0.050
FIG8_CITY_LABEL_FONTSIZE = 5.8
FIG8_AXIS_TICK_FONTSIZE = 6.8
FIG8_TITLE_FONTSIZE = 8.6
FIG8_STRIP_LABEL_FONTSIZE = 6.8
FIG8_COLORBAR_LABEL_FONTSIZE = 6.3
FIG8_COLORBAR_TICK_FONTSIZE = 6.1
FIG8_SHAPE_LEGEND_FONTSIZE = 6.8
FIG8_MORPHOTYPE_LEGEND_FONTSIZE = 6.3
FIG8_MORPHOTYPE_LEGEND_TITLE_FONTSIZE = 6.7
FIG12_METRIC_DIRECTIONS = {
    "entropy_norm": "higher entropy means more mixed morphotype composition",
    "access_norm": "higher access_metric is better",
    "circuity_norm": "higher circuity_metric is worse",
    "inequality_norm": "higher inequality_metric is worse",
    "quality_norm": "higher quality_score is better",
}
FIG7_PANEL_ACCESS = "access_panel"
FIG7_PANEL_DETOUR = "detour_circuity_panel"
FIG7_PANEL_QUADRANT = "quadrant_panel"
FIG7_THRESHOLD_METHOD = "median split across MT01-MT10 using morphotype_mean_access_share and weighted_circuity"
FIG7_LAYOUT_WIDTH_RATIOS_PREVIOUS = {"single_grid_2x4": [3.65, 0.15, 0.90, 3.00]}
FIG7_LAYOUT_WIDTH_RATIOS = {
    "top_panel_a_colorbar": [1.0, 0.035],
    "bottom_panel_b_c": [1.0, 1.02],
}
FIG7_LAYOUT_HEIGHT_RATIOS = [0.95, 1.05]
FIG7_LAYOUT_WSPACE_PREVIOUS = 0.12
FIG7_LAYOUT_TOP_WSPACE = 0.040
FIG7_LAYOUT_BOTTOM_WSPACE = 0.300
FIG7_LAYOUT_WSPACE = FIG7_LAYOUT_BOTTOM_WSPACE
FIG7_LAYOUT_HSPACE_PREVIOUS = 0.34
FIG7_LAYOUT_HSPACE = 0.42
FIG7_CURRENT_VISUAL_PANELS = ("A", "B")
FIG7_PANEL_A_REMOVAL_NOTE = "Former access heatmap is migrated to Fig6C; current Fig8 renders only panels A and B."
FIG7_REMOVED_BOTTOM_NOTE_TEXT = (
    "Panel A shows p10-p90 detour intervals. Panel B uses median-split access and circuity thresholds. "
    "Former access heatmap is now Fig6C."
)
FIG7_FOOTNOTE_REMOVAL_NOTE = (
    "Fig8 rendered canvas removes the bottom footnote/note; panel A/B titles, axis labels, tick labels, "
    "morphotype labels, marker labels and leader lines remain."
)
FIG7_BC_LAYOUT_GRID_SPEC = "GridSpec(1,2): current Fig8 panels A and B only"
FIG7_PANEL_A_CELL_SHARE_FONTSIZE = 5.8
FIG7_PANEL_A_CELL_DIFF_FONTSIZE = 4.4
FIG7_DETOUR_VALUE_LABEL_FONTSIZE = 5.0
FIG7_QUADRANT_LABEL_FONTSIZE = 4.8
FIG7_PANEL_C_MARKER_SIZE_PREVIOUS = 42.0
FIG7_PANEL_C_MARKER_SIZE = 30.0
FIG7_PANEL_C_MARKER_SIZE_REDUCTION_PERCENT = (
    (FIG7_PANEL_C_MARKER_SIZE_PREVIOUS - FIG7_PANEL_C_MARKER_SIZE) / FIG7_PANEL_C_MARKER_SIZE_PREVIOUS * 100.0
)
FIG7_PANEL_C_MARKER_SIZE_UNITS = "matplotlib scatter marker area, points^2"
FIG7_QUADRANT_LABEL_OFFSETS = {
    "MT01": (-10, 10, "right"),
    "MT02": (7, -1, "left"),
    "MT03": (9, 5, "left"),
    "MT04": (-11, -3, "right"),
    "MT05": (7, -12, "left"),
    "MT06": (-26, -7, "right"),
    "MT07": (5, -6, "left"),
    "MT08": (5, 6, "left"),
    "MT09": (5, 6, "left"),
    "MT10": (5, 0, "left"),
}
FIG7_PANEL_C_LABEL_OFFSET_REVISIONS = {
    "MT03": {
        "previous": (-20, 9, "right"),
        "current": FIG7_QUADRANT_LABEL_OFFSETS["MT03"],
        "reason": "right-side leader to avoid the neighboring MT04/MT06/MT02 cluster",
    },
    "MT04": {
        "previous": (-26, 7, "right"),
        "current": FIG7_QUADRANT_LABEL_OFFSETS["MT04"],
        "reason": "shorter and lower leader to reduce upward reach",
    },
}
FIG7_QUADRANT_LEADER_LINE_LABELS = ("MT01", "MT03", "MT04", "MT05", "MT06", "MT07", "MT08", "MT09")
FIG7_QUADRANT_LEADER_LINE_STYLE = {
    "arrowstyle": "-",
    "color": "#94A3B8",
    "linewidth": 0.45,
    "alpha": 0.78,
    "shrinkA": 1.4,
    "shrinkB": 3.0,
    "zorder": 4.5,
}
FIG7_QUADRANT_LEADER_LINE_CONNECTIONSTYLE = "arc3,rad=0"
FIG7_PANEL_C_LABEL_ADJUSTMENT_NOTE = (
    "Current Panel B label-offset refinement: MT03 offset changed from (-20, 9, right) to (9, 5, left) "
    "so its leader exits to the right and avoids neighboring circles; MT04 offset changed from "
    "(-26, 7, right) to (-11, -3, right) for a shorter, lower leader."
)
FIG7_PANEL_C_MARKER_LABEL_NOTE = (
    f"Current Panel B marker area reduced from {FIG7_PANEL_C_MARKER_SIZE_PREVIOUS:g} to "
    f"{FIG7_PANEL_C_MARKER_SIZE:g} points^2 "
    f"({FIG7_PANEL_C_MARKER_SIZE_REDUCTION_PERCENT:.1f}% smaller); thin neutral leader lines connect labels "
    f"{', '.join(FIG7_QUADRANT_LEADER_LINE_LABELS)} to their corresponding circles. "
    f"{FIG7_PANEL_C_LABEL_ADJUSTMENT_NOTE}"
)
FIG7_LAYOUT_NOTE = (
    "Fig8 layout revision: former Panel A heatmap is removed from current Fig8 and migrated to Fig6C. "
    "Current Fig8 renders only Panels A/B side-by-side, with Panel A retaining a visible morphotype y-axis. "
    "Fig8 is saved with bbox padding to prevent clipping in PNG/SVG/PDF exports. "
    f"{FIG7_PANEL_C_MARKER_LABEL_NOTE}"
)
FIG7_COLORBAR_CLIPPING_FIX_NOTE = (
    f"Historical former access-heatmap colorbar note before migration: label changed from "
    f"'{FIG7_COLORBAR_LABEL_PREVIOUS}' to '{FIG7_COLORBAR_LABEL}' with labelpad={FIG7_COLORBAR_LABELPAD}; "
    "current Fig8 has no heatmap colorbar."
)
FIG10_DISPLAY_TITLE = "Morphotype radar signatures"
FIG10_REMOVED_TOP_TITLE = "Morphotype structural signatures"
FIG10_REMOVED_SUBTITLE = "Outer bands group dimensions; A-G axes use a common 0-1 normalized scale."
FIG10_REMOVED_BOTTOM_NOTE = (
    "Outer bands group domains; common 0-1 radial scale; 0.5 ring = centered Step09 score after clipping/rescaling."
)
FIG10_FIGURE_WIDTH_CM = 18.3
FIG10_FIGURE_HEIGHT_CM_PREVIOUS = 14.8
FIG10_FIGURE_HEIGHT_CM = 13.6
FIG10_LAYOUT_TOP_PREVIOUS = 0.825
FIG10_LAYOUT_TOP = 0.890
FIG10_LAYOUT_BOTTOM = 0.220
FIG10_LAYOUT_WSPACE = 0.30
FIG10_LAYOUT_HSPACE_PREVIOUS = 0.48
FIG10_LAYOUT_HSPACE = 0.16
FIG10_LEGEND_BOUNDS = [0.045, 0.070, 0.940, 0.125]
FIG10_RADAR_RING_LABEL_FONTSIZE = 5.8
FIG10_RADAR_MAX_LABEL_FONTSIZE = 6.5
FIG10_RADAR_AXIS_CODE_FONTSIZE = 7.1
FIG10_RADAR_PANEL_LABEL_FONTSIZE = 7.8
FIG10_LEGEND_TITLE_FONTSIZE = 7.1
FIG10_LEGEND_TEXT_FONTSIZE = 6.7
RADAR_SCORE_METRICS = [
    "density_score",
    "fine_grain_score",
    "grid_score",
    "hierarchy_score",
    "culdesac_score",
    "circuity_score",
    "segment_length_score",
]
RADAR_SCORE_LABELS = {
    "density_score": "Density",
    "fine_grain_score": "Fine grain",
    "grid_score": "Grid",
    "culdesac_score": "Cul-de-sac",
    "hierarchy_score": "Hierarchy",
    "circuity_score": "Circuity",
    "segment_length_score": "Segment length",
}
RADAR_AXIS_CODES = {metric: chr(ord("A") + idx) for idx, metric in enumerate(RADAR_SCORE_METRICS)}
RADAR_METRIC_FAMILIES = {
    "density_score": {
        "metric_family": "intensity_grain",
        "family_label": "Intensity / grain",
        "family_color": "#8FB6C8",
        "metric_family_order": 1,
    },
    "fine_grain_score": {
        "metric_family": "intensity_grain",
        "family_label": "Intensity / grain",
        "family_color": "#8FB6C8",
        "metric_family_order": 1,
    },
    "grid_score": {
        "metric_family": "order_hierarchy",
        "family_label": "Order / hierarchy",
        "family_color": "#9FB899",
        "metric_family_order": 2,
    },
    "hierarchy_score": {
        "metric_family": "order_hierarchy",
        "family_label": "Order / hierarchy",
        "family_color": "#9FB899",
        "metric_family_order": 2,
    },
    "culdesac_score": {
        "metric_family": "friction_access_cost",
        "family_label": "Friction / access cost",
        "family_color": "#CFA7A5",
        "metric_family_order": 3,
    },
    "circuity_score": {
        "metric_family": "friction_access_cost",
        "family_label": "Friction / access cost",
        "family_color": "#CFA7A5",
        "metric_family_order": 3,
    },
    "segment_length_score": {
        "metric_family": "friction_access_cost",
        "family_label": "Friction / access cost",
        "family_color": "#CFA7A5",
        "metric_family_order": 3,
    },
}
RADAR_FAMILY_ORDER = ["intensity_grain", "order_hierarchy", "friction_access_cost"]
RADAR_FAMILY_LABELS = {
    family: next(v["family_label"] for v in RADAR_METRIC_FAMILIES.values() if v["metric_family"] == family)
    for family in RADAR_FAMILY_ORDER
}
RADAR_FAMILY_COLORS = {
    family: next(v["family_color"] for v in RADAR_METRIC_FAMILIES.values() if v["metric_family"] == family)
    for family in RADAR_FAMILY_ORDER
}
RADAR_FAMILY_AXIS_CODES = {
    family: tuple(RADAR_AXIS_CODES[metric] for metric in RADAR_SCORE_METRICS if RADAR_METRIC_FAMILIES[metric]["metric_family"] == family)
    for family in RADAR_FAMILY_ORDER
}
OBSOLETE_STEP12_OUTPUTS = [
    *(f"figures/Fig1_quality_aware_multiscale_typology_framework.{ext}" for ext in ["png", "svg", "pdf", "tiff"]),
    "figure_data_Fig1_quality_aware_multiscale_typology_framework.csv",
    *(f"figures/Fig2_{FIG2_CANONICAL_SLUG}.{ext}" for ext in ["png", "svg", "pdf", "tiff"]),
    f"figure_data_Fig2_{FIG2_CANONICAL_SLUG}.csv",
    *(f"figures/Fig3_osm_quality_confidence.{ext}" for ext in ["png", "svg", "pdf", "tiff"]),
    "figure_data_Fig3_osm_quality_confidence.csv",
    *(f"figures/Fig4_{FIG4_CANONICAL_SLUG}.{ext}" for ext in ["png", "svg", "pdf", "tiff"]),
    f"figure_data_Fig4_{FIG4_CANONICAL_SLUG}.csv",
    *(f"figures/Fig5_morphotype_atlas_signatures.{ext}" for ext in ["png", "svg", "pdf", "tiff"]),
    "figure_data_Fig5_morphotype_atlas_signatures.csv",
    *(f"figures/Fig6_{FIG6_RADAR_SLUG}.{ext}" for ext in ["png", "svg", "pdf", "tiff"]),
    f"figure_data_Fig6_{FIG6_RADAR_SLUG}.csv",
    *(f"figures/Fig7_{FIG7_CANONICAL_SLUG}.{ext}" for ext in ["png", "svg", "pdf", "tiff"]),
    f"figure_data_Fig7_{FIG7_CANONICAL_SLUG}.csv",
    *(f"figures/Fig8_{FIG8_CORE_CITY_SLUG}.{ext}" for ext in ["png", "svg", "pdf", "tiff"]),
    f"figure_data_Fig8_{FIG8_CORE_CITY_SLUG}.csv",
    *(f"figures/Fig9_{FIG9_ACCESS_DETOUR_SLUG}.{ext}" for ext in ["png", "svg", "pdf", "tiff"]),
    f"figure_data_Fig9_{FIG9_ACCESS_DETOUR_SLUG}.csv",
    *(f"figures/Fig10_{FIG10_OBSERVED_GROUP_MEANS_SLUG}.{ext}" for ext in ["png", "svg", "pdf", "tiff"]),
    f"figure_data_Fig10_{FIG10_OBSERVED_GROUP_MEANS_SLUG}.csv",
    *(f"figures/Fig11_{FIG11_CITY_INEQUALITY_SLUG}.{ext}" for ext in ["png", "svg", "pdf", "tiff"]),
    f"figure_data_Fig11_{FIG11_CITY_INEQUALITY_SLUG}.csv",
    *(f"figures/Fig12_{FIG12_MODEL_COEFFICIENT_SLUG}.{ext}" for ext in ["png", "svg", "pdf", "tiff"]),
    f"figure_data_Fig12_{FIG12_MODEL_COEFFICIENT_SLUG}.csv",
    "figures/Fig2_global_sample_city_overview.png",
    "figures/Fig2_global_sample_city_overview.svg",
    "figures/Fig2_global_sample_city_overview.pdf",
    "figure_data_Fig2_global_sample_city_overview.csv",
    "figures/Fig4_multiscale_metric_curves.png",
    "figures/Fig4_multiscale_metric_curves.svg",
    "figures/Fig4_multiscale_metric_curves.pdf",
    "figure_data_Fig4_multiscale_metric_curves.csv",
    "figures/Fig5_morphotype_atlas.png",
    "figures/Fig5_morphotype_atlas.svg",
    "figures/Fig5_morphotype_atlas.pdf",
    "figure_data_Fig5_morphotype_atlas.csv",
    "figures/Fig6_scale_network_transition_matrix.png",
    "figures/Fig6_scale_network_transition_matrix.svg",
    "figures/Fig6_scale_network_transition_matrix.pdf",
    "figure_data_Fig6_scale_network_transition_matrix.csv",
    "figures/Fig6_scale_network_composition_matrix.png",
    "figures/Fig6_scale_network_composition_matrix.svg",
    "figures/Fig6_scale_network_composition_matrix.pdf",
    "figure_data_Fig6_scale_network_composition_matrix.csv",
    "figures/Fig6_scale_transition_matrix.png",
    "figures/Fig6_scale_transition_matrix.svg",
    "figures/Fig6_scale_transition_matrix.pdf",
    "figure_data_Fig6_scale_transition_matrix.csv",
    "figures/Fig7_accessibility_by_morphotype_category.png",
    "figures/Fig7_accessibility_by_morphotype_category.svg",
    "figures/Fig7_accessibility_by_morphotype_category.pdf",
    "figure_data_Fig7_accessibility_by_morphotype_category.csv",
    "figures/Fig8_circuity_efficiency_by_morphotype.png",
    "figures/Fig8_circuity_efficiency_by_morphotype.svg",
    "figures/Fig8_circuity_efficiency_by_morphotype.pdf",
    "figure_data_Fig8_circuity_efficiency_by_morphotype.csv",
    "figures/Fig7_accessibility_detour_performance_by_morphotype.png",
    "figures/Fig7_accessibility_detour_performance_by_morphotype.svg",
    "figures/Fig7_accessibility_detour_performance_by_morphotype.pdf",
    "figure_data_Fig7_accessibility_detour_performance_by_morphotype.csv",
    "figures/Fig8_city_level_accessibility_inequality.png",
    "figures/Fig8_city_level_accessibility_inequality.svg",
    "figures/Fig8_city_level_accessibility_inequality.pdf",
    "figure_data_Fig8_city_level_accessibility_inequality.csv",
    "figures/Fig9_model_coefficient_forest.png",
    "figures/Fig9_model_coefficient_forest.svg",
    "figures/Fig9_model_coefficient_forest.pdf",
    "figure_data_Fig9_model_coefficient_forest.csv",
    "figures/Fig10_morphotype_radar_signatures.png",
    "figures/Fig10_morphotype_radar_signatures.svg",
    "figures/Fig10_morphotype_radar_signatures.pdf",
    "figure_data_Fig10_morphotype_radar_signatures.csv",
    "figures/Fig11_observed_weighted_group_means.png",
    "figures/Fig11_observed_weighted_group_means.svg",
    "figures/Fig11_observed_weighted_group_means.pdf",
    "figure_data_Fig11_observed_weighted_group_means.csv",
    "figures/Fig12_core_city_morphotype_stability_and_performance.png",
    "figures/Fig12_core_city_morphotype_stability_and_performance.svg",
    "figures/Fig12_core_city_morphotype_stability_and_performance.pdf",
    "figure_data_Fig12_core_city_morphotype_stability_and_performance.csv",
]
LEGACY_SUPPLEMENTAL_OUTPUTS = [
    *(f"figures/{'S'}Fig1_city_level_inequality_overview.{ext}" for ext in ["png", "svg", "pdf"]),
    f"figure_data_{'S'}Fig1_city_level_inequality_overview.csv",
    *(f"figures/{'S'}Fig1_{FIG6_RADAR_SLUG}.{ext}" for ext in ["png", "svg", "pdf"]),
    f"figure_data_{'S'}Fig1_{FIG6_RADAR_SLUG}.csv",
    *(f"figures/{'S'}Fig2_{FIG10_OBSERVED_GROUP_MEANS_SLUG}.{ext}" for ext in ["png", "svg", "pdf"]),
    f"figure_data_{'S'}Fig2_{FIG10_OBSERVED_GROUP_MEANS_SLUG}.csv",
]

RISK_NOTES = [
    "Step11 currently has one non-blocking warning about small groups; Step12 records it but does not block figure generation.",
    "Upstream Chinese morphotype names are not unique; figures use MTxx codes plus unique English labels.",
    "figure_data_inequality.csv is city-level / city-morphotype-level, not unit-level microdata.",
    "Model coefficients are explanatory associations, not causal effects.",
    "model_marginal_effects.csv is treated only as observed weighted group means / group marginal summary, not adjusted marginal effects.",
]

FIG9_README_NOTES = [
    "MT01 is the reference category.",
    "Fig11 filters out intercepts, city fixed effects, and other fixed-effect terms.",
    "Fig11 coefficients are explanatory associations, not causal effects.",
]


@dataclass
class StepPaths:
    root: Path
    output_dir: Path
    figure_dir: Path
    step01_dir: Path
    step02_dir: Path
    step06_dir: Path
    step07_dir: Path
    step08_dir: Path
    step09_dir: Path
    step10_dir: Path
    step11_dir: Path
    step13_dir: Path

    @property
    def step01_city_master(self) -> Path:
        return self.step01_dir / "city_sample" / "city_master.parquet"

    @property
    def step01_city_boundaries(self) -> Path:
        return self.step01_dir / "city_sample" / "city_boundaries.gpkg"

    @property
    def step07_city_quality_parquet(self) -> Path:
        return self.step07_dir / "quality_scores.parquet"

    @property
    def step07_city_quality_csv(self) -> Path:
        return self.step07_dir / "quality_scores.csv"

    @property
    def step08_scale_signature_long(self) -> Path:
        return self.step08_dir / "scale_signatures_long.parquet"

    @property
    def step08_city_profile_base(self) -> Path:
        return self.step08_dir / "city_profiles_base.parquet"

    @property
    def step09_type_centers_parquet(self) -> Path:
        return self.step09_dir / "morphotype_centroids.parquet"

    @property
    def step09_type_centers_csv(self) -> Path:
        return self.step09_dir / "morphotype_centroids.csv"

    @property
    def step09_city_profiles(self) -> Path:
        return self.step09_dir / "city_morphotype_profiles.parquet"

    @property
    def step09_local_types(self) -> Path:
        return self.step09_dir / "local_morphotypes.parquet"

    @property
    def step10_od_metrics(self) -> Path:
        return self.step10_dir / "OD_detour_metrics.parquet"

    @property
    def step11_accessibility(self) -> Path:
        return self.step11_dir / "figure_data_accessibility_by_type.csv"

    @property
    def step11_circuity(self) -> Path:
        return self.step11_dir / "figure_data_morphotype_circuity.csv"

    @property
    def step11_inequality(self) -> Path:
        return self.step11_dir / "figure_data_inequality.csv"

    @property
    def step11_table_model(self) -> Path:
        return self.step11_dir / "table_model_results.csv"

    @property
    def step11_coef_table(self) -> Path:
        return self.step11_dir / "model_coefficients.csv"

    @property
    def step11_group_means(self) -> Path:
        return self.step11_dir / "model_marginal_effects.csv"

    @property
    def step11_qc(self) -> Path:
        return self.step11_dir / "step11_quality_checks.csv"


@dataclass
class FigureProduct:
    figure_id: str
    title: str
    kind: str
    png_path: Path
    svg_path: Path
    pdf_path: Path
    data_path: Path
    source_inputs: list[str]
    transformation: str
    fallback_note: str = ""
    palette: str = ""
    cmap: str = ""
    heatmap_cmap: str = ""
    color_variable: str = ""
    palette_note: str = ""
    visual_text_note: str = ""
    aliases: list[dict[str, Any]] = field(default_factory=list)


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
        help=f"Output directory. Defaults to data/{STEP_NAME} under --root; custom paths must stay inside that directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing Step12 outputs.",
    )
    return parser.parse_args()


def resolve_step12_output_dir(root: Path, output_dir: Path | None) -> Path:
    allowed_root = (root / "data" / STEP_NAME).resolve()
    if output_dir is None:
        return allowed_root

    candidate = output_dir.expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    out = candidate.resolve()
    if out != allowed_root and not out.is_relative_to(allowed_root):
        raise ValueError(
            f"Unsafe --output-dir: {out}. Step12 outputs may only be written to "
            f"{allowed_root} or one of its subdirectories."
        )
    return out


def get_paths(root: Path, output_dir: Path | None) -> StepPaths:
    root = root.resolve()
    out = resolve_step12_output_dir(root, output_dir)
    return StepPaths(
        root=root,
        output_dir=out,
        figure_dir=out / "figures",
        step01_dir=root / "data" / STEP01_NAME,
        step02_dir=root / "data" / STEP02_NAME,
        step06_dir=root / "data" / STEP06_NAME,
        step07_dir=root / "data" / STEP07_NAME,
        step08_dir=root / "data" / STEP08_NAME,
        step09_dir=root / "data" / STEP09_NAME,
        step10_dir=root / "data" / STEP10_NAME,
        step11_dir=root / "data" / STEP11_NAME,
        step13_dir=root / "data" / STEP13_NAME,
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
        return json_ready(obj.tolist())
    return scalar(obj)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def file_metadata(path: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return meta
    stat = path.stat()
    meta.update(
        {
            "size_bytes": stat.st_size,
            "mtime": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
            "sha256": file_sha256(path),
        }
    )
    try:
        if path.suffix == ".csv":
            df = pd.read_csv(path, nrows=5)
            meta.update({"format": "csv", "columns": list(df.columns)})
        elif path.suffix == ".parquet":
            pf = pq.ParquetFile(path)
            meta.update({"format": "parquet", "row_count": pf.metadata.num_rows, "columns": pf.schema_arrow.names})
        elif path.suffix == ".json":
            obj = json.loads(path.read_text(encoding="utf-8"))
            meta.update({"format": "json", "top_level_keys": list(obj) if isinstance(obj, dict) else []})
    except Exception as exc:
        meta["metadata_error"] = repr(exc)
    return meta


def package_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for package in ["pandas", "numpy", "pyarrow", "matplotlib", "geopandas", "shapely", "pillow"]:
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


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def cm_to_in(width_cm: float, height_cm: float) -> tuple[float, float]:
    return width_cm / 2.54, height_cm / 2.54


def save_figure(fig: plt.Figure, base: Path, *, pad_inches: float = 0.1) -> tuple[Path, Path, Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    png = base.with_suffix(".png")
    svg = base.with_suffix(".svg")
    pdf = base.with_suffix(".pdf")
    tiff = base.with_suffix(".tiff")
    fig.savefig(png, dpi=450, bbox_inches="tight", pad_inches=pad_inches, facecolor="white")
    fig.savefig(svg, bbox_inches="tight", pad_inches=pad_inches, facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=pad_inches, facecolor="white")
    fig.savefig(tiff, dpi=600, bbox_inches="tight", pad_inches=pad_inches, facecolor="white")
    plt.close(fig)
    return png, svg, pdf


def read_table(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if path.suffix == ".parquet":
        if columns is not None:
            available = set(pq.ParquetFile(path).schema_arrow.names)
            keep = [c for c in columns if c in available]
            return pd.read_parquet(path, columns=keep)
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def choose_existing(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def snapshot_dir(path: Path) -> dict[str, tuple[int, int]]:
    if not path.exists():
        return {}
    snapshot = {}
    for file in path.rglob("*"):
        if file.is_file():
            stat = file.stat()
            snapshot[str(file.resolve())] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


def compare_snapshot(before: dict[str, tuple[int, int]], after: dict[str, tuple[int, int]]) -> list[str]:
    changed: list[str] = []
    for key, value in after.items():
        if key not in before or before[key] != value:
            changed.append(key)
    for key in before:
        if key not in after:
            changed.append(key)
    return sorted(changed)


def wait_for_stable_directory(path: Path, *, stable_seconds: float = 2.0, timeout_seconds: float = 30.0) -> dict[str, Any]:
    start = time.time()
    last_snapshot = snapshot_dir(path)
    last_change = time.time()
    checks = 0
    while time.time() - start < timeout_seconds:
        time.sleep(0.5)
        checks += 1
        current = snapshot_dir(path)
        if current != last_snapshot:
            last_snapshot = current
            last_change = time.time()
            continue
        if time.time() - last_change >= stable_seconds:
            return {
                "status": "stable",
                "wait_seconds": round(time.time() - start, 3),
                "checks": checks,
                "file_count": len(current),
            }
    return {
        "status": "timeout",
        "wait_seconds": round(time.time() - start, 3),
        "checks": checks,
        "file_count": len(last_snapshot),
    }


def numeric(series: pd.Series, default: float = np.nan) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if not np.isnan(default):
        out = out.fillna(default)
    return out


def fig5_style_sequential_cmap() -> mpl.colors.LinearSegmentedColormap:
    return mpl.colors.LinearSegmentedColormap.from_list(
        FIG5_STYLE_SEQUENTIAL_CMAP,
        FIG5_STYLE_SEQUENTIAL_CMAP_COLORS,
        N=256,
    )


def heatmap_text_color(cmap: mpl.colors.Colormap, norm: mpl.colors.Normalize, value: Any) -> str:
    if pd.isna(value):
        return "#0F172A"
    rgba = cmap(norm(float(value)))
    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "white" if luminance < 0.45 else "#0F172A"


def symmetric_pp_limit(values: pd.Series, *, minimum: float = 5.0, step: float = 5.0) -> float:
    finite = numeric(values).replace([np.inf, -np.inf], np.nan).dropna().abs()
    if finite.empty:
        return minimum
    return max(minimum, math.ceil(float(finite.max()) / step) * step)


def weighted_mean(values: pd.Series, weights: pd.Series | None = None) -> float:
    v = numeric(values)
    if weights is None:
        return float(v.mean()) if v.notna().any() else np.nan
    w = numeric(weights)
    mask = v.notna() & w.notna() & (w > 0)
    if mask.any() and float(w[mask].sum()) > 0:
        return float(np.average(v[mask], weights=w[mask]))
    return float(v.mean()) if v.notna().any() else np.nan


def quantile(values: pd.Series, q: float) -> float:
    v = numeric(values).dropna()
    return float(v.quantile(q)) if not v.empty else np.nan


def make_output_path(paths: StepPaths, filename: str) -> Path:
    return paths.output_dir / filename


def source_data_name(figure_id: str, slug: str) -> str:
    return f"figure_data_{figure_id}_{slug}.csv"


def replace_text_in_string_columns(df: pd.DataFrame, replacements: list[tuple[str, str]]) -> pd.DataFrame:
    out = df.copy()
    if not replacements:
        return out
    for col in out.columns:
        if not (pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col])):
            continue
        series = out[col]
        mask = series.notna()
        if not mask.any():
            continue
        values = series.loc[mask].astype(str)
        for old, new in replacements:
            values = values.str.replace(old, new, regex=False)
        out.loc[mask, col] = values
    return out


def figure_alias_output_paths(product: FigureProduct) -> list[Path]:
    paths: list[Path] = []
    for alias in product.aliases:
        for key in ["png_path", "svg_path", "pdf_path", "data_path"]:
            value = alias.get(key)
            if isinstance(value, Path):
                paths.append(value)
    return paths


def all_product_output_paths(products: list[FigureProduct], *, include_aliases: bool = True) -> list[Path]:
    paths: list[Path] = []
    for product in products:
        paths.extend([product.png_path, product.svg_path, product.pdf_path, product.data_path])
        if include_aliases:
            paths.extend(figure_alias_output_paths(product))
    return paths


def collect_figure_aliases(products: list[FigureProduct]) -> list[dict[str, Any]]:
    aliases: list[dict[str, Any]] = []
    for product in products:
        for alias in product.aliases:
            aliases.append({"figure_id": product.figure_id, **alias})
    return aliases


def cleanup_obsolete_step12_outputs(paths: StepPaths) -> list[str]:
    removed: list[str] = []
    for rel in OBSOLETE_STEP12_OUTPUTS:
        path = paths.output_dir / rel
        if path.exists() and path.is_file():
            path.unlink()
            removed.append(rel)
    legacy_removed = 0
    for rel in LEGACY_SUPPLEMENTAL_OUTPUTS:
        path = paths.output_dir / rel
        if path.exists() and path.is_file():
            path.unlink()
            legacy_removed += 1
    if legacy_removed:
        removed.append(f"legacy_supplemental_canonical_outputs_removed={legacy_removed}")
    return removed


def compact_number(value: Any, *, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(number):
        return "nan"
    return f"{number:.{digits}g}"


def fig6_panel_a_norm_metadata(data: pd.DataFrame | None = None) -> dict[str, Any]:
    data_max = np.nan
    panel_c_color_limit = 5.0
    if data is not None and not data.empty and {"panel", "mean_unit_share_percent"}.issubset(data.columns):
        composition_values = numeric(data.loc[data["panel"].astype(str).eq(FIG6_PANEL_COMPOSITION), "mean_unit_share_percent"])
        if composition_values.notna().any():
            data_max = float(composition_values.max())
    if data is not None and not data.empty and {"panel", FIG7_COLOR_VARIABLE}.issubset(data.columns):
        panel_c_values = numeric(data.loc[data["panel"].astype(str).eq(FIG6_PANEL_PERFORMANCE), FIG7_COLOR_VARIABLE])
        if panel_c_values.notna().any():
            panel_c_color_limit = symmetric_pp_limit(panel_c_values)
    candidate_vmax = (
        FIG6_PANEL_A_NORM_VMAX_DATA_FACTOR * data_max
        if np.isfinite(data_max)
        else np.nan
    )
    current_vmax = max(FIG6_PANEL_A_NORM_VMAX_FLOOR, candidate_vmax) if np.isfinite(candidate_vmax) else FIG6_PANEL_A_NORM_VMAX_FLOOR
    formula = (
        f"vmax=max({compact_number(FIG6_PANEL_A_NORM_VMAX_FLOOR)}, "
        f"{compact_number(FIG6_PANEL_A_NORM_VMAX_DATA_FACTOR)}*data_max)"
    )
    current_clause = (
        f"current data_max={compact_number(data_max)}, "
        f"current 1.05*data_max={compact_number(candidate_vmax)}, "
        f"current vmax={compact_number(current_vmax)}"
    )
    summary = (
        f"Panel A uses {FIG6_FIG7_HEATMAP_CMAP} with nonnegative "
        f"Normalize(vmin={compact_number(FIG6_PANEL_A_NORM_VMIN)}, {formula}); "
        f"{current_clause}."
    )
    return {
        "panel_a_cmap": FIG6_FIG7_HEATMAP_CMAP,
        "panel_a_norm_type": "Normalize",
        "panel_a_norm_vmin": FIG6_PANEL_A_NORM_VMIN,
        "panel_a_norm_vmax_floor": FIG6_PANEL_A_NORM_VMAX_FLOOR,
        "panel_a_norm_vmax_data_factor": FIG6_PANEL_A_NORM_VMAX_DATA_FACTOR,
        "panel_a_norm_data_max": data_max,
        "panel_a_norm_candidate_vmax": candidate_vmax,
        "panel_a_norm_vmax": current_vmax,
        "panel_a_norm_vmax_formula": formula,
        "panel_a_norm_summary": summary,
        "panel_b_norm_summary": "Panel B uses the same RdBu_r cmap with TwoSlopeNorm centered at 0 for signed transition_delta_pp.",
        "panel_c_cmap": FIG7_HEATMAP_CMAP,
        "panel_c_norm_type": "TwoSlopeNorm",
        "panel_c_norm_vmin": -panel_c_color_limit,
        "panel_c_norm_vcenter": 0.0,
        "panel_c_norm_vmax": panel_c_color_limit,
        "panel_c_color_limit_abs_pp": panel_c_color_limit,
        "panel_c_color_variable": FIG7_COLOR_VARIABLE,
        "panel_c_colorbar_label": FIG7_COLORBAR_LABEL,
        "panel_c_norm_summary": (
            f"Panel C uses {FIG7_HEATMAP_CMAP} with TwoSlopeNorm(vmin=-{compact_number(panel_c_color_limit)}, "
            f"vcenter=0, vmax={compact_number(panel_c_color_limit)}) for access_share_diff_pp, migrated from the former access heatmap."
        ),
    }


def fig6_norm_metadata_from_products(products: list[FigureProduct]) -> dict[str, Any]:
    product = next((p for p in products if p.figure_id == "Fig6"), None)
    if product is None or not product.data_path.exists():
        return fig6_panel_a_norm_metadata(None)
    try:
        return fig6_panel_a_norm_metadata(pd.read_csv(product.data_path))
    except Exception:
        return fig6_panel_a_norm_metadata(None)


def fig6_enrichment_metadata_from_products(products: list[FigureProduct]) -> dict[str, Any]:
    product = next((p for p in products if p.figure_id == "Fig6"), None)
    if product is None or not product.data_path.exists():
        return fig6_enrichment_metadata(None)
    try:
        return fig6_enrichment_metadata(pd.read_csv(product.data_path))
    except Exception:
        return fig6_enrichment_metadata(None)


def fig6_palette_note(norm_metadata: dict[str, Any] | None = None) -> str:
    meta = norm_metadata or fig6_panel_a_norm_metadata(None)
    return (
        f"Fig6 panels A/B/C use heatmap cmap/palettes with panel-specific norms; A/B share {FIG6_FIG7_HEATMAP_CMAP}, "
        f"and C uses {meta['panel_c_cmap']} for the former access-deviation metric. "
        f"{meta['panel_a_norm_summary']} "
        "Panel B keeps a zero-centered signed-delta norm. "
        f"{meta['panel_c_norm_summary']} "
        f"Thin outlines mark the top {FIG6_TOP_CELL_COUNT} cells in panels A and B."
    )


def fig6_title_footnote_cleanup_metadata() -> dict[str, Any]:
    return {
        "figure_level_title_current": False,
        "bottom_note_current": False,
        "removed_figure_level_title_text": FIG6_REMOVED_SUPTITLE_TEXT,
        "removed_bottom_note_text": FIG6_REMOVED_BOTTOM_NOTE_TEXT,
        "retained_visual_text_elements": (
            "panel A/B/C titles; x/y axis labels; tick labels; colorbar labels; "
            "in-cell numeric labels; top-cell outline encoding"
        ),
        "removal_note": FIG6_TITLE_FOOTNOTE_REMOVAL_NOTE,
    }


def fig7_footnote_cleanup_metadata() -> dict[str, Any]:
    return {
        "bottom_note_current": False,
        "removed_bottom_note_text": FIG7_REMOVED_BOTTOM_NOTE_TEXT,
        "retained_visual_text_elements": (
            "panel A/B titles; x/y axis labels; tick labels; morphotype labels; "
            "current Panel B marker labels and leader lines"
        ),
        "removal_note": FIG7_FOOTNOTE_REMOVAL_NOTE,
    }


def fig6_enrichment_metadata(data: pd.DataFrame | None = None) -> dict[str, Any]:
    text_cleanup = fig6_title_footnote_cleanup_metadata()
    meta: dict[str, Any] = {
        "top_cell_count": FIG6_TOP_CELL_COUNT,
        "panel_a_top_cell_metric": "mean_unit_share_percent",
        "panel_a_top_cell_style": (
            f"thin neutral outline ({FIG6_PANEL_A_TOP_CELL_OUTLINE_COLOR}) around the "
            f"top {FIG6_TOP_CELL_COUNT} composition-share cells"
        ),
        "panel_b_top_cell_metric": "absolute_transition_delta_pp",
        "panel_b_top_cell_style": (
            f"thin signed outline around the top {FIG6_TOP_CELL_COUNT} absolute-delta cells; "
            f"positive={FIG6_PANEL_B_POSITIVE_OUTLINE_COLOR}, negative={FIG6_PANEL_B_NEGATIVE_OUTLINE_COLOR}"
        ),
        "morphotype_axis_order": FIG6_MORPHOTYPE_AXIS_ORDER_TEXT,
        "panel_c_morphotype_axis_order": FIG6_MORPHOTYPE_AXIS_ORDER_TEXT,
        "panel_c_morphotype_axis_order_note": FIG6_PANEL_C_MORPHOTYPE_AXIS_ORDER_NOTE,
        "current_visual_encoding": (
            "heatmap cell color and in-cell numeric labels for A/B/C; top-cell outline annotations for A/B; "
            "no figure-level title or bottom footnote/note"
        ),
        "text_cleanup": text_cleanup,
    }
    if data is None or data.empty:
        return meta

    frame = data.copy()
    if {"panel", "fig7_top_cell_flag", "fig7_top_cell_rank"}.issubset(frame.columns):
        top_records: dict[str, list[dict[str, Any]]] = {}
        for panel in ["composition", "transition"]:
            panel_top = frame[
                frame["panel"].astype(str).eq(panel)
                & frame["fig7_top_cell_flag"].astype(str).str.lower().isin(["true", "1"])
            ].copy()
            if "fig7_top_cell_rank" in panel_top.columns:
                panel_top["_rank"] = numeric(panel_top["fig7_top_cell_rank"])
                panel_top = panel_top.sort_values("_rank")
            top_records[panel] = [
                {
                    "rank": scalar(row.get("fig7_top_cell_rank")),
                    "row_label": row.get("row_label"),
                    "morphotype": row.get("morphotype"),
                    "value_for_plot": scalar(row.get("value_for_plot")),
                    "ranking_metric_value": scalar(row.get("fig7_top_cell_metric_value")),
                    "outline_color": row.get("fig7_top_cell_outline_color"),
                }
                for _, row in panel_top.iterrows()
            ]
        meta["top_cells"] = top_records

    return meta


def add_fig6_enrichment_fields(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["fig7_top_cell_count"] = FIG6_TOP_CELL_COUNT
    out["fig7_top_cell_flag"] = False
    out["fig7_top_cell_rank"] = np.nan
    out["fig7_top_cell_metric"] = ""
    out["fig7_top_cell_metric_value"] = np.nan
    out["fig7_top_cell_annotation"] = ""
    out["fig7_top_cell_outline_color"] = ""
    out["fig7_top_cell_delta_sign"] = ""

    panel = out["panel"].astype(str)
    comp_mask = panel.eq(FIG6_PANEL_COMPOSITION)
    trans_mask = panel.eq(FIG6_PANEL_TRANSITION)

    if "mean_unit_share_percent" in out.columns:
        comp_metric = numeric(out.loc[comp_mask, "mean_unit_share_percent"]).dropna()
        for rank, idx in enumerate(comp_metric.sort_values(ascending=False).head(FIG6_TOP_CELL_COUNT).index, start=1):
            out.loc[idx, "fig7_top_cell_flag"] = True
            out.loc[idx, "fig7_top_cell_rank"] = rank
            out.loc[idx, "fig7_top_cell_metric"] = "mean_unit_share_percent"
            out.loc[idx, "fig7_top_cell_metric_value"] = float(out.loc[idx, "mean_unit_share_percent"])
            out.loc[idx, "fig7_top_cell_annotation"] = f"panel A top {FIG6_TOP_CELL_COUNT} composition-share cell"
            out.loc[idx, "fig7_top_cell_outline_color"] = FIG6_PANEL_A_TOP_CELL_OUTLINE_COLOR

    if "absolute_transition_delta_pp" in out.columns:
        trans_metric = numeric(out.loc[trans_mask, "absolute_transition_delta_pp"]).dropna()
        for rank, idx in enumerate(trans_metric.sort_values(ascending=False).head(FIG6_TOP_CELL_COUNT).index, start=1):
            delta = scalar(out.loc[idx, "transition_delta_pp"]) if "transition_delta_pp" in out.columns else np.nan
            sign = "positive" if pd.notna(delta) and float(delta) >= 0 else "negative"
            outline_color = FIG6_PANEL_B_POSITIVE_OUTLINE_COLOR if sign == "positive" else FIG6_PANEL_B_NEGATIVE_OUTLINE_COLOR
            out.loc[idx, "fig7_top_cell_flag"] = True
            out.loc[idx, "fig7_top_cell_rank"] = rank
            out.loc[idx, "fig7_top_cell_metric"] = "absolute_transition_delta_pp"
            out.loc[idx, "fig7_top_cell_metric_value"] = float(out.loc[idx, "absolute_transition_delta_pp"])
            out.loc[idx, "fig7_top_cell_annotation"] = f"panel B top {FIG6_TOP_CELL_COUNT} absolute transition-delta cell"
            out.loc[idx, "fig7_top_cell_outline_color"] = outline_color
            out.loc[idx, "fig7_top_cell_delta_sign"] = sign

    return out


def draw_fig6_top_cell_outlines(ax: plt.Axes, panel_data: pd.DataFrame, row_order: list[str]) -> None:
    if panel_data.empty or "fig7_top_cell_flag" not in panel_data.columns:
        return
    row_lookup = {label: i for i, label in enumerate(row_order)}
    col_lookup = {mt: i for i, mt in enumerate(MORPHOTYPES)}
    selected = panel_data[panel_data["fig7_top_cell_flag"].astype(bool)]
    for _, row in selected.iterrows():
        y = row_lookup.get(row.get("row_label"))
        x = col_lookup.get(row.get("morphotype"))
        if y is None or x is None:
            continue
        outline_color = str(row.get("fig7_top_cell_outline_color") or FIG6_PANEL_A_TOP_CELL_OUTLINE_COLOR)
        ax.add_patch(
            Rectangle(
                (x - 0.5, y - 0.5),
                1,
                1,
                fill=False,
                edgecolor=outline_color,
                linewidth=FIG6_TOP_CELL_OUTLINE_LINEWIDTH,
                zorder=5,
                clip_on=False,
            )
        )


def palette_scheme_metadata(fig6_norm_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    fig6_norm = fig6_norm_metadata or fig6_panel_a_norm_metadata(None)
    return {
        "scope": "Fig1-Fig11 main-text figures; Fig1 map bytes and Fig1 palette/encoding semantics are protected on rerun when present.",
        "sequential_cmap": SEQUENTIAL_CMAP,
        "sequential_usage": "continuous non-target Fig3/Fig9/Fig10/Fig11 summaries; Fig8 detour panel uses a neutral gray interval palette with pure gray p10-p90 interval range (#B8B8B8) and median line/marker color #3B4A6B; Fig7 metric strips use a neutral 0-1 sequential strip colormap; Fig4/Fig6 use the heatmap palettes below",
        "top_journal_sequential_cmap": TOP_JOURNAL_SEQUENTIAL_CMAP,
        "top_journal_diverging_cmap": TOP_JOURNAL_DIVERGING_CMAP,
        "fig5_style_sequential_cmap": FIG5_STYLE_SEQUENTIAL_CMAP,
        "fig5_style_sequential_cmap_source": FIG5_STYLE_SEQUENTIAL_CMAP_SOURCE,
        "fig5_style_sequential_cmap_colors": FIG5_STYLE_SEQUENTIAL_CMAP_COLORS,
        "fig3_palette_exception": FIG3_QUALITY_TIER_COMMUNICATION_NOTE,
        "fig3_quality_tier_colors": FIG2_QUALITY_TIER_COLORS,
        "fig5_heatmap_cmap": FIG5_HEATMAP_CMAP,
        "fig9_interval_palette": FIG8_INTERVAL_PALETTE_NAME,
        "fig9_interval_line_color": FIG8_INTERVAL_LINE_COLOR,
        "fig9_interval_range_color": FIG8_INTERVAL_RANGE_COLOR,
        "fig5_heatmap_norm": "TwoSlopeNorm centered at 0 for signed standardized type-center scores",
        "fig7_heatmap_cmap": FIG6_FIG7_HEATMAP_CMAP,
        "fig7_heatmap_norm": (
            f"{fig6_norm['panel_a_norm_summary']} "
            "Panel B transition uses signed percentage-point deltas with TwoSlopeNorm centered at zero. "
            f"Panel C uses the migrated former access-deviation metric with {fig6_norm['panel_c_norm_summary']} "
            "Fig6 A/B/C norms are panel-specific and must not be treated as identical."
        ),
        "fig7_visual_enrichment": fig6_enrichment_metadata(None),
        "fig7_morphotype_axis_order": FIG6_MORPHOTYPE_AXIS_ORDER_TEXT,
        "fig7_panel_c_morphotype_axis_order_note": FIG6_PANEL_C_MORPHOTYPE_AXIS_ORDER_NOTE,
        "fig7_panel_a_norm_vmax_formula": fig6_norm["panel_a_norm_vmax_formula"],
        "fig7_panel_a_norm_vmax_current": fig6_norm["panel_a_norm_vmax"],
        "fig7_panel_a_norm_data_max": fig6_norm["panel_a_norm_data_max"],
        "fig7_panel_c_heatmap_cmap": fig6_norm["panel_c_cmap"],
        "fig7_panel_c_heatmap_norm": fig6_norm["panel_c_norm_summary"],
        "fig7_panel_c_colorbar_label": fig6_norm["panel_c_colorbar_label"],
        "fig9_heatmap_cmap": "",
        "fig8_heatmap_norm": "Current Fig8 has no heatmap; former access heatmap is migrated to Fig6C.",
        "fig6_fig8_heatmap_cmap": f"Fig6 A/B={FIG6_FIG7_HEATMAP_CMAP}; Fig6C={fig6_norm['panel_c_cmap']}; current Fig8=none",
        "fig7_fig9_heatmap_norm": (
            f"Fig6 panel A uses nonnegative linear normalization with {fig6_norm['panel_a_norm_vmax_formula']} "
            f"(current vmax={compact_number(fig6_norm['panel_a_norm_vmax'])}) and Fig6 panel B uses zero-centered "
            f"diverging normalization with the same RdBu_r colormap; Fig6 panel C uses {fig6_norm['panel_c_norm_summary']}; "
            "current Fig8 has no heatmap/colorbar."
        ),
        "heatmap_usage": "Fig4 uses a zero-centered blue-white-red diverging heatmap for signed standardized scores; Fig6 panel A and panel B share the same RdBu_r heatmap colormap while keeping panel-specific norms for nonnegative shares versus signed transition deltas; Fig6 panel C contains the migrated former access-deviation heatmap with its own zero-centered norm and Difference from mean (pp) colorbar; current Fig8 has no heatmap; Fig7 right-side strips use a single neutral sequential 0-1 normalized colormap.",
        "morphotype_palette": MORPHOTYPE_PALETTE_NAME,
        "morphotype_colors": MORPHOTYPE_COLORS,
        "quality_tier_analysis_colors": QUALITY_TIER_COLORS_FOR_ANALYSIS,
    }


def add_palette_metadata(
    data: pd.DataFrame,
    *,
    palette: str = "",
    cmap: str = "",
    heatmap_cmap: str = "",
    color_variable: str = "",
    color_variable_label: str = "",
    palette_note: str = "",
) -> pd.DataFrame:
    out = data.copy()
    if palette:
        out["palette"] = palette
    if cmap:
        out["cmap"] = cmap
    if heatmap_cmap:
        out["heatmap_cmap"] = heatmap_cmap
    if color_variable:
        out["color_variable"] = color_variable
    if color_variable_label:
        out["color_variable_label"] = color_variable_label
    if palette_note:
        out["palette_note"] = palette_note
    return out


def add_morphotype_color_field(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    if "morphotype" in out.columns:
        out["morphotype_color"] = out["morphotype"].map(MORPHOTYPE_COLORS)
        out["morphotype_palette"] = MORPHOTYPE_PALETTE_NAME
    return out


def protected_fig1_fig2_output_paths(paths: StepPaths) -> dict[str, Path]:
    return {
        "Fig1_png": paths.figure_dir / f"Fig1_{FIG2_CANONICAL_SLUG}.png",
        "Fig1_svg": paths.figure_dir / f"Fig1_{FIG2_CANONICAL_SLUG}.svg",
        "Fig1_pdf": paths.figure_dir / f"Fig1_{FIG2_CANONICAL_SLUG}.pdf",
        "Fig1_csv": make_output_path(paths, source_data_name("Fig1", FIG2_CANONICAL_SLUG)),
    }


def snapshot_protected_fig1_fig2_outputs(paths: StepPaths) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for label, path in protected_fig1_fig2_output_paths(paths).items():
        exists = path.exists()
        stat = path.stat() if exists else None
        snapshot[label] = {
            "path": str(path),
            "exists": exists,
            "size_bytes": stat.st_size if stat is not None else None,
            "mtime_ns": stat.st_mtime_ns if stat is not None else None,
            "mtime": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds")
            if stat is not None
            else None,
            "sha256": file_sha256(path) if exists else None,
        }
    return snapshot


def public_preservation_snapshot(snapshot: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        label: {key: value for key, value in record.items() if key != "_bytes"}
        for label, record in snapshot.items()
    }


def protected_record_mismatch_fields(before: dict[str, Any], after: dict[str, Any] | None) -> list[str]:
    if after is None:
        after = {}
    fields = ["exists", "size_bytes", "mtime_ns", "sha256"]
    return [field for field in fields if before.get(field) != after.get(field)]


def load_protected_fig1_fig2_reference_hashes(paths: StepPaths) -> dict[str, dict[str, Any]]:
    """Load the latest Step13 material-list hashes as accepted Fig1 references."""
    material_list = paths.step13_dir / "paper_materials_manifest.csv"
    references: dict[str, dict[str, Any]] = {}
    if not material_list.exists():
        return {
            label: {
                "source": str(material_list),
                "available": False,
                "reason": "Step13 material list not found.",
            }
            for label in protected_fig1_fig2_output_paths(paths)
        }

    try:
        material = pd.read_csv(material_list)
    except Exception as exc:
        return {
            label: {
                "source": str(material_list),
                "available": False,
                "reason": f"Could not read Step13 material list: {exc!r}",
            }
            for label in protected_fig1_fig2_output_paths(paths)
        }

    for label, path in protected_fig1_fig2_output_paths(paths).items():
        matches = material[material.get("artifact_name", pd.Series(dtype=str)).astype(str).eq(path.name)]
        if matches.empty and "absolute_path" in material.columns:
            matches = material[material["absolute_path"].astype(str).eq(str(path))]
        if matches.empty and "relative_path" in material.columns:
            rel = str(path.relative_to(paths.root)) if path.is_relative_to(paths.root) else str(path)
            matches = material[material["relative_path"].astype(str).eq(rel)]
        if matches.empty:
            references[label] = {
                "source": str(material_list),
                "available": False,
                "reason": f"No Step13 material-list row found for {path.name}.",
            }
            continue
        row = matches.iloc[0].to_dict()
        references[label] = {
            "source": str(material_list),
            "available": True,
            "artifact_name": scalar(row.get("artifact_name")),
            "relative_path": scalar(row.get("relative_path")),
            "mtime": scalar(row.get("mtime")),
            "size_bytes": scalar(row.get("size_bytes")),
            "sha256": scalar(row.get("hash_sha256")),
        }
    return references


def create_fig1_fig2_protection(paths: StepPaths, temp_dir: Path) -> dict[str, Any]:
    temp_dir.mkdir(parents=True, exist_ok=True)
    run_start = snapshot_protected_fig1_fig2_outputs(paths)
    protected_copies: dict[str, dict[str, Any]] = {}
    for label, record in run_start.items():
        if not record.get("exists"):
            continue
        source = Path(record["path"])
        copy_path = temp_dir / f"{label}{source.suffix}"
        shutil.copy2(source, copy_path)
        protected_copies[label] = {
            "copy_path": str(copy_path),
            "sha256": file_sha256(copy_path),
            "size_bytes": copy_path.stat().st_size,
        }
    reference_hashes = load_protected_fig1_fig2_reference_hashes(paths)
    reference_available = {
        label: bool(reference_hashes.get(label, {}).get("available"))
        for label in protected_fig1_fig2_output_paths(paths)
    }
    reference_matches = {
        label: (
            bool(reference_hashes.get(label, {}).get("available"))
            and reference_hashes.get(label, {}).get("sha256") == run_start.get(label, {}).get("sha256")
        )
        for label in protected_fig1_fig2_output_paths(paths)
    }
    reference_mismatches = [
        label
        for label, matches in reference_matches.items()
        if reference_hashes.get(label, {}).get("available") and not matches
    ]
    reference_missing = [
        label for label, available in reference_available.items() if not available
    ]
    return {
        "created_at": now_iso(),
        "temp_dir": str(temp_dir),
        "run_start": run_start,
        "protected_copies": protected_copies,
        "expected_protected_file_count": len(run_start),
        "run_start_missing": [label for label, record in run_start.items() if not record.get("exists")],
        "accepted_reference": reference_hashes,
        "accepted_reference_available": reference_available,
        "accepted_reference_matches": reference_matches,
        "accepted_reference_mismatches": reference_mismatches,
        "accepted_reference_missing": reference_missing,
    }


def preserve_fig1_fig2_outputs(
    paths: StepPaths,
    protection: dict[str, Any],
    *,
    phase: str = "after_generation",
) -> dict[str, Any]:
    before = protection["run_start"]
    before_restore = snapshot_protected_fig1_fig2_outputs(paths)
    changed_labels: list[str] = []
    changed_details: dict[str, list[str]] = {}
    restored_labels: list[str] = []
    for label, before_record in before.items():
        if not before_record.get("exists"):
            continue
        mismatch_fields = protected_record_mismatch_fields(before_record, before_restore.get(label))
        if not mismatch_fields:
            continue
        changed_labels.append(label)
        changed_details[label] = mismatch_fields
        copy_record = protection.get("protected_copies", {}).get(label, {})
        copy_value = copy_record.get("copy_path")
        if not copy_value:
            continue
        copy_path = Path(copy_value)
        if not copy_path.exists():
            continue
        shutil.copy2(copy_path, Path(before_record["path"]))
        restored_labels.append(label)

    after_restore = snapshot_protected_fig1_fig2_outputs(paths)
    final_mismatches = [
        label
        for label, before_record in before.items()
        if before_record.get("exists")
        and protected_record_mismatch_fields(before_record, after_restore.get(label))
    ]
    final_mismatch_details = {
        label: protected_record_mismatch_fields(before_record, after_restore.get(label))
        for label, before_record in before.items()
        if before_record.get("exists") and protected_record_mismatch_fields(before_record, after_restore.get(label))
    }
    run_start_missing = protection.get("run_start_missing", [])
    expected_count = int(protection.get("expected_protected_file_count", len(before)))
    protected_copy_count = len(protection.get("protected_copies", {}))
    run_start_missing_set = set(run_start_missing)
    generated_missing_after = [
        label
        for label in run_start_missing
        if not after_restore.get(label, {}).get("exists")
        or int(after_restore.get(label, {}).get("size_bytes") or 0) <= 0
    ]
    expected_present_at_start = expected_count - len(run_start_missing_set)
    mode = (
        "generated_without_run_start_snapshot"
        if run_start_missing
        else "preserved_from_run_start_unchanged"
        if not changed_labels
        else "restored_from_run_start_snapshot"
    )
    status = (
        "pass"
        if not final_mismatches
        and not generated_missing_after
        and protected_copy_count == expected_present_at_start
        else "fail"
    )
    reference_mismatches = protection.get("accepted_reference_mismatches", [])
    reference_missing = protection.get("accepted_reference_missing", [])
    caveat = ""
    if reference_mismatches:
        caveat = (
            "Step13 material-list hashes do not match the current Step12 run-start protected bytes for "
            f"{', '.join(reference_mismatches)}. The files were preserved from this run-start snapshot; "
            "no older accepted file bytes were available in the workspace."
        )
    elif reference_missing:
        caveat = (
            "Step13 material-list rows were not available for "
            f"{', '.join(reference_missing)}. The files were preserved from this run-start snapshot."
        )
    return {
        "status": status,
        "mode": mode,
        "phase": phase,
        "preserved_from_run_start": status == "pass" and not final_mismatches and not run_start_missing,
        "protected_file_count": expected_count,
        "protected_copy_count": protected_copy_count,
        "run_start_missing": run_start_missing,
        "generated_missing_after": generated_missing_after,
        "changed_before_restore": changed_labels,
        "changed_before_restore_details": changed_details,
        "changed_after_generation": changed_labels,
        "restored": restored_labels,
        "final_mismatches": final_mismatches,
        "final_mismatch_details": final_mismatch_details,
        "protection_created_at": protection.get("created_at"),
        "protected_copy_hashes": {
            label: {
                "sha256": record.get("sha256"),
                "size_bytes": record.get("size_bytes"),
            }
            for label, record in protection.get("protected_copies", {}).items()
        },
        "accepted_reference": protection.get("accepted_reference", {}),
        "accepted_reference_available": protection.get("accepted_reference_available", {}),
        "accepted_reference_matches": protection.get("accepted_reference_matches", {}),
        "accepted_reference_mismatches": reference_mismatches,
        "accepted_reference_missing": reference_missing,
        "accepted_reference_matches_all": not reference_mismatches and not reference_missing,
        "preservation_caveat": caveat,
        "before": public_preservation_snapshot(before),
        "before_restore": public_preservation_snapshot(before_restore),
        "after_generation": public_preservation_snapshot(before_restore),
        "after_restore": public_preservation_snapshot(after_restore),
        "note": (
            "Fig1 PNG/SVG/PDF/source CSV files are copied to a protected temporary "
            "directory at Step12 start when they already exist. During the first run after "
            "continuous renumbering, missing run-start Fig1 files are accepted if the newly "
            "generated outputs exist and are non-empty before QC, logs and reproducibility "
            "metadata are written."
        ),
    }


def fig7_material_output_paths(paths: StepPaths) -> dict[str, Path]:
    base = paths.figure_dir / f"Fig8_{FIG9_ACCESS_DETOUR_SLUG}"
    return {
        "Fig8_png": base.with_suffix(".png"),
        "Fig8_svg": base.with_suffix(".svg"),
        "Fig8_pdf": base.with_suffix(".pdf"),
        "Fig8_csv": make_output_path(paths, source_data_name("Fig8", FIG9_ACCESS_DETOUR_SLUG)),
    }


def protected_fig7_vector_output_paths(paths: StepPaths) -> dict[str, Path]:
    return {}


def snapshot_named_paths(named_paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for label, path in named_paths.items():
        exists = path.exists()
        stat = path.stat() if exists else None
        snapshot[label] = {
            "path": str(path),
            "exists": exists,
            "size_bytes": stat.st_size if stat is not None else None,
            "mtime_ns": stat.st_mtime_ns if stat is not None else None,
            "mtime": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds")
            if stat is not None
            else None,
            "sha256": file_sha256(path) if exists else None,
        }
    return snapshot


def load_fig7_material_reference_hashes(paths: StepPaths) -> dict[str, dict[str, Any]]:
    """Load Step13 material-list hashes for Fig8 PNG/SVG/PDF/source CSV."""
    material_list = paths.step13_dir / "paper_materials_manifest.csv"
    references: dict[str, dict[str, Any]] = {}
    labels = fig7_material_output_paths(paths)
    if not material_list.exists():
        return {
            label: {
                "source": str(material_list),
                "available": False,
                "reason": "Step13 material list not found.",
            }
            for label in labels
        }

    try:
        material = pd.read_csv(material_list)
    except Exception as exc:
        return {
            label: {
                "source": str(material_list),
                "available": False,
                "reason": f"Could not read Step13 material list: {exc!r}",
            }
            for label in labels
        }

    for label, path in labels.items():
        matches = material[material.get("artifact_name", pd.Series(dtype=str)).astype(str).eq(path.name)]
        if matches.empty and "absolute_path" in material.columns:
            matches = material[material["absolute_path"].astype(str).eq(str(path))]
        if matches.empty and "relative_path" in material.columns:
            rel = str(path.relative_to(paths.root)) if path.is_relative_to(paths.root) else str(path)
            matches = material[material["relative_path"].astype(str).eq(rel)]
        if matches.empty:
            references[label] = {
                "source": str(material_list),
                "available": False,
                "reason": f"No Step13 material-list row found for {path.name}.",
            }
            continue
        row = matches.iloc[0].to_dict()
        references[label] = {
            "source": str(material_list),
            "available": True,
            "artifact_name": scalar(row.get("artifact_name")),
            "relative_path": scalar(row.get("relative_path")),
            "mtime": scalar(row.get("mtime")),
            "size_bytes": scalar(row.get("size_bytes")),
            "sha256": scalar(row.get("hash_sha256")),
        }
    return references


def create_fig7_vector_protection(paths: StepPaths, temp_dir: Path) -> dict[str, Any]:
    temp_dir.mkdir(parents=True, exist_ok=True)
    run_start = snapshot_named_paths(protected_fig7_vector_output_paths(paths))
    material_run_start = snapshot_named_paths(fig7_material_output_paths(paths))
    protected_copies: dict[str, dict[str, Any]] = {}
    for label, record in run_start.items():
        if not record.get("exists"):
            continue
        source = Path(record["path"])
        copy_path = temp_dir / f"{label}{source.suffix}"
        shutil.copy2(source, copy_path)
        protected_copies[label] = {
            "copy_path": str(copy_path),
            "sha256": file_sha256(copy_path),
            "size_bytes": copy_path.stat().st_size,
        }

    reference_hashes = load_fig7_material_reference_hashes(paths)
    reference_available = {
        label: bool(reference_hashes.get(label, {}).get("available"))
        for label in fig7_material_output_paths(paths)
    }
    reference_matches = {
        label: (
            bool(reference_hashes.get(label, {}).get("available"))
            and reference_hashes.get(label, {}).get("sha256") == material_run_start.get(label, {}).get("sha256")
        )
        for label in fig7_material_output_paths(paths)
    }
    reference_mismatches = [
        label
        for label, matches in reference_matches.items()
        if reference_hashes.get(label, {}).get("available") and not matches
    ]
    reference_missing = [
        label for label, available in reference_available.items() if not available
    ]
    return {
        "created_at": now_iso(),
        "temp_dir": str(temp_dir),
        "run_start": run_start,
        "material_run_start": material_run_start,
        "protected_copies": protected_copies,
        "expected_protected_file_count": len(run_start),
        "run_start_missing": [label for label, record in run_start.items() if not record.get("exists")],
        "accepted_reference": reference_hashes,
        "accepted_reference_available": reference_available,
        "accepted_reference_matches": reference_matches,
        "accepted_reference_mismatches": reference_mismatches,
        "accepted_reference_missing": reference_missing,
        "accepted_vector_mismatches": [label for label in ["Fig8_svg", "Fig8_pdf"] if label in reference_mismatches],
        "accepted_support_mismatches": [label for label in ["Fig8_png", "Fig8_csv"] if label in reference_mismatches],
    }


def preserve_fig7_vector_outputs(
    paths: StepPaths,
    protection: dict[str, Any],
    *,
    phase: str = "after_generation",
) -> dict[str, Any]:
    before = protection["run_start"]
    before_restore = snapshot_named_paths(protected_fig7_vector_output_paths(paths))
    changed_labels: list[str] = []
    changed_details: dict[str, list[str]] = {}
    restored_labels: list[str] = []
    for label, before_record in before.items():
        if not before_record.get("exists"):
            continue
        mismatch_fields = protected_record_mismatch_fields(before_record, before_restore.get(label))
        if not mismatch_fields:
            continue
        changed_labels.append(label)
        changed_details[label] = mismatch_fields
        copy_record = protection.get("protected_copies", {}).get(label, {})
        copy_value = copy_record.get("copy_path")
        if not copy_value:
            continue
        copy_path = Path(copy_value)
        if not copy_path.exists():
            continue
        shutil.copy2(copy_path, Path(before_record["path"]))
        restored_labels.append(label)

    after_restore = snapshot_named_paths(protected_fig7_vector_output_paths(paths))
    material_after_restore = snapshot_named_paths(fig7_material_output_paths(paths))
    final_mismatches = [
        label
        for label, before_record in before.items()
        if before_record.get("exists")
        and protected_record_mismatch_fields(before_record, after_restore.get(label))
    ]
    final_mismatch_details = {
        label: protected_record_mismatch_fields(before_record, after_restore.get(label))
        for label, before_record in before.items()
        if before_record.get("exists") and protected_record_mismatch_fields(before_record, after_restore.get(label))
    }
    run_start_missing = protection.get("run_start_missing", [])
    expected_count = int(protection.get("expected_protected_file_count", len(before)))
    protected_copy_count = len(protection.get("protected_copies", {}))
    mode = "preserved_from_run_start_unchanged" if not changed_labels else "restored_from_run_start_snapshot"
    status = "pass" if not final_mismatches and not run_start_missing and protected_copy_count == expected_count else "fail"

    accepted_reference = protection.get("accepted_reference", {})
    accepted_reference_available = protection.get("accepted_reference_available", {})
    accepted_reference_missing = protection.get("accepted_reference_missing", [])
    accepted_reference_matches_after_restore = {
        label: (
            bool(accepted_reference.get(label, {}).get("available"))
            and accepted_reference.get(label, {}).get("sha256") == material_after_restore.get(label, {}).get("sha256")
        )
        for label in fig7_material_output_paths(paths)
    }
    accepted_reference_mismatches_after_restore = [
        label
        for label, matches in accepted_reference_matches_after_restore.items()
        if accepted_reference.get(label, {}).get("available") and not matches
    ]
    accepted_vector_mismatches = [
        label for label in ["Fig8_svg", "Fig8_pdf"] if label in accepted_reference_mismatches_after_restore
    ]
    accepted_support_mismatches = [
        label for label in ["Fig8_png", "Fig8_csv"] if label in accepted_reference_mismatches_after_restore
    ]
    drift_scope = "none"
    if accepted_vector_mismatches and not accepted_support_mismatches:
        drift_scope = "vector_only"
    elif accepted_reference_mismatches_after_restore:
        drift_scope = "material_outputs"
    caveat = ""
    if accepted_vector_mismatches and not accepted_support_mismatches:
        restore_note = (
            "the run-start Fig8 vector bytes were restored after redraw."
            if protected_copy_count
            else "Fig8 target vector files were intentionally allowed to update in this repair; rerun Step13 to sync material-list hashes."
        )
        caveat = (
            "Fig8 Step13 material-list hashes match current PNG/source CSV but not "
            f"{', '.join(accepted_vector_mismatches)}. No older accepted SVG bytes were found in the workspace; "
            "the remaining SVG/PDF mismatch is treated as stale pre-sync vector metadata/id drift and "
            f"{restore_note}"
        )
    elif accepted_reference_mismatches_after_restore:
        caveat = (
            "Fig8 Step13 material-list hashes do not match current files for "
            f"{', '.join(accepted_reference_mismatches_after_restore)}."
        )
    elif accepted_reference_missing:
        caveat = (
            "Step13 material-list rows were not available for "
            f"{', '.join(accepted_reference_missing)}. Fig8 vector files were preserved from this run-start snapshot."
        )

    return {
        "status": status,
        "mode": mode,
        "phase": phase,
        "preserved_from_run_start": status == "pass" and not final_mismatches,
        "protected_file_count": expected_count,
        "protected_copy_count": protected_copy_count,
        "run_start_missing": run_start_missing,
        "changed_before_restore": changed_labels,
        "changed_before_restore_details": changed_details,
        "changed_after_generation": changed_labels,
        "restored": restored_labels,
        "final_mismatches": final_mismatches,
        "final_mismatch_details": final_mismatch_details,
        "protection_created_at": protection.get("created_at"),
        "protected_copy_hashes": {
            label: {
                "sha256": record.get("sha256"),
                "size_bytes": record.get("size_bytes"),
            }
            for label, record in protection.get("protected_copies", {}).items()
        },
        "accepted_reference": accepted_reference,
        "accepted_reference_available": accepted_reference_available,
        "accepted_reference_matches": accepted_reference_matches_after_restore,
        "accepted_reference_mismatches": accepted_reference_mismatches_after_restore,
        "accepted_reference_missing": accepted_reference_missing,
        "accepted_reference_matches_all": not accepted_reference_mismatches_after_restore and not accepted_reference_missing,
        "accepted_vector_mismatches": accepted_vector_mismatches,
        "accepted_support_mismatches": accepted_support_mismatches,
        "drift_scope": drift_scope,
        "non_target_vector_metadata_drift_accepted": bool(
            status == "pass"
            and drift_scope == "vector_only"
            and not accepted_support_mismatches
            and set(accepted_vector_mismatches).issubset({"Fig8_svg", "Fig8_pdf"})
        ),
        "preservation_caveat": caveat,
        "before": public_preservation_snapshot(before),
        "material_before": public_preservation_snapshot(protection.get("material_run_start", {})),
        "before_restore": public_preservation_snapshot(before_restore),
        "after_generation": public_preservation_snapshot(before_restore),
        "after_restore": public_preservation_snapshot(after_restore),
        "material_after_restore": public_preservation_snapshot(material_after_restore),
        "note": FIG7_VECTOR_DRIFT_GUARD_NOTE,
    }


FALLBACK_CONTINENT_OUTLINES: list[tuple[str, list[tuple[float, float]]]] = [
    (
        "North America",
        [
            (-168, 72),
            (-140, 70),
            (-123, 58),
            (-126, 49),
            (-117, 33),
            (-105, 24),
            (-96, 16),
            (-82, 9),
            (-75, 18),
            (-66, 45),
            (-54, 52),
            (-58, 64),
            (-78, 72),
            (-110, 75),
        ],
    ),
    (
        "South America",
        [
            (-82, 12),
            (-66, 10),
            (-50, -2),
            (-35, -12),
            (-42, -34),
            (-55, -55),
            (-70, -52),
            (-76, -32),
            (-80, -10),
        ],
    ),
    (
        "Europe",
        [
            (-11, 36),
            (8, 58),
            (30, 70),
            (46, 58),
            (38, 42),
            (18, 35),
            (0, 36),
        ],
    ),
    (
        "Africa",
        [
            (-18, 36),
            (8, 38),
            (34, 31),
            (51, 11),
            (43, -12),
            (31, -35),
            (17, -35),
            (3, -25),
            (-13, -5),
            (-17, 18),
        ],
    ),
    (
        "Asia",
        [
            (28, 70),
            (62, 78),
            (112, 74),
            (150, 62),
            (165, 48),
            (136, 32),
            (116, 8),
            (96, 6),
            (78, 20),
            (61, 8),
            (45, 24),
            (36, 40),
        ],
    ),
    (
        "Australia",
        [
            (112, -11),
            (154, -12),
            (153, -31),
            (139, -44),
            (115, -35),
            (111, -22),
        ],
    ),
    (
        "Greenland",
        [
            (-54, 59),
            (-30, 64),
            (-22, 76),
            (-44, 83),
            (-66, 76),
            (-70, 65),
        ],
    ),
]


def local_world_basemap_candidates(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for pattern in [
        "data/**/ne_110m_land.shp",
        "data/**/ne_110m_admin_0_countries.shp",
        "data/**/naturalearth_lowres.shp",
    ]:
        candidates.extend(root.glob(pattern))

    try:
        import cartopy

        downloader = cartopy.config.get("downloaders", {}).get(("shapefiles", "natural_earth"))
        if downloader is not None:
            for resolution in ["110m", "50m"]:
                for category, name in [("physical", "land"), ("cultural", "admin_0_countries")]:
                    fmt = {"config": cartopy.config, "resolution": resolution, "category": category, "name": name}
                    candidates.append(Path(downloader.target_path(fmt)))
                    pre_path = Path(downloader.pre_downloaded_path(fmt))
                    if pre_path.is_absolute():
                        candidates.append(pre_path)
                    else:
                        candidates.append(Path(cartopy.config.get("pre_existing_data_dir", ".")) / pre_path)
    except Exception:
        pass

    pyogrio_spec = importlib.util.find_spec("pyogrio")
    if pyogrio_spec and pyogrio_spec.origin:
        candidates.append(Path(pyogrio_spec.origin).parent / "tests" / "fixtures" / "naturalearth_lowres" / "naturalearth_lowres.shp")

    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in candidates:
        path = candidate.expanduser()
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def load_local_world_basemap(root: Path) -> tuple[Any | None, str]:
    if not HAVE_GEOPANDAS:
        return None, "World basemap fallback: geopandas is unavailable; used built-in coarse continent outlines."

    for candidate in local_world_basemap_candidates(root):
        if not candidate.exists():
            continue
        try:
            world = gpd.read_file(candidate)
            if world.empty or "geometry" not in world:
                continue
            world = world[world.geometry.notna() & ~world.geometry.is_empty].copy()
            if world.empty:
                continue
            if world.crs is None:
                world = world.set_crs("EPSG:4326")
            else:
                world = world.to_crs("EPSG:4326")
            return world, f"World basemap: local Natural Earth-compatible polygons ({candidate.name}); no network access required."
        except Exception:
            continue

    return None, "World basemap fallback: no local polygon basemap was found; used built-in coarse continent outlines and graticule."


def draw_fallback_world_outlines(ax: plt.Axes) -> None:
    for _, coords in FALLBACK_CONTINENT_OUTLINES:
        patch = Polygon(
            coords,
            closed=True,
            facecolor="#EEF2F6",
            edgecolor="#CBD5E1",
            linewidth=0.55,
            zorder=0,
        )
        ax.add_patch(patch)


def add_world_map_background(ax: plt.Axes, paths: StepPaths) -> str:
    world, basemap_note = load_local_world_basemap(paths.root)
    ax.set_facecolor("#F8FAFC")
    if world is not None:
        world.plot(ax=ax, facecolor="#EEF2F6", edgecolor="#CBD5E1", linewidth=0.35, zorder=0)
    else:
        draw_fallback_world_outlines(ax)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-58, 84)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-60, 91, 30))
    ax.grid(True, color="#DCE4EC", lw=0.45, zorder=0)
    ax.axhline(0, color="#B8C3CF", lw=0.7, zorder=1)
    ax.axvline(0, color="#B8C3CF", lw=0.7, zorder=1)
    ax.tick_params(
        axis="both",
        which="both",
        left=False,
        bottom=False,
        right=False,
        top=False,
        labelleft=False,
        labelbottom=False,
        labelright=False,
        labeltop=False,
        length=0,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.1)
    return basemap_note


def fig2_marker_area(population: float, pop_min: float, pop_max: float) -> float:
    if not np.isfinite(population):
        return FIG2_POINT_SIZE_MIN
    if not np.isfinite(pop_min) or not np.isfinite(pop_max) or pop_max <= pop_min:
        return (FIG2_POINT_SIZE_MIN + FIG2_POINT_SIZE_MAX) / 2
    clipped = min(max(float(population), pop_min), pop_max)
    ratio = (clipped - pop_min) / (pop_max - pop_min)
    return FIG2_POINT_SIZE_MIN + ratio * (FIG2_POINT_SIZE_MAX - FIG2_POINT_SIZE_MIN)


def fig2_population_legend_values(population: pd.Series) -> list[float]:
    max_pop = float(numeric(population).max())
    if not np.isfinite(max_pop):
        return [1_000_000, 10_000_000, 30_000_000]
    candidates = [1_000_000, 10_000_000, 30_000_000]
    values = [v for v in candidates if v <= max_pop * 1.05]
    if not values:
        values = [float(max_pop)]
    return values[:3]


def fig2_add_city_number_fields(data: pd.DataFrame) -> pd.DataFrame:
    out = data.reset_index(drop=True).copy()
    out["city_number"] = np.arange(1, len(out) + 1, dtype=int)
    out["city_label"] = out["city_number"].astype(str)
    out["city_index_label"] = out.apply(
        lambda row: f"{int(row['city_number']):02d} {row.get('city_name_en', row.get('city_id', ''))}",
        axis=1,
    )
    out["city_number_rule"] = FIG2_CITY_NUMBER_RULE
    return out


def bbox_overlap_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float((x1 - x0) * (y1 - y0))


def bbox_outside_area(bbox: tuple[float, float, float, float], bounds: tuple[float, float, float, float]) -> float:
    left = max(bounds[0] - bbox[0], 0.0)
    bottom = max(bounds[1] - bbox[1], 0.0)
    right = max(bbox[2] - bounds[2], 0.0)
    top = max(bbox[3] - bounds[3], 0.0)
    width = max(bbox[2] - bbox[0], 0.0)
    height = max(bbox[3] - bbox[1], 0.0)
    return float(left * height + right * height + bottom * width + top * width)


def fig2_estimated_label_bbox(
    center_px: tuple[float, float],
    label: str,
    fig: plt.Figure,
    fontsize: float = FIG2_CITY_LABEL_FONTSIZE,
) -> tuple[float, float, float, float]:
    px_per_pt = fig.dpi / 72.0
    width_pt = max(6.2, len(str(label)) * fontsize * 0.55) + 2.3
    height_pt = fontsize * 1.35 + 2.1
    half_w = width_pt * px_per_pt / 2.0
    half_h = height_pt * px_per_pt / 2.0
    x, y = center_px
    return (x - half_w, y - half_h, x + half_w, y + half_h)


def fig2_point_bboxes(points_px: np.ndarray, point_sizes: pd.Series, fig: plt.Figure) -> list[tuple[float, float, float, float]]:
    px_per_pt = fig.dpi / 72.0
    bboxes: list[tuple[float, float, float, float]] = []
    for (x, y), area in zip(points_px, numeric(point_sizes).fillna(FIG2_POINT_SIZE_MIN)):
        radius_pt = math.sqrt(max(float(area), FIG2_POINT_SIZE_MIN) / math.pi) + 1.2
        radius_px = radius_pt * px_per_pt
        bboxes.append((x - radius_px, y - radius_px, x + radius_px, y + radius_px))
    return bboxes


def fig2_city_label_candidate_offsets(base_angle: float, city_number: int) -> list[tuple[float, float]]:
    angle_offsets = [0, math.pi / 4, -math.pi / 4, math.pi / 2, -math.pi / 2, math.pi, 3 * math.pi / 4, -3 * math.pi / 4]
    radii = [7.0, 10.0, 13.0, 17.0, 22.0, 28.0, 35.0, 43.0]
    rotation = (city_number % 5) * math.pi / 30.0
    candidates: list[tuple[float, float]] = []
    for radius in radii:
        for angle_offset in angle_offsets:
            angle = base_angle + angle_offset + rotation
            candidates.append((radius * math.cos(angle), radius * math.sin(angle)))
    candidates.append((0.0, 0.0))
    return candidates


def fig2_city_label_offsets(ax: plt.Axes, data: pd.DataFrame) -> pd.DataFrame:
    fig = ax.figure
    fig.canvas.draw()
    points_px = ax.transData.transform(data[["lon", "lat"]].to_numpy(dtype=float))
    bounds_bbox = ax.bbox
    margin = 3.0
    bounds = (
        float(bounds_bbox.x0 + margin),
        float(bounds_bbox.y0 + margin),
        float(bounds_bbox.x1 - margin),
        float(bounds_bbox.y1 - margin),
    )
    px_per_pt = fig.dpi / 72.0
    point_bboxes = fig2_point_bboxes(points_px, data["point_size"], fig)
    diff = points_px[:, None, :] - points_px[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    density = np.where((dist > 0) & (dist < 62), 1.0 / np.maximum(dist, 1.0), 0.0).sum(axis=1)
    order = np.lexsort((data["city_number"].to_numpy(dtype=int), -density))
    placed_bboxes: list[tuple[float, float, float, float]] = []
    offsets: dict[int, tuple[float, float, float]] = {}

    for row_pos in order:
        city_number = int(data.iloc[row_pos]["city_number"])
        label = str(data.iloc[row_pos]["city_label"])
        point = points_px[row_pos]
        neighbor_mask = (dist[row_pos] > 0) & (dist[row_pos] < 70)
        if neighbor_mask.any():
            vectors = (point - points_px[neighbor_mask]) / np.maximum(dist[row_pos, neighbor_mask][:, None], 1.0) ** 2
            vector = vectors.sum(axis=0)
            base_angle = math.atan2(float(vector[1]), float(vector[0])) if np.linalg.norm(vector) > 0 else city_number * 2.399963
        else:
            base_angle = city_number * 2.399963

        best_offset = (0.0, 0.0)
        best_bbox = fig2_estimated_label_bbox((float(point[0]), float(point[1])), label, fig)
        best_score = float("inf")
        for dx_pt, dy_pt in fig2_city_label_candidate_offsets(base_angle, city_number):
            center = (float(point[0] + dx_pt * px_per_pt), float(point[1] + dy_pt * px_per_pt))
            bbox = fig2_estimated_label_bbox(center, label, fig)
            area = max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 1.0)
            label_overlap = sum(bbox_overlap_area(bbox, old) for old in placed_bboxes) / area
            point_overlap = sum(
                bbox_overlap_area(bbox, point_box)
                for idx, point_box in enumerate(point_bboxes)
                if idx != row_pos
            ) / area
            outside = bbox_outside_area(bbox, bounds) / area
            distance_penalty = math.hypot(dx_pt, dy_pt) * 0.012
            score = label_overlap * 120.0 + point_overlap * 7.5 + outside * 180.0 + distance_penalty
            if score < best_score:
                best_score = score
                best_offset = (dx_pt, dy_pt)
                best_bbox = bbox
            if score < 0.15:
                break
        placed_bboxes.append(best_bbox)
        offsets[int(row_pos)] = (best_offset[0], best_offset[1], best_score)

    offset_df = pd.DataFrame.from_dict(
        offsets,
        orient="index",
        columns=["city_label_dx_points", "city_label_dy_points", "city_label_overlap_score"],
    ).sort_index()
    return offset_df


def fig2_draw_city_number_labels(ax: plt.Axes, data: pd.DataFrame) -> None:
    for _, row in data.iterrows():
        dx = float(row.get("city_label_dx_points", 0.0))
        dy = float(row.get("city_label_dy_points", 0.0))
        color = FIG2_QUALITY_TIER_COLORS.get(str(row.get("quality_tier_plot", "unknown")), FIG2_QUALITY_TIER_COLORS["unknown"])
        distance = math.hypot(dx, dy)
        arrowprops = None
        if distance >= 6.0:
            arrowprops = {
                "arrowstyle": "-",
                "color": "#64748B",
                "lw": 0.28,
                "alpha": 0.72,
                "shrinkA": 1.6,
                "shrinkB": 1.3,
            }
        ax.annotate(
            str(row["city_label"]),
            xy=(float(row["lon"]), float(row["lat"])),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=FIG2_CITY_LABEL_FONTSIZE,
            fontweight="bold",
            color="#111827",
            zorder=5,
            clip_on=True,
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "#FFFFFF",
                "edgecolor": color,
                "linewidth": 0.38,
                "alpha": 0.94,
            },
            arrowprops=arrowprops,
        )


def fig2_draw_city_index(ax: plt.Axes, data: pd.DataFrame) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.0, 0.985, "City index", fontsize=FIG2_CITY_INDEX_TITLE_FONTSIZE, fontweight="bold", va="top", color="#0F172A")
    ax.add_patch(
        Rectangle((0.0, 0.0), 1.0, 0.975, facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=0.6, zorder=0)
    )
    rows_per_col = int(math.ceil(len(data) / FIG2_CITY_INDEX_COLUMNS))
    y_top = 0.885
    y_bottom = 0.055
    row_gap = (y_top - y_bottom) / max(rows_per_col - 1, 1)
    col_gap = 0.018
    col_w = (1.0 - col_gap * (FIG2_CITY_INDEX_COLUMNS - 1)) / FIG2_CITY_INDEX_COLUMNS
    for col in range(1, FIG2_CITY_INDEX_COLUMNS):
        x = col * (col_w + col_gap) - col_gap / 2.0
        ax.plot([x, x], [0.04, 0.90], color="#E2E8F0", lw=0.35, zorder=1)
    for idx, (_, row) in enumerate(data.sort_values("city_number").iterrows()):
        col = idx // rows_per_col
        row_in_col = idx % rows_per_col
        x0 = col * (col_w + col_gap) + FIG2_CITY_INDEX_INNER_PAD
        y = y_top - row_in_col * row_gap
        number = f"{int(row['city_number']):02d}"
        city_name = str(row.get("city_name_en", row.get("city_id", "")))
        color = FIG2_QUALITY_TIER_COLORS.get(str(row.get("quality_tier_plot", "unknown")), FIG2_QUALITY_TIER_COLORS["unknown"])
        ax.text(x0, y, number, fontsize=FIG2_CITY_INDEX_ENTRY_FONTSIZE, fontweight="bold", color=color, ha="left", va="center", zorder=2)
        ax.text(x0 + 0.034, y, city_name, fontsize=FIG2_CITY_INDEX_ENTRY_FONTSIZE, color="#1F2937", ha="left", va="center", zorder=2)


def fig2_layout_boxes() -> dict[str, tuple[float, float, float, float]]:
    left = FIG2_LAYOUT_LEFT
    right = FIG2_LAYOUT_RIGHT
    index_y = FIG2_INDEX_Y
    index_h = FIG2_INDEX_H
    bottom_y = FIG2_BOTTOM_Y
    bottom_h = FIG2_BOTTOM_H
    map_y = bottom_y + bottom_h + FIG2_MAP_BOTTOM_GAP_CURRENT
    map_h = FIG2_MAP_TOP - map_y
    gap = FIG2_BOTTOM_HORIZONTAL_GAP
    bar_ratio = 1.18
    legend_ratio = 1.0
    usable_w = right - left
    bar_w = (usable_w - gap) * bar_ratio / (bar_ratio + legend_ratio)
    legend_w = usable_w - gap - bar_w
    return {
        "map": (left, map_y, usable_w, map_h),
        "bar": (left, bottom_y, bar_w, bottom_h),
        "legend": (left + bar_w + gap, bottom_y, legend_w, bottom_h),
        "index": (left, index_y, usable_w, index_h),
    }


def fig3_layout_boxes() -> dict[str, tuple[float, float, float, float]]:
    left = 0.045
    right = 0.985
    thumbnail_y = 0.055
    thumbnail_h = 0.550
    quality_y = 0.690
    quality_h = 0.255
    score_h = FIG3_SCORE_PANEL_COMPACT_HEIGHT
    score_y = quality_y + quality_h - score_h
    gap = 0.060
    usable_w = right - left
    box_w = (usable_w - gap) * 0.37
    legend_w = usable_w - gap - box_w
    return {
        "box": (left, score_y, box_w, score_h),
        "legend": (left + box_w + gap, quality_y, legend_w, quality_h),
        "thumbnails": (left, thumbnail_y, usable_w, thumbnail_h),
        "quality_band": (left, quality_y, usable_w, quality_h),
    }


def figure_fraction_bbox(fig: plt.Figure, bbox: Bbox | None) -> Bbox | None:
    if bbox is None:
        return None
    values = np.array([bbox.x0, bbox.y0, bbox.x1, bbox.y1], dtype=float)
    if not np.isfinite(values).all() or bbox.width <= 0 or bbox.height <= 0:
        return None
    return bbox.transformed(fig.transFigure.inverted())


def artist_figure_fraction_bbox(fig: plt.Figure, artist: Any, renderer: Any) -> Bbox | None:
    if not getattr(artist, "get_visible", lambda: True)():
        return None
    try:
        bbox = artist.get_window_extent(renderer=renderer)
    except Exception:
        try:
            bbox = artist.get_tightbbox(renderer=renderer)
        except Exception:
            return None
    return figure_fraction_bbox(fig, bbox)


def artist_anchor_figure_fraction(fig: plt.Figure, artist: Any) -> tuple[float, float]:
    if artist is None:
        return np.nan, np.nan
    try:
        xy_display = artist.get_transform().transform(artist.get_position())
        xy_figure = fig.transFigure.inverted().transform(xy_display)
    except Exception:
        return np.nan, np.nan
    return float(xy_figure[0]), float(xy_figure[1])


def union_figure_fraction_bboxes(bboxes: list[Bbox | None]) -> Bbox | None:
    valid: list[Bbox] = []
    for bbox in bboxes:
        if bbox is None:
            continue
        values = np.array([bbox.x0, bbox.y0, bbox.x1, bbox.y1], dtype=float)
        if np.isfinite(values).all() and bbox.width > 0 and bbox.height > 0:
            valid.append(bbox)
    if not valid:
        return None
    return Bbox.from_extents(
        min(float(bbox.x0) for bbox in valid),
        min(float(bbox.y0) for bbox in valid),
        max(float(bbox.x1) for bbox in valid),
        max(float(bbox.y1) for bbox in valid),
    )


def bbox_to_record(bbox: Bbox | None) -> dict[str, float] | None:
    if bbox is None:
        return None
    return {
        "x0": round(float(bbox.x0), 12),
        "y0": round(float(bbox.y0), 12),
        "x1": round(float(bbox.x1), 12),
        "y1": round(float(bbox.y1), 12),
        "width": round(float(bbox.width), 12),
        "height": round(float(bbox.height), 12),
    }


def axes_tight_figure_fraction_bbox(fig: plt.Figure, ax: plt.Axes, renderer: Any) -> Bbox | None:
    try:
        bbox = ax.get_tightbbox(renderer=renderer)
    except Exception:
        return None
    return figure_fraction_bbox(fig, bbox)


def text_artists_left_edge(fig: plt.Figure, ax: plt.Axes, renderer: Any) -> tuple[float, str]:
    candidates: list[tuple[float, str]] = []
    text_artists = [
        ax.title,
        ax._left_title,
        ax._right_title,
        ax.xaxis.label,
        ax.yaxis.label,
        *ax.get_xticklabels(),
        *ax.get_yticklabels(),
        *ax.texts,
    ]
    for artist in text_artists:
        text = getattr(artist, "get_text", lambda: "")()
        if not str(text).strip():
            continue
        bbox = artist_figure_fraction_bbox(fig, artist, renderer)
        if bbox is not None:
            candidates.append((float(bbox.x0), str(text)))
    if not candidates:
        return float(ax.get_position().x0), "bar axes left edge"
    return min(candidates, key=lambda item: item[0])


def axes_visual_right_edge(fig: plt.Figure, ax: plt.Axes, renderer: Any) -> float:
    rights = [float(ax.get_position().x1)]
    for artist in [*ax.texts, *ax.collections, *ax.lines, *ax.patches]:
        bbox = artist_figure_fraction_bbox(fig, artist, renderer)
        if bbox is not None:
            rights.append(float(bbox.x1))
    return max(rights)


def fig2_align_map_to_bottom_visual_edges(
    fig: plt.Figure,
    ax_map: plt.Axes,
    ax_bar: plt.Axes,
    ax_legend: plt.Axes,
) -> dict[str, Any]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    bottom_text_left, bottom_text_left_source = text_artists_left_edge(fig, ax_bar, renderer)
    bottom_visual_left = bottom_text_left
    legend_axes_right = float(ax_legend.get_position().x1)
    bottom_visual_right = axes_visual_right_edge(fig, ax_legend, renderer)

    original_map_position = ax_map.get_position()
    spine_linewidth = max(spine.get_linewidth() for spine in ax_map.spines.values())
    spine_half_width = (spine_linewidth / 2.0) / (fig.get_figwidth() * 72.0)
    map_axes_left = bottom_visual_left + spine_half_width
    map_axes_right = bottom_visual_right - spine_half_width
    if map_axes_right <= map_axes_left:
        map_axes_left = bottom_visual_left
        map_axes_right = bottom_visual_right
        spine_half_width = 0.0

    ax_map.set_position(
        [
            map_axes_left,
            float(original_map_position.y0),
            map_axes_right - map_axes_left,
            float(original_map_position.height),
        ]
    )
    fig.canvas.draw()

    final_map_position = ax_map.get_position()
    map_left = float(final_map_position.x0 - spine_half_width)
    map_right = float(final_map_position.x1 + spine_half_width)
    left_delta = bottom_visual_left - map_left
    right_delta = bottom_visual_right - map_right
    threshold = FIG2_ALIGNMENT_THRESHOLD
    status = "pass" if abs(left_delta) <= threshold and abs(right_delta) <= threshold else "fail"
    return {
        "map_left": round(map_left, 12),
        "map_right": round(map_right, 12),
        "map_axes_left": round(float(final_map_position.x0), 12),
        "map_axes_right": round(float(final_map_position.x1), 12),
        "bottom_text_left": round(bottom_text_left, 12),
        "bottom_text_left_source": bottom_text_left_source,
        "bottom_visual_left": round(bottom_visual_left, 12),
        "bottom_visual_right": round(bottom_visual_right, 12),
        "legend_axes_right": round(legend_axes_right, 12),
        "spine_half_width": round(spine_half_width, 12),
        "left_delta": round(left_delta, 12),
        "right_delta": round(right_delta, 12),
        "threshold": threshold,
        "status": status,
    }


def fig2_align_city_index_to_region_panel(
    fig: plt.Figure,
    ax_bar: plt.Axes,
    ax_legend: plt.Axes,
    ax_index: plt.Axes,
) -> dict[str, Any]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    region_panel_left, region_panel_left_source = text_artists_left_edge(fig, ax_bar, renderer)
    bottom_panels_right = axes_visual_right_edge(fig, ax_legend, renderer)
    if bottom_panels_right <= region_panel_left:
        region_panel_left = float(ax_bar.get_position().x0)
        bottom_panels_right = float(ax_legend.get_position().x1)
        region_panel_left_source = "fallback: bar axes left edge"

    original_index_position = ax_index.get_position()
    ax_index.set_position(
        [
            region_panel_left,
            float(original_index_position.y0),
            bottom_panels_right - region_panel_left,
            float(original_index_position.height),
        ]
    )
    fig.canvas.draw()

    final_index_position = ax_index.get_position()
    city_index_left = float(final_index_position.x0)
    city_index_right = float(final_index_position.x1)
    left_delta = city_index_left - region_panel_left
    right_delta = city_index_right - bottom_panels_right
    threshold = FIG2_ALIGNMENT_THRESHOLD
    status = "pass" if abs(left_delta) <= threshold and abs(right_delta) <= threshold else "fail"
    return {
        "region_panel_left": round(region_panel_left, 12),
        "region_panel_left_source": region_panel_left_source,
        "region_panel_left_basis": "renderer-measured Cities by region text bbox; includes y tick labels when they are the leftmost visual element",
        "bottom_panels_right": round(bottom_panels_right, 12),
        "city_index_left": round(city_index_left, 12),
        "city_index_right": round(city_index_right, 12),
        "left_delta": round(left_delta, 12),
        "right_delta": round(right_delta, 12),
        "threshold": threshold,
        "status": status,
    }


def fig2_panel_overlap_payload(
    fig: plt.Figure,
    ax_map: plt.Axes,
    ax_bar: plt.Axes,
    ax_legend: plt.Axes,
    ax_index: plt.Axes,
) -> dict[str, Any]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    map_bbox = axes_tight_figure_fraction_bbox(fig, ax_map, renderer)
    bottom_bbox = union_figure_fraction_bboxes(
        [
            axes_tight_figure_fraction_bbox(fig, ax_bar, renderer),
            axes_tight_figure_fraction_bbox(fig, ax_legend, renderer),
        ]
    )
    index_bbox = axes_tight_figure_fraction_bbox(fig, ax_index, renderer)
    map_bottom_to_bottom_top_gap = (
        float(map_bbox.y0 - bottom_bbox.y1) if map_bbox is not None and bottom_bbox is not None else np.nan
    )
    bottom_bottom_to_index_top_gap = (
        float(bottom_bbox.y0 - index_bbox.y1) if bottom_bbox is not None and index_bbox is not None else np.nan
    )
    no_overlap = (
        np.isfinite(map_bottom_to_bottom_top_gap)
        and np.isfinite(bottom_bottom_to_index_top_gap)
        and map_bottom_to_bottom_top_gap > 0
        and bottom_bottom_to_index_top_gap > 0
    )
    return {
        "status": "pass" if no_overlap else "fail",
        "map_bottom_to_bottom_row_visual_top_gap": round(map_bottom_to_bottom_top_gap, 12)
        if np.isfinite(map_bottom_to_bottom_top_gap)
        else None,
        "bottom_row_bottom_to_city_index_visual_top_gap": round(bottom_bottom_to_index_top_gap, 12)
        if np.isfinite(bottom_bottom_to_index_top_gap)
        else None,
        "map_tight_bbox": bbox_to_record(map_bbox),
        "bottom_row_tight_bbox": bbox_to_record(bottom_bbox),
        "city_index_tight_bbox": bbox_to_record(index_bbox),
    }


def collect_input_specs(paths: StepPaths) -> list[dict[str, Any]]:
    centers = choose_existing([paths.step09_type_centers_parquet, paths.step09_type_centers_csv])
    quality = choose_existing([paths.step07_city_quality_parquet, paths.step07_city_quality_csv])
    model = choose_existing([paths.step11_table_model, paths.step11_coef_table])
    return [
        {"input_name": "figure_data_accessibility_by_type", "path": paths.step11_accessibility, "required": True},
        {"input_name": "figure_data_morphotype_circuity", "path": paths.step11_circuity, "required": True},
        {"input_name": "figure_data_inequality", "path": paths.step11_inequality, "required": True},
        {"input_name": "model_coefficients", "path": model, "required": True, "preferred": str(paths.step11_table_model)},
        {"input_name": "observed_weighted_group_means", "path": paths.step11_group_means, "required": False},
        {"input_name": "type_centers", "path": centers, "required": True, "preferred": str(paths.step09_type_centers_parquet)},
        {"input_name": "city_morphotype_profiles", "path": paths.step09_city_profiles, "required": True},
        {"input_name": "local_morphotype_units", "path": paths.step09_local_types, "required": True},
        {"input_name": "od_detour_metrics", "path": paths.step10_od_metrics, "required": True},
        {"input_name": "scale_signature_long", "path": paths.step08_scale_signature_long, "required": True},
        {"input_name": "city_profile_base", "path": paths.step08_city_profile_base, "required": True},
        {"input_name": "city_quality_scores", "path": quality, "required": True, "preferred": str(paths.step07_city_quality_parquet)},
        {"input_name": "city_master", "path": paths.step01_city_master, "required": False},
        {"input_name": "city_boundaries", "path": paths.step01_city_boundaries, "required": False},
        {"input_name": "step11_qc", "path": paths.step11_qc, "required": False},
    ]


def input_metadata(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for spec in specs:
        path = Path(spec["path"])
        record: dict[str, Any] = {
            "input_name": spec["input_name"],
            "path": str(path),
            "required": bool(spec.get("required", False)),
            "exists": path.exists(),
            "preferred": spec.get("preferred", ""),
        }
        if not path.exists():
            record["status"] = "fail" if spec.get("required", False) else "warn"
            records.append(record)
            continue
        try:
            meta = file_metadata(path)
            record.update({k: v for k, v in meta.items() if k not in {"path", "exists", "sha256"}})
            record["sha256"] = meta.get("sha256")
            if path.suffix == ".csv":
                record["row_count"] = int(sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1)
            record["status"] = "pass"
        except Exception as exc:
            record["status"] = "fail" if spec.get("required", False) else "warn"
            record["metadata_error"] = repr(exc)
        records.append(record)
    return records


def write_input_report(paths: StepPaths, records: list[dict[str, Any]]) -> dict[str, Any]:
    required_failures = [r for r in records if r.get("required") and r.get("status") != "pass"]
    optional_missing = [r for r in records if not r.get("required") and not r.get("exists")]
    report = {
        "generated_at": now_iso(),
        "step": STEP_NAME,
        "status": "pass" if not required_failures else "fail",
        "required_failure_count": len(required_failures),
        "optional_missing_count": len(optional_missing),
        "records": records,
        "risk_notes": RISK_NOTES,
    }
    write_json(paths.output_dir / OUTPUT_FILES["input_report_json"], report)

    df = pd.DataFrame(records)
    display_cols = ["input_name", "status", "required", "exists", "row_count", "path"]
    display_cols = [c for c in display_cols if c in df.columns]
    lines = [
        f"# {STEP_NAME} input acceptance report",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Status: {report['status']}",
        f"- Required failures: {report['required_failure_count']}",
        f"- Optional missing inputs: {report['optional_missing_count']}",
        "",
        "## Input Files",
        "",
        markdown_table(df[display_cols] if display_cols else df, max_rows=30),
        "",
        "## Figure-Interpretation Notes",
        "",
        *[f"- {note}" for note in RISK_NOTES],
        "",
    ]
    (paths.output_dir / OUTPUT_FILES["input_report_md"]).write_text("\n".join(lines), encoding="utf-8")
    return report


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No records._"
    show = df.head(max_rows).copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    headers = list(show.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in headers) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Showing first {max_rows} of {len(df)} rows._")
    return "\n".join(lines)


def load_inputs(paths: StepPaths) -> dict[str, pd.DataFrame]:
    local_cols = [
        "city_id",
        "city_name_en",
        "country",
        "iso3",
        "region",
        "sample_group",
        "quality_tier",
        "network_type",
        "scale",
        "analysis_scope",
        "unit_id",
        "center_lon",
        "center_lat",
        "population_sum",
        "population_weight",
        "morphotype",
        "morphotype_name",
        "type_confidence",
        "edge_density_km_per_km2",
        "intersection_density_per_km2",
        "four_way_share",
        "dead_end_share",
        "segment_length_median",
        "edge_circuity_mean",
        "orientation_order",
        "road_hierarchy_entropy",
    ]
    centers_path = choose_existing([paths.step09_type_centers_parquet, paths.step09_type_centers_csv])
    quality_path = choose_existing([paths.step07_city_quality_parquet, paths.step07_city_quality_csv])
    model_path = choose_existing([paths.step11_table_model, paths.step11_coef_table])
    inputs: dict[str, pd.DataFrame] = {
        "accessibility": read_table(paths.step11_accessibility),
        "circuity": read_table(paths.step11_circuity),
        "inequality": read_table(paths.step11_inequality),
        "model": read_table(model_path),
        "type_centers": read_table(centers_path),
        "city_profiles": read_table(paths.step09_city_profiles),
        "local_types": read_table(paths.step09_local_types, columns=local_cols),
        "od_metrics": read_table(paths.step10_od_metrics),
        "scale_signature": read_table(paths.step08_scale_signature_long),
        "city_base": read_table(paths.step08_city_profile_base),
        "city_quality": read_table(quality_path),
    }
    if paths.step11_group_means.exists():
        inputs["group_means"] = read_table(paths.step11_group_means)
    else:
        inputs["group_means"] = pd.DataFrame()
    if paths.step11_qc.exists():
        inputs["step11_qc"] = read_table(paths.step11_qc)
    else:
        inputs["step11_qc"] = pd.DataFrame()
    if paths.step01_city_master.exists():
        inputs["city_master"] = read_table(paths.step01_city_master)
    else:
        inputs["city_master"] = pd.DataFrame()
    return inputs


def add_morphotype_labels(df: pd.DataFrame, mt_col: str = "morphotype") -> pd.DataFrame:
    out = df.copy()
    out["morphotype_label_en"] = out[mt_col].map(MORPHOTYPE_LABELS_EN).fillna(out[mt_col].astype(str))
    out["morphotype_plot_label"] = out[mt_col].map(MORPHOTYPE_SHORT_LABELS).fillna(out[mt_col].astype(str))
    return out


def type_dimension(type_centers: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["morphotype", "morphotype_name", "morphotype_label"] if c in type_centers.columns]
    dim = type_centers[cols].drop_duplicates("morphotype").copy()
    dim = dim[dim["morphotype"].isin(MORPHOTYPES)].sort_values("morphotype")
    dim = add_morphotype_labels(dim)
    dim = dim.rename(columns={"morphotype_name": "morphotype_name_zh"})
    return dim


def city_centroids(paths: StepPaths, inputs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, str]:
    fallback_note = "Point overview uses city centroids; no external world basemap is used."
    centroids = pd.DataFrame()
    if HAVE_GEOPANDAS and paths.step01_city_boundaries.exists():
        try:
            gdf = gpd.read_file(paths.step01_city_boundaries)
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            projected = gdf.to_crs("EPSG:6933")
            pts = projected.geometry.centroid
            centroid_gdf = gpd.GeoDataFrame(gdf[["city_id"]].copy(), geometry=pts, crs=projected.crs).to_crs("EPSG:4326")
            centroids = pd.DataFrame(
                {
                    "city_id": centroid_gdf["city_id"].astype(str),
                    "lon": centroid_gdf.geometry.x,
                    "lat": centroid_gdf.geometry.y,
                    "centroid_source": "Step01 city boundary centroid",
                }
            )
            fallback_note = (
                "City boundary centroids are plotted as a readable global point overview; "
                "urban polygons are not drawn because their global-scale footprints are tiny."
            )
        except Exception as exc:
            fallback_note = f"Boundary centroid read failed ({exc!r}); fallback to Step09 local unit center medians."
            centroids = pd.DataFrame()
    if centroids.empty:
        local = inputs["local_types"].copy()
        local = local[local["center_lon"].notna() & local["center_lat"].notna()]
        if not local.empty:
            centroids = (
                local.groupby("city_id", as_index=False)
                .agg(lon=("center_lon", "median"), lat=("center_lat", "median"))
                .assign(centroid_source="Step09 local unit center median")
            )
        else:
            base = inputs["city_base"].copy()
            lon_col = next((c for c in ["lon", "longitude", "center_lon"] if c in base.columns), None)
            lat_col = next((c for c in ["lat", "latitude", "center_lat"] if c in base.columns), None)
            if lon_col and lat_col:
                centroids = base[["city_id", lon_col, lat_col]].rename(columns={lon_col: "lon", lat_col: "lat"})
                centroids["centroid_source"] = "Step08 city profile coordinates"
    return centroids, fallback_note


def build_city_overview(inputs: dict[str, pd.DataFrame], centroids: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "city_id",
        "city_name_en",
        "country",
        "iso3",
        "region",
        "sample_group",
        "morphology_prior",
        "ucdb_area_km2",
        "ucdb_pop_2025",
        "main_analysis_sample",
        "sensitivity_sample",
    ]
    base = inputs["city_base"][[c for c in base_cols if c in inputs["city_base"].columns]].drop_duplicates("city_id").copy()
    if base.empty and not inputs.get("city_master", pd.DataFrame()).empty:
        master = inputs["city_master"]
        base = master[[c for c in base_cols if c in master.columns]].drop_duplicates("city_id").copy()
    quality_cols = [
        "city_id",
        "quality_score",
        "quality_tier",
        "quality_weight",
        "network_integrity_score",
        "historical_maturity_score",
        "poi_completeness_score",
        "population_built_support_score",
        "local_coverage_score",
        "review_flag",
        "review_reason",
    ]
    quality = inputs["city_quality"][[c for c in quality_cols if c in inputs["city_quality"].columns]].drop_duplicates("city_id")
    out = base.merge(quality, on="city_id", how="left").merge(centroids, on="city_id", how="left")
    out["population_millions"] = numeric(out.get("ucdb_pop_2025", pd.Series(np.nan, index=out.index))) / 1_000_000
    out["area_km2"] = numeric(out.get("ucdb_area_km2", pd.Series(np.nan, index=out.index)))
    out["region"] = out["region"].fillna("Unknown")
    out["sample_group"] = out["sample_group"].fillna("unknown")
    return out


def plot_fig2(paths: StepPaths, city_data: pd.DataFrame, fallback_note: str) -> tuple[FigureProduct, dict[str, Any]]:
    data = city_data.copy()
    data = data[data["lon"].notna() & data["lat"].notna()].copy()
    data = fig2_add_city_number_fields(data)
    data[FIG2_SIZE_VARIABLE] = numeric(data.get(FIG2_SIZE_VARIABLE, pd.Series(np.nan, index=data.index))).clip(lower=0)
    pop_valid = data[FIG2_SIZE_VARIABLE].dropna()
    pop_min = float(pop_valid.min()) if not pop_valid.empty else np.nan
    pop_max = float(pop_valid.max()) if not pop_valid.empty else np.nan
    data["point_size"] = data[FIG2_SIZE_VARIABLE].map(lambda value: fig2_marker_area(value, pop_min, pop_max))
    data["size_variable"] = FIG2_SIZE_VARIABLE
    data["size_variable_label"] = FIG2_SIZE_VARIABLE_LABEL
    data["point_size_units"] = "matplotlib scatter marker area, points^2"
    data["point_size_scale"] = (
        f"Linear area scale from {FIG2_SIZE_VARIABLE} to {FIG2_POINT_SIZE_MIN:.0f}-{FIG2_POINT_SIZE_MAX:.0f} points^2 "
        "within the Fig1 sample."
    )
    data["quality_tier_plot"] = data["quality_tier"].fillna("unknown").astype(str)
    data["sample_group_plot"] = data["sample_group"].fillna("unknown").astype(str)
    data["quality_tier_color"] = data["quality_tier_plot"].map(
        lambda tier: FIG2_QUALITY_TIER_COLORS.get(str(tier), FIG2_QUALITY_TIER_COLORS["unknown"])
    )
    data_path = make_output_path(paths, source_data_name("Fig1", FIG2_CANONICAL_SLUG))

    fig = plt.figure(figsize=cm_to_in(18.3, 20.4))
    layout = fig2_layout_boxes()
    ax = fig.add_axes(layout["map"])
    ax_bar = fig.add_axes(layout["bar"])
    ax_legend = fig.add_axes(layout["legend"])
    ax_index = fig.add_axes(layout["index"])
    ax.set_title("Global sample city map", loc="left", fontsize=10, fontweight="bold")
    basemap_note = add_world_map_background(ax, paths)
    ax.set_aspect("auto")
    for (tier, sample_group), group in data.groupby(["quality_tier_plot", "sample_group_plot"], dropna=False):
        ax.scatter(
            group["lon"],
            group["lat"],
            s=group["point_size"],
            color=FIG2_QUALITY_TIER_COLORS.get(str(tier), FIG2_QUALITY_TIER_COLORS["unknown"]),
            marker=SAMPLE_GROUP_MARKERS.get(str(sample_group), SAMPLE_GROUP_MARKERS["unknown"]),
            edgecolor="white",
            linewidth=0.42,
            alpha=0.92,
            zorder=3,
        )
    city_count = int(data["city_id"].nunique())
    core_count = int(data["quality_tier_plot"].eq("core").sum())
    ax.text(
        0.01,
        0.01,
        f"n={city_count} cities; core tier={core_count}. Point area = 2025 UCDB population; color = quality tier; marker = sample group.",
        transform=ax.transAxes,
        fontsize=6.6,
        color="#475569",
        va="bottom",
    )

    counts = data.groupby("region", as_index=False).agg(city_count=("city_id", "nunique")).sort_values("city_count")
    ax_bar.barh(
        counts["region"],
        counts["city_count"],
        color=[REGION_COLORS.get(str(r), "#94A3B8") for r in counts["region"]],
        edgecolor="white",
    )
    ax_bar.set_title("Cities by region", loc="left", fontsize=8, fontweight="bold")
    ax_bar.set_xlabel("Cities")
    ax_bar.grid(axis="x", color="#E2E8F0", lw=0.5)
    ax_bar.set_xlim(0, max(float(counts["city_count"].max()) * 1.18, 1.0))
    for y, x in enumerate(counts["city_count"]):
        ax_bar.text(x + 0.35, y, str(int(x)), va="center", fontsize=6.2)

    ax_legend.axis("off")
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    ax_legend.text(0.0, 0.98, "Legend", fontsize=7.6, fontweight="bold", va="top")

    quality_values = set(data["quality_tier_plot"])
    sample_values = set(data["sample_group_plot"])
    quality_items = [t for t in ["core", "sensitivity", "excluded_candidate", "review", "unknown"] if t in quality_values]
    sample_items = [g for g in ["main_80", "china_pressure_test", "unknown"] if g in sample_values]

    left_x = 0.0
    right_x = 0.52
    header_y = 0.855
    item_start_y = 0.705
    item_gap = 0.155
    population_header_y = 0.255
    population_marker_y = 0.12
    population_label_y = 0.02

    ax_legend.text(left_x, header_y, "Color: quality tier", fontsize=6.8, fontweight="bold", va="top")
    for idx, tier in enumerate(quality_items):
        y = item_start_y - idx * item_gap
        label = f"{QUALITY_TIER_LABELS.get(tier, tier)} (n={int(data['quality_tier_plot'].eq(tier).sum())})"
        ax_legend.scatter(
            [left_x + 0.045],
            [y],
            s=24,
            marker="o",
            color=FIG2_QUALITY_TIER_COLORS.get(tier, FIG2_QUALITY_TIER_COLORS["unknown"]),
            edgecolor="white",
            linewidth=0.42,
        )
        ax_legend.text(left_x + 0.105, y, label, fontsize=6.2, va="center", color="#334155")

    ax_legend.text(right_x, header_y, "Marker: sample group", fontsize=6.8, fontweight="bold", va="top")
    for idx, sample_group in enumerate(sample_items):
        y = item_start_y - idx * item_gap
        label = f"{SAMPLE_GROUP_LABELS.get(sample_group, sample_group)} (n={int(data['sample_group_plot'].eq(sample_group).sum())})"
        ax_legend.scatter(
            [right_x + 0.045],
            [y],
            s=28,
            marker=SAMPLE_GROUP_MARKERS.get(sample_group, SAMPLE_GROUP_MARKERS["unknown"]),
            facecolor="#FFFFFF",
            edgecolor="#334155",
            linewidth=0.7,
        )
        ax_legend.text(right_x + 0.105, y, label, fontsize=6.2, va="center", color="#334155")

    population_values = fig2_population_legend_values(data[FIG2_SIZE_VARIABLE])
    population_x = [0.10, 0.43, 0.76][: len(population_values)]
    ax_legend.text(0.0, population_header_y, "Area: 2025 UCDB population", fontsize=6.8, fontweight="bold", va="top")
    for x_size, pop_value in zip(population_x, population_values):
        area = fig2_marker_area(pop_value, pop_min, pop_max)
        ax_legend.scatter(
            [x_size],
            [population_marker_y],
            s=area,
            marker="o",
            facecolor="#F8FAFC",
            edgecolor="#334155",
            linewidth=0.7,
            clip_on=False,
        )
        ax_legend.text(
            x_size,
            population_label_y,
            f"{pop_value / 1_000_000:g} million",
            fontsize=6.2,
            va="center",
            ha="center",
            color="#334155",
        )

    alignment = fig2_align_map_to_bottom_visual_edges(fig, ax, ax_bar, ax_legend)
    city_index_alignment = fig2_align_city_index_to_region_panel(fig, ax_bar, ax_legend, ax_index)
    label_offsets = fig2_city_label_offsets(ax, data)
    data = data.join(label_offsets)
    data["city_label_dx_points"] = numeric(data["city_label_dx_points"]).fillna(0.0).round(3)
    data["city_label_dy_points"] = numeric(data["city_label_dy_points"]).fillna(0.0).round(3)
    data["city_label_overlap_score"] = numeric(data["city_label_overlap_score"]).fillna(0.0).round(4)
    data["city_label_placement_method"] = "Greedy screen-space offset with leader lines for dense areas."
    data["city_index_columns"] = FIG2_CITY_INDEX_COLUMNS
    fig2_draw_city_number_labels(ax, data)
    fig2_draw_city_index(ax_index, data)
    fig.canvas.draw()
    city_index_texts = [str(artist.get_text()).strip() for artist in ax_index.texts]
    city_index_right_note_present = FIG2_CITY_INDEX_RIGHT_NOTE_TEXT in city_index_texts
    city_index_title_present = "City index" in city_index_texts
    visible_map_tick_label_count = sum(
        bool(label.get_visible() and str(label.get_text()).strip())
        for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]
    )
    bottom_row_axes_top = max(float(ax_bar.get_position().y1), float(ax_legend.get_position().y1))
    map_bottom_gap_current = float(ax.get_position().y0) - bottom_row_axes_top
    fig2_no_overlap_payload = fig2_panel_overlap_payload(fig, ax, ax_bar, ax_legend, ax_index)
    bottom_small_panels_preserved = (
        ax_bar.get_visible()
        and ax_legend.get_visible()
        and float(ax_bar.get_position().y1) < float(ax.get_position().y0)
        and float(ax_legend.get_position().y1) < float(ax.get_position().y0)
        and float(ax_index.get_position().y1) < float(ax_bar.get_position().y0)
    )
    alignment.update(
        {
            "bottom_small_panels_preserved": bool(bottom_small_panels_preserved),
            "bar_panel_bottom": round(float(ax_bar.get_position().y0), 12),
            "bar_panel_top": round(float(ax_bar.get_position().y1), 12),
            "legend_panel_bottom": round(float(ax_legend.get_position().y0), 12),
            "legend_panel_top": round(float(ax_legend.get_position().y1), 12),
            "city_index_panel_bottom": round(float(ax_index.get_position().y0), 12),
            "city_index_panel_top": round(float(ax_index.get_position().y1), 12),
            "map_axes_bottom": round(float(ax.get_position().y0), 12),
            "map_axes_top": round(float(ax.get_position().y1), 12),
            "bottom_row_axes_top": round(bottom_row_axes_top, 12),
            "map_bottom_gap_previous": FIG2_MAP_BOTTOM_GAP_PREVIOUS,
            "map_bottom_gap_current": round(map_bottom_gap_current, 12),
            "map_bottom_gap_reduction": round(FIG2_MAP_BOTTOM_GAP_PREVIOUS - map_bottom_gap_current, 12),
            "map_bottom_gap_units": "figure-fraction axis coordinate",
            "map_bottom_gap_reduced": bool(
                np.isfinite(map_bottom_gap_current)
                and map_bottom_gap_current > 0
                and map_bottom_gap_current < FIG2_MAP_BOTTOM_GAP_PREVIOUS
            ),
            "city_index_columns": FIG2_CITY_INDEX_COLUMNS,
            "city_number_rule": FIG2_CITY_NUMBER_RULE,
            "city_index_left_alignment": city_index_alignment,
            "city_index_right_note_text": FIG2_CITY_INDEX_RIGHT_NOTE_TEXT,
            "city_index_right_note_present": bool(city_index_right_note_present),
            "city_index_title_present": bool(city_index_title_present),
            "panel_overlap": fig2_no_overlap_payload,
            "no_overlap": fig2_no_overlap_payload.get("status") == "pass",
            "visible_map_tick_label_count": int(visible_map_tick_label_count),
            "map_xlabel": ax.get_xlabel(),
            "map_ylabel": ax.get_ylabel(),
        }
    )
    write_csv(data, data_path)
    png, svg, pdf = save_figure(fig, paths.figure_dir / f"Fig1_{FIG2_CANONICAL_SLUG}")
    fig2_note = (
        f"{basemap_note} {fallback_note} Fig1 point area is scaled from {FIG2_SIZE_VARIABLE} "
        f"({FIG2_SIZE_VARIABLE_LABEL}); color encodes quality_tier and marker shape encodes sample_group. "
        f"City numbers follow this rule: {FIG2_CITY_NUMBER_RULE}"
    )
    return (
        FigureProduct(
            figure_id="Fig1",
            title="Global sample cities and quality tiers",
            kind="main",
            png_path=png,
            svg_path=svg,
            pdf_path=pdf,
            data_path=data_path,
            source_inputs=[
                "data/08_generate_scale_signatures_and_city_profiles/city_profiles_base.parquet",
                "data/07_generate_quality_scores_and_type_confidence/quality_scores.parquet",
                "data/01_city_boundaries_and_sample_list/city_sample/city_boundaries.gpkg",
                "local Natural Earth-compatible world basemap or built-in coarse continent-outline fallback",
            ],
            transformation=(
                "Merged city profile, quality score and centroid coordinates; plotted over a world-map background. "
                f"Point area is scaled from {FIG2_SIZE_VARIABLE}; color encodes quality_tier and marker shape encodes sample_group. "
                f"Map labels use stable city numbers and the bottom index lists number-to-city-name mappings."
            ),
            fallback_note=fig2_note,
        ),
        alignment,
    )


def fig3_quality_score_axis_limits(scores: pd.Series) -> tuple[float, float]:
    values = numeric(scores).dropna()
    if values.empty:
        return 0.0, 100.0
    lower = max(0.0, math.floor(float(values.min()) - FIG3_SCORE_AXIS_PADDING))
    upper = min(100.0, math.ceil(float(values.max()) + FIG3_SCORE_AXIS_PADDING))
    if upper - lower < 12.0:
        center = (upper + lower) / 2.0
        lower = max(0.0, math.floor(center - 6.0))
        upper = min(100.0, math.ceil(center + 6.0))
    return lower, upper


def fig3_quality_tier_color(tier: Any) -> str:
    return FIG2_QUALITY_TIER_COLORS.get(str(tier), FIG2_QUALITY_TIER_COLORS["unknown"])


def fig3_low_high_ranked_cities(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = data[data["quality_score_plot"].notna()].copy()
    if valid.empty:
        return pd.DataFrame(), pd.DataFrame()
    valid["city_id"] = valid["city_id"].astype(str)
    low = valid.sort_values(["quality_score_plot", "city_id"], ascending=[True, True]).head(FIG3_KEY_LABEL_BOTTOM_N).copy()
    high = valid.sort_values(["quality_score_plot", "city_id"], ascending=[False, True]).head(FIG3_KEY_LABEL_TOP_N).copy()
    low["road_thumbnail_rank_group"] = "low"
    high["road_thumbnail_rank_group"] = "high"
    low["road_thumbnail_rank"] = np.arange(1, len(low) + 1)
    high["road_thumbnail_rank"] = np.arange(1, len(high) + 1)
    return low, high


def fig3_rank_reason(rank_group: str) -> str:
    if rank_group == "low":
        return f"low_{FIG3_KEY_LABEL_BOTTOM_N}_quality_score"
    if rank_group == "high":
        return f"high_{FIG3_KEY_LABEL_TOP_N}_quality_score"
    return f"{rank_group}_quality_score"


def fig3_selection_reason(row: pd.Series) -> str:
    reasons = [fig3_rank_reason(str(row.get("road_thumbnail_rank_group", "")))]
    if str(row.get("quality_tier_plot", "")).strip() == "excluded_candidate":
        reasons.append("excluded_candidate_tier")
    return ";".join(reason for reason in reasons if reason)


def fig3_thumbnail_group_label(rank_group: Any) -> str:
    return {
        "low": "Low quality cities",
        "high": "High quality cities",
    }.get(str(rank_group), str(rank_group).title())


def fig3_thumbnail_group_color(rank_group: Any) -> str:
    return FIG3_ROAD_THUMBNAIL_GROUP_COLORS.get(str(rank_group), "#334155")


def fig3_component_strip_fields() -> list[str]:
    return [field for field, _ in FIG3_QUALITY_COMPONENTS]


def fig3_component_strip_abbreviations() -> list[str]:
    return [FIG3_QUALITY_COMPONENT_ABBREVIATIONS[field] for field in fig3_component_strip_fields()]


def fig3_component_strip_values(record: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for field in fig3_component_strip_fields():
        value = pd.to_numeric(pd.Series([record.get(field)]), errors="coerce").iloc[0]
        values.append(float(value) if np.isfinite(value) else np.nan)
    return values


def fig3_component_strip_value_text(record: dict[str, Any]) -> str:
    parts: list[str] = []
    for field, value in zip(fig3_component_strip_fields(), fig3_component_strip_values(record)):
        abbreviation = FIG3_QUALITY_COMPONENT_ABBREVIATIONS[field]
        parts.append(f"{abbreviation}={value:.1f}" if np.isfinite(value) else f"{abbreviation}=NA")
    return ";".join(parts)


def fig3_thumbnail_encoding_metadata(record: dict[str, Any]) -> dict[str, Any]:
    score = pd.to_numeric(
        pd.Series([record.get("quality_score_plot", record.get("quality_score"))]),
        errors="coerce",
    ).iloc[0]
    group = str(record.get("road_thumbnail_rank_group", ""))
    component_values = fig3_component_strip_values(record)
    component_values_non_null = bool(component_values and all(np.isfinite(value) for value in component_values))
    return {
        "road_thumbnail_row_group_label": fig3_thumbnail_group_label(group),
        "road_thumbnail_expected_row": "top" if group == "high" else "bottom" if group == "low" else "",
        "road_thumbnail_score_badge_present": bool(np.isfinite(score)),
        "road_thumbnail_score_badge_text": f"{score:.1f}" if np.isfinite(score) else "",
        "road_thumbnail_score_badge_color": fig3_thumbnail_group_color(group),
        "road_thumbnail_component_strip_present": component_values_non_null,
        "road_thumbnail_component_strip_color": fig3_thumbnail_group_color(group),
        "road_thumbnail_component_strip_fields": ";".join(fig3_component_strip_fields()),
        "road_thumbnail_component_strip_abbreviations": ";".join(fig3_component_strip_abbreviations()),
        "road_thumbnail_component_strip_values": fig3_component_strip_value_text(record),
        "road_thumbnail_component_strip_values_non_null": component_values_non_null,
        "road_thumbnail_component_strip_source_step": STEP07_NAME,
        "road_thumbnail_component_strip_legend": FIG3_QUALITY_COMPONENT_STRIP_LEGEND,
        "road_thumbnail_component_strip_encoding": "Five compact mini bars below each tile; bar height encodes component score/100 in R,H,P,L,B order.",
    }


def fig3_component_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for field, label in FIG3_QUALITY_COMPONENTS:
        if field not in data.columns:
            continue
        values = numeric(data[field]).dropna()
        if values.empty:
            continue
        rows.append(
            {
                "component_field": field,
                "component_label": label,
                "mean_score": float(values.mean()),
                "median_score": float(values.median()),
                "p10_score": float(values.quantile(0.10)),
                "p90_score": float(values.quantile(0.90)),
                "min_score": float(values.min()),
                "max_score": float(values.max()),
                "city_count": int(values.size),
            }
        )
    return pd.DataFrame(rows)


def fig3_component_thumbnail_group_means(
    thumbnail_records: list[dict[str, Any]],
    component_fields: list[str],
) -> dict[str, dict[str, float]]:
    means: dict[str, dict[str, float]] = {field: {"low": np.nan, "high": np.nan} for field in component_fields}
    for field in component_fields:
        for group in ["low", "high"]:
            values: list[float] = []
            for record in thumbnail_records:
                if str(record.get("road_thumbnail_rank_group", "")) != group:
                    continue
                value = pd.to_numeric(pd.Series([record.get(field)]), errors="coerce").iloc[0]
                if np.isfinite(value):
                    values.append(float(value))
            if values:
                means[field][group] = float(np.mean(values))
    return means


def fig3_select_road_thumbnail_cities(data: pd.DataFrame) -> pd.DataFrame:
    """Select Low 6 and High 6 Fig3 road-thumbnail cities from quality_score ranks."""
    low, high = fig3_low_high_ranked_cities(data)
    if low.empty and high.empty:
        return pd.DataFrame()
    selected = pd.concat([high, low], axis=0, ignore_index=False).copy()
    selected["road_thumbnail_selection_reason"] = selected.apply(fig3_selection_reason, axis=1)
    selected["road_thumbnail_display_order"] = np.arange(1, len(selected) + 1)
    selected["road_thumbnail_selection_rule"] = FIG3_ROAD_THUMBNAIL_SELECTION_RULE
    meta = selected[
        [
            "city_id",
            "road_thumbnail_rank_group",
            "road_thumbnail_rank",
            "road_thumbnail_display_order",
            "road_thumbnail_selection_reason",
            "road_thumbnail_selection_rule",
        ]
    ].copy()
    return meta.merge(data, on="city_id", how="left", suffixes=("", "_city"))


def fig3_road_index(paths: StepPaths) -> pd.DataFrame:
    path = paths.step06_dir / "road_network_file_index.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        index = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if "network_type" in index.columns:
        index = index[index["network_type"].astype(str).eq("drive")].copy()
    return index


def fig3_drive_gpkg_path(paths: StepPaths, city_id: str, road_index: pd.DataFrame) -> tuple[Path, str]:
    candidates: list[Path] = []
    if not road_index.empty and {"city_id", "gpkg_path"} <= set(road_index.columns):
        rows = road_index[road_index["city_id"].astype(str).eq(str(city_id))]
        for value in rows["gpkg_path"].dropna().astype(str):
            candidates.append(Path(value))
    candidates.append(paths.step02_dir / "city_road_network_gpkg" / "drive" / f"{city_id}_drive.gpkg")

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        resolved = path if path.is_absolute() else paths.root / path
        try:
            key = resolved.resolve()
        except Exception:
            key = resolved
        if key in seen:
            continue
        seen.add(key)
        unique.append(resolved)

    for path in unique:
        if path.exists():
            rel = str(path.relative_to(paths.root)) if path.is_relative_to(paths.root) else str(path)
            return path, f"Step02 drive GPKG edges layer ({rel})"
    fallback = unique[0]
    rel = str(fallback.relative_to(paths.root)) if fallback.is_relative_to(paths.root) else str(fallback)
    return fallback, f"missing Step02 drive GPKG candidate ({rel})"


def fig3_thumbnail_extent(lon: float, lat: float, half_side_km: float) -> tuple[float, float, float, float]:
    cos_lat = max(0.20, abs(math.cos(math.radians(lat))))
    half_lat = half_side_km / 111.32
    half_lon = half_side_km / (111.32 * cos_lat)
    return lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat


def fig3_thumbnail_center_candidates(row: pd.Series, local_types: pd.DataFrame) -> list[dict[str, Any]]:
    city_id = str(row["city_id"])
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[float, float]] = set()

    def add(source: str, lon: Any, lat: Any) -> None:
        lon_num = pd.to_numeric(pd.Series([lon]), errors="coerce").iloc[0]
        lat_num = pd.to_numeric(pd.Series([lat]), errors="coerce").iloc[0]
        if not np.isfinite(lon_num) or not np.isfinite(lat_num):
            return
        key = (round(float(lon_num), 5), round(float(lat_num), 5))
        if key in seen:
            return
        seen.add(key)
        candidates.append({"center_source": source, "lon": float(lon_num), "lat": float(lat_num)})

    add(
        "Fig3 city centroid",
        row.get("lon"),
        row.get("lat"),
    )
    if local_types.empty or "city_id" not in local_types.columns:
        return candidates

    sub = local_types[local_types["city_id"].astype(str).eq(city_id)].copy()
    if "network_type" in sub.columns:
        sub = sub[sub["network_type"].astype(str).eq("drive")]
    if sub.empty or not {"center_lon", "center_lat"} <= set(sub.columns):
        return candidates

    sub = sub[sub["center_lon"].notna() & sub["center_lat"].notna()].copy()
    if sub.empty:
        return candidates
    sub["_population_sort"] = numeric(sub.get("population_sum", pd.Series(0.0, index=sub.index)), default=0.0)
    sub["_edge_density_sort"] = numeric(sub.get("edge_density_km_per_km2", pd.Series(0.0, index=sub.index)), default=0.0)
    sub["_scale_rank"] = sub.get("scale", pd.Series("", index=sub.index)).map({"hex_1km": 0, "hex_2km": 1}).fillna(9)
    sub["_lon_round"] = numeric(sub["center_lon"]).round(5)
    sub["_lat_round"] = numeric(sub["center_lat"]).round(5)
    sub = (
        sub.sort_values(["_scale_rank", "_population_sort", "_edge_density_sort"], ascending=[True, False, False])
        .drop_duplicates(["_lon_round", "_lat_round"])
        .head(12)
    )
    for _, local_row in sub.iterrows():
        source = f"Step09 local {local_row.get('scale', 'unit')} center"
        add(source, local_row["center_lon"], local_row["center_lat"])
    return candidates


def fig3_read_road_edges(path: Path, bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    if not HAVE_GEOPANDAS:
        return pd.DataFrame()
    try:
        return gpd.read_file(path, layer="edges", bbox=bbox, columns=["geometry", "highway", "length"])
    except TypeError:
        return gpd.read_file(path, layer="edges", bbox=bbox)


def fig3_major_road_mask(gdf: pd.DataFrame) -> pd.Series:
    highway = gdf.get("highway", pd.Series("", index=gdf.index)).fillna("").astype(str).str.lower()
    return highway.str.contains("motorway|trunk|primary|secondary|tertiary", regex=True)


def fig3_downsample_road_edges(gdf: pd.DataFrame, city_id: str) -> tuple[pd.DataFrame, bool]:
    if len(gdf) <= FIG3_ROAD_THUMBNAIL_MAX_PLOTTED_EDGES:
        return gdf.copy(), False

    seed = int(hashlib.sha256(str(city_id).encode("utf-8")).hexdigest()[:8], 16)
    major_mask = fig3_major_road_mask(gdf)
    major = gdf[major_mask].copy()
    major_keep = min(len(major), max(0, int(FIG3_ROAD_THUMBNAIL_MAX_PLOTTED_EDGES * 0.35)))
    if major_keep and len(major) > major_keep:
        major["_thumbnail_length"] = numeric(major.get("length", pd.Series(np.nan, index=major.index)))
        major = major.sort_values("_thumbnail_length", ascending=False).head(major_keep)
    remaining_keep = FIG3_ROAD_THUMBNAIL_MAX_PLOTTED_EDGES - len(major)
    remaining = gdf.drop(index=major.index, errors="ignore")
    if remaining_keep <= 0:
        sampled = major
    elif len(remaining) > remaining_keep:
        sampled = pd.concat([major, remaining.sample(n=remaining_keep, random_state=seed)], axis=0)
    else:
        sampled = pd.concat([major, remaining], axis=0)
    return sampled.drop(columns=["_thumbnail_length"], errors="ignore").sort_index(), True


def fig3_geometry_segments(geometries: pd.Series) -> list[np.ndarray]:
    segments: list[np.ndarray] = []

    def add_geometry(geom: Any) -> None:
        if geom is None or getattr(geom, "is_empty", True):
            return
        geom_type = getattr(geom, "geom_type", "")
        if geom_type == "LineString":
            coords = np.asarray(geom.coords, dtype=float)
            if coords.ndim == 2 and coords.shape[0] >= 2:
                segments.append(coords[:, :2])
        elif geom_type in {"MultiLineString", "GeometryCollection"}:
            for part in getattr(geom, "geoms", []):
                add_geometry(part)

    for geometry in geometries:
        add_geometry(geometry)
    return segments


def fig3_project_thumbnail_segments(
    segments: list[np.ndarray],
    center_lon: float,
    center_lat: float,
) -> list[np.ndarray]:
    cos_lat = max(0.20, abs(math.cos(math.radians(center_lat))))
    projected: list[np.ndarray] = []
    for segment in segments:
        coords = np.asarray(segment, dtype=float)
        if coords.ndim != 2 or coords.shape[0] < 2 or coords.shape[1] < 2:
            continue
        x_km = (coords[:, 0] - center_lon) * 111.32 * cos_lat
        y_km = (coords[:, 1] - center_lat) * 111.32
        projected.append(np.column_stack([x_km, y_km]))
    return projected


def fig3_load_road_thumbnail(
    paths: StepPaths,
    row: pd.Series,
    local_types: pd.DataFrame,
    road_index: pd.DataFrame,
) -> dict[str, Any]:
    city_id = str(row["city_id"])
    gpkg_path, source_label = fig3_drive_gpkg_path(paths, city_id, road_index)
    base_record: dict[str, Any] = {
        "city_id": city_id,
        "road_thumbnail_source": source_label,
        "road_thumbnail_status": "not_loaded",
        "road_thumbnail_status_detail": "",
        "road_thumbnail_extent": "",
        "road_thumbnail_center_source": "",
        "road_thumbnail_center_lon": np.nan,
        "road_thumbnail_center_lat": np.nan,
        "road_thumbnail_half_side_km": np.nan,
        "road_thumbnail_fixed_crop_half_side_km": FIG3_ROAD_THUMBNAIL_FIXED_HALF_SIDE_KM,
        "road_thumbnail_fixed_crop_width_km": FIG3_ROAD_THUMBNAIL_FIXED_SIDE_KM,
        "road_thumbnail_fixed_crop_height_km": FIG3_ROAD_THUMBNAIL_FIXED_SIDE_KM,
        "road_thumbnail_fixed_crop_scale_rule": (
            f"Fixed square local-km crop: {FIG3_ROAD_THUMBNAIL_FIXED_SIDE_KM:g} km x "
            f"{FIG3_ROAD_THUMBNAIL_FIXED_SIDE_KM:g} km for every Fig3 road thumbnail."
        ),
        "road_thumbnail_recentered": False,
        "road_thumbnail_recenter_reason": "",
        "road_thumbnail_scale_bar_present": False,
        "road_thumbnail_scale_bar_km": FIG3_ROAD_THUMBNAIL_SCALE_BAR_KM,
        "road_thumbnail_scale_bar_label": f"{FIG3_ROAD_THUMBNAIL_SCALE_BAR_KM:g} km",
        "road_thumbnail_extent_width_km": np.nan,
        "road_thumbnail_extent_height_km": np.nan,
        "road_thumbnail_extent_aspect": np.nan,
        "road_thumbnail_crop_shape": "",
        "road_thumbnail_coordinate_units": "",
        "road_thumbnail_edges_in_extent": 0,
        "road_thumbnail_edges_plotted": 0,
        "road_thumbnail_downsampled": False,
        "_local_segments": [],
        "_major_segments": [],
        "_xlim": (0.0, 1.0),
        "_ylim": (0.0, 1.0),
    }
    if not HAVE_GEOPANDAS:
        base_record.update(
            {
                "road_thumbnail_status": "placeholder_geopandas_unavailable",
                "road_thumbnail_status_detail": "geopandas unavailable; could not read Step02 GPKG edges.",
            }
        )
        return base_record
    if not gpkg_path.exists():
        base_record.update(
            {
                "road_thumbnail_status": "placeholder_missing_gpkg",
                "road_thumbnail_status_detail": f"Drive GPKG not found: {gpkg_path}",
            }
        )
        return base_record

    fixed_half_side_km = FIG3_ROAD_THUMBNAIL_FIXED_HALF_SIDE_KM
    best: dict[str, Any] | None = None
    errors: list[str] = []
    for center in fig3_thumbnail_center_candidates(row, local_types):
        bbox = fig3_thumbnail_extent(center["lon"], center["lat"], fixed_half_side_km)
        try:
            gdf = fig3_read_road_edges(gpkg_path, bbox)
        except Exception as exc:
            errors.append(f"{center['center_source']} fixed {fixed_half_side_km:g}km half-side: {exc!r}")
            continue
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy() if "geometry" in gdf else pd.DataFrame()
        candidate = {
            "gdf": gdf,
            "bbox": bbox,
            "center": center,
            "half_side_km": fixed_half_side_km,
            "edge_count": len(gdf),
            "recentered": False,
            "recenter_reason": "",
        }
        if best is None or candidate["edge_count"] > best["edge_count"]:
            best = candidate

    if best is not None and best["edge_count"] > 0 and "geometry" in best["gdf"]:
        bounds = best["gdf"].total_bounds
        if np.isfinite(bounds).all() and bounds[2] > bounds[0] and bounds[3] > bounds[1]:
            road_center = {
                "center_source": f"{best['center']['center_source']} recentered on local road-edge bounds",
                "lon": float((bounds[0] + bounds[2]) / 2.0),
                "lat": float((bounds[1] + bounds[3]) / 2.0),
            }
            bbox = fig3_thumbnail_extent(road_center["lon"], road_center["lat"], fixed_half_side_km)
            try:
                gdf = fig3_read_road_edges(gpkg_path, bbox)
                gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy() if "geometry" in gdf else pd.DataFrame()
                recentered_edge_count = len(gdf)
                improvement = recentered_edge_count / max(1, int(best["edge_count"]))
                should_recenter = (
                    (int(best["edge_count"]) < FIG3_ROAD_THUMBNAIL_MIN_EDGES and recentered_edge_count > int(best["edge_count"]))
                    or (
                        recentered_edge_count >= FIG3_ROAD_THUMBNAIL_MIN_EDGES
                        and improvement >= FIG3_ROAD_THUMBNAIL_RECENTER_MIN_IMPROVEMENT
                    )
                )
                if should_recenter:
                    best = {
                        "gdf": gdf,
                        "bbox": bbox,
                        "center": road_center,
                        "half_side_km": fixed_half_side_km,
                        "edge_count": recentered_edge_count,
                        "recentered": True,
                        "recenter_reason": (
                            f"Same fixed {FIG3_ROAD_THUMBNAIL_FIXED_SIDE_KM:g} km x "
                            f"{FIG3_ROAD_THUMBNAIL_FIXED_SIDE_KM:g} km crop recentered on local road-edge bounds; "
                            f"edge count improved from {int(best['edge_count'])} to {recentered_edge_count}."
                        ),
                    }
            except Exception as exc:
                errors.append(f"{road_center['center_source']} fixed {fixed_half_side_km:g}km half-side: {exc!r}")

    if best is None or best["edge_count"] <= 0:
        base_record.update(
            {
                "road_thumbnail_status": "placeholder_empty_gpkg_bbox",
                "road_thumbnail_status_detail": "; ".join(errors[:3]) if errors else "No drive edges found in candidate thumbnail windows.",
            }
        )
        return base_record

    gdf_plot, downsampled = fig3_downsample_road_edges(best["gdf"], city_id)
    major_mask = fig3_major_road_mask(gdf_plot)
    local_segments = fig3_geometry_segments(gdf_plot.loc[~major_mask, "geometry"])
    major_segments = fig3_geometry_segments(gdf_plot.loc[major_mask, "geometry"])
    if not local_segments and not major_segments:
        base_record.update(
            {
                "road_thumbnail_status": "placeholder_empty_geometry",
                "road_thumbnail_status_detail": "Edges were read but no LineString geometry could be plotted.",
            }
        )
        return base_record

    bbox = best["bbox"]
    center = best["center"]
    half_side_km = float(best["half_side_km"])
    center_lon = float(center["lon"])
    center_lat = float(center["lat"])
    local_segments = fig3_project_thumbnail_segments(local_segments, center_lon, center_lat)
    major_segments = fig3_project_thumbnail_segments(major_segments, center_lon, center_lat)
    width_km = half_side_km * 2.0
    height_km = half_side_km * 2.0
    extent = f"lon_min={bbox[0]:.6f};lat_min={bbox[1]:.6f};lon_max={bbox[2]:.6f};lat_max={bbox[3]:.6f}"
    base_record.update(
        {
            "road_thumbnail_status": "ok_gpkg_edges_bbox",
            "road_thumbnail_status_detail": (
                f"Read Step02 drive GPKG edges layer in a fixed "
                f"{FIG3_ROAD_THUMBNAIL_FIXED_SIDE_KM:g} km x {FIG3_ROAD_THUMBNAIL_FIXED_SIDE_KM:g} km "
                f"local-km crop; recentered={bool(best.get('recentered', False))}."
            ),
            "road_thumbnail_extent": extent,
            "road_thumbnail_center_source": center["center_source"],
            "road_thumbnail_center_lon": center_lon,
            "road_thumbnail_center_lat": center_lat,
            "road_thumbnail_half_side_km": half_side_km,
            "road_thumbnail_fixed_crop_half_side_km": FIG3_ROAD_THUMBNAIL_FIXED_HALF_SIDE_KM,
            "road_thumbnail_fixed_crop_width_km": FIG3_ROAD_THUMBNAIL_FIXED_SIDE_KM,
            "road_thumbnail_fixed_crop_height_km": FIG3_ROAD_THUMBNAIL_FIXED_SIDE_KM,
            "road_thumbnail_recentered": bool(best.get("recentered", False)),
            "road_thumbnail_recenter_reason": str(best.get("recenter_reason", "")),
            "road_thumbnail_scale_bar_present": True,
            "road_thumbnail_scale_bar_km": FIG3_ROAD_THUMBNAIL_SCALE_BAR_KM,
            "road_thumbnail_scale_bar_label": f"{FIG3_ROAD_THUMBNAIL_SCALE_BAR_KM:g} km",
            "road_thumbnail_extent_width_km": width_km,
            "road_thumbnail_extent_height_km": height_km,
            "road_thumbnail_extent_aspect": width_km / height_km if height_km else np.nan,
            "road_thumbnail_crop_shape": "square_local_km_window",
            "road_thumbnail_coordinate_units": "local_km_from_thumbnail_center",
            "road_thumbnail_edges_in_extent": int(best["edge_count"]),
            "road_thumbnail_edges_plotted": int(len(gdf_plot)),
            "road_thumbnail_downsampled": bool(downsampled),
            "_local_segments": local_segments,
            "_major_segments": major_segments,
            "_xlim": (-half_side_km, half_side_km),
            "_ylim": (-half_side_km, half_side_km),
        }
    )
    return base_record


def fig3_prepare_road_thumbnails(
    paths: StepPaths,
    data: pd.DataFrame,
    local_types: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]], str]:
    selection = fig3_select_road_thumbnail_cities(data)
    if selection.empty:
        return pd.DataFrame(), [], "Fig3 road thumbnails: no eligible city with quality_score was available."

    road_index = fig3_road_index(paths)
    records: list[dict[str, Any]] = []
    for _, row in selection.iterrows():
        record = row.to_dict()
        record.update(fig3_load_road_thumbnail(paths, row, local_types, road_index))
        record.update(fig3_thumbnail_encoding_metadata(record))
        records.append(record)

    public_records = [
        {key: value for key, value in record.items() if not str(key).startswith("_")}
        for record in records
    ]
    meta = pd.DataFrame(public_records)
    status_counts = meta["road_thumbnail_status"].value_counts(dropna=False).to_dict() if not meta.empty else {}
    note = (
        "Fig3 road thumbnails use Step02 local drive-network GPKG `edges` layers cropped around city centers; "
        f"all selected thumbnails use a fixed {FIG3_ROAD_THUMBNAIL_FIXED_SIDE_KM:g} km x "
        f"{FIG3_ROAD_THUMBNAIL_FIXED_SIDE_KM:g} km local-km window with a "
        f"{FIG3_ROAD_THUMBNAIL_SCALE_BAR_KM:g} km scale bar; selection rule: "
        f"{FIG3_ROAD_THUMBNAIL_SELECTION_RULE} Status counts: {status_counts}."
    )
    return meta, records, note


def fig3_thumbnail_grid_geometry(
    fig: plt.Figure,
    layout: dict[str, tuple[float, float, float, float]],
    count: int,
) -> dict[str, float | int]:
    left, bottom, width, height = layout["thumbnails"]
    cols = FIG3_ROAD_THUMBNAIL_COLUMNS
    rows = max(1, math.ceil(count / cols))
    fig_w, fig_h = fig.get_size_inches()
    col_gap = FIG3_ROAD_THUMBNAIL_COLUMN_GAP
    cell_w = max(0.01, (width - col_gap * (cols - 1)) / cols)
    cell_h = cell_w * fig_w / fig_h
    if rows > 1:
        row_gap_available = max(0.0, height - rows * cell_h) / (rows - 1)
        row_gap = min(FIG3_ROAD_THUMBNAIL_ROW_GAP, row_gap_available)
    else:
        row_gap = 0.0
    grid_w = cols * cell_w + (cols - 1) * col_gap
    grid_h = rows * cell_h + (rows - 1) * row_gap
    vertical_slack = max(0.0, height - grid_h)
    centered_start_y = bottom + vertical_slack / 2.0
    top_aligned_start_y = bottom + vertical_slack
    return {
        "left": float(left),
        "bottom": float(bottom),
        "width": float(width),
        "height": float(height),
        "cols": int(cols),
        "rows": int(rows),
        "cell_w": float(cell_w),
        "cell_h": float(cell_h),
        "col_gap": float(col_gap),
        "row_gap": float(row_gap),
        "grid_w": float(grid_w),
        "grid_h": float(grid_h),
        "vertical_slack": float(vertical_slack),
        "centered_start_y": float(centered_start_y),
        "centered_top": float(centered_start_y + grid_h),
        "top_aligned_start_y": float(top_aligned_start_y),
        "top_aligned_top": float(top_aligned_start_y + grid_h),
    }


def fig3_thumbnail_axes_boxes(
    fig: plt.Figure,
    layout: dict[str, tuple[float, float, float, float]],
    count: int,
) -> list[tuple[float, float, float, float]]:
    geometry = fig3_thumbnail_grid_geometry(fig, layout, count)
    left = float(geometry["left"])
    cols = int(geometry["cols"])
    rows = int(geometry["rows"])
    cell_w = float(geometry["cell_w"])
    cell_h = float(geometry["cell_h"])
    col_gap = float(geometry["col_gap"])
    row_gap = float(geometry["row_gap"])
    grid_h = float(geometry["grid_h"])
    start_x = left
    start_y = (
        float(geometry["top_aligned_start_y"])
        if FIG3_ROAD_THUMBNAIL_VERTICAL_ANCHOR == "top"
        else float(geometry["centered_start_y"])
    )
    boxes: list[tuple[float, float, float, float]] = []
    for idx in range(count):
        row = idx // cols
        col = idx % cols
        x = start_x + col * (cell_w + col_gap)
        y = start_y + (rows - 1 - row) * (cell_h + row_gap)
        boxes.append((x, y, cell_w, cell_h))
    return boxes


def fig3_thumbnail_title(record: dict[str, Any]) -> str:
    group_label = {
        "low": "Low",
        "high": "High",
    }.get(str(record.get("road_thumbnail_rank_group", "")), str(record.get("road_thumbnail_rank_group", "City")).title())
    rank = record.get("road_thumbnail_rank", "")
    city_name = str(record.get("city_name_en", record.get("city_id", "")))
    reason = str(record.get("road_thumbnail_selection_reason", ""))
    tier = str(record.get("quality_tier_plot", ""))
    excluded_marker = "*" if tier == "excluded_candidate" or "excluded_candidate_tier" in reason else ""
    prefix = f"{group_label} {rank}{excluded_marker}".strip()
    max_city_chars = 16
    city_short = city_name if len(city_name) <= max_city_chars else f"{city_name[: max_city_chars - 3].rstrip()}..."
    return f"{prefix}: {city_short}"


def fig3_draw_thumbnail_badge(ax: plt.Axes, record: dict[str, Any]) -> None:
    if not bool(record.get("road_thumbnail_score_badge_present", False)):
        return
    badge_text = str(record.get("road_thumbnail_score_badge_text", "")).strip()
    if not badge_text:
        return
    ax.text(
        0.965,
        0.955,
        badge_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=4.45,
        fontweight="bold",
        color="#FFFFFF",
        zorder=8,
        bbox={
            "boxstyle": "round,pad=0.16,rounding_size=0.20",
            "facecolor": str(record.get("road_thumbnail_score_badge_color", "#334155")),
            "edgecolor": "#111827",
            "linewidth": 0.28,
            "alpha": 0.96,
        },
    )


def fig3_draw_thumbnail_scale_bar(ax: plt.Axes, record: dict[str, Any]) -> None:
    if not bool(record.get("road_thumbnail_scale_bar_present", False)):
        return
    xlim = record.get("_xlim", (np.nan, np.nan))
    ylim = record.get("_ylim", (np.nan, np.nan))
    try:
        x_min, x_max = float(xlim[0]), float(xlim[1])
        y_min, y_max = float(ylim[0]), float(ylim[1])
    except Exception:
        return
    if not all(np.isfinite(value) for value in [x_min, x_max, y_min, y_max]) or x_max <= x_min or y_max <= y_min:
        return
    bar_km = float(record.get("road_thumbnail_scale_bar_km", FIG3_ROAD_THUMBNAIL_SCALE_BAR_KM))
    if not np.isfinite(bar_km) or bar_km <= 0:
        return
    x_span = x_max - x_min
    y_span = y_max - y_min
    bar_km = min(bar_km, x_span * 0.42)
    x0 = x_min + x_span * 0.075
    y0 = y_min + y_span * 0.080
    tick = y_span * 0.018
    ax.plot([x0, x0 + bar_km], [y0, y0], color="#000000", lw=0.86, solid_capstyle="butt", zorder=7)
    ax.plot([x0, x0], [y0 - tick, y0 + tick], color="#000000", lw=0.62, zorder=7)
    ax.plot([x0 + bar_km, x0 + bar_km], [y0 - tick, y0 + tick], color="#000000", lw=0.62, zorder=7)
    ax.text(
        x0 + bar_km / 2.0,
        y0 + y_span * 0.028,
        str(record.get("road_thumbnail_scale_bar_label", f"{bar_km:g} km")),
        ha="center",
        va="bottom",
        fontsize=3.65,
        color="#000000",
        zorder=8,
        bbox={"facecolor": "#FFFFFF", "edgecolor": "none", "alpha": 0.78, "pad": 0.25},
    )


def fig3_draw_thumbnail_component_strip(fig: plt.Figure, ax: plt.Axes, record: dict[str, Any]) -> None:
    if not bool(record.get("road_thumbnail_component_strip_present", False)):
        return
    values = fig3_component_strip_values(record)
    if not values or not all(np.isfinite(value) for value in values):
        return
    pos = ax.get_position()
    abbreviations = fig3_component_strip_abbreviations()
    group_color = fig3_thumbnail_group_color(record.get("road_thumbnail_rank_group", ""))
    slot_w = float(pos.width) / len(values)
    bar_w = slot_w * 0.46
    max_h = min(0.0105, float(pos.height) * 0.095)
    bar_base_y = max(0.002, float(pos.y0) - 0.0205)
    label_y = bar_base_y + max_h + 0.0032
    for idx, (abbr, value) in enumerate(zip(abbreviations, values)):
        x_center = float(pos.x0) + slot_w * (idx + 0.5)
        x0 = x_center - bar_w / 2.0
        fig.add_artist(
            Rectangle(
                (x0, bar_base_y),
                bar_w,
                max_h,
                transform=fig.transFigure,
                facecolor="#F8FAFC",
                edgecolor="#CBD5E1",
                linewidth=0.24,
                zorder=20,
            )
        )
        fill_h = max_h * max(0.0, min(float(value), 100.0)) / 100.0
        fig.add_artist(
            Rectangle(
                (x0, bar_base_y),
                bar_w,
                fill_h,
                transform=fig.transFigure,
                facecolor=group_color,
                edgecolor="none",
                alpha=0.86,
                zorder=21,
            )
        )
        fig.text(
            x_center,
            label_y,
            abbr,
            ha="center",
            va="bottom",
            fontsize=3.3,
            color="#334155",
            zorder=22,
        )


def fig3_draw_road_thumbnail(ax: plt.Axes, record: dict[str, Any]) -> None:
    ax.set_facecolor(FIG3_ROAD_THUMBNAIL_BACKGROUND_COLOR)
    local_segments = record.get("_local_segments", [])
    major_segments = record.get("_major_segments", [])
    if local_segments:
        ax.add_collection(
            LineCollection(
                local_segments,
                colors=FIG3_ROAD_THUMBNAIL_ROAD_COLOR,
                linewidths=0.095,
                alpha=1.0,
                capstyle="round",
                joinstyle="round",
                zorder=2,
            )
        )
    if major_segments:
        ax.add_collection(
            LineCollection(
                major_segments,
                colors=FIG3_ROAD_THUMBNAIL_ROAD_COLOR,
                linewidths=0.235,
                alpha=1.0,
                capstyle="round",
                joinstyle="round",
                zorder=3,
            )
        )
    if not local_segments and not major_segments:
        ax.text(0.5, 0.54, "road data\nunavailable", ha="center", va="center", fontsize=5.5, color="#111827")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        ax.set_xlim(*record.get("_xlim", (0.0, 1.0)))
        ax.set_ylim(*record.get("_ylim", (0.0, 1.0)))
        fig3_draw_thumbnail_scale_bar(ax, record)
    fig3_draw_thumbnail_badge(ax, record)
    try:
        ax.set_box_aspect(1)
    except Exception:
        pass
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.56)
        spine.set_color(FIG3_ROAD_THUMBNAIL_BORDER_COLOR)
    ax.set_title(fig3_thumbnail_title(record), fontsize=4.55, color="#111827", pad=1.0, linespacing=0.82)


def fig3_axes_layout_metadata(
    ax_box: plt.Axes,
    ax_summary: plt.Axes,
    summary_title_artist: Any | None,
    summary_bottom_artist: Any | None,
    definition_artist: Any | None,
    component_fields: list[str],
    quality_score_axis: tuple[float, float],
    quality_score_extent: tuple[float, float],
    thumbnail_records: list[dict[str, Any]],
    thumbnail_axes: list[plt.Axes],
    layout: dict[str, tuple[float, float, float, float]],
    thumbnail_header_artists: list[Any] | None = None,
) -> dict[str, Any]:
    fig = ax_box.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    thumbnail_header_artists = thumbnail_header_artists or []
    box_pos = ax_box.get_position()
    summary_pos = ax_summary.get_position()
    thumbnail_positions = [axis.get_position() for axis in thumbnail_axes]
    quality_panel_gap = float(summary_pos.x0) - float(box_pos.x1)
    thumbnail_top = max((float(pos.y1) for pos in thumbnail_positions), default=np.nan)
    thumbnail_bottom = min((float(pos.y0) for pos in thumbnail_positions), default=np.nan)
    thumbnail_left = min((float(pos.x0) for pos in thumbnail_positions), default=np.nan)
    thumbnail_right = max((float(pos.x1) for pos in thumbnail_positions), default=np.nan)
    layout_thumbnail_left, _, layout_thumbnail_width, _ = layout["thumbnails"]
    layout_thumbnail_right = layout_thumbnail_left + layout_thumbnail_width
    thumbnail_x_positions = sorted(
        {round(float(pos.x0), 12): pos for pos in thumbnail_positions}.values(),
        key=lambda pos: float(pos.x0),
    )
    thumbnail_column_gaps = [
        float(right_pos.x0) - float(left_pos.x1)
        for left_pos, right_pos in zip(thumbnail_x_positions, thumbnail_x_positions[1:])
    ]
    thumbnail_left_layout_delta = abs(float(thumbnail_left) - float(layout_thumbnail_left)) if np.isfinite(thumbnail_left) else np.nan
    thumbnail_right_layout_delta = abs(float(thumbnail_right) - float(layout_thumbnail_right)) if np.isfinite(thumbnail_right) else np.nan
    thumbnail_layout_aligned = (
        np.isfinite(thumbnail_left_layout_delta)
        and np.isfinite(thumbnail_right_layout_delta)
        and thumbnail_left_layout_delta <= FIG3_ROAD_THUMBNAIL_ALIGNMENT_TOLERANCE
        and thumbnail_right_layout_delta <= FIG3_ROAD_THUMBNAIL_ALIGNMENT_TOLERANCE
    )
    fig_w, fig_h = fig.get_size_inches()
    thumbnail_axis_metrics: list[dict[str, float]] = []
    for pos in thumbnail_positions:
        width_in = float(pos.width) * float(fig_w)
        height_in = float(pos.height) * float(fig_h)
        aspect = width_in / height_in if height_in else np.nan
        thumbnail_axis_metrics.append(
            {
                "width_in": width_in,
                "height_in": height_in,
                "aspect_width_over_height": aspect,
                "abs_aspect_error": abs(aspect - 1.0) if np.isfinite(aspect) else np.nan,
            }
        )
    thumbnail_square_errors = [
        metric["abs_aspect_error"] for metric in thumbnail_axis_metrics if np.isfinite(metric["abs_aspect_error"])
    ]
    thumbnail_extent_errors: list[float] = []
    for record in thumbnail_records:
        xlim = record.get("_xlim", (np.nan, np.nan))
        ylim = record.get("_ylim", (np.nan, np.nan))
        try:
            x_span = abs(float(xlim[1]) - float(xlim[0]))
            y_span = abs(float(ylim[1]) - float(ylim[0]))
        except Exception:
            continue
        if np.isfinite(x_span) and np.isfinite(y_span):
            thumbnail_extent_errors.append(abs(x_span - y_span))
    thumbnail_groups = pd.Series(
        [record.get("road_thumbnail_rank_group", "") for record in thumbnail_records], dtype="object"
    ).value_counts(dropna=False).to_dict()
    thumbnail_group_row_positions: dict[str, dict[str, float]] = {}
    for record, pos in zip(thumbnail_records, thumbnail_positions):
        group = str(record.get("road_thumbnail_rank_group", ""))
        if not group:
            continue
        group_pos = thumbnail_group_row_positions.setdefault(
            group,
            {
                "count": 0,
                "y_center_min": float("inf"),
                "y_center_max": float("-inf"),
                "y_center_mean": 0.0,
                "x_left": float("inf"),
                "x_right": float("-inf"),
            },
        )
        y_center = float((pos.y0 + pos.y1) / 2.0)
        group_pos["count"] += 1
        group_pos["y_center_min"] = min(group_pos["y_center_min"], y_center)
        group_pos["y_center_max"] = max(group_pos["y_center_max"], y_center)
        group_pos["y_center_mean"] += y_center
        group_pos["x_left"] = min(group_pos["x_left"], float(pos.x0))
        group_pos["x_right"] = max(group_pos["x_right"], float(pos.x1))
    for group_pos in thumbnail_group_row_positions.values():
        if group_pos["count"]:
            group_pos["y_center_mean"] = group_pos["y_center_mean"] / group_pos["count"]
    low_row_mean = thumbnail_group_row_positions.get("low", {}).get("y_center_mean", np.nan)
    high_row_mean = thumbnail_group_row_positions.get("high", {}).get("y_center_mean", np.nan)
    low_row_above_high = bool(np.isfinite(low_row_mean) and np.isfinite(high_row_mean) and low_row_mean > high_row_mean)
    high_row_above_low = bool(np.isfinite(low_row_mean) and np.isfinite(high_row_mean) and high_row_mean > low_row_mean)
    thumbnail_badge_count = sum(bool(record.get("road_thumbnail_score_badge_present", False)) for record in thumbnail_records)
    thumbnail_component_strip_count = sum(
        bool(record.get("road_thumbnail_component_strip_present", False)) for record in thumbnail_records
    )
    thumbnail_scale_bar_count = sum(bool(record.get("road_thumbnail_scale_bar_present", False)) for record in thumbnail_records)
    fixed_half_values = [
        float(value)
        for value in [
            pd.to_numeric(pd.Series([record.get("road_thumbnail_fixed_crop_half_side_km")]), errors="coerce").iloc[0]
            for record in thumbnail_records
        ]
        if np.isfinite(value)
    ]
    fixed_width_values = [
        float(value)
        for value in [
            pd.to_numeric(pd.Series([record.get("road_thumbnail_fixed_crop_width_km")]), errors="coerce").iloc[0]
            for record in thumbnail_records
        ]
        if np.isfinite(value)
    ]
    fixed_height_values = [
        float(value)
        for value in [
            pd.to_numeric(pd.Series([record.get("road_thumbnail_fixed_crop_height_km")]), errors="coerce").iloc[0]
            for record in thumbnail_records
        ]
        if np.isfinite(value)
    ]
    scale_bar_values = [
        float(value)
        for value in [
            pd.to_numeric(pd.Series([record.get("road_thumbnail_scale_bar_km")]), errors="coerce").iloc[0]
            for record in thumbnail_records
        ]
        if np.isfinite(value)
    ]
    recentered_records = [
        {
            "city_id": record.get("city_id"),
            "city_name_en": record.get("city_name_en"),
            "center_source": record.get("road_thumbnail_center_source"),
            "reason": record.get("road_thumbnail_recenter_reason"),
        }
        for record in thumbnail_records
        if bool(record.get("road_thumbnail_recentered", False))
    ]
    component_y_positions = (
        np.linspace(FIG3_COMPONENT_SUMMARY_Y_TOP, FIG3_COMPONENT_SUMMARY_Y_BOTTOM, len(component_fields)).tolist()
        if component_fields
        else []
    )
    component_gaps = [abs(float(a) - float(b)) for a, b in zip(component_y_positions, component_y_positions[1:])]
    component_min_gap = min(component_gaps) if component_gaps else np.nan
    component_group_means = fig3_component_thumbnail_group_means(thumbnail_records, component_fields)
    component_low_marker_count = sum(
        np.isfinite(float(component_group_means.get(field, {}).get("low", np.nan))) for field in component_fields
    )
    component_high_marker_count = sum(
        np.isfinite(float(component_group_means.get(field, {}).get("high", np.nan))) for field in component_fields
    )
    thumbnail_status_counts = pd.Series(
        [record.get("road_thumbnail_status", "") for record in thumbnail_records], dtype="object"
    ).value_counts(dropna=False).to_dict()
    quality_band = layout.get(
        "quality_band",
        (
            min(float(box_pos.x0), float(summary_pos.x0)),
            min(float(box_pos.y0), float(summary_pos.y0)),
            max(float(box_pos.x1), float(summary_pos.x1)) - min(float(box_pos.x0), float(summary_pos.x0)),
            max(float(box_pos.y1), float(summary_pos.y1)) - min(float(box_pos.y0), float(summary_pos.y0)),
        ),
    )
    quality_band_height = float(quality_band[3])
    score_panel_height_ratio = float(box_pos.height) / quality_band_height if quality_band_height > 0 else np.nan
    score_panel_compact = bool(
        np.isfinite(score_panel_height_ratio)
        and score_panel_height_ratio < 0.75
        and float(box_pos.height) <= FIG3_SCORE_PANEL_COMPACT_HEIGHT + 1e-9
        and float(box_pos.height) < FIG3_SCORE_PANEL_PREVIOUS_HEIGHT
    )
    component_summary_texts = [str(text.get_text()).strip() for text in ax_summary.texts if str(text.get_text()).strip()]
    removed_component_footnote_count = sum(
        text.startswith("quality_score: project-derived Step07")
        or text.startswith("Not an official OSM field; not a pure")
        for text in component_summary_texts
    )
    summary_box_top_delta = float(summary_pos.y1) - float(box_pos.y1)
    summary_box_bottom_delta = float(summary_pos.y0) - float(box_pos.y0)
    summary_box_height_delta = float(summary_pos.height) - float(box_pos.height)
    summary_box_vertical_aligned = bool(
        abs(summary_box_top_delta) <= FIG3_ROAD_THUMBNAIL_ALIGNMENT_TOLERANCE
        and abs(summary_box_bottom_delta) <= FIG3_ROAD_THUMBNAIL_ALIGNMENT_TOLERANCE
        and abs(summary_box_height_delta) <= FIG3_ROAD_THUMBNAIL_ALIGNMENT_TOLERANCE
    )
    score_title_artist = ax_box._left_title
    score_title_bbox = artist_figure_fraction_bbox(fig, score_title_artist, renderer)
    summary_title_bbox = artist_figure_fraction_bbox(fig, summary_title_artist, renderer)
    score_title_anchor_x, score_title_anchor_y = artist_anchor_figure_fraction(fig, score_title_artist)
    summary_title_anchor_x, summary_title_anchor_y = artist_anchor_figure_fraction(fig, summary_title_artist)
    score_visible_bbox = axes_tight_figure_fraction_bbox(fig, ax_box, renderer)
    summary_visible_bbox = union_figure_fraction_bboxes(
        [
            artist_figure_fraction_bbox(fig, artist, renderer)
            for artist in [
                summary_title_artist,
                summary_bottom_artist,
                definition_artist,
                *ax_summary.texts,
                *ax_summary.lines,
                *ax_summary.patches,
                *ax_summary.collections,
            ]
        ]
    )

    def bbox_edge(bbox: Bbox | None, attr: str) -> float:
        return float(getattr(bbox, attr)) if bbox is not None else np.nan

    thumbnail_visible_bbox = union_figure_fraction_bboxes(
        [
            axes_tight_figure_fraction_bbox(fig, axis, renderer)
            for axis in thumbnail_axes
        ]
        + [
            artist_figure_fraction_bbox(fig, artist, renderer)
            for artist in thumbnail_header_artists
        ]
    )
    quality_visible_bbox = union_figure_fraction_bboxes([score_visible_bbox, summary_visible_bbox])

    score_title_top = bbox_edge(score_title_bbox, "y1")
    summary_title_top = bbox_edge(summary_title_bbox, "y1")
    score_title_bottom = bbox_edge(score_title_bbox, "y0")
    summary_title_bottom = bbox_edge(summary_title_bbox, "y0")
    title_top_delta = summary_title_top - score_title_top
    title_bottom_delta = summary_title_bottom - score_title_bottom
    title_anchor_y_delta = summary_title_anchor_y - score_title_anchor_y
    title_alignment_ok = bool(
        np.isfinite(title_top_delta)
        and np.isfinite(title_anchor_y_delta)
        and abs(title_top_delta) <= FIG3_VISUAL_ALIGNMENT_TOLERANCE
        and abs(title_anchor_y_delta) <= FIG3_VISUAL_ALIGNMENT_TOLERANCE
    )
    score_visible_top = bbox_edge(score_visible_bbox, "y1")
    summary_visible_top = bbox_edge(summary_visible_bbox, "y1")
    score_visible_bottom = bbox_edge(score_visible_bbox, "y0")
    summary_visible_bottom = bbox_edge(summary_visible_bbox, "y0")
    visible_top_delta = summary_visible_top - score_visible_top
    visible_bottom_delta = summary_visible_bottom - score_visible_bottom
    visible_height_delta = (
        float(summary_visible_bbox.height) - float(score_visible_bbox.height)
        if summary_visible_bbox is not None and score_visible_bbox is not None
        else np.nan
    )
    visible_block_alignment_ok = bool(
        np.isfinite(visible_top_delta)
        and np.isfinite(visible_bottom_delta)
        and abs(visible_top_delta) <= 0.045
        and abs(visible_bottom_delta) <= 0.075
    )
    quality_visible_bottom = bbox_edge(quality_visible_bbox, "y0")
    quality_visible_top = bbox_edge(quality_visible_bbox, "y1")
    thumbnail_visible_top = bbox_edge(thumbnail_visible_bbox, "y1")
    thumbnail_visible_bottom = bbox_edge(thumbnail_visible_bbox, "y0")
    quality_thumbnail_visible_gap = (
        quality_visible_bottom - thumbnail_visible_top
        if np.isfinite(quality_visible_bottom) and np.isfinite(thumbnail_visible_top)
        else np.nan
    )
    thumbnail_grid_geometry = fig3_thumbnail_grid_geometry(fig, layout, len(thumbnail_records)) if thumbnail_records else {}
    quality_block_bottom = min(float(box_pos.y0), float(summary_pos.y0))
    previous_thumbnail_top = float(thumbnail_grid_geometry.get("centered_top", np.nan)) if thumbnail_grid_geometry else np.nan
    quality_to_thumbnail_vertical_gap = (
        quality_block_bottom - float(thumbnail_top)
        if np.isfinite(thumbnail_top)
        else np.nan
    )
    previous_quality_to_thumbnail_vertical_gap = (
        quality_block_bottom - previous_thumbnail_top
        if np.isfinite(previous_thumbnail_top)
        else np.nan
    )
    quality_to_thumbnail_gap_reduction = (
        previous_quality_to_thumbnail_vertical_gap - quality_to_thumbnail_vertical_gap
        if np.isfinite(previous_quality_to_thumbnail_vertical_gap) and np.isfinite(quality_to_thumbnail_vertical_gap)
        else np.nan
    )
    quality_to_thumbnail_gap_reduced = bool(
        np.isfinite(quality_to_thumbnail_vertical_gap)
        and np.isfinite(previous_quality_to_thumbnail_vertical_gap)
        and quality_to_thumbnail_vertical_gap <= FIG3_QUALITY_TO_THUMBNAIL_VERTICAL_GAP_MAX
        and quality_to_thumbnail_gap_reduction >= FIG3_QUALITY_TO_THUMBNAIL_VERTICAL_GAP_MIN_REDUCTION
    )
    quality_thumbnail_blocks_nonoverlap = bool(
        np.isfinite(quality_thumbnail_visible_gap)
        and quality_thumbnail_visible_gap >= FIG3_QUALITY_THUMBNAIL_VISIBLE_GAP_MIN
        and bool(np.isfinite(thumbnail_top) and thumbnail_top < quality_block_bottom)
    )
    quality_explanation_area = float(box_pos.width * box_pos.height + summary_pos.width * summary_pos.height)
    thumbnail_axes_area = float(sum(float(pos.width * pos.height) for pos in thumbnail_positions))
    thumbnail_quality_ratio = (
        thumbnail_axes_area / quality_explanation_area
        if quality_explanation_area > 0
        else np.nan
    )
    component_definitions = {
        FIG3_QUALITY_COMPONENT_ABBREVIATIONS[field]: label
        for field, label in FIG3_QUALITY_COMPONENTS
        if field in component_fields
    }
    definition_bbox = artist_figure_fraction_bbox(fig, definition_artist, renderer)
    summary_bottom_bbox = artist_figure_fraction_bbox(fig, summary_bottom_artist, renderer)
    component_figure_texts = component_summary_texts + [
        str(artist.get_text()).strip()
        for artist in [summary_title_artist, summary_bottom_artist, definition_artist]
        if artist is not None and str(artist.get_text()).strip()
    ]
    component_combined_legend_hits = [
        text for text in component_figure_texts if text in FIG3_COMPONENT_VISUAL_LEGEND_COMBINED_TEXTS
    ]
    component_visual_legend_items = [
        {"key": key, "symbol": symbol, "label": label}
        for key, symbol, label in FIG3_COMPONENT_VISUAL_LEGEND_ITEMS
    ]
    component_visual_legend_labels = {key: label for key, _, label in FIG3_COMPONENT_VISUAL_LEGEND_ITEMS}
    quality_explanation_present = bool(
        box_pos.width > 0
        and summary_pos.width > 0
        and component_fields
        and definition_artist is not None
        and "project-derived" in FIG3_QUALITY_SCORE_DEFINITION
    )
    return {
        "world_map_panel_present": False,
        "map_axes_created": False,
        "map_encoding_present": False,
        "map_point_area_legend_present": False,
        "map_city_labels_present": False,
        "quality_explanation_panel_present": quality_explanation_present,
        "quality_score_by_tier_panel_present": True,
        "quality_score_by_tier_panel_compact": score_panel_compact,
        "quality_score_by_tier_panel_height": round(float(box_pos.height), 12),
        "quality_score_by_tier_panel_previous_height": float(FIG3_SCORE_PANEL_PREVIOUS_HEIGHT),
        "quality_score_by_tier_panel_compact_height": float(FIG3_SCORE_PANEL_COMPACT_HEIGHT),
        "quality_score_by_tier_compact_height_ratio": float(score_panel_height_ratio)
        if np.isfinite(score_panel_height_ratio)
        else np.nan,
        "quality_upper_band_height": round(quality_band_height, 12),
        "quality_score_definition_note_present": bool(definition_artist is not None and definition_bbox is not None),
        "box_panel_bottom": round(float(box_pos.y0), 12),
        "box_panel_top": round(float(box_pos.y1), 12),
        "box_panel_left": round(float(box_pos.x0), 12),
        "box_panel_right": round(float(box_pos.x1), 12),
        "summary_panel_left": round(float(summary_pos.x0), 12),
        "summary_panel_right": round(float(summary_pos.x1), 12),
        "summary_panel_bottom": round(float(summary_pos.y0), 12),
        "summary_panel_top": round(float(summary_pos.y1), 12),
        "summary_panel_height": round(float(summary_pos.height), 12),
        "quality_panel_horizontal_gap": round(quality_panel_gap, 12),
        "quality_panels_nonoverlap": bool(quality_panel_gap > 0.0 and float(summary_pos.x0) > float(box_pos.x1)),
        "quality_explanation_area_fraction": quality_explanation_area,
        "road_thumbnail_axes_area_fraction": thumbnail_axes_area,
        "road_thumbnail_to_quality_explanation_area_ratio": float(thumbnail_quality_ratio) if np.isfinite(thumbnail_quality_ratio) else np.nan,
        "road_thumbnail_panel_primary_visual_area": bool(
            np.isfinite(thumbnail_quality_ratio) and thumbnail_quality_ratio >= FIG3_THUMBNAIL_PRIMARY_AREA_RATIO_MIN
        ),
        "quality_component_summary_aligns_quality_score_panel": summary_box_vertical_aligned,
        "quality_component_summary_score_panel_top_delta": round(summary_box_top_delta, 12),
        "quality_component_summary_score_panel_bottom_delta": round(summary_box_bottom_delta, 12),
        "quality_component_summary_score_panel_height_delta": round(summary_box_height_delta, 12),
        "quality_component_summary_alignment_tolerance": float(FIG3_ROAD_THUMBNAIL_ALIGNMENT_TOLERANCE),
        "quality_component_summary_visual_alignment_tolerance": float(FIG3_VISUAL_ALIGNMENT_TOLERANCE),
        "quality_summary_title_aligned_with_score_by_tier_title": title_alignment_ok,
        "quality_summary_title_top": round(summary_title_top, 12),
        "score_by_tier_title_top": round(score_title_top, 12),
        "quality_summary_title_bottom": round(summary_title_bottom, 12),
        "score_by_tier_title_bottom": round(score_title_bottom, 12),
        "quality_summary_title_anchor_y": round(float(summary_title_anchor_y), 12),
        "score_by_tier_title_anchor_y": round(float(score_title_anchor_y), 12),
        "quality_summary_title_top_delta": round(title_top_delta, 12),
        "quality_summary_title_bottom_delta": round(title_bottom_delta, 12),
        "quality_summary_title_anchor_y_delta": round(title_anchor_y_delta, 12),
        "quality_summary_visible_block_aligned_with_score_by_tier_visible_block": visible_block_alignment_ok,
        "quality_summary_visible_block_top": round(summary_visible_top, 12),
        "score_by_tier_visible_block_top": round(score_visible_top, 12),
        "quality_summary_visible_block_bottom": round(summary_visible_bottom, 12),
        "score_by_tier_visible_block_bottom": round(score_visible_bottom, 12),
        "quality_summary_visible_block_top_delta": round(visible_top_delta, 12),
        "quality_summary_visible_block_bottom_delta": round(visible_bottom_delta, 12),
        "quality_summary_visible_block_height_delta": round(visible_height_delta, 12),
        "definition_note_top": float(definition_bbox.y1) if definition_bbox is not None else np.nan,
        "definition_note_bottom": float(definition_bbox.y0) if definition_bbox is not None else np.nan,
        "summary_bottom_note_top": float(summary_bottom_bbox.y1) if summary_bottom_bbox is not None else np.nan,
        "summary_bottom_note_bottom": float(summary_bottom_bbox.y0) if summary_bottom_bbox is not None else np.nan,
        "quality_component_panel_present": bool(component_fields),
        "quality_component_compact_legend_present": bool(component_definitions),
        "quality_component_fields": component_fields,
        "quality_component_abbreviations": fig3_component_strip_abbreviations(),
        "quality_component_definitions": component_definitions,
        "quality_component_summary_title_y": round(float(summary_title_anchor_y), 12),
        "quality_component_summary_y_top": float(max(component_y_positions)) if component_y_positions else np.nan,
        "quality_component_summary_y_bottom": float(min(component_y_positions)) if component_y_positions else np.nan,
        "quality_component_summary_y_positions": [float(value) for value in component_y_positions],
        "quality_component_summary_min_vertical_gap": float(component_min_gap) if np.isfinite(component_min_gap) else np.nan,
        "quality_component_summary_bar_height": float(FIG3_COMPONENT_SUMMARY_BAR_HEIGHT),
        "quality_component_summary_previous_bar_height": float(FIG3_COMPONENT_SUMMARY_PREVIOUS_BAR_HEIGHT),
        "quality_component_summary_visual_y_span": float(FIG3_COMPONENT_SUMMARY_Y_TOP - FIG3_COMPONENT_SUMMARY_Y_BOTTOM),
        "quality_component_summary_previous_y_span": float(FIG3_COMPONENT_SUMMARY_PREVIOUS_Y_SPAN),
        "quality_component_summary_axis_ticks": [float(value) for value in FIG3_COMPONENT_SUMMARY_AXIS_TICKS],
        "quality_component_summary_enrichments": [
            "p10_p90_distribution_band",
            "all_city_mean_dot",
            "low_6_component_marker",
            "high_6_component_marker",
            "0_50_100_reference_ticks",
        ],
        "quality_component_visual_legend_items": component_visual_legend_items,
        "quality_component_visual_legend_label_strategy": "attached_to_each_symbol",
        "quality_component_visual_legend_attached_labels": component_visual_legend_labels,
        "quality_component_visual_legend_combined_summary_absent": bool(not component_combined_legend_hits),
        "quality_component_visual_legend_combined_summary_hits": component_combined_legend_hits,
        "quality_component_summary_low_marker_count": int(component_low_marker_count),
        "quality_component_summary_high_marker_count": int(component_high_marker_count),
        "quality_component_summary_bar_height_enlarged": bool(
            FIG3_COMPONENT_SUMMARY_BAR_HEIGHT > FIG3_COMPONENT_SUMMARY_PREVIOUS_BAR_HEIGHT
        ),
        "quality_component_summary_visual_span_enlarged": bool(
            (FIG3_COMPONENT_SUMMARY_Y_TOP - FIG3_COMPONENT_SUMMARY_Y_BOTTOM) > FIG3_COMPONENT_SUMMARY_PREVIOUS_Y_SPAN
        ),
        "quality_component_footnote_lines_removed": bool(removed_component_footnote_count == 0),
        "quality_component_removed_footnote_line_count": int(removed_component_footnote_count),
        "quality_component_score_definition_note_location": "under_score_by_tier_panel",
        "quality_component_summary_vertical_spacing_pass": bool(
            np.isfinite(component_min_gap) and component_min_gap >= FIG3_COMPONENT_SUMMARY_MIN_VERTICAL_GAP
        ),
        "quality_score_axis_min": float(quality_score_axis[0]),
        "quality_score_axis_max": float(quality_score_axis[1]),
        "quality_score_data_min": float(quality_score_extent[0]),
        "quality_score_data_max": float(quality_score_extent[1]),
        "quality_score_axis_method": f"data min/max padded by {FIG3_SCORE_AXIS_PADDING:g} score points and clipped to 0-100",
        "quality_score_not_official_osm_field": True,
        "quality_score_not_pure_road_morphology_score": True,
        "road_thumbnail_panel_present": bool(thumbnail_records),
        "road_thumbnail_city_count": len(thumbnail_records),
        "road_thumbnail_axes_count": len(thumbnail_axes),
        "road_thumbnail_grid": f"{max(1, math.ceil(len(thumbnail_records) / FIG3_ROAD_THUMBNAIL_COLUMNS))}x{FIG3_ROAD_THUMBNAIL_COLUMNS}",
        "road_thumbnail_group_counts": thumbnail_groups,
        "road_thumbnail_low_count": int(thumbnail_groups.get("low", 0)),
        "road_thumbnail_high_count": int(thumbnail_groups.get("high", 0)),
        "road_thumbnail_group_row_labels": {"low": "Low quality cities", "high": "High quality cities"},
        "road_thumbnail_group_colors": dict(FIG3_ROAD_THUMBNAIL_GROUP_COLORS),
        "road_thumbnail_group_row_positions": thumbnail_group_row_positions,
        "road_thumbnail_two_group_rows": bool(
            len(thumbnail_group_row_positions) == 2
            and int(thumbnail_groups.get("low", 0)) == FIG3_KEY_LABEL_BOTTOM_N
            and int(thumbnail_groups.get("high", 0)) == FIG3_KEY_LABEL_TOP_N
        ),
        "road_thumbnail_low_row_above_high": low_row_above_high,
        "road_thumbnail_high_row_above_low": high_row_above_low,
        "road_thumbnail_top_row_group": "high" if high_row_above_low else "low" if low_row_above_high else "",
        "road_thumbnail_bottom_row_group": "low" if high_row_above_low else "high" if low_row_above_high else "",
        "road_thumbnail_badge_count": int(thumbnail_badge_count),
        "road_thumbnail_badge_numeric": bool(thumbnail_badge_count == len(thumbnail_records)),
        "road_thumbnail_component_strip_count": int(thumbnail_component_strip_count),
        "road_thumbnail_component_strip_fields": fig3_component_strip_fields(),
        "road_thumbnail_component_strip_abbreviations": fig3_component_strip_abbreviations(),
        "road_thumbnail_component_strip_legend": FIG3_QUALITY_COMPONENT_STRIP_LEGEND,
        "road_thumbnail_scale_bar_count": int(thumbnail_scale_bar_count),
        "road_thumbnail_scale_bar_km": float(FIG3_ROAD_THUMBNAIL_SCALE_BAR_KM),
        "road_thumbnail_fixed_crop_half_side_km": float(FIG3_ROAD_THUMBNAIL_FIXED_HALF_SIDE_KM),
        "road_thumbnail_fixed_crop_side_km": float(FIG3_ROAD_THUMBNAIL_FIXED_SIDE_KM),
        "road_thumbnail_fixed_crop_half_side_values": fixed_half_values,
        "road_thumbnail_fixed_crop_width_values": fixed_width_values,
        "road_thumbnail_fixed_crop_height_values": fixed_height_values,
        "road_thumbnail_scale_bar_values": scale_bar_values,
        "road_thumbnail_recentered_count": len(recentered_records),
        "road_thumbnail_recentered_records": recentered_records,
        "road_thumbnail_axes_square_max_abs_aspect_error": float(max(thumbnail_square_errors)) if thumbnail_square_errors else np.nan,
        "road_thumbnail_axes_square": bool(thumbnail_square_errors and max(thumbnail_square_errors) <= 1e-6),
        "road_thumbnail_extent_square_max_abs_error_km": float(max(thumbnail_extent_errors)) if thumbnail_extent_errors else np.nan,
        "road_thumbnail_extent_square": bool(thumbnail_extent_errors and max(thumbnail_extent_errors) <= 1e-6),
        "road_thumbnail_coordinate_units": "local_km_from_thumbnail_center",
        "road_thumbnail_panel_left": float(thumbnail_left) if np.isfinite(thumbnail_left) else np.nan,
        "road_thumbnail_panel_right": float(thumbnail_right) if np.isfinite(thumbnail_right) else np.nan,
        "road_thumbnail_layout_left": float(layout_thumbnail_left),
        "road_thumbnail_layout_right": float(layout_thumbnail_right),
        "road_thumbnail_panel_left_layout_delta": float(thumbnail_left_layout_delta) if np.isfinite(thumbnail_left_layout_delta) else np.nan,
        "road_thumbnail_panel_right_layout_delta": float(thumbnail_right_layout_delta) if np.isfinite(thumbnail_right_layout_delta) else np.nan,
        "road_thumbnail_panel_spans_wide_layout": bool(thumbnail_layout_aligned),
        "road_thumbnail_column_gap_target": float(FIG3_ROAD_THUMBNAIL_COLUMN_GAP),
        "road_thumbnail_column_gap_min": float(min(thumbnail_column_gaps)) if thumbnail_column_gaps else np.nan,
        "road_thumbnail_column_gap_max": float(max(thumbnail_column_gaps)) if thumbnail_column_gaps else np.nan,
        "road_thumbnail_horizontal_spacing_expanded": bool(
            thumbnail_column_gaps and min(thumbnail_column_gaps) >= FIG3_ROAD_THUMBNAIL_COLUMN_GAP - 1e-9
        ),
        "road_thumbnail_background_color": FIG3_ROAD_THUMBNAIL_BACKGROUND_COLOR,
        "road_thumbnail_road_color": FIG3_ROAD_THUMBNAIL_ROAD_COLOR,
        "road_thumbnail_border_color": FIG3_ROAD_THUMBNAIL_BORDER_COLOR,
        "road_thumbnail_style_note": "white tile background; black drive-road network; black tile border",
        "road_thumbnail_panel_top": float(thumbnail_top) if np.isfinite(thumbnail_top) else np.nan,
        "road_thumbnail_panel_bottom": float(thumbnail_bottom) if np.isfinite(thumbnail_bottom) else np.nan,
        "road_thumbnail_vertical_anchor": FIG3_ROAD_THUMBNAIL_VERTICAL_ANCHOR,
        "road_thumbnail_grid_geometry": thumbnail_grid_geometry,
        "quality_to_thumbnail_vertical_gap": float(quality_to_thumbnail_vertical_gap) if np.isfinite(quality_to_thumbnail_vertical_gap) else np.nan,
        "quality_to_thumbnail_vertical_gap_previous_centered": float(previous_quality_to_thumbnail_vertical_gap) if np.isfinite(previous_quality_to_thumbnail_vertical_gap) else np.nan,
        "quality_to_thumbnail_vertical_gap_reduction": float(quality_to_thumbnail_gap_reduction) if np.isfinite(quality_to_thumbnail_gap_reduction) else np.nan,
        "quality_to_thumbnail_vertical_gap_reduced": quality_to_thumbnail_gap_reduced,
        "quality_to_thumbnail_vertical_gap_max": float(FIG3_QUALITY_TO_THUMBNAIL_VERTICAL_GAP_MAX),
        "quality_to_thumbnail_vertical_gap_min_reduction": float(FIG3_QUALITY_TO_THUMBNAIL_VERTICAL_GAP_MIN_REDUCTION),
        "quality_thumbnail_visible_gap": float(quality_thumbnail_visible_gap) if np.isfinite(quality_thumbnail_visible_gap) else np.nan,
        "quality_thumbnail_visible_gap_min": float(FIG3_QUALITY_THUMBNAIL_VISIBLE_GAP_MIN),
        "quality_visible_block_top": float(quality_visible_top) if np.isfinite(quality_visible_top) else np.nan,
        "quality_visible_block_bottom": float(quality_visible_bottom) if np.isfinite(quality_visible_bottom) else np.nan,
        "thumbnail_visible_block_top": float(thumbnail_visible_top) if np.isfinite(thumbnail_visible_top) else np.nan,
        "thumbnail_visible_block_bottom": float(thumbnail_visible_bottom) if np.isfinite(thumbnail_visible_bottom) else np.nan,
        "quality_and_thumbnail_blocks_nonoverlap": quality_thumbnail_blocks_nonoverlap,
        "road_thumbnail_panel_below_quality_explanation": bool(
            np.isfinite(thumbnail_top) and thumbnail_top < min(float(box_pos.y0), float(summary_pos.y0))
        ),
        "road_thumbnail_status_counts": thumbnail_status_counts,
        "road_thumbnail_selection_rule": FIG3_ROAD_THUMBNAIL_SELECTION_RULE,
        "road_thumbnail_display_order": [
            {
                "display_order": record.get("road_thumbnail_display_order"),
                "city_id": record.get("city_id"),
                "city_name_en": record.get("city_name_en"),
                "rank_group": record.get("road_thumbnail_rank_group"),
                "rank": record.get("road_thumbnail_rank"),
                "quality_score": record.get("quality_score_plot", record.get("quality_score")),
            }
            for record in thumbnail_records
        ],
        "quality_score_definition": FIG3_QUALITY_SCORE_DEFINITION,
    }


def plot_fig3(
    paths: StepPaths,
    city_data: pd.DataFrame,
    local_types: pd.DataFrame,
    fallback_note: str,
) -> tuple[FigureProduct, dict[str, Any]]:
    data = city_data.copy()
    data = data[data["lon"].notna() & data["lat"].notna()].copy()
    data["quality_score"] = numeric(data["quality_score"])
    data["quality_weight"] = numeric(data.get("quality_weight", pd.Series(1, index=data.index)), default=1.0)
    data["quality_score_plot"] = data["quality_score"].clip(0, 100)
    data["quality_tier_plot"] = data["quality_tier"].fillna("unknown").astype(str)
    data["quality_tier_color"] = data["quality_tier_plot"].map(fig3_quality_tier_color)
    data["color_variable"] = "quality_tier"
    data["color_variable_label"] = "Quality tier (categorical)"
    data["cmap"] = ""
    data["palette"] = "Categorical quality tier palette reused from Fig1"
    data["palette_note"] = FIG3_QUALITY_TIER_COMMUNICATION_NOTE
    data["quality_tier_analysis_color"] = data["quality_tier_plot"].map(QUALITY_TIER_COLORS_FOR_ANALYSIS).fillna(
        QUALITY_TIER_COLORS_FOR_ANALYSIS["unknown"]
    )
    component_summary = fig3_component_summary(data)
    component_fields = component_summary["component_field"].tolist() if not component_summary.empty else []
    data["component_panel_fields"] = ";".join(component_fields)
    data["component_panel_summary"] = "Overall city mean of Step07 OSM quality component scores."
    quality_score_values = data["quality_score_plot"].dropna()
    if quality_score_values.empty:
        quality_score_extent = (np.nan, np.nan)
    else:
        quality_score_extent = (float(quality_score_values.min()), float(quality_score_values.max()))
    quality_score_axis = fig3_quality_score_axis_limits(data["quality_score_plot"])
    data["quality_score_axis_min"] = quality_score_axis[0]
    data["quality_score_axis_max"] = quality_score_axis[1]
    data["quality_score_axis_method"] = (
        f"data min/max padded by {FIG3_SCORE_AXIS_PADDING:g} score points and clipped to 0-100"
    )
    data["quality_score_source_step"] = STEP07_NAME
    data["quality_score_definition"] = FIG3_QUALITY_SCORE_DEFINITION
    data["quality_score_not_official_osm_field"] = True
    data["quality_score_not_morphology_type_score"] = True

    thumbnail_meta, thumbnail_records, thumbnail_note = fig3_prepare_road_thumbnails(paths, data, local_types)
    thumbnail_columns = [
        "city_id",
        "road_thumbnail_included",
        "road_thumbnail_display_order",
        "road_thumbnail_rank_group",
        "road_thumbnail_rank",
        "road_thumbnail_selection_reason",
        "road_thumbnail_selection_rule",
        "road_thumbnail_source",
        "road_thumbnail_status",
        "road_thumbnail_status_detail",
        "road_thumbnail_extent",
        "road_thumbnail_center_source",
        "road_thumbnail_center_lon",
        "road_thumbnail_center_lat",
        "road_thumbnail_half_side_km",
        "road_thumbnail_fixed_crop_half_side_km",
        "road_thumbnail_fixed_crop_width_km",
        "road_thumbnail_fixed_crop_height_km",
        "road_thumbnail_fixed_crop_scale_rule",
        "road_thumbnail_recentered",
        "road_thumbnail_recenter_reason",
        "road_thumbnail_scale_bar_present",
        "road_thumbnail_scale_bar_km",
        "road_thumbnail_scale_bar_label",
        "road_thumbnail_extent_width_km",
        "road_thumbnail_extent_height_km",
        "road_thumbnail_extent_aspect",
        "road_thumbnail_crop_shape",
        "road_thumbnail_coordinate_units",
        "road_thumbnail_edges_in_extent",
        "road_thumbnail_edges_plotted",
        "road_thumbnail_downsampled",
        "road_thumbnail_row_group_label",
        "road_thumbnail_expected_row",
        "road_thumbnail_score_badge_present",
        "road_thumbnail_score_badge_text",
        "road_thumbnail_score_badge_color",
        "road_thumbnail_component_strip_present",
        "road_thumbnail_component_strip_color",
        "road_thumbnail_component_strip_fields",
        "road_thumbnail_component_strip_abbreviations",
        "road_thumbnail_component_strip_values",
        "road_thumbnail_component_strip_values_non_null",
        "road_thumbnail_component_strip_source_step",
        "road_thumbnail_component_strip_legend",
        "road_thumbnail_component_strip_encoding",
    ]
    if not thumbnail_meta.empty:
        thumbnail_meta["road_thumbnail_included"] = True
        keep = [col for col in thumbnail_columns if col in thumbnail_meta.columns]
        data = data.merge(thumbnail_meta[keep], on="city_id", how="left")
    if "road_thumbnail_included" not in data.columns:
        data["road_thumbnail_included"] = False
    data["road_thumbnail_included"] = data["road_thumbnail_included"].astype("boolean").fillna(False).astype(bool)
    for col in thumbnail_columns:
        if col in {"city_id", "road_thumbnail_included"}:
            continue
        if col not in data.columns:
            data[col] = np.nan
    for col in [
        "road_thumbnail_rank_group",
        "road_thumbnail_selection_reason",
        "road_thumbnail_selection_rule",
        "road_thumbnail_source",
        "road_thumbnail_status",
        "road_thumbnail_status_detail",
        "road_thumbnail_extent",
        "road_thumbnail_center_source",
        "road_thumbnail_fixed_crop_scale_rule",
        "road_thumbnail_recenter_reason",
        "road_thumbnail_scale_bar_label",
        "road_thumbnail_crop_shape",
        "road_thumbnail_coordinate_units",
        "road_thumbnail_row_group_label",
        "road_thumbnail_expected_row",
        "road_thumbnail_score_badge_text",
        "road_thumbnail_score_badge_color",
        "road_thumbnail_component_strip_color",
        "road_thumbnail_component_strip_fields",
        "road_thumbnail_component_strip_abbreviations",
        "road_thumbnail_component_strip_values",
        "road_thumbnail_component_strip_source_step",
        "road_thumbnail_component_strip_legend",
        "road_thumbnail_component_strip_encoding",
    ]:
        data[col] = data[col].fillna("")
    data["road_thumbnail_background_color"] = np.where(
        data["road_thumbnail_included"], FIG3_ROAD_THUMBNAIL_BACKGROUND_COLOR, ""
    )
    data["road_thumbnail_road_color"] = np.where(data["road_thumbnail_included"], FIG3_ROAD_THUMBNAIL_ROAD_COLOR, "")
    data["road_thumbnail_border_color"] = np.where(data["road_thumbnail_included"], FIG3_ROAD_THUMBNAIL_BORDER_COLOR, "")
    data["road_thumbnail_style_note"] = np.where(
        data["road_thumbnail_included"],
        "White tile background, black drive-road network, black tile border.",
        "",
    )

    fig = plt.figure(figsize=cm_to_in(18.3, 15.4))
    layout = fig3_layout_boxes()
    ax_box = fig.add_axes(layout["box"])
    ax_summary = fig.add_axes(layout["legend"])
    box_pos = ax_box.get_position()
    summary_pos = ax_summary.get_position()
    thumbnail_axes: list[plt.Axes] = []

    short_tier_labels = {
        "core": "Core",
        "sensitivity": "Sensitivity",
        "excluded_candidate": "Excluded",
        "review": "Review",
        "unknown": "Unknown",
    }
    tiers = [
        t
        for t in ["core", "sensitivity", "excluded_candidate", "review", "unknown"]
        if t in set(data["quality_tier_plot"])
        and data.loc[data["quality_tier_plot"].eq(t), "quality_score"].dropna().size > 0
    ]
    if not tiers:
        tiers = sorted(data["quality_tier_plot"].dropna().astype(str).unique())
    box_data = [data.loc[data["quality_tier_plot"].eq(t), "quality_score"].dropna() for t in tiers]
    if box_data:
        labels = [f"{short_tier_labels.get(t, t)}\nn={int(data['quality_tier_plot'].eq(t).sum())}" for t in tiers]
        positions = np.arange(1, len(tiers) + 1)
        box = ax_box.boxplot(box_data, tick_labels=labels, positions=positions, vert=True, patch_artist=True, widths=0.46)
        for patch, tier in zip(box.get("boxes", []), tiers):
            color = fig3_quality_tier_color(tier)
            patch.set_facecolor(mpl.colors.to_rgba(color, 0.18))
            patch.set_edgecolor(color)
            patch.set_linewidth(0.8)
        for median in box.get("medians", []):
            median.set_color("#111827")
            median.set_linewidth(1.1)
        rng = np.random.default_rng(1203)
        for pos, tier, values in zip(positions, tiers, box_data):
            jitter = rng.uniform(-0.10, 0.10, size=len(values))
            ax_box.scatter(
                np.full(len(values), pos) + jitter,
                values,
                s=6,
                color=fig3_quality_tier_color(tier),
                edgecolor="white",
                linewidth=0.18,
                alpha=0.58,
                zorder=3,
            )
    else:
        ax_box.text(0.5, 0.5, "No quality score by tier data", ha="center", va="center", fontsize=7)
    ax_box.set_title("Quality score by tier", loc="left", fontsize=8, fontweight="bold")
    ax_box.set_ylabel("Quality score", labelpad=2)
    ax_box.set_ylim(*quality_score_axis)
    ax_box.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax_box.grid(axis="y", color="#E2E8F0", lw=0.5)
    ax_box.tick_params(axis="x", labelsize=5.45, pad=2)
    ax_box.tick_params(axis="y", labelsize=5.8, pad=1.5)

    ax_summary.axis("off")
    ax_summary.set_xlim(0, 1)
    ax_summary.set_ylim(0, 1)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    score_title_artist = ax_box._left_title
    _, score_title_anchor_y = artist_anchor_figure_fraction(fig, score_title_artist)
    score_visible_bbox = axes_tight_figure_fraction_bbox(fig, ax_box, renderer)
    if not np.isfinite(score_title_anchor_y):
        score_title_anchor_y = float(box_pos.y1) + 0.010
    score_visible_bottom = float(score_visible_bbox.y0) if score_visible_bbox is not None else float(box_pos.y0)
    summary_title_artist = fig.text(
        float(summary_pos.x0),
        score_title_anchor_y,
        "Quality components (R/H/P/L/B)",
        fontsize=8,
        fontweight="bold",
        ha="left",
        va=score_title_artist.get_va() or "baseline",
    )
    summary_bottom_artist: Any | None = None
    definition_artist: Any | None = None
    if component_summary.empty:
        ax_summary.text(0.5, 0.66, "No component score data", ha="center", va="center", fontsize=7)
    else:
        component_stats = component_summary.set_index("component_field").to_dict("index")
        component_group_means = fig3_component_thumbnail_group_means(thumbnail_records, component_fields)
        y_positions = np.linspace(FIG3_COMPONENT_SUMMARY_Y_TOP, FIG3_COMPONENT_SUMMARY_Y_BOTTOM, len(FIG3_QUALITY_COMPONENTS))
        bar_start = 0.390
        bar_end = 0.860
        bar_height = FIG3_COMPONENT_SUMMARY_BAR_HEIGHT
        chart_bottom = FIG3_COMPONENT_SUMMARY_Y_BOTTOM - bar_height * 0.90
        chart_top = FIG3_COMPONENT_SUMMARY_Y_TOP + bar_height * 0.90
        legend_y = min(0.900, FIG3_COMPONENT_SUMMARY_Y_TOP + 0.095)
        tick_label_y = max(0.145, FIG3_COMPONENT_SUMMARY_Y_BOTTOM - 0.092)

        def component_x(value: float) -> float:
            return bar_start + (bar_end - bar_start) * max(0.0, min(value, 100.0)) / 100.0

        for tick in FIG3_COMPONENT_SUMMARY_AXIS_TICKS:
            x_tick = component_x(float(tick))
            ax_summary.plot(
                [x_tick, x_tick],
                [chart_bottom, chart_top],
                color="#E2E8F0",
                lw=0.42,
                zorder=0,
                clip_on=False,
            )
            ax_summary.text(
                x_tick,
                tick_label_y,
                f"{int(tick):d}",
                fontsize=4.7,
                ha="center",
                va="top",
                color="#64748B",
            )

        legend_symbol_x = {
            "p10_p90_distribution_band": bar_start + 0.020,
            "all_city_mean_dot": bar_start + 0.210,
            "low_6_component_marker": bar_start + 0.378,
            "high_6_component_marker": bar_start + 0.492,
        }
        legend_label_x = {
            "p10_p90_distribution_band": bar_start + 0.052,
            "all_city_mean_dot": bar_start + 0.232,
            "low_6_component_marker": bar_start + 0.400,
            "high_6_component_marker": bar_start + 0.514,
        }
        for legend_key, legend_symbol, legend_label in FIG3_COMPONENT_VISUAL_LEGEND_ITEMS:
            symbol_x = legend_symbol_x[legend_key]
            if legend_symbol == "band":
                ax_summary.add_patch(
                    Rectangle(
                        (symbol_x - 0.020, legend_y - 0.011),
                        0.040,
                        0.022,
                        transform=ax_summary.transAxes,
                        facecolor="#CBD5E1",
                        edgecolor="#94A3B8",
                        linewidth=0.28,
                        alpha=0.72,
                    )
                )
            elif legend_symbol == "dot":
                ax_summary.scatter(
                    [symbol_x],
                    [legend_y],
                    s=12,
                    color="#111827",
                    edgecolor="#FFFFFF",
                    linewidth=0.35,
                    zorder=5,
                )
            elif legend_symbol == "low_marker":
                ax_summary.scatter(
                    [symbol_x],
                    [legend_y],
                    s=15,
                    marker="v",
                    color=FIG3_ROAD_THUMBNAIL_GROUP_COLORS["low"],
                    edgecolor="#FFFFFF",
                    linewidth=0.35,
                    zorder=5,
                )
            elif legend_symbol == "high_marker":
                ax_summary.scatter(
                    [symbol_x],
                    [legend_y],
                    s=15,
                    marker="^",
                    color=FIG3_ROAD_THUMBNAIL_GROUP_COLORS["high"],
                    edgecolor="#FFFFFF",
                    linewidth=0.35,
                    zorder=5,
                )
            ax_summary.text(
                legend_label_x[legend_key],
                legend_y,
                legend_label,
                fontsize=4.9,
                ha="left",
                va="center",
                color="#475569",
            )
        for y_pos, (field, label) in zip(y_positions, FIG3_QUALITY_COMPONENTS):
            stats = component_stats.get(field, {})
            value = float(stats.get("mean_score", np.nan))
            p10 = float(stats.get("p10_score", np.nan))
            p90 = float(stats.get("p90_score", np.nan))
            low_value = float(component_group_means.get(field, {}).get("low", np.nan))
            high_value = float(component_group_means.get(field, {}).get("high", np.nan))
            abbr = FIG3_QUALITY_COMPONENT_ABBREVIATIONS[field]
            ax_summary.text(
                0.022,
                y_pos,
                abbr,
                fontsize=6.55,
                fontweight="bold",
                ha="center",
                va="center",
                color="#111827",
                bbox={
                    "boxstyle": "round,pad=0.13,rounding_size=0.12",
                    "facecolor": "#F8FAFC",
                    "edgecolor": "#CBD5E1",
                    "linewidth": 0.42,
                },
            )
            ax_summary.text(0.065, y_pos, label, fontsize=5.7, ha="left", va="center", color="#111827")
            ax_summary.add_patch(
                Rectangle(
                    (bar_start, y_pos - bar_height / 2.0),
                    bar_end - bar_start,
                    bar_height,
                    transform=ax_summary.transAxes,
                    facecolor="#F8FAFC",
                    edgecolor="#CBD5E1",
                    linewidth=0.35,
                )
            )
            if np.isfinite(p10) and np.isfinite(p90):
                x_p10 = component_x(p10)
                x_p90 = component_x(p90)
                ax_summary.add_patch(
                    Rectangle(
                        (min(x_p10, x_p90), y_pos - bar_height * 0.34),
                        abs(x_p90 - x_p10),
                        bar_height * 0.68,
                        transform=ax_summary.transAxes,
                        facecolor="#CBD5E1",
                        edgecolor="#94A3B8",
                        linewidth=0.30,
                        alpha=0.72,
                    )
                )
            if np.isfinite(value):
                x_mean = component_x(value)
                ax_summary.plot(
                    [x_mean, x_mean],
                    [y_pos - bar_height * 0.53, y_pos + bar_height * 0.53],
                    color="#111827",
                    lw=0.68,
                    zorder=4,
                )
                ax_summary.scatter(
                    [x_mean],
                    [y_pos],
                    s=13,
                    color="#111827",
                    edgecolor="#FFFFFF",
                    linewidth=0.38,
                    zorder=5,
                )
            if np.isfinite(low_value):
                ax_summary.scatter(
                    [component_x(low_value)],
                    [y_pos - bar_height * 0.62],
                    s=16,
                    marker="v",
                    color=FIG3_ROAD_THUMBNAIL_GROUP_COLORS["low"],
                    edgecolor="#FFFFFF",
                    linewidth=0.35,
                    zorder=6,
                )
            if np.isfinite(high_value):
                ax_summary.scatter(
                    [component_x(high_value)],
                    [y_pos + bar_height * 0.62],
                    s=16,
                    marker="^",
                    color=FIG3_ROAD_THUMBNAIL_GROUP_COLORS["high"],
                    edgecolor="#FFFFFF",
                    linewidth=0.35,
                    zorder=6,
                )
            ax_summary.text(
                0.892,
                y_pos,
                f"{value:.1f}" if np.isfinite(value) else "NA",
                va="center",
                ha="left",
                fontsize=5.55,
                color="#334155",
            )

    quality_band = layout["quality_band"]
    score_note_y = max(float(quality_band[1]) + 0.008, min(score_visible_bottom - 0.044, float(box_pos.y0) - 0.028))
    definition_artist = fig.text(
        float(box_pos.x0),
        score_note_y,
        "Project-derived quality/confidence; not official OSM or morphology-only.",
        fontsize=5.35,
        color="#334155",
        ha="left",
        va="bottom",
    )
    summary_bottom_artist = fig.text(
        float(summary_pos.x0),
        score_note_y,
        "R/H/P/L/B component scores use Step07 0-100 scale.",
        fontsize=5.35,
        color="#475569",
        ha="left",
        va="bottom",
    )

    if thumbnail_records:
        thumb_left, thumb_bottom, thumb_width, _ = layout["thumbnails"]
        thumbnail_boxes = fig3_thumbnail_axes_boxes(fig, layout, len(thumbnail_records))
        thumbnail_header_artists: list[Any] = []
        row_bounds: dict[str, dict[str, float]] = {}
        for box, record in zip(thumbnail_boxes, thumbnail_records):
            group = str(record.get("road_thumbnail_rank_group", ""))
            bounds = row_bounds.setdefault(group, {"x0": box[0], "x1": box[0] + box[2], "y0": box[1], "y1": box[1] + box[3]})
            bounds["x0"] = min(bounds["x0"], box[0])
            bounds["x1"] = max(bounds["x1"], box[0] + box[2])
            bounds["y0"] = min(bounds["y0"], box[1])
            bounds["y1"] = max(bounds["y1"], box[1] + box[3])
        grid_top = max((bounds["y1"] for bounds in row_bounds.values()), default=thumb_bottom)
        thumbnail_header_artists.append(fig.text(
            thumb_left,
            grid_top + 0.043,
            "Road-network examples (fixed 10 km x 10 km crops)",
            fontsize=7.7,
            fontweight="bold",
            ha="left",
            va="bottom",
        ))
        thumbnail_header_artists.append(fig.text(
            thumb_left + thumb_width,
            grid_top + 0.043,
            f"Badge=quality_score; strip=R/H/P/L/B; scale bar={FIG3_ROAD_THUMBNAIL_SCALE_BAR_KM:g} km; * = excluded_candidate",
            fontsize=5.25,
            color="#475569",
            ha="right",
            va="bottom",
        ))
        for group in ["high", "low"]:
            if group not in row_bounds:
                continue
            bounds = row_bounds[group]
            thumbnail_header_artists.append(fig.text(
                bounds["x0"] - 0.012,
                (bounds["y0"] + bounds["y1"]) / 2.0,
                fig3_thumbnail_group_label(group),
                fontsize=6.15,
                fontweight="bold",
                color=fig3_thumbnail_group_color(group),
                ha="center",
                va="center",
                rotation=90,
            ))
        for box, record in zip(thumbnail_boxes, thumbnail_records):
            ax_thumb = fig.add_axes(box)
            fig3_draw_road_thumbnail(ax_thumb, record)
            fig3_draw_thumbnail_component_strip(fig, ax_thumb, record)
            thumbnail_axes.append(ax_thumb)
    else:
        thumbnail_header_artists = []

    fig3_layout = fig3_axes_layout_metadata(
        ax_box,
        ax_summary,
        summary_title_artist,
        summary_bottom_artist,
        definition_artist,
        component_fields,
        quality_score_axis,
        quality_score_extent,
        thumbnail_records,
        thumbnail_axes,
        layout,
        thumbnail_header_artists,
    )
    data_path = make_output_path(paths, source_data_name("Fig2", "osm_quality_confidence"))
    write_csv(data, data_path)
    png, svg, pdf = save_figure(fig, paths.figure_dir / "Fig2_osm_quality_confidence")
    fig3_note = (
        f"{FIG3_QUALITY_TIER_COMMUNICATION_NOTE} The compact upper band keeps `Quality score by tier`, "
        "a compact score plot, an enlarged R/H/P/L/B component mini-chart with p10-p90 bands, all-city mean dots "
        "and Low/High thumbnail markers, plus a score-definition note below the score plot. The lower 2x6 square road-thumbnail "
        "group is the primary visual area, with an upper High quality cities row and lower Low quality cities row; "
        "each tile has a numeric quality-score badge, a compact R/H/P/L/B component strip, a 2 km scale bar, white "
        f"tile background, black road lines and black tile borders. {FIG3_QUALITY_SCORE_DEFINITION} {thumbnail_note}"
    )
    return (
        FigureProduct(
            figure_id="Fig2",
            title="OSM quality contrast and road-network examples",
            kind="main",
            png_path=png,
            svg_path=svg,
            pdf_path=pdf,
            data_path=data_path,
            source_inputs=[
                "data/07_generate_quality_scores_and_type_confidence/quality_scores.parquet",
                "data/01_city_boundaries_and_sample_list/city_sample/city_boundaries.gpkg",
                "data/02_download_road_network_data/city_road_network_gpkg/drive/*_drive.gpkg",
            ],
            transformation=(
                "Merged quality scores, quality tiers, Step07 component scores and city centroids for the score-by-tier "
                "and road-thumbnail panels. The compact upper band summarizes quality-score distribution by tier, "
                "draws an enlarged R/H/P/L/B component mini-chart with all-city distribution and Low/High thumbnail "
                "references, and explains the score definition below the score plot. The lower 2x6 panel draws square, white-background, "
                "black-road, black-bordered Step02 drive-road GPKG edge thumbnails for the fixed High 6 and Low 6 "
                "quality-score cities, arranged with the High row above the Low row, numeric quality-score badges, "
                f"compact R/H/P/L/B Step07 component strips, a shared {FIG3_ROAD_THUMBNAIL_FIXED_SIDE_KM:g} km x "
                f"{FIG3_ROAD_THUMBNAIL_FIXED_SIDE_KM:g} km local-km crop and {FIG3_ROAD_THUMBNAIL_SCALE_BAR_KM:g} km scale bars."
            ),
            fallback_note=fig3_note,
            palette="Categorical quality tier palette reused from Fig1 for the score-by-tier panel",
            cmap="",
            color_variable="quality_tier",
            palette_note=FIG3_QUALITY_TIER_COMMUNICATION_NOTE,
        ),
        fig3_layout,
    )


def build_fig4_data(scale_signature: pd.DataFrame) -> pd.DataFrame:
    df = scale_signature.copy()
    if "main_analysis_sample" in df.columns:
        df = df[df["main_analysis_sample"].fillna(False)]
    if "network_type" in df.columns:
        df = df[df["network_type"].eq("drive")]
    df = df[df["metric"].isin(FIG4_METRICS) & df["scale"].isin(SCALE_ORDER)]
    df = df[df["statistic"].isin(["mean", "value"])]
    df = df[
        ((df["scale"].eq("full_city")) & df["statistic"].isin(["value", "mean"]))
        | ((~df["scale"].eq("full_city")) & df["statistic"].eq("mean"))
    ].copy()
    if "value_z" not in df.columns or df["value_z"].isna().all():
        df["value_z"] = df.groupby("metric")["value"].transform(lambda s: (numeric(s) - numeric(s).mean()) / numeric(s).std(ddof=0))
    df["value_z"] = numeric(df["value_z"])
    df["scale_order"] = df["scale"].map(SCALE_ORDER)
    df["scale_label"] = df["scale"].map(SCALE_LABELS)
    df["metric_label"] = df["metric"].map(METRIC_LABELS)
    grouped = (
        df.groupby(["metric", "metric_label", "scale", "scale_label", "scale_order"], as_index=False)
        .agg(
            city_count=("city_id", "nunique"),
            median_z=("value_z", "median"),
            p10_z=("value_z", lambda s: quantile(s, 0.10)),
            p90_z=("value_z", lambda s: quantile(s, 0.90)),
            mean_z=("value_z", "mean"),
        )
        .sort_values(["metric", "scale_order"])
    )
    present_scales = [SCALE_LABELS[key] for key in SCALE_ORDER if key in set(grouped["scale"].astype(str))]
    omitted_scales = [SCALE_LABELS[key] for key in SCALE_ORDER if key not in set(grouped["scale"].astype(str))]
    grouped["plotted_scale_labels"] = "; ".join(present_scales)
    grouped["omitted_scale_labels"] = "; ".join(omitted_scales)
    grouped["scale_tick_policy"] = (
        "Only scales with source rows are shown on the x-axis; omitted scales are recorded here to avoid empty ticks."
    )
    return grouped


def plot_fig4(paths: StepPaths, scale_signature: pd.DataFrame) -> FigureProduct:
    data = build_fig4_data(scale_signature)
    data = add_palette_metadata(
        data,
        palette=f"Sequential line/band colors sampled from {SEQUENTIAL_CMAP}",
        cmap=SEQUENTIAL_CMAP,
        color_variable="median_z; p10_z; p90_z",
        color_variable_label="Standardized metric value summaries",
        palette_note="Unified Step12 sequential palette for quantitative summaries.",
    )
    data_path = make_output_path(paths, source_data_name("Fig3", FIG4_CANONICAL_SLUG))
    write_csv(data, data_path)

    metrics = [m for m in FIG4_METRICS if m in set(data["metric"])]
    present = data[["scale_order", "scale_label"]].drop_duplicates().sort_values("scale_order")
    x_ticks = present["scale_order"].astype(float).tolist()
    x_labels = present["scale_label"].astype(str).tolist()
    fig, axes = plt.subplots(2, 3, figsize=cm_to_in(18.3, 12.0), sharex=True)
    axes = axes.ravel()
    for ax, metric in zip(axes, metrics):
        g = data[data["metric"].eq(metric)].sort_values("scale_order")
        x = g["scale_order"].to_numpy(dtype=float)
        ax.fill_between(x, g["p10_z"], g["p90_z"], color=SEQUENTIAL_BAND_COLOR, alpha=0.30, lw=0)
        ax.plot(x, g["median_z"], marker="o", color=SEQUENTIAL_LINE_COLOR, lw=1.6, ms=3.8)
        ax.axhline(0, color="#CBD5E1", lw=0.7)
        ax.set_title(METRIC_LABELS.get(metric, metric), loc="left", fontsize=8, fontweight="bold")
        ax.grid(axis="y", color="#E2E8F0", lw=0.5)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=25, ha="right")
    for ax in axes[len(metrics) :]:
        ax.set_axis_off()
    axes[0].set_ylabel("Standardized value")
    axes[3].set_ylabel("Standardized value")
    png, svg, pdf = save_figure(fig, paths.figure_dir / f"Fig3_{FIG4_CANONICAL_SLUG}")
    return FigureProduct(
        figure_id="Fig3",
        title="Multiscale street-network signatures",
        kind="main",
        png_path=png,
        svg_path=svg,
        pdf_path=pdf,
        data_path=data_path,
        source_inputs=["data/08_generate_scale_signatures_and_city_profiles/scale_signatures_long.parquet"],
        transformation="Filtered to main-sample drive-network city signatures; summarized median and p10/p90 standardized values by scale.",
        palette=f"Sequential line/band colors sampled from {SEQUENTIAL_CMAP}",
        cmap=SEQUENTIAL_CMAP,
        color_variable="median_z; p10_z; p90_z",
        palette_note="Unified Step12 sequential palette for quantitative summaries.",
    )


def plot_fig5(paths: StepPaths, type_centers: pd.DataFrame) -> FigureProduct:
    score_cols = [
        "density_score",
        "fine_grain_score",
        "grid_score",
        "culdesac_score",
        "hierarchy_score",
        "circuity_score",
        "segment_length_score",
    ]
    keep_cols = ["morphotype", "morphotype_name", "balanced_training_share", *score_cols]
    data = type_centers[[c for c in keep_cols if c in type_centers.columns]].copy()
    data = data[data["morphotype"].isin(MORPHOTYPES)].sort_values("morphotype")
    data = add_morphotype_labels(data)
    data = add_morphotype_color_field(data)
    data = add_palette_metadata(
        data,
        palette=f"Zero-centered diverging heatmap cmap: {FIG5_HEATMAP_CMAP}; {MORPHOTYPE_PALETTE_NAME} for share bars",
        cmap=FIG5_HEATMAP_CMAP,
        heatmap_cmap=FIG5_HEATMAP_CMAP,
        color_variable="standardized_type_center_score",
        color_variable_label="Standardized type-center score (low-to-high relative score)",
        palette_note=f"Diverging heatmap uses {FIG5_HEATMAP_CMAP} with zero centered for signed standardized scores; share bars use fixed morphotype colors.",
    )
    data["morphotype_name_not_unique_note"] = "Chinese names may repeat; use MTxx + English label for plotting."
    data_path = make_output_path(paths, source_data_name("Fig4", "morphotype_atlas_signatures"))
    write_csv(data, data_path)

    matrix = data[score_cols].astype(float).to_numpy()
    labels = [MORPHOTYPE_SHORT_LABELS[mt] for mt in data["morphotype"]]
    col_labels = ["Density", "Fine grain", "Grid", "Cul-de-sac", "Hierarchy", "Circuity", "Segment length"]
    fig = plt.figure(figsize=cm_to_in(18.3, 12.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[4.8, 0.18, 0.42, 1.05], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[0, 3], sharey=ax)
    cmap = plt.get_cmap(FIG5_HEATMAP_CMAP)
    norm = mpl.colors.TwoSlopeNorm(vmin=-TYPE_CENTER_SCORE_LIMIT, vcenter=0, vmax=TYPE_CENTER_SCORE_LIMIT)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
    ax.set_title("Morphotype atlas signatures", loc="left", fontsize=10, fontweight="bold")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=FIG5_HEATMAP_TICK_FONTSIZE)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=FIG5_HEATMAP_TICK_FONTSIZE)
    ax.set_xlabel("Standardized type-center scores", fontsize=FIG5_HEATMAP_AXIS_LABEL_FONTSIZE)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            rgba = cmap(norm(val))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            text_color = "white" if luminance < 0.45 else "#0F172A"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=FIG5_HEATMAP_VALUE_FONTSIZE, color=text_color)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Type-center score (zero-centered)", fontsize=FIG5_COLORBAR_LABEL_FONTSIZE)
    cb.ax.tick_params(labelsize=FIG5_COLORBAR_TICK_FONTSIZE)
    share = numeric(data.get("balanced_training_share", pd.Series(np.nan, index=data.index))) * 100
    ax_bar.barh(np.arange(len(data)), share, color=[TYPE_COLORS[mt] for mt in data["morphotype"]], edgecolor="white")
    ax_bar.set_xlabel("Training share (%)", fontsize=FIG5_HEATMAP_AXIS_LABEL_FONTSIZE)
    ax_bar.set_title("Share", loc="left", fontsize=8, fontweight="bold")
    ax_bar.grid(axis="x", color="#E2E8F0", lw=0.5)
    ax_bar.tick_params(axis="x", labelsize=FIG5_COLORBAR_TICK_FONTSIZE)
    ax_bar.tick_params(labelleft=False)
    ax_bar.invert_yaxis()
    png, svg, pdf = save_figure(fig, paths.figure_dir / "Fig4_morphotype_atlas_signatures")
    return FigureProduct(
        figure_id="Fig4",
        title="Morphotype atlas signatures",
        kind="main",
        png_path=png,
        svg_path=svg,
        pdf_path=pdf,
        data_path=data_path,
        source_inputs=["data/09_cluster_morphotypes/morphotype_centroids.parquet"],
        transformation="Selected type-center score metrics, attached unique English labels for MTxx classes, and placed the morphotype score heatmap in the main figure.",
        palette=f"Zero-centered diverging heatmap cmap: {FIG5_HEATMAP_CMAP}; {MORPHOTYPE_PALETTE_NAME} for share bars",
        cmap=FIG5_HEATMAP_CMAP,
        heatmap_cmap=FIG5_HEATMAP_CMAP,
        color_variable="standardized_type_center_score",
        palette_note=f"Diverging heatmap uses {FIG5_HEATMAP_CMAP} with zero centered for signed standardized scores; share bars use fixed morphotype colors.",
    )


def build_fig6_data(city_profiles: pd.DataFrame, access: pd.DataFrame | None = None) -> pd.DataFrame:
    df = city_profiles.copy()
    if "sample_group" in df.columns:
        df = df[df["sample_group"].eq("main_80")]
    df = df[df["network_type"].isin(["drive", "walk"]) & df["scale"].isin(["hex_1km", "hex_2km"])]
    composition_rows: list[dict[str, Any]] = []
    for (network, scale), g in df.groupby(["network_type", "scale"], dropna=False):
        weight = numeric(g.get("n_units", pd.Series(1, index=g.index)), default=1.0)
        for mt in MORPHOTYPES:
            col = f"unit_share__{mt}"
            if col not in g.columns:
                value = np.nan
            else:
                value = weighted_mean(g[col], weight)
            composition_rows.append(
                {
                    "panel": FIG6_PANEL_COMPOSITION,
                    "panel_order": 1,
                    "comparison_type": "composition",
                    "network_type": network,
                    "scale": scale,
                    "network_scale": f"{str(network).title()} {SCALE_LABELS.get(scale, scale)}",
                    "network_scale_order": (0 if network == "drive" else 2) + (0 if scale == "hex_1km" else 1),
                    "row_label": f"{str(network).title()} {SCALE_LABELS.get(scale, scale)}",
                    "row_order": (0 if network == "drive" else 2) + (0 if scale == "hex_1km" else 1),
                    "from_network_type": network,
                    "from_scale": scale,
                    "to_network_type": network,
                    "to_scale": scale,
                    "morphotype": mt,
                    "morphotype_label_en": MORPHOTYPE_LABELS_EN[mt],
                    "mean_unit_share": value,
                    "mean_unit_share_percent": value * 100 if pd.notna(value) else np.nan,
                    "from_mean_unit_share_percent": value * 100 if pd.notna(value) else np.nan,
                    "to_mean_unit_share_percent": value * 100 if pd.notna(value) else np.nan,
                    "transition_delta_pp": np.nan,
                    "absolute_transition_delta_pp": np.nan,
                    "value_for_plot": value * 100 if pd.notna(value) else np.nan,
                    "value_units": "percent",
                    "cell_definition": (
                        f"Weighted mean share of {mt} units among main-sample cities for "
                        f"{network} {scale}; weights are n_units."
                    ),
                    "city_count": int(g["city_id"].nunique()) if "city_id" in g.columns else len(g),
                    "unit_weight_sum": float(weight.sum()),
                    "from_city_count": int(g["city_id"].nunique()) if "city_id" in g.columns else len(g),
                    "to_city_count": int(g["city_id"].nunique()) if "city_id" in g.columns else len(g),
                    "from_unit_weight_sum": float(weight.sum()),
                    "to_unit_weight_sum": float(weight.sum()),
                }
            )
    comp = pd.DataFrame(composition_rows)
    lookup = {
        (row["network_type"], row["scale"], row["morphotype"]): row
        for _, row in comp.iterrows()
    }
    comparisons = [
        ("scale_transition", "Drive 1 km -> 2 km", "drive", "hex_1km", "drive", "hex_2km", 0),
        ("scale_transition", "Walk 1 km -> 2 km", "walk", "hex_1km", "walk", "hex_2km", 1),
        ("network_transition", "Hex 1 km drive -> walk", "drive", "hex_1km", "walk", "hex_1km", 2),
        ("network_transition", "Hex 2 km drive -> walk", "drive", "hex_2km", "walk", "hex_2km", 3),
    ]
    transition_rows: list[dict[str, Any]] = []
    for comparison_type, row_label, from_network, from_scale, to_network, to_scale, order in comparisons:
        for mt in MORPHOTYPES:
            from_row = lookup.get((from_network, from_scale, mt), {})
            to_row = lookup.get((to_network, to_scale, mt), {})
            from_value = scalar(from_row.get("mean_unit_share_percent", np.nan))
            to_value = scalar(to_row.get("mean_unit_share_percent", np.nan))
            if pd.isna(from_value) or pd.isna(to_value):
                delta = np.nan
            else:
                delta = float(to_value) - float(from_value)
            transition_rows.append(
                {
                    "panel": FIG6_PANEL_TRANSITION,
                    "panel_order": 2,
                    "comparison_type": comparison_type,
                    "network_type": f"{from_network}->{to_network}",
                    "scale": f"{from_scale}->{to_scale}",
                    "network_scale": row_label,
                    "network_scale_order": order,
                    "row_label": row_label,
                    "row_order": order,
                    "from_network_type": from_network,
                    "from_scale": from_scale,
                    "to_network_type": to_network,
                    "to_scale": to_scale,
                    "morphotype": mt,
                    "morphotype_label_en": MORPHOTYPE_LABELS_EN[mt],
                    "mean_unit_share": np.nan,
                    "mean_unit_share_percent": np.nan,
                    "from_mean_unit_share_percent": from_value,
                    "to_mean_unit_share_percent": to_value,
                    "transition_delta_pp": delta,
                    "absolute_transition_delta_pp": abs(delta) if pd.notna(delta) else np.nan,
                    "value_for_plot": delta,
                    "value_units": "percentage points",
                    "cell_definition": (
                        f"Transition cell for {mt}: {to_network} {to_scale} mean share minus "
                        f"{from_network} {from_scale} mean share, in percentage points."
                    ),
                    "city_count": min(
                        int(from_row.get("city_count", 0) or 0),
                        int(to_row.get("city_count", 0) or 0),
                    ),
                    "unit_weight_sum": np.nan,
                    "from_city_count": from_row.get("city_count", np.nan),
                    "to_city_count": to_row.get("city_count", np.nan),
                    "from_unit_weight_sum": from_row.get("unit_weight_sum", np.nan),
                    "to_unit_weight_sum": to_row.get("unit_weight_sum", np.nan),
                }
            )
    frames = [comp, pd.DataFrame(transition_rows)]
    if access is not None:
        access_panel = build_access_panel_data(access).copy()
        performance_rows = access_panel.rename(
            columns={
                "fig9_color_variable": "previous_fig7a_color_variable",
                "fig9_cmap": "previous_fig7a_cmap",
                "fig9_norm": "previous_fig7a_norm",
                "fig9_colorbar_label": "previous_fig7a_colorbar_label",
                "fig9_color_limit_abs_pp": "previous_fig7a_color_limit_abs_pp",
                "fig9_cell_label_variable": "previous_fig7a_cell_label_variable",
                "fig9_sort_variable": "previous_fig7a_sort_variable",
                "fig9_sort_direction": "previous_fig7a_sort_direction",
            }
        )
        performance_rows["panel"] = FIG6_PANEL_PERFORMANCE
        performance_rows["panel_order"] = 3
        performance_rows["comparison_type"] = "performance_deviation"
        performance_rows["network_type"] = ""
        performance_rows["scale"] = ""
        performance_rows["network_scale"] = performance_rows["category_label"]
        performance_rows["network_scale_order"] = performance_rows["category_order"]
        performance_rows["row_label"] = performance_rows["category_label"]
        performance_rows["row_order"] = performance_rows["category_order"]
        performance_rows["from_network_type"] = ""
        performance_rows["from_scale"] = ""
        performance_rows["to_network_type"] = ""
        performance_rows["to_scale"] = ""
        performance_rows["mean_unit_share"] = np.nan
        performance_rows["mean_unit_share_percent"] = np.nan
        performance_rows["from_mean_unit_share_percent"] = np.nan
        performance_rows["to_mean_unit_share_percent"] = np.nan
        performance_rows["transition_delta_pp"] = np.nan
        performance_rows["absolute_transition_delta_pp"] = np.nan
        performance_rows["value_for_plot"] = performance_rows["access_share_diff_pp"]
        performance_rows["value_units"] = "percentage points"
        performance_rows["cell_definition"] = (
            "Migrated access-performance heatmap cell; color is the morphotype/category "
            "population-weighted access-share difference from the facility-category mean, in percentage points."
        )
        performance_rows["city_count"] = performance_rows.get("city_group_count", np.nan)
        performance_rows["unit_weight_sum"] = np.nan
        performance_rows["from_city_count"] = performance_rows.get("city_group_count", np.nan)
        performance_rows["to_city_count"] = performance_rows.get("city_group_count", np.nan)
        performance_rows["from_unit_weight_sum"] = np.nan
        performance_rows["to_unit_weight_sum"] = np.nan
        performance_rows["fig7_panel_c_source_panel"] = "migrated access_panel heatmap"
        performance_rows["fig7_panel_c_source_input"] = "data/11_statistical_models_and_robustness_checks/figure_data_accessibility_by_type.csv"
        performance_rows["fig7_panel_c_migration_note"] = FIG6_PANEL_C_SOURCE_NOTE
        performance_rows["fig7_panel_c_metric"] = FIG7_COLOR_VARIABLE
        performance_rows["fig7_panel_c_raw_label_metric"] = "population_weighted_access_share_percent"
        frames.append(performance_rows)

    data = pd.concat(frames, ignore_index=True, sort=False)
    return data.sort_values(["panel_order", "row_order", "morphotype"])


def plot_fig6(paths: StepPaths, city_profiles: pd.DataFrame, access: pd.DataFrame) -> FigureProduct:
    data = build_fig6_data(city_profiles, access)
    data = add_morphotype_color_field(data)
    fig6_norm = fig6_panel_a_norm_metadata(data)
    text_cleanup = fig6_title_footnote_cleanup_metadata()
    data = add_palette_metadata(
        data,
        palette=(
            f"Fig6 A/B heatmap cmap: {FIG6_FIG7_HEATMAP_CMAP}; Fig6C migrated access-deviation cmap: "
            f"{fig6_norm['panel_c_cmap']}; {MORPHOTYPE_PALETTE_NAME} recorded for MTxx classes"
        ),
        cmap=FIG6_FIG7_HEATMAP_CMAP,
        heatmap_cmap=f"A/B={FIG6_FIG7_HEATMAP_CMAP}; C={fig6_norm['panel_c_cmap']}",
        color_variable="mean_unit_share_percent; transition_delta_pp; access_share_diff_pp",
        color_variable_label="Composition share (%), transition delta (pp), and performance deviation from mean (pp)",
        palette_note=fig6_palette_note(fig6_norm),
    )
    panel_labels = {
        FIG6_PANEL_COMPOSITION: "A",
        FIG6_PANEL_TRANSITION: "B",
        FIG6_PANEL_PERFORMANCE: "C",
    }
    panel_norms = {
        FIG6_PANEL_COMPOSITION: fig6_norm["panel_a_norm_summary"],
        FIG6_PANEL_TRANSITION: fig6_norm["panel_b_norm_summary"],
        FIG6_PANEL_PERFORMANCE: fig6_norm["panel_c_norm_summary"],
    }
    panel_norm_types = {
        FIG6_PANEL_COMPOSITION: fig6_norm["panel_a_norm_type"],
        FIG6_PANEL_TRANSITION: "TwoSlopeNorm",
        FIG6_PANEL_PERFORMANCE: fig6_norm["panel_c_norm_type"],
    }
    panel_cmaps = {
        FIG6_PANEL_COMPOSITION: FIG6_FIG7_HEATMAP_CMAP,
        FIG6_PANEL_TRANSITION: FIG6_FIG7_HEATMAP_CMAP,
        FIG6_PANEL_PERFORMANCE: fig6_norm["panel_c_cmap"],
    }
    panel_colorbar_labels = {
        FIG6_PANEL_COMPOSITION: "Mean unit share (%)",
        FIG6_PANEL_TRANSITION: "Delta share (pp)",
        FIG6_PANEL_PERFORMANCE: fig6_norm["panel_c_colorbar_label"],
    }
    panel_color_variables = {
        FIG6_PANEL_COMPOSITION: "mean_unit_share_percent",
        FIG6_PANEL_TRANSITION: "transition_delta_pp",
        FIG6_PANEL_PERFORMANCE: FIG7_COLOR_VARIABLE,
    }
    data["fig7_panel_label"] = data["panel"].map(panel_labels).fillna("")
    data["fig7_layout_grid_spec"] = FIG6_LAYOUT_GRID_SPEC
    data["fig7_layout_choice_note"] = FIG6_LAYOUT_CHOICE_NOTE
    data["fig7_overlap_control_note"] = FIG6_OVERLAP_CONTROL_NOTE
    data["fig7_panel_heatmap_cmap"] = data["panel"].map(panel_cmaps).fillna("")
    data["fig7_panel_norm"] = data["panel"].map(panel_norms).fillna("")
    data["fig7_panel_norm_type"] = data["panel"].map(panel_norm_types).fillna("")
    data["fig7_panel_colorbar_label"] = data["panel"].map(panel_colorbar_labels).fillna("")
    data["fig7_panel_color_variable"] = data["panel"].map(panel_color_variables).fillna("")
    data["fig7_morphotype_axis_order"] = FIG6_MORPHOTYPE_AXIS_ORDER_TEXT
    data["fig7_morphotype_axis_order_note"] = "Fig6 A/B/C heatmap x-axes use MT01->MT10."
    data["fig7_panel_c_morphotype_axis_order"] = np.where(
        data["panel"].eq(FIG6_PANEL_PERFORMANCE),
        FIG6_MORPHOTYPE_AXIS_ORDER_TEXT,
        "",
    )
    data["fig7_panel_c_morphotype_axis_order_note"] = np.where(
        data["panel"].eq(FIG6_PANEL_PERFORMANCE),
        FIG6_PANEL_C_MORPHOTYPE_AXIS_ORDER_NOTE,
        "",
    )
    data["fig7_panel_c_morphotype_display_order"] = np.where(
        data["panel"].eq(FIG6_PANEL_PERFORMANCE),
        data["morphotype"].map({mt: idx + 1 for idx, mt in enumerate(MORPHOTYPES)}),
        np.nan,
    )
    data["fig7_panel_palette_note"] = (
        "A/B share the RdBu_r cmap but use different norms; C records the migrated access-performance "
        "deviation norm and colorbar."
    )
    data["fig7_figure_level_title_current"] = text_cleanup["figure_level_title_current"]
    data["fig7_bottom_note_current"] = text_cleanup["bottom_note_current"]
    data["fig7_removed_figure_level_title_text"] = text_cleanup["removed_figure_level_title_text"]
    data["fig7_removed_bottom_note_text"] = text_cleanup["removed_bottom_note_text"]
    data["fig7_retained_visual_text_elements"] = text_cleanup["retained_visual_text_elements"]
    data["fig7_title_footnote_removal_note"] = text_cleanup["removal_note"]
    data["fig7_panel_a_heatmap_cmap"] = fig6_norm["panel_a_cmap"]
    data["fig7_panel_a_norm"] = fig6_norm["panel_a_norm_summary"]
    data["fig7_panel_a_norm_type"] = fig6_norm["panel_a_norm_type"]
    data["fig7_panel_a_norm_vmin"] = fig6_norm["panel_a_norm_vmin"]
    data["fig7_panel_a_norm_vmax_formula"] = fig6_norm["panel_a_norm_vmax_formula"]
    data["fig7_panel_a_norm_vmax_floor"] = fig6_norm["panel_a_norm_vmax_floor"]
    data["fig7_panel_a_norm_vmax_data_factor"] = fig6_norm["panel_a_norm_vmax_data_factor"]
    data["fig7_panel_a_norm_data_max"] = fig6_norm["panel_a_norm_data_max"]
    data["fig7_panel_a_norm_candidate_vmax"] = fig6_norm["panel_a_norm_candidate_vmax"]
    data["fig7_panel_a_norm_vmax_current"] = fig6_norm["panel_a_norm_vmax"]
    data["fig7_panel_b_norm"] = fig6_norm["panel_b_norm_summary"]
    data["fig7_panel_c_heatmap_cmap"] = fig6_norm["panel_c_cmap"]
    data["fig7_panel_c_norm"] = fig6_norm["panel_c_norm_summary"]
    data["fig7_panel_c_norm_type"] = fig6_norm["panel_c_norm_type"]
    data["fig7_panel_c_norm_vmin"] = fig6_norm["panel_c_norm_vmin"]
    data["fig7_panel_c_norm_vcenter"] = fig6_norm["panel_c_norm_vcenter"]
    data["fig7_panel_c_norm_vmax"] = fig6_norm["panel_c_norm_vmax"]
    data["fig7_panel_c_color_limit_abs_pp"] = fig6_norm["panel_c_color_limit_abs_pp"]
    data["fig7_panel_c_color_variable"] = fig6_norm["panel_c_color_variable"]
    data["fig7_panel_c_colorbar_label"] = fig6_norm["panel_c_colorbar_label"]
    data = add_fig6_enrichment_fields(data)
    data["figure_id"] = "Fig6"
    data["figure_slug"] = FIG7_CANONICAL_SLUG
    data = replace_text_in_string_columns(
        data,
        [
            ("Former Fig7A", "Former access heatmap"),
            ("former Fig7A", "former access heatmap"),
        ],
    )
    data_path = make_output_path(paths, source_data_name("Fig6", FIG7_CANONICAL_SLUG))
    write_csv(data, data_path)

    composition = data[data["panel"].eq(FIG6_PANEL_COMPOSITION)].copy()
    transition = data[data["panel"].eq(FIG6_PANEL_TRANSITION)].copy()
    performance = data[data["panel"].eq(FIG6_PANEL_PERFORMANCE)].copy()
    comp_order = composition.drop_duplicates("row_label").sort_values("row_order")["row_label"].tolist()
    trans_order = transition.drop_duplicates("row_label").sort_values("row_order")["row_label"].tolist()
    perf_order = performance.drop_duplicates("row_label").sort_values("row_order")["row_label"].tolist()
    perf_morph_order = MORPHOTYPES
    comp_pivot = composition.pivot(index="row_label", columns="morphotype", values="mean_unit_share_percent").reindex(comp_order)[MORPHOTYPES]
    trans_pivot = transition.pivot(index="row_label", columns="morphotype", values="transition_delta_pp").reindex(trans_order)[MORPHOTYPES]
    perf_pivot = performance.pivot(index="row_label", columns="morphotype", values=FIG7_COLOR_VARIABLE).reindex(perf_order)[perf_morph_order]
    perf_share_pivot = (
        performance.pivot(index="row_label", columns="morphotype", values="population_weighted_access_share_percent")
        .reindex(perf_order)[perf_morph_order]
    )

    fig = plt.figure(figsize=cm_to_in(18.3, 18.0))
    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=[1.0, 0.035],
        height_ratios=[1.0, 1.0, 1.18],
        wspace=0.08,
        hspace=0.50,
    )
    ax_comp = fig.add_subplot(gs[0, 0])
    cax_comp = fig.add_subplot(gs[0, 1])
    ax_trans = fig.add_subplot(gs[1, 0])
    cax_trans = fig.add_subplot(gs[1, 1])
    ax_perf = fig.add_subplot(gs[2, 0])
    cax_perf = fig.add_subplot(gs[2, 1])

    comp_vmax = float(fig6_norm["panel_a_norm_vmax"])
    shared_fig6_cmap = plt.get_cmap(FIG6_FIG7_HEATMAP_CMAP)
    comp_cmap = shared_fig6_cmap
    comp_norm = mpl.colors.Normalize(vmin=FIG6_PANEL_A_NORM_VMIN, vmax=comp_vmax)
    im_comp = ax_comp.imshow(comp_pivot.to_numpy(dtype=float), aspect="auto", cmap=comp_cmap, norm=comp_norm)
    ax_comp.set_title("A  Scale/network composition", loc="left", fontsize=9, fontweight="bold")
    ax_comp.set_yticks(np.arange(len(comp_order)))
    ax_comp.set_yticklabels(comp_order)
    ax_comp.set_xticks(np.arange(len(MORPHOTYPES)))
    ax_comp.set_xticklabels(MORPHOTYPES)
    for i in range(comp_pivot.shape[0]):
        for j in range(comp_pivot.shape[1]):
            val = comp_pivot.iloc[i, j]
            ax_comp.text(j, i, "" if pd.isna(val) else f"{val:.1f}", ha="center", va="center", fontsize=5.8, color=heatmap_text_color(comp_cmap, comp_norm, val))
    draw_fig6_top_cell_outlines(ax_comp, composition, comp_order)
    cb_comp = fig.colorbar(im_comp, cax=cax_comp)
    cb_comp.set_label("Mean unit share (%)")

    trans_limit = symmetric_pp_limit(transition["transition_delta_pp"], minimum=2.0, step=2.0)
    trans_cmap = shared_fig6_cmap
    trans_norm = mpl.colors.TwoSlopeNorm(vmin=-trans_limit, vcenter=0, vmax=trans_limit)
    im_trans = ax_trans.imshow(trans_pivot.to_numpy(dtype=float), aspect="auto", cmap=trans_cmap, norm=trans_norm)
    ax_trans.set_title("B  Scale/network transitions", loc="left", fontsize=9, fontweight="bold")
    ax_trans.set_yticks(np.arange(len(trans_order)))
    ax_trans.set_yticklabels(trans_order)
    ax_trans.set_xticks(np.arange(len(MORPHOTYPES)))
    ax_trans.set_xticklabels(MORPHOTYPES)
    ax_trans.set_xlabel("Morphotype")
    for i in range(trans_pivot.shape[0]):
        for j in range(trans_pivot.shape[1]):
            val = trans_pivot.iloc[i, j]
            ax_trans.text(j, i, "" if pd.isna(val) else f"{val:+.1f}", ha="center", va="center", fontsize=5.8, color=heatmap_text_color(trans_cmap, trans_norm, val))
    draw_fig6_top_cell_outlines(ax_trans, transition, trans_order)
    cb_trans = fig.colorbar(im_trans, cax=cax_trans)
    cb_trans.set_label("Delta share (pp)")

    perf_cmap = plt.get_cmap(fig6_norm["panel_c_cmap"])
    perf_limit = float(fig6_norm["panel_c_color_limit_abs_pp"])
    perf_norm = mpl.colors.TwoSlopeNorm(vmin=-perf_limit, vcenter=0.0, vmax=perf_limit)
    im_perf = ax_perf.imshow(perf_pivot.to_numpy(dtype=float), aspect="auto", cmap=perf_cmap, norm=perf_norm)
    ax_perf.set_title("C  Performance deviation by facility category", loc="left", fontsize=9, fontweight="bold")
    ax_perf.set_yticks(np.arange(len(perf_order)))
    ax_perf.set_yticklabels(perf_order)
    ax_perf.set_xticks(np.arange(len(perf_morph_order)))
    ax_perf.set_xticklabels(perf_morph_order)
    ax_perf.set_xlabel("Morphotype")
    ax_perf.set_ylabel("Facility category", labelpad=4)
    for i in range(perf_pivot.shape[0]):
        for j in range(perf_pivot.shape[1]):
            diff = perf_pivot.iloc[i, j]
            share = perf_share_pivot.iloc[i, j]
            color = heatmap_text_color(perf_cmap, perf_norm, diff)
            if pd.notna(share):
                ax_perf.text(j, i - 0.10, f"{share:.1f}%", ha="center", va="center", fontsize=5.4, color=color, fontweight="bold")
            if pd.notna(diff):
                ax_perf.text(j, i + 0.20, f"{diff:+.1f} pp", ha="center", va="center", fontsize=4.1, color=color)
    cb_perf = fig.colorbar(im_perf, cax=cax_perf)
    cb_perf.set_label(FIG7_COLORBAR_LABEL, labelpad=FIG7_COLORBAR_LABELPAD)
    cb_perf.ax.yaxis.set_major_locator(MaxNLocator(nbins=7))

    for ax, n_rows, n_cols in [
        (ax_comp, len(comp_order), len(MORPHOTYPES)),
        (ax_trans, len(trans_order), len(MORPHOTYPES)),
        (ax_perf, len(perf_order), len(perf_morph_order)),
    ]:
        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which="minor", color="white", lw=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
    png, svg, pdf = save_figure(fig, paths.figure_dir / f"Fig6_{FIG7_CANONICAL_SLUG}")
    return FigureProduct(
        figure_id="Fig6",
        title="Scale and network transition matrix",
        kind="main",
        png_path=png,
        svg_path=svg,
        pdf_path=pdf,
        data_path=data_path,
        source_inputs=[
            "data/09_cluster_morphotypes/city_morphotype_profiles.parquet",
            "data/11_statistical_models_and_robustness_checks/figure_data_accessibility_by_type.csv",
        ],
        transformation=(
            "Converted city-level MTxx unit-share columns into composition cells and true scale/network transition cells; "
            "transition values are target mean share minus source mean share in percentage points. "
            f"Fig6 drawing marks top-{FIG6_TOP_CELL_COUNT} composition-share and absolute-delta cells. "
            "Former accessibility performance-deviation heatmap rows are migrated into Fig6C without changing "
            "their access_share_diff_pp semantics. "
            f"{FIG6_PANEL_C_MORPHOTYPE_AXIS_ORDER_NOTE} "
            f"{FIG6_TITLE_FOOTNOTE_REMOVAL_NOTE}"
        ),
        palette=(
            f"Fig6 A/B heatmap cmap: {FIG6_FIG7_HEATMAP_CMAP}; Fig6C migrated access-deviation cmap: "
            f"{fig6_norm['panel_c_cmap']}; {MORPHOTYPE_PALETTE_NAME} recorded for MTxx classes"
        ),
        cmap=FIG6_FIG7_HEATMAP_CMAP,
        heatmap_cmap=f"A/B={FIG6_FIG7_HEATMAP_CMAP}; C={fig6_norm['panel_c_cmap']}",
        color_variable="mean_unit_share_percent; transition_delta_pp; access_share_diff_pp",
        palette_note=fig6_palette_note(fig6_norm),
        visual_text_note=FIG6_TITLE_FOOTNOTE_REMOVAL_NOTE,
    )


def build_access_panel_data(access: pd.DataFrame) -> pd.DataFrame:
    data = access.copy()
    data = data[data["morphotype"].isin(MORPHOTYPES) & data["category"].isin(FACILITY_CATEGORIES)].copy()
    data["category_label"] = data["category"].map(CATEGORY_LABELS)
    data["category_order"] = data["category"].map({category: idx for idx, category in enumerate(FACILITY_CATEGORIES)})
    data = add_morphotype_labels(data)
    numeric_cols = [
        "population_weighted_access_share",
        "unweighted_access_share",
        "population_weighted_accessibility_score",
        "median_distance_m",
        "p90_distance_m",
        "long_snap_share",
        "mean_type_confidence",
        "unit_count",
        "city_group_count",
    ]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = numeric(data[col])
    data["category_mean_access_share"] = data.groupby("category")["population_weighted_access_share"].transform("mean")
    data["access_share_diff_from_category_mean"] = (
        data["population_weighted_access_share"] - data["category_mean_access_share"]
    )
    data["access_share_diff_pp"] = data["access_share_diff_from_category_mean"] * 100.0
    data["morphotype_mean_access_share"] = data.groupby("morphotype")["population_weighted_access_share"].transform("mean")
    order = (
        data[["morphotype", "morphotype_mean_access_share"]]
        .drop_duplicates()
        .sort_values(["morphotype_mean_access_share", "morphotype"], ascending=[True, True])
        .reset_index(drop=True)
    )
    rank_map = {mt: idx + 1 for idx, mt in enumerate(order["morphotype"])}
    data["morphotype_rank"] = data["morphotype"].map(rank_map).astype("Int64")
    color_limit = symmetric_pp_limit(data["access_share_diff_pp"])
    data["fig9_color_variable"] = FIG7_COLOR_VARIABLE
    data["fig9_cmap"] = FIG7_HEATMAP_CMAP
    data["fig9_norm"] = f"TwoSlopeNorm(vmin=-{color_limit:g}, vcenter=0, vmax={color_limit:g})"
    data["fig9_colorbar_label"] = FIG7_COLORBAR_LABEL
    data["fig9_color_limit_abs_pp"] = color_limit
    data["fig9_cell_label_variable"] = "population_weighted_access_share_percent; access_share_diff_pp"
    data["fig9_sort_variable"] = "morphotype_mean_access_share"
    data["fig9_sort_direction"] = "ascending; rank 1 is lowest overall mean access"
    data["population_weighted_access_share_percent"] = data["population_weighted_access_share"] * 100.0
    return data.sort_values(["morphotype_rank", "category_order", "morphotype"])


def build_circuity_summary_data(circuity: pd.DataFrame) -> pd.DataFrame:
    data = circuity.copy()
    data = data[data["morphotype"].isin(MORPHOTYPES)].copy()
    data = add_morphotype_labels(data)
    rename = {
        "od_circuity_weighted_mean_weighted_mean": "weighted_circuity",
        "od_circuity_weighted_mean_p10": "p10_circuity",
        "od_circuity_weighted_mean_p90": "p90_circuity",
        "route_found_weighted_share_weighted_mean": "route_found_weighted_share",
        "detour_ratio_median_weighted_mean": "detour_ratio_median",
    }
    data = data.rename(columns=rename)
    for col in [
        "weighted_circuity",
        "p10_circuity",
        "p90_circuity",
        "route_found_weighted_share",
        "detour_ratio_median",
        "city_group_count",
        "od_pair_count",
        "weight_sum",
    ]:
        if col in data.columns:
            data[col] = numeric(data[col])
    data["circuity_interval_definition"] = "p10-p90 distribution interval across city-type OD summaries"
    keep = [
        "morphotype",
        "morphotype_name",
        "morphotype_label_en",
        "weighted_circuity",
        "p10_circuity",
        "p90_circuity",
        "route_found_weighted_share",
        "detour_ratio_median",
        "circuity_interval_definition",
        "city_group_count",
        "od_pair_count",
        "weight_sum",
    ]
    return data[[c for c in keep if c in data.columns]].sort_values("morphotype")


def fig7_quadrant_label(high_access: bool, low_detour: bool) -> str:
    if high_access and low_detour:
        return "high_access_low_detour"
    if high_access and not low_detour:
        return "high_access_high_detour"
    if not high_access and low_detour:
        return "low_access_low_detour"
    return "low_access_high_detour"


def build_fig7_data(access: pd.DataFrame, circuity: pd.DataFrame) -> pd.DataFrame:
    access_panel = build_access_panel_data(access)
    circuity_summary = build_circuity_summary_data(circuity)
    access_mean = (
        access_panel[["morphotype", "morphotype_mean_access_share", "morphotype_rank"]]
        .drop_duplicates("morphotype")
        .merge(circuity_summary, on="morphotype", how="left", suffixes=("", "_circuity"))
    )
    access_threshold = float(access_mean["morphotype_mean_access_share"].median())
    circuity_threshold = float(access_mean["weighted_circuity"].median())
    access_mean["is_high_access"] = access_mean["morphotype_mean_access_share"] >= access_threshold
    access_mean["is_low_detour"] = access_mean["weighted_circuity"] <= circuity_threshold
    access_mean["quadrant_label"] = [
        fig7_quadrant_label(bool(high), bool(low))
        for high, low in zip(access_mean["is_high_access"], access_mean["is_low_detour"])
    ]
    access_mean["threshold_method"] = FIG7_THRESHOLD_METHOD
    access_mean["access_threshold"] = access_threshold
    access_mean["circuity_threshold"] = circuity_threshold
    access_mean = add_morphotype_color_field(access_mean)

    quadrant_fields = [
        "morphotype",
        "morphotype_mean_access_share",
        "weighted_circuity",
        "p10_circuity",
        "p90_circuity",
        "route_found_weighted_share",
        "detour_ratio_median",
        "circuity_interval_definition",
        "morphotype_rank",
        "quadrant_label",
        "threshold_method",
        "access_threshold",
        "circuity_threshold",
        "is_high_access",
        "is_low_detour",
        "morphotype_color",
        "morphotype_palette",
    ]
    quadrant = access_mean[[c for c in quadrant_fields if c in access_mean.columns]].copy()

    access_rows = access_panel.merge(
        quadrant[
            [
                "morphotype",
                "weighted_circuity",
                "p10_circuity",
                "p90_circuity",
                "quadrant_label",
                "threshold_method",
                "access_threshold",
                "circuity_threshold",
                "is_high_access",
                "is_low_detour",
            ]
        ],
        on="morphotype",
        how="left",
    )
    access_rows["panel"] = FIG7_PANEL_ACCESS
    access_rows["panel_order"] = 1
    access_rows["access_metric"] = access_rows["population_weighted_access_share"]
    access_rows["access_metric_name"] = "population_weighted_access_share"
    access_rows["access_metric_units"] = "share"
    access_rows["circuity_metric"] = access_rows["weighted_circuity"]
    access_rows["circuity_metric_name"] = "weighted_circuity"
    access_rows["circuity_metric_units"] = "ratio"
    access_rows["panel_value"] = access_rows["access_share_diff_pp"]
    access_rows["panel_value_units"] = "percentage points"
    access_rows["panel_definition"] = "Access heatmap cell; color is category-mean access-share difference and label is raw access share."

    detour_rows = quadrant.copy()
    detour_rows["panel"] = FIG7_PANEL_DETOUR
    detour_rows["panel_order"] = 2
    detour_rows["category"] = "all"
    detour_rows["category_label"] = "All OD pairs"
    detour_rows["category_order"] = 99
    detour_rows["access_metric"] = detour_rows["morphotype_mean_access_share"]
    detour_rows["access_metric_name"] = "morphotype_mean_access_share"
    detour_rows["access_metric_units"] = "share"
    detour_rows["circuity_metric"] = detour_rows["weighted_circuity"]
    detour_rows["circuity_metric_name"] = "weighted_circuity"
    detour_rows["circuity_metric_units"] = "ratio"
    detour_rows["panel_value"] = detour_rows["weighted_circuity"]
    detour_rows["panel_value_units"] = "ratio"
    detour_rows["panel_definition"] = "Detour/circuity interval panel; point is weighted OD circuity and interval is p10-p90."

    quadrant_rows = quadrant.copy()
    quadrant_rows["panel"] = FIG7_PANEL_QUADRANT
    quadrant_rows["panel_order"] = 3
    quadrant_rows["category"] = "all"
    quadrant_rows["category_label"] = "All facility categories and OD pairs"
    quadrant_rows["category_order"] = 100
    quadrant_rows["access_metric"] = quadrant_rows["morphotype_mean_access_share"]
    quadrant_rows["access_metric_name"] = "morphotype_mean_access_share"
    quadrant_rows["access_metric_units"] = "share"
    quadrant_rows["circuity_metric"] = quadrant_rows["weighted_circuity"]
    quadrant_rows["circuity_metric_name"] = "weighted_circuity"
    quadrant_rows["circuity_metric_units"] = "ratio"
    quadrant_rows["panel_value"] = quadrant_rows["morphotype_mean_access_share"]
    quadrant_rows["panel_value_units"] = "share"
    quadrant_rows["panel_definition"] = "Accessibility-circuity quadrant panel; thresholds are medians across MT01-MT10."

    combined = pd.concat([detour_rows, quadrant_rows], ignore_index=True, sort=False)
    combined["access_metric_percent"] = numeric(combined["access_metric"]) * 100.0
    combined["quadrant_threshold_note"] = FIG7_THRESHOLD_METHOD
    combined["detour_panel_interval_metric"] = "p10_circuity; p90_circuity"
    combined["fig9_panel_a_removed"] = True
    combined["fig9_panel_a_heatmap_current"] = False
    combined["fig9_panel_a_colorbar_current"] = False
    combined["fig9_former_panel_a_migrated_to"] = "Fig6C"
    combined["fig9_former_panel_a_migration_note"] = FIG7_PANEL_A_REMOVAL_NOTE
    return combined.sort_values(["panel_order", "morphotype_rank", "category_order", "morphotype"])


def fig7_layout_metadata() -> dict[str, Any]:
    current = {"panel_b_c": FIG7_LAYOUT_WIDTH_RATIOS["bottom_panel_b_c"]}
    previous = FIG7_LAYOUT_WIDTH_RATIOS_PREVIOUS
    text_cleanup = fig7_footnote_cleanup_metadata()
    leader_line_style = {
        **FIG7_QUADRANT_LEADER_LINE_STYLE,
        "connectionstyle": FIG7_QUADRANT_LEADER_LINE_CONNECTIONSTYLE,
    }
    return {
        "grid_spec": FIG7_BC_LAYOUT_GRID_SPEC,
        "width_ratios_previous": previous,
        "width_ratios_current": current,
        "bottom_width_ratios_current": current["panel_b_c"],
        "wspace_previous": FIG7_LAYOUT_WSPACE_PREVIOUS,
        "wspace_current": FIG7_LAYOUT_WSPACE,
        "bottom_wspace_current": FIG7_LAYOUT_BOTTOM_WSPACE,
        "current_visual_panels": list(FIG7_CURRENT_VISUAL_PANELS),
        "panel_a_removed": True,
        "panel_a_heatmap_current": False,
        "panel_a_colorbar_current": False,
        "former_panel_a_migrated_to": "Fig6C",
        "former_panel_a_migration_note": FIG7_PANEL_A_REMOVAL_NOTE,
        "savefig_pad_inches": FIG7_SAVEFIG_PAD_INCHES,
        "panel_b_c_bottom_side_by_side": True,
        "panel_b_y_axis_restored": True,
        "panel_c_marker_size_previous": FIG7_PANEL_C_MARKER_SIZE_PREVIOUS,
        "panel_c_marker_size_current": FIG7_PANEL_C_MARKER_SIZE,
        "panel_c_marker_size_reduction_percent": FIG7_PANEL_C_MARKER_SIZE_REDUCTION_PERCENT,
        "panel_c_marker_size_units": FIG7_PANEL_C_MARKER_SIZE_UNITS,
        "panel_c_leader_line_labels": list(FIG7_QUADRANT_LEADER_LINE_LABELS),
        "panel_c_leader_line_count": len(FIG7_QUADRANT_LEADER_LINE_LABELS),
        "panel_c_leader_line_style": leader_line_style,
        "panel_c_marker_label_note": FIG7_PANEL_C_MARKER_LABEL_NOTE,
        "panel_c_label_offset_revisions": FIG7_PANEL_C_LABEL_OFFSET_REVISIONS,
        "panel_c_label_adjustment_note": FIG7_PANEL_C_LABEL_ADJUSTMENT_NOTE,
        "bottom_note_current": text_cleanup["bottom_note_current"],
        "removed_bottom_note_text": text_cleanup["removed_bottom_note_text"],
        "retained_visual_text_elements": text_cleanup["retained_visual_text_elements"],
        "footnote_removal_note": text_cleanup["removal_note"],
        "overlap_controls": (
            "single-row A/B GridSpec, balanced A/B width ratios, restored Panel A left spine/ticks/MTxx y tick labels, "
            "dedicated horizontal wspace, and Fig8 savefig padding to avoid title, axis-label and annotation "
            "overlap/clipping. Former access heatmap and colorbar are not drawn in current Fig8. Current Panel B uses smaller "
            "circle markers and thin neutral leader lines for distant or dense labels; MT03 is pulled to the right "
            "and MT04 is shortened/lowered to reduce local label-line conflicts."
        ),
        "layout_note": FIG7_LAYOUT_NOTE,
    }


def plot_fig7(paths: StepPaths, access: pd.DataFrame, circuity: pd.DataFrame) -> FigureProduct:
    data = build_fig7_data(access, circuity)
    layout = fig7_layout_metadata()
    text_cleanup = fig7_footnote_cleanup_metadata()
    data["fig9_layout_width_ratios_previous"] = json.dumps(layout["width_ratios_previous"], ensure_ascii=False)
    data["fig9_layout_width_ratios_current"] = json.dumps(layout["width_ratios_current"], ensure_ascii=False)
    data["fig9_layout_grid_spec"] = layout["grid_spec"]
    data["fig9_layout_bottom_width_ratios_current"] = json.dumps(layout["bottom_width_ratios_current"], ensure_ascii=False)
    data["fig9_layout_wspace"] = layout["wspace_current"]
    data["fig9_layout_bottom_wspace"] = layout["bottom_wspace_current"]
    data["fig9_current_visual_panels"] = ", ".join(layout["current_visual_panels"])
    data["fig9_panel_a_removed"] = layout["panel_a_removed"]
    data["fig9_panel_a_heatmap_current"] = layout["panel_a_heatmap_current"]
    data["fig9_panel_a_colorbar_current"] = layout["panel_a_colorbar_current"]
    data["fig9_former_panel_a_migrated_to"] = layout["former_panel_a_migrated_to"]
    data["fig9_former_panel_a_migration_note"] = layout["former_panel_a_migration_note"]
    data["fig9_savefig_pad_inches"] = layout["savefig_pad_inches"]
    data["fig9_layout_panel_b_c_bottom_side_by_side"] = layout["panel_b_c_bottom_side_by_side"]
    data["fig9_layout_panel_b_y_axis_restored"] = layout["panel_b_y_axis_restored"]
    data["fig9_panel_label"] = data["panel"].map({FIG7_PANEL_DETOUR: "A", FIG7_PANEL_QUADRANT: "B"}).fillna("")
    data["fig9_panel_c_marker_size_previous"] = layout["panel_c_marker_size_previous"]
    data["fig9_panel_c_marker_size_current"] = layout["panel_c_marker_size_current"]
    data["fig9_panel_c_marker_size_reduction_percent"] = layout["panel_c_marker_size_reduction_percent"]
    data["fig9_panel_c_marker_size_units"] = layout["panel_c_marker_size_units"]
    data["fig9_panel_c_leader_line_labels"] = ", ".join(layout["panel_c_leader_line_labels"])
    data["fig9_panel_c_leader_line_count"] = layout["panel_c_leader_line_count"]
    data["fig9_panel_c_leader_line_style"] = json.dumps(layout["panel_c_leader_line_style"], ensure_ascii=False, sort_keys=True)
    data["fig9_panel_c_marker_label_note"] = layout["panel_c_marker_label_note"]
    data["fig9_panel_c_label_offset_points"] = data["morphotype"].map(
        lambda mt: json.dumps(FIG7_QUADRANT_LABEL_OFFSETS.get(str(mt), (5, 0, "left")), ensure_ascii=False)
    )
    data["fig9_panel_c_label_offset_revisions"] = json.dumps(
        layout["panel_c_label_offset_revisions"], ensure_ascii=False, sort_keys=True
    )
    data["fig9_panel_c_label_adjustment_note"] = layout["panel_c_label_adjustment_note"]
    leader_line_labels = set(FIG7_QUADRANT_LEADER_LINE_LABELS)
    data["fig9_panel_c_label_has_leader_line"] = data["panel"].eq(FIG7_PANEL_QUADRANT) & data["morphotype"].astype(str).isin(
        leader_line_labels
    )
    data["fig9_panel_b_y_axis_setting"] = "current Panel A: left spine, y ticks, and MTxx morphotype tick labels visible"
    data["fig9_layout_note"] = layout["layout_note"]
    data["fig9_bottom_note_current"] = text_cleanup["bottom_note_current"]
    data["fig9_removed_bottom_note_text"] = text_cleanup["removed_bottom_note_text"]
    data["fig9_retained_visual_text_elements"] = text_cleanup["retained_visual_text_elements"]
    data["fig9_footnote_removal_note"] = text_cleanup["removal_note"]
    data = add_morphotype_color_field(data)
    data = add_palette_metadata(
        data,
        palette=f"{FIG8_INTERVAL_PALETTE_NAME} for Panel A intervals; {MORPHOTYPE_PALETTE_NAME} for A/B morphotype markers",
        cmap="",
        heatmap_cmap="",
        color_variable="weighted_circuity; morphotype",
        color_variable_label="Weighted OD circuity and morphotype class",
        palette_note=(
            "Current Fig8 has no heatmap or heatmap colorbar. The former access-performance heatmap is migrated "
            "to Fig6C. Panel A uses neutral interval styling plus morphotype-colored points; Panel B uses fixed "
            "morphotype marker colors."
        ),
    )
    data["figure_id"] = "Fig8"
    data["figure_slug"] = FIG9_ACCESS_DETOUR_SLUG
    data = replace_text_in_string_columns(
        data,
        [
            ("Former Fig7A", "Former access heatmap"),
            ("former Fig7A", "former access heatmap"),
            ("Fig7C", "Fig6C"),
            ("current Fig7", "current Fig8"),
            ("Current Fig7", "Current Fig8"),
            ("current Fig9", "current Fig8"),
            ("Current Fig9", "Current Fig8"),
            ("Fig9 layout", "Fig8 layout"),
            ("Fig9 rendered", "Fig8 rendered"),
            ("Fig9 source", "Fig8 source"),
            ("Fig9 savefig", "Fig8 savefig"),
        ],
    )
    data = data.rename(columns=lambda col: col.replace("fig9_", "fig9_"))
    data_path = make_output_path(paths, source_data_name("Fig8", FIG9_ACCESS_DETOUR_SLUG))
    write_csv(data, data_path)

    detour_rows = data[data["panel"].eq(FIG7_PANEL_DETOUR)].copy()
    quadrant_rows = data[data["panel"].eq(FIG7_PANEL_QUADRANT)].copy()
    row_order = (
        detour_rows[["morphotype", "morphotype_rank"]]
        .drop_duplicates()
        .sort_values(["morphotype_rank", "morphotype"])["morphotype"]
        .tolist()
    )
    if not row_order:
        row_order = (
            quadrant_rows[["morphotype", "morphotype_rank"]]
            .drop_duplicates()
            .sort_values(["morphotype_rank", "morphotype"])["morphotype"]
            .tolist()
        )
    detour_plot = (
        detour_rows.set_index("morphotype")
        .reindex(row_order)
    )
    quadrant_plot = quadrant_rows.set_index("morphotype").reindex(row_order).reset_index()
    fig = plt.figure(figsize=cm_to_in(18.3, 9.8))
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=FIG7_LAYOUT_WIDTH_RATIOS["bottom_panel_b_c"],
        wspace=FIG7_LAYOUT_BOTTOM_WSPACE,
    )
    ax_detour = fig.add_subplot(gs[0, 0])
    ax_quad = fig.add_subplot(gs[0, 1])
    fig.subplots_adjust(bottom=0.22, top=0.88)

    y = np.arange(len(row_order))
    x = numeric(detour_plot["weighted_circuity"]).to_numpy(dtype=float)
    low = np.maximum(0, x - numeric(detour_plot["p10_circuity"]).to_numpy(dtype=float))
    high = np.maximum(0, numeric(detour_plot["p90_circuity"]).to_numpy(dtype=float) - x)
    x_min = max(1.0, math.floor(float(np.nanmin(x - low) - 0.025) * 100.0) / 100.0)
    x_max = math.ceil(float(np.nanmax(x + high) + 0.045) * 100.0) / 100.0
    ax_detour.errorbar(
        x,
        y,
        xerr=np.vstack([low, high]),
        fmt="o",
        ms=4.5,
        lw=1.0,
        capsize=2.6,
        color=FIG8_INTERVAL_LINE_COLOR,
        ecolor=FIG8_INTERVAL_RANGE_COLOR,
        zorder=1,
    )
    ax_detour.scatter(
        x,
        y,
        s=31,
        color=detour_plot["morphotype_color"].fillna("#64748B").tolist(),
        edgecolor="#111827",
        linewidth=0.4,
        zorder=2,
    )
    label_offset = max((x_max - x_min) * 0.018, 0.006)
    for yi, value in zip(y, x):
        if pd.notna(value):
            ax_detour.text(
                min(float(value) + label_offset, x_max - label_offset * 0.25),
                yi,
                f"{value:.2f}",
                va="center",
                fontsize=FIG7_DETOUR_VALUE_LABEL_FONTSIZE,
                color="#334155",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.4},
                clip_on=True,
            )
    ax_detour.set_xlim(x_min, x_max)
    ax_detour.set_xlabel("Weighted OD circuity", labelpad=4)
    ax_detour.set_title("A  Detour/circuity interval", loc="left", fontsize=8, fontweight="bold")
    ax_detour.grid(axis="x", color="#E2E8F0", lw=0.5)
    ax_detour.set_yticks(y)
    ax_detour.set_yticklabels(row_order)
    ax_detour.set_ylabel("Morphotype", labelpad=4)
    ax_detour.set_ylim(len(row_order) - 0.5, -0.5)
    ax_detour.tick_params(axis="y", which="both", left=True, right=False, labelleft=True, length=2.5, pad=2)
    ax_detour.spines["left"].set_visible(True)

    quad_colors = quadrant_plot["morphotype_color"].fillna("#64748B").tolist()
    access_percent = numeric(quadrant_plot["access_metric"]) * 100.0
    circuity_value = numeric(quadrant_plot["circuity_metric"])
    access_threshold = float(quadrant_plot["access_threshold"].dropna().iloc[0]) * 100.0
    circuity_threshold = float(quadrant_plot["circuity_threshold"].dropna().iloc[0])
    ax_quad.axvline(access_threshold, color="#94A3B8", lw=0.9, ls="--")
    ax_quad.axhline(circuity_threshold, color="#94A3B8", lw=0.9, ls="--")
    ax_quad.scatter(
        access_percent,
        circuity_value,
        s=FIG7_PANEL_C_MARKER_SIZE,
        color=quad_colors,
        edgecolor="#111827",
        linewidth=0.45,
        zorder=5,
    )
    x_pad = max((float(access_percent.max()) - float(access_percent.min())) * 0.10, 1.2)
    y_pad = max((float(circuity_value.max()) - float(circuity_value.min())) * 0.10, 0.015)
    ax_quad.set_xlim(float(access_percent.min()) - x_pad, float(access_percent.max()) + x_pad * 1.4)
    ax_quad.set_ylim(float(circuity_value.min()) - y_pad, float(circuity_value.max()) + y_pad)
    leader_line_labels = set(FIG7_QUADRANT_LEADER_LINE_LABELS)
    leader_line_arrowprops = {
        **FIG7_QUADRANT_LEADER_LINE_STYLE,
        "connectionstyle": FIG7_QUADRANT_LEADER_LINE_CONNECTIONSTYLE,
    }
    for _, row in quadrant_plot.iterrows():
        mt = str(row["morphotype"])
        dx, dy, ha = FIG7_QUADRANT_LABEL_OFFSETS.get(mt, (5, 0, "left"))
        ax_quad.annotate(
            mt,
            xy=(float(row["access_metric"]) * 100.0, float(row["circuity_metric"])),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=FIG7_QUADRANT_LABEL_FONTSIZE,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.25},
            arrowprops=leader_line_arrowprops if mt in leader_line_labels else None,
            clip_on=False,
            zorder=6,
        )
    ax_quad.set_xlabel("Mean access (%)", labelpad=4)
    ax_quad.set_ylabel("Weighted OD circuity", labelpad=4)
    ax_quad.set_title("B  Access-detour quadrants", loc="left", fontsize=8, fontweight="bold")
    ax_quad.grid(color="#E2E8F0", lw=0.5)
    png, svg, pdf = save_figure(
        fig,
        paths.figure_dir / f"Fig8_{FIG9_ACCESS_DETOUR_SLUG}",
        pad_inches=FIG7_SAVEFIG_PAD_INCHES,
    )
    return FigureProduct(
        figure_id="Fig8",
        title="Accessibility and detour performance by morphotype",
        kind="main",
        png_path=png,
        svg_path=svg,
        pdf_path=pdf,
        data_path=data_path,
        source_inputs=[
            "data/11_statistical_models_and_robustness_checks/figure_data_accessibility_by_type.csv",
            "data/11_statistical_models_and_robustness_checks/figure_data_morphotype_circuity.csv",
        ],
        transformation=(
            "Combined morphotype circuity intervals and median-split accessibility-circuity quadrant assignments "
            "into one long source-data table with current visual panels A and B only. The former access heatmap "
            "panel is migrated to Fig6C and is not drawn or exported as a current Fig8 panel. "
            f"{FIG7_LAYOUT_NOTE} {FIG7_FOOTNOTE_REMOVAL_NOTE}"
        ),
        palette=f"{FIG8_INTERVAL_PALETTE_NAME} for Panel A intervals; {MORPHOTYPE_PALETTE_NAME} for A/B morphotype markers",
        cmap="",
        heatmap_cmap="",
        color_variable="weighted_circuity; morphotype",
        palette_note=(
            "Current Fig8 has no heatmap and no heatmap colorbar. "
            f"Layout uses A/B width ratios={FIG7_LAYOUT_WIDTH_RATIOS['bottom_panel_b_c']} and "
            f"wspace={FIG7_LAYOUT_BOTTOM_WSPACE}; Panel A restores the morphotype y-axis. "
            f"{FIG7_PANEL_C_MARKER_LABEL_NOTE}"
        ),
        visual_text_note=FIG7_FOOTNOTE_REMOVAL_NOTE,
    )


def build_fig8_data(inequality: pd.DataFrame) -> pd.DataFrame:
    data = inequality.copy()
    if "group_level" in data.columns:
        data = data[data["group_level"].eq("city")].copy()
    data = data[data["category"].isin(FACILITY_CATEGORIES)].copy()
    data["category_label"] = data["category"].map(CATEGORY_LABELS)
    data["category_order"] = data["category"].map({category: idx for idx, category in enumerate(FACILITY_CATEGORIES)})
    for col in ["accessibility_gini", "no_access_population_share", "population_weighted_access_share", "quality_score", "city_category_weight"]:
        if col in data.columns:
            data[col] = numeric(data[col])
    data["population_weighted_access_share_percent"] = data["population_weighted_access_share"] * 100.0
    data["no_access_population_share_percent"] = data["no_access_population_share"] * 100.0
    data["level_note"] = "city-level rows from Step11 figure_data_inequality.csv"
    data["figure_id"] = "Fig10"
    data["figure_slug"] = FIG11_CITY_INEQUALITY_SLUG
    data["fig11_row_definition"] = "One city x facility-category accessibility inequality summary row."
    return data.sort_values(["category_order", "city_id"]).reset_index(drop=True)


def plot_fig8(paths: StepPaths, inequality: pd.DataFrame) -> FigureProduct:
    data = build_fig8_data(inequality)
    data = add_palette_metadata(
        data,
        palette=FIG8_GRAYSCALE_PALETTE_NAME,
        color_variable="facility category; city-level accessibility inequality metrics",
        color_variable_label="City-level accessibility inequality",
        palette_note="Neutral grayscale boxplots; individual city points use low-alpha category positions without category hue encoding.",
    )
    data_path = make_output_path(paths, source_data_name("Fig10", FIG11_CITY_INEQUALITY_SLUG))
    write_csv(data, data_path)

    fig, axes = plt.subplots(1, 3, figsize=cm_to_in(18.3, 8.8), sharex=True)
    metric_specs = [
        ("population_weighted_access_share_percent", "Access share (%)"),
        ("accessibility_gini", "Accessibility Gini"),
        ("no_access_population_share_percent", "No-access share (%)"),
    ]
    rng = np.random.default_rng(20260602)
    for ax, (metric, title) in zip(axes, metric_specs):
        box_data = [data.loc[data["category"].eq(cat), metric].dropna() for cat in FACILITY_CATEGORIES]
        bp = ax.boxplot(box_data, tick_labels=[CATEGORY_LABELS[c] for c in FACILITY_CATEGORIES], patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(FIG8_BOX_FACE_COLOR)
            patch.set_edgecolor(FIG8_BOX_EDGE_COLOR)
            patch.set_linewidth(0.8)
        for median in bp["medians"]:
            median.set_color(FIG8_MEDIAN_COLOR)
            median.set_linewidth(1.0)
        for element in [*bp["whiskers"], *bp["caps"]]:
            element.set_color(FIG8_WHISKER_COLOR)
            element.set_linewidth(0.8)
        for flier in bp["fliers"]:
            flier.set_markerfacecolor(FIG8_FLIER_FACE_COLOR)
            flier.set_markeredgecolor(FIG8_WHISKER_COLOR)
            flier.set_markersize(2.3)
            flier.set_alpha(0.55)
        for idx, cat in enumerate(FACILITY_CATEGORIES, start=1):
            values = data.loc[data["category"].eq(cat), metric].dropna().to_numpy(dtype=float)
            if values.size:
                jitter = rng.normal(0, 0.025, size=values.size)
                ax.scatter(np.full(values.size, idx) + jitter, values, s=7, color="#334155", alpha=0.28, linewidth=0, zorder=1)
        ax.set_title(title, loc="left", fontsize=8, fontweight="bold")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", color=FIG8_GRID_COLOR, lw=0.5)
    fig.subplots_adjust(bottom=0.25)
    png, svg, pdf = save_figure(fig, paths.figure_dir / f"Fig10_{FIG11_CITY_INEQUALITY_SLUG}")
    return FigureProduct(
        figure_id="Fig10",
        title="City-level accessibility inequality",
        kind="main",
        png_path=png,
        svg_path=svg,
        pdf_path=pdf,
        data_path=data_path,
        source_inputs=["data/11_statistical_models_and_robustness_checks/figure_data_inequality.csv"],
        transformation="Filtered Step11 inequality figure data to group_level == city and plotted city-level accessibility, Gini, and no-access distributions by facility category.",
        palette=FIG8_GRAYSCALE_PALETTE_NAME,
        color_variable="facility category; city-level accessibility inequality metrics",
        palette_note="Neutral grayscale boxplots; individual city points use low-alpha category positions without category hue encoding.",
    )


def parse_morphotype_from_term(term: str) -> str | None:
    match = re.search(r"T\.(MT\d{2})\]", str(term))
    if match:
        return match.group(1)
    match = re.search(r"\b(MT\d{2})\b", str(term))
    return match.group(1) if match else None


def build_fig9_data(model: pd.DataFrame) -> pd.DataFrame:
    selected_outcomes = {
        ("A_OD_circuity", "log_od_circuity_weighted_mean"): "OD circuity, log scale",
        ("B_facility_accessibility", "accessibility_score"): "Accessibility score",
        ("C_accessibility_inequality", "accessibility_gini"): "Accessibility inequality",
    }
    frames = []
    for (family, outcome), label in selected_outcomes.items():
        g = model[
            model["model_family"].eq(family)
            & model["outcome"].eq(outcome)
            & model["status"].eq("ok")
            & model["term_kind"].eq("morphotype")
            & ~model["term"].astype(str).str.contains(":", regex=False)
        ].copy()
        if g.empty:
            continue
        g["morphotype"] = g["term"].map(parse_morphotype_from_term)
        g = g[g["morphotype"].isin(MORPHOTYPES)].copy()
        g["outcome_label"] = label
        g["reference_flag"] = False
        g["category_context"] = "overall or reference facility category"
        frames.append(g)
        ref = g.head(1).copy()
        if not ref.empty:
            ref = ref.assign(
                term="MT01 reference (omitted by treatment coding)",
                term_kind="reference",
                coef=0.0,
                std_err=np.nan,
                t_value=np.nan,
                p_value=np.nan,
                ci_low=0.0,
                ci_high=0.0,
                morphotype="MT01",
                outcome_label=label,
                reference_flag=True,
                category_context="reference morphotype",
            )
            frames.append(ref)
    if frames:
        data = pd.concat(frames, ignore_index=True, sort=False)
    else:
        data = pd.DataFrame()
    data = add_morphotype_labels(data)
    keep = [
        "model_id",
        "model_family",
        "outcome",
        "outcome_label",
        "term",
        "term_kind",
        "morphotype",
        "morphotype_label_en",
        "coef",
        "std_err",
        "p_value",
        "ci_low",
        "ci_high",
        "nobs",
        "r_squared",
        "category_context",
        "reference_flag",
        "formula",
    ]
    data = data[[c for c in keep if c in data.columns]].copy()
    data["reference_note"] = "MT01 is the reference category; displayed at zero for orientation."
    data["fixed_effect_filter_note"] = "Intercepts, city fixed effects, and other fixed-effect terms are excluded."
    data["morphotype_order"] = data["morphotype"].map({mt: i for i, mt in enumerate(MORPHOTYPES)})
    outcome_order = {label: i for i, label in enumerate(selected_outcomes.values())}
    data["outcome_order"] = data["outcome_label"].map(outcome_order)
    return data.sort_values(["outcome_order", "morphotype_order"]).drop(columns=["outcome_order", "morphotype_order"])


def plot_fig9(paths: StepPaths, model: pd.DataFrame) -> FigureProduct:
    data = build_fig9_data(model)
    data = add_morphotype_color_field(data)
    data = add_palette_metadata(
        data,
        palette=MORPHOTYPE_PALETTE_NAME,
        color_variable="morphotype",
        color_variable_label="Morphotype",
        palette_note="Coefficient sign/magnitude remains on the x-axis; point color uses fixed morphotype classes.",
    )
    data["figure_id"] = "Fig11"
    data["figure_slug"] = FIG12_MODEL_COEFFICIENT_SLUG
    data_path = make_output_path(paths, source_data_name("Fig11", FIG12_MODEL_COEFFICIENT_SLUG))
    write_csv(data, data_path)

    outcomes = data["outcome_label"].dropna().drop_duplicates().tolist()
    fig, axes = plt.subplots(1, len(outcomes), figsize=cm_to_in(18.3, 10.8), sharey=True)
    if len(outcomes) == 1:
        axes = [axes]
    for ax, outcome in zip(axes, outcomes):
        g = data[data["outcome_label"].eq(outcome)].set_index("morphotype").reindex(MORPHOTYPES).reset_index()
        y = np.arange(len(MORPHOTYPES))
        coef = numeric(g["coef"]).to_numpy(dtype=float)
        ci_low = numeric(g["ci_low"]).to_numpy(dtype=float)
        ci_high = numeric(g["ci_high"]).to_numpy(dtype=float)
        err_low = np.maximum(0, coef - ci_low)
        err_high = np.maximum(0, ci_high - coef)
        colors = [TYPE_COLORS.get(mt, SEQUENTIAL_LINE_COLOR) for mt in MORPHOTYPES]
        ax.axvline(
            0,
            color=FIG9_ZERO_REFERENCE_LINE_COLOR,
            lw=FIG9_ZERO_REFERENCE_LINEWIDTH,
            alpha=FIG9_ZERO_REFERENCE_ALPHA,
            zorder=0,
        )
        ax.errorbar(coef, y, xerr=np.vstack([err_low, err_high]), fmt="none", ecolor="#94A3B8", lw=1.0, capsize=2.8, zorder=1)
        ax.scatter(coef, y, s=30, color=colors, edgecolor="white", linewidth=0.5, zorder=2)
        ax.set_title(outcome, loc="left", fontsize=8, fontweight="bold")
        ax.set_xlabel("Coefficient")
        ax.grid(axis="x", color="#E2E8F0", lw=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{mt}  {MORPHOTYPE_LABELS_EN[mt]}" for mt in MORPHOTYPES])
        ax.invert_yaxis()
    png, svg, pdf = save_figure(fig, paths.figure_dir / f"Fig11_{FIG12_MODEL_COEFFICIENT_SLUG}")
    return FigureProduct(
        figure_id="Fig11",
        title="Model coefficient forest",
        kind="main",
        png_path=png,
        svg_path=svg,
        pdf_path=pdf,
        data_path=data_path,
        source_inputs=["data/11_statistical_models_and_robustness_checks/table_model_results.csv"],
        transformation="Filtered status-ok morphotype main effects, excluded intercepts/city FE/other fixed-effect terms, parsed MTxx terms, and added MT01 reference rows.",
        palette=MORPHOTYPE_PALETTE_NAME,
        color_variable="morphotype",
        palette_note="Coefficient sign/magnitude remains on the x-axis; point color uses fixed morphotype classes.",
    )


def build_fig10_radar_data(type_centers: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["morphotype", "morphotype_name", "balanced_training_share", *RADAR_SCORE_METRICS]
    data = type_centers[[c for c in base_cols if c in type_centers.columns]].copy()
    data = data[data["morphotype"].isin(MORPHOTYPES)].sort_values("morphotype")
    data = add_morphotype_labels(data)
    data = add_morphotype_color_field(data)
    rows: list[dict[str, Any]] = []
    for _, row in data.iterrows():
        for order, metric in enumerate(RADAR_SCORE_METRICS, start=1):
            value = scalar(row.get(metric, np.nan))
            clipped = np.clip(float(value), -TYPE_CENTER_SCORE_LIMIT, TYPE_CENTER_SCORE_LIMIT) if pd.notna(value) else np.nan
            radius = (clipped + TYPE_CENTER_SCORE_LIMIT) / (2 * TYPE_CENTER_SCORE_LIMIT) if pd.notna(clipped) else np.nan
            family = RADAR_METRIC_FAMILIES[metric]
            rows.append(
                {
                    "morphotype": row["morphotype"],
                    "morphotype_name": row.get("morphotype_name", ""),
                    "morphotype_label_en": row["morphotype_label_en"],
                    "morphotype_plot_label": row["morphotype_plot_label"],
                    "morphotype_color": row["morphotype_color"],
                    "morphotype_palette": row["morphotype_palette"],
                    "balanced_training_share": row.get("balanced_training_share", np.nan),
                    "axis_code": RADAR_AXIS_CODES.get(metric, ""),
                    "score_metric": metric,
                    "score_metric_label": RADAR_SCORE_LABELS.get(metric, metric),
                    "score_metric_order": order,
                    "metric_family": family["metric_family"],
                    "family_label": family["family_label"],
                    "family_color": family["family_color"],
                    "metric_family_order": family["metric_family_order"],
                    "score_value": value,
                    "score_value_clipped": clipped,
                    "radar_radius": radius,
                    "radar_radius_definition": f"(score_value_clipped + {TYPE_CENTER_SCORE_LIMIT:g}) / {2 * TYPE_CENTER_SCORE_LIMIT:g}; zero score maps to 0.5",
                    "score_source": "Step09 type center score column",
                }
            )
    return pd.DataFrame(rows).sort_values(["morphotype", "score_metric_order"])


def plot_fig10(paths: StepPaths, type_centers: pd.DataFrame) -> FigureProduct:
    data = build_fig10_radar_data(type_centers)
    fig10_palette_note = (
        "Nature-style radial signature rendering uses lightly tinted morphotype color, polygonal grid rings, "
        "max-axis accent markers and segmented outer domain bands; outer axis labels use A-G codes with a shared bottom legend "
        "and family-color domain key. The rendered panel is title-free at the top; the compact bottom legend keeps only "
        "the A-G metric key and the Domains family-color key."
    )
    data = add_palette_metadata(
        data,
        palette=MORPHOTYPE_PALETTE_NAME,
        color_variable="morphotype",
        color_variable_label="Morphotype",
        palette_note=fig10_palette_note,
    )
    data["figure_id"] = "Fig5"
    data["figure_slug"] = FIG6_RADAR_SLUG
    data_path = make_output_path(paths, source_data_name("Fig5", FIG6_RADAR_SLUG))
    write_csv(data, data_path)

    axis_codes = [RADAR_AXIS_CODES[m] for m in RADAR_SCORE_METRICS]
    axis_mapping = [
        (RADAR_AXIS_CODES[m], RADAR_SCORE_LABELS[m], RADAR_METRIC_FAMILIES[m]["family_label"])
        for m in RADAR_SCORE_METRICS
    ]
    angles = np.linspace(0, 2 * np.pi, len(RADAR_SCORE_METRICS), endpoint=False)
    angles_closed = np.r_[angles, angles[0]]
    baseline = 0.5
    segment_width = (2 * np.pi / len(RADAR_SCORE_METRICS)) * 0.90

    def soften_color(hex_color: str, mix_with: str, amount: float) -> tuple[float, float, float]:
        base = np.array(mpl.colors.to_rgb(hex_color))
        target = np.array(mpl.colors.to_rgb(mix_with))
        return tuple((1 - amount) * base + amount * target)

    fig = plt.figure(figsize=cm_to_in(FIG10_FIGURE_WIDTH_CM, FIG10_FIGURE_HEIGHT_CM))
    radar_gs = fig.add_gridspec(
        2,
        5,
        left=0.045,
        right=0.985,
        top=FIG10_LAYOUT_TOP,
        bottom=FIG10_LAYOUT_BOTTOM,
        wspace=FIG10_LAYOUT_WSPACE,
        hspace=FIG10_LAYOUT_HSPACE,
    )
    axes = [fig.add_subplot(radar_gs[row, col], projection="polar") for row in range(2) for col in range(5)]
    for ax, mt in zip(axes, MORPHOTYPES):
        g = data[data["morphotype"].eq(mt)].sort_values("score_metric_order")
        values = g["radar_radius"].to_numpy(dtype=float)
        values_closed = np.r_[values, values[0]]
        raw_color = TYPE_COLORS.get(mt, "#64748B")
        line_color = soften_color(raw_color, "#1F2937", 0.18)
        fill_color = soften_color(raw_color, "#FFFFFF", 0.54)

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1)
        ax.set_xticks(angles)
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.grid(False)
        ax.spines["polar"].set_visible(False)
        ax.tick_params(axis="both", length=0)

        for theta in angles:
            ax.plot([theta, theta], [0, 1], color="#E5E7EB", lw=0.35, zorder=0)
        for radius, lw, color, linestyle in [
            (0.25, 0.35, "#E5E7EB", "-"),
            (baseline, 0.58, "#CBD5E1", (0, (2, 2))),
            (0.75, 0.35, "#E5E7EB", "-"),
            (1.00, 0.65, "#9CA3AF", "-"),
        ]:
            ax.plot(angles_closed, np.full_like(angles_closed, radius), color=color, lw=lw, ls=linestyle, zorder=0)
        ax.text(np.pi / 2, baseline + 0.025, "0.5", ha="left", va="center", fontsize=FIG10_RADAR_RING_LABEL_FONTSIZE, color="#94A3B8", clip_on=False)
        for theta, metric in zip(angles, RADAR_SCORE_METRICS):
            family = RADAR_METRIC_FAMILIES[metric]
            bars = ax.bar(
                theta,
                0.075,
                width=segment_width,
                bottom=1.025,
                align="center",
                color=family["family_color"],
                edgecolor="white",
                linewidth=0.35,
                alpha=0.84,
                zorder=5,
            )
            for patch in bars:
                patch.set_clip_on(False)
        ax.fill(angles_closed, values_closed, color=fill_color, alpha=0.22, zorder=1)
        ax.plot(angles_closed, values_closed, color=line_color, lw=1.15, solid_capstyle="round", zorder=3)
        for theta, radius in zip(angles, values):
            ax.plot([theta, theta], [baseline, radius], color=line_color, lw=0.9, alpha=0.74, zorder=2)
        ax.scatter(angles, values, s=12, color=line_color, edgecolor="white", linewidth=0.45, zorder=4)
        finite_idx = np.flatnonzero(np.isfinite(values))
        if finite_idx.size:
            max_idx = int(finite_idx[np.argmax(values[finite_idx])])
            max_code = axis_codes[max_idx]
            ax.scatter(
                [angles[max_idx]],
                [values[max_idx]],
                s=26,
                facecolor="white",
                edgecolor=line_color,
                linewidth=0.9,
                zorder=7,
            )
            ax.text(
                0.98,
                1.120,
                f"max: {max_code}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=FIG10_RADAR_MAX_LABEL_FONTSIZE,
                color="#475569",
            )
        for theta, code in zip(angles, axis_codes):
            ax.text(theta, 1.135, code, ha="center", va="center", fontsize=FIG10_RADAR_AXIS_CODE_FONTSIZE, fontweight="bold", color="#111827", clip_on=False, zorder=8)
        ax.text(0.02, 1.120, mt, transform=ax.transAxes, ha="left", va="bottom", fontsize=FIG10_RADAR_PANEL_LABEL_FONTSIZE, fontweight="bold", color="#111827")

    legend_ax = fig.add_axes(FIG10_LEGEND_BOUNDS)
    legend_ax.set_axis_off()
    legend_ax.axhline(0.96, color="#E5E7EB", lw=0.6, clip_on=False)
    legend_ax.text(
        0.0,
        0.72,
        "A-G metrics",
        ha="left",
        va="center",
        fontsize=FIG10_LEGEND_TITLE_FONTSIZE,
        fontweight="bold",
        color="#111827",
        transform=legend_ax.transAxes,
    )
    metric_x = np.linspace(0.145, 0.985, len(axis_mapping))
    for idx, (code, label, _family_label) in enumerate(axis_mapping):
        legend_ax.text(
            metric_x[idx],
            0.72,
            f"{code}  {label}",
            ha="center",
            va="center",
            fontsize=FIG10_LEGEND_TEXT_FONTSIZE,
            color="#111827",
            transform=legend_ax.transAxes,
        )
    legend_ax.text(
        0.0,
        0.34,
        "Domains",
        ha="left",
        va="center",
        fontsize=FIG10_LEGEND_TITLE_FONTSIZE,
        fontweight="bold",
        color="#111827",
        transform=legend_ax.transAxes,
    )
    family_x = [0.145, 0.425, 0.690]
    for x, family in zip(family_x, RADAR_FAMILY_ORDER):
        legend_ax.add_patch(
            Rectangle(
                (x, 0.255),
                0.026,
                0.17,
                transform=legend_ax.transAxes,
                facecolor=RADAR_FAMILY_COLORS[family],
                edgecolor="none",
                alpha=0.90,
            )
        )
        code_text = ",".join(RADAR_FAMILY_AXIS_CODES[family])
        legend_ax.text(
            x + 0.035,
            0.34,
            f"{RADAR_FAMILY_LABELS[family]} ({code_text})",
            ha="left",
            va="center",
            fontsize=FIG10_LEGEND_TEXT_FONTSIZE,
            color="#111827",
            transform=legend_ax.transAxes,
        )
    png, svg, pdf = save_figure(fig, paths.figure_dir / f"Fig5_{FIG6_RADAR_SLUG}")
    return FigureProduct(
        figure_id="Fig5",
        title="Morphotype radial / structural signatures",
        kind="main",
        png_path=png,
        svg_path=svg,
        pdf_path=pdf,
        data_path=data_path,
        source_inputs=["data/09_cluster_morphotypes/morphotype_centroids.parquet"],
        transformation=(
            "Converted Step09 type-center score columns into a 10 morphotype x 7 score radar long table "
            "with A-G axis codes and metric-family fields mapped to full metric labels, then rendered a Nature-style radial signature plate with segmented outer domain bands."
        ),
        palette=MORPHOTYPE_PALETTE_NAME,
        color_variable="morphotype",
        palette_note=fig10_palette_note,
    )


def build_fig11_data(group_means: pd.DataFrame) -> pd.DataFrame:
    if group_means.empty:
        return pd.DataFrame()
    data = group_means.copy()
    data = data[data["morphotype"].isin(MORPHOTYPES)].copy()
    data = add_morphotype_labels(data)
    data["weight_sum"] = numeric(data.get("weight_sum", pd.Series(np.nan, index=data.index)))
    data["marginal_mean"] = numeric(data.get("marginal_mean", pd.Series(np.nan, index=data.index)))
    focus = data[data["outcome"].isin(["od_circuity_weighted_mean", "accessibility_score", "accessibility_gini"])].copy()
    rows = []
    for (mt, label, outcome), g in focus.groupby(["morphotype", "morphotype_label_en", "outcome"], dropna=False):
        rows.append(
            {
                "morphotype": mt,
                "morphotype_label_en": label,
                "outcome": outcome,
                "observed_weighted_group_mean": weighted_mean(g["marginal_mean"], g["weight_sum"]),
                "source_model_count": int(g["source_model"].nunique()) if "source_model" in g.columns else np.nan,
                "category_count": int(g["category"].nunique()) if "category" in g.columns else np.nan,
                "weight_sum": float(g["weight_sum"].sum()),
                "label_policy": "Observed weighted group means / group marginal summary; not adjusted marginal effects.",
            }
        )
    return pd.DataFrame(rows).sort_values(["outcome", "morphotype"])


def plot_fig11(paths: StepPaths, group_means: pd.DataFrame) -> FigureProduct | None:
    data = build_fig11_data(group_means)
    if data.empty:
        return None
    data = add_morphotype_color_field(data)
    data = add_palette_metadata(
        data,
        palette=MORPHOTYPE_PALETTE_NAME,
        color_variable="morphotype",
        color_variable_label="Morphotype",
        palette_note="Bar colors use the same fixed morphotype colors as Fig5/Fig8/Fig9.",
    )
    data["figure_id"] = "Fig9"
    data["figure_slug"] = FIG10_OBSERVED_GROUP_MEANS_SLUG
    data_path = make_output_path(paths, source_data_name("Fig9", FIG10_OBSERVED_GROUP_MEANS_SLUG))
    write_csv(data, data_path)

    outcome_labels = {
        "od_circuity_weighted_mean": "Weighted circuity",
        "accessibility_score": "Accessibility score",
        "accessibility_gini": "Accessibility Gini",
    }
    outcomes = [o for o in outcome_labels if o in set(data["outcome"])]
    fig, axes = plt.subplots(1, len(outcomes), figsize=cm_to_in(18.3, 9.5), sharey=True)
    if len(outcomes) == 1:
        axes = [axes]
    for ax, outcome in zip(axes, outcomes):
        g = data[data["outcome"].eq(outcome)].set_index("morphotype").reindex(MORPHOTYPES).reset_index()
        ax.barh(
            np.arange(len(MORPHOTYPES)),
            g["observed_weighted_group_mean"],
            color=[TYPE_COLORS[mt] for mt in MORPHOTYPES],
            edgecolor="white",
        )
        ax.set_title(outcome_labels[outcome], loc="left", fontsize=8, fontweight="bold")
        ax.grid(axis="x", color="#E2E8F0", lw=0.5)
        ax.set_yticks(np.arange(len(MORPHOTYPES)))
        ax.set_yticklabels([f"{mt}  {MORPHOTYPE_LABELS_EN[mt]}" for mt in MORPHOTYPES])
        ax.invert_yaxis()
    png, svg, pdf = save_figure(fig, paths.figure_dir / f"Fig9_{FIG10_OBSERVED_GROUP_MEANS_SLUG}")
    return FigureProduct(
        figure_id="Fig9",
        title="Observed weighted group means / group marginal summary",
        kind="main",
        png_path=png,
        svg_path=svg,
        pdf_path=pdf,
        data_path=data_path,
        source_inputs=["data/11_statistical_models_and_robustness_checks/model_marginal_effects.csv"],
        transformation="Aggregated Step11 group summary rows with weight_sum and labeled them as observed weighted group means only.",
        palette=MORPHOTYPE_PALETTE_NAME,
        color_variable="morphotype",
        palette_note="Bar colors use the same fixed morphotype colors as Fig5/Fig8/Fig9.",
    )


def fig12_heatmap_cmap() -> mpl.colors.LinearSegmentedColormap:
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        FIG12_HEATMAP_CMAP,
        FIG12_HEATMAP_CMAP_COLORS,
        N=256,
    )
    cmap.set_bad("#F1F5F9")
    return cmap


def normalize_0_1(series: pd.Series) -> tuple[pd.Series, float, float]:
    values = numeric(series)
    finite = values.dropna()
    if finite.empty:
        return pd.Series(np.nan, index=series.index), np.nan, np.nan
    vmin = float(finite.min())
    vmax = float(finite.max())
    if math.isclose(vmin, vmax):
        return pd.Series(0.5, index=series.index), vmin, vmax
    return (values - vmin) / (vmax - vmin), vmin, vmax


def first_nonempty_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns and df[col].notna().any():
            return col
    return None


def build_fig12_core_metrics(
    core_profiles: pd.DataFrame,
    inequality: pd.DataFrame,
    od_metrics: pd.DataFrame,
    city_quality: pd.DataFrame,
) -> pd.DataFrame:
    hex1 = core_profiles[core_profiles["scale"].astype(str).eq("hex_1km")].copy()
    base_cols = ["city_id", "city_name_en", "region", "quality_tier"]
    city = hex1[base_cols].drop_duplicates("city_id").copy()

    ineq = inequality.copy()
    if "group_level" in ineq.columns:
        ineq = ineq[ineq["group_level"].astype(str).eq("city")].copy()
    if "scale" in ineq.columns:
        ineq = ineq[ineq["scale"].astype(str).eq("hex_1km")].copy()
    if not ineq.empty:
        ineq["population_weighted_access_share"] = numeric(ineq.get("population_weighted_access_share", pd.Series(np.nan, index=ineq.index)))
        ineq["accessibility_gini"] = numeric(ineq.get("accessibility_gini", pd.Series(np.nan, index=ineq.index)))
        access = (
            ineq.groupby("city_id", dropna=False)
            .agg(
                access_metric=("population_weighted_access_share", "mean"),
                inequality_metric=("accessibility_gini", "mean"),
                access_category_count=("category", "nunique") if "category" in ineq.columns else ("city_id", "size"),
            )
            .reset_index()
        )
    else:
        access = pd.DataFrame({"city_id": city["city_id"], "access_metric": np.nan, "inequality_metric": np.nan, "access_category_count": 0})
    city = city.merge(access, on="city_id", how="left")

    od = od_metrics.copy()
    if "group_level" in od.columns:
        od = od[od["group_level"].astype(str).eq("city")].copy()
    if "scale" in od.columns:
        od = od[od["scale"].astype(str).eq("hex_1km")].copy()
    od_cols = [
        "city_id",
        "od_circuity_weighted_mean",
        "od_pair_count",
        "route_found_weighted_share",
    ]
    od_cols = [c for c in od_cols if c in od.columns]
    if od_cols and "city_id" in od_cols:
        circuity = od[od_cols].drop_duplicates("city_id").rename(columns={"od_circuity_weighted_mean": "circuity_metric"})
    else:
        circuity = pd.DataFrame({"city_id": city["city_id"], "circuity_metric": np.nan})
    city = city.merge(circuity, on="city_id", how="left")

    q_cols = [c for c in ["city_id", "quality_score", "quality_weight"] if c in city_quality.columns]
    if "city_id" in q_cols:
        city = city.merge(city_quality[q_cols].drop_duplicates("city_id"), on="city_id", how="left")
    else:
        city["quality_score"] = np.nan
        city["quality_weight"] = np.nan

    city = city.merge(
        hex1[["city_id", "morphotype_entropy"]].rename(columns={"morphotype_entropy": "entropy_metric"}),
        on="city_id",
        how="left",
    )

    norm_specs = [
        ("entropy_metric", "entropy_norm"),
        ("access_metric", "access_norm"),
        ("circuity_metric", "circuity_norm"),
        ("inequality_metric", "inequality_norm"),
        ("quality_score", "quality_norm"),
    ]
    norm_meta: dict[str, tuple[float, float]] = {}
    for raw_col, norm_col in norm_specs:
        city[norm_col], vmin, vmax = normalize_0_1(city.get(raw_col, pd.Series(np.nan, index=city.index)))
        norm_meta[norm_col] = (vmin, vmax)
        city[f"{norm_col}_min"] = vmin
        city[f"{norm_col}_max"] = vmax

    city["access_metric_source"] = "Step11 figure_data_inequality.csv city-level hex_1km mean across five facility categories"
    city["inequality_metric_source"] = "Step11 figure_data_inequality.csv city-level hex_1km mean accessibility_gini across five facility categories"
    city["circuity_metric_source"] = "Step10 OD_detour_metrics.parquet group_level=city hex_1km od_circuity_weighted_mean"
    city["quality_metric_source"] = "Step07 quality_scores quality_score"
    city["metric_normalization_scope"] = "Min-max normalization across the 48 Fig8 core cities; dark strip cells indicate higher raw metric values."
    for norm_col, direction in FIG12_METRIC_DIRECTIONS.items():
        city[f"{norm_col}_direction_note"] = direction
    return city


def build_fig12_data(
    city_profiles: pd.DataFrame,
    inequality: pd.DataFrame,
    od_metrics: pd.DataFrame,
    city_quality: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    core = city_profiles.copy()
    core = core[
        core.get("quality_tier", pd.Series("", index=core.index)).astype(str).eq("core")
        & core.get("network_type", pd.Series("", index=core.index)).astype(str).eq("drive")
        & core.get("scale", pd.Series("", index=core.index)).astype(str).isin(["hex_1km", "hex_2km"])
    ].copy()
    core["dominant_morphotype"] = core.get("dominant_morphotype", pd.Series("", index=core.index)).astype(str)

    pop_share_cols = [f"population_share__{mt}" for mt in MORPHOTYPES if f"population_share__{mt}" in core.columns]
    unit_share_cols = [f"unit_share__{mt}" for mt in MORPHOTYPES if f"unit_share__{mt}" in core.columns]
    if "dominant_population_share" in core.columns and numeric(core["dominant_population_share"]).notna().any():
        core["dominant_share"] = numeric(core["dominant_population_share"])
        core["dominant_share_source"] = "dominant_population_share"
        core["dominant_share_fallback_note"] = "Primary Step09 dominant_population_share used."
    elif pop_share_cols:
        shares = core[pop_share_cols].apply(pd.to_numeric, errors="coerce")
        core["dominant_share"] = shares.max(axis=1)
        fallback_mt = shares.idxmax(axis=1).str.replace("population_share__", "", regex=False)
        core.loc[core["dominant_morphotype"].eq("") | core["dominant_morphotype"].eq("nan"), "dominant_morphotype"] = fallback_mt
        core["dominant_share_source"] = "max population_share__MTxx"
        core["dominant_share_fallback_note"] = "dominant_population_share unavailable; derived dominant share from population_share__MTxx columns."
    elif unit_share_cols:
        shares = core[unit_share_cols].apply(pd.to_numeric, errors="coerce")
        core["dominant_share"] = shares.max(axis=1)
        fallback_mt = shares.idxmax(axis=1).str.replace("unit_share__", "", regex=False)
        core.loc[core["dominant_morphotype"].eq("") | core["dominant_morphotype"].eq("nan"), "dominant_morphotype"] = fallback_mt
        core["dominant_share_source"] = "max unit_share__MTxx"
        core["dominant_share_fallback_note"] = "No population-share fields available; derived dominant share from unit_share__MTxx columns."
    else:
        core["dominant_share"] = np.nan
        core["dominant_share_source"] = "unavailable"
        core["dominant_share_fallback_note"] = "No dominant-share or MTxx share columns were available."

    entropy_col = first_nonempty_column(core, ["morphotype_entropy", "unit_morphotype_entropy"])
    if entropy_col is None and pop_share_cols:
        shares = core[pop_share_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        p = shares.div(shares.sum(axis=1).replace(0, np.nan), axis=0)
        core["morphotype_entropy"] = (-(p * np.log(p.replace(0, np.nan))).sum(axis=1) / math.log(len(MORPHOTYPES))).replace([np.inf, -np.inf], np.nan)
        core["entropy_source"] = "computed normalized Shannon entropy from population_share__MTxx"
    elif entropy_col is None and unit_share_cols:
        shares = core[unit_share_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        p = shares.div(shares.sum(axis=1).replace(0, np.nan), axis=0)
        core["morphotype_entropy"] = (-(p * np.log(p.replace(0, np.nan))).sum(axis=1) / math.log(len(MORPHOTYPES))).replace([np.inf, -np.inf], np.nan)
        core["entropy_source"] = "computed normalized Shannon entropy from unit_share__MTxx"
    else:
        if entropy_col != "morphotype_entropy":
            core["morphotype_entropy"] = numeric(core[entropy_col])
        core["entropy_source"] = entropy_col or "unavailable"

    city_metrics = build_fig12_core_metrics(core, inequality, od_metrics, city_quality)
    hex1 = core[core["scale"].astype(str).eq("hex_1km")].drop_duplicates("city_id").set_index("city_id")
    hex2 = core[core["scale"].astype(str).eq("hex_2km")].drop_duplicates("city_id").set_index("city_id")
    city_scale = pd.DataFrame(index=sorted(set(hex1.index) | set(hex2.index)))
    city_scale["dominant_morphotype_hex_1km"] = hex1.get("dominant_morphotype")
    city_scale["dominant_morphotype_hex_2km"] = hex2.get("dominant_morphotype")
    city_scale["dominant_share_hex_1km"] = numeric(hex1.get("dominant_share", pd.Series(np.nan, index=hex1.index))).reindex(city_scale.index)
    city_scale["dominant_share_hex_2km"] = numeric(hex2.get("dominant_share", pd.Series(np.nan, index=hex2.index))).reindex(city_scale.index)
    city_scale["morphotype_entropy_hex_1km"] = numeric(hex1.get("morphotype_entropy", pd.Series(np.nan, index=hex1.index))).reindex(city_scale.index)
    city_scale["morphotype_entropy_hex_2km"] = numeric(hex2.get("morphotype_entropy", pd.Series(np.nan, index=hex2.index))).reindex(city_scale.index)
    city_scale["type_changed_between_scales"] = (
        city_scale["dominant_morphotype_hex_1km"].astype(str)
        != city_scale["dominant_morphotype_hex_2km"].astype(str)
    )
    city_scale["share_delta"] = city_scale["dominant_share_hex_2km"] - city_scale["dominant_share_hex_1km"]
    city_scale = city_scale.reset_index(names="city_id")

    order = (
        hex1.reset_index()[["city_id", "city_name_en", "dominant_share"]]
        .sort_values(["dominant_share", "city_name_en"], ascending=[False, True], na_position="last")
        .reset_index(drop=True)
    )
    order["sorting_rank"] = np.arange(1, len(order) + 1)
    order["sorting_rule"] = FIG12_SORTING_RULE

    data = (
        core.merge(city_scale, on="city_id", how="left")
        .merge(city_metrics, on=["city_id", "city_name_en", "region", "quality_tier"], how="left")
        .merge(order[["city_id", "sorting_rank", "sorting_rule"]], on="city_id", how="left")
    )
    data["scale_marker"] = data["scale"].map({"hex_1km": "circle", "hex_2km": "triangle"})
    data["scale_marker_shape"] = data["scale"].map({"hex_1km": "o", "hex_2km": "^"})
    data["scale_share_difference_from_hex_1km"] = np.where(
        data["scale"].astype(str).eq("hex_2km"),
        data["share_delta"],
        0.0,
    )
    data["dominant_morphotype_label_en"] = data["dominant_morphotype"].map(MORPHOTYPE_LABELS_EN).fillna(data["dominant_morphotype"])
    data["dominant_morphotype_color"] = data["dominant_morphotype"].map(MORPHOTYPE_COLORS)
    data["figure_id"] = "Fig7"
    data["figure_slug"] = FIG8_CORE_CITY_SLUG
    data["figure_core_conclusion"] = "48 high-confidence core cities differ jointly in morphotype dominance, scale stability, accessibility, detour, inequality and data quality."
    data["archetype"] = "quantitative grid / ordered city plate"
    data["dominant_share_priority"] = FIG12_DOMINANT_SHARE_PRIORITY
    data["share_delta_definition"] = "dominant_share_hex_2km - dominant_share_hex_1km; repeated on both scale rows for each city."
    data["fig8_strip_cmap"] = FIG12_HEATMAP_CMAP
    data["fig8_strip_cmap_colors"] = " -> ".join(FIG12_HEATMAP_CMAP_COLORS)
    data["fig8_right_strip_metrics"] = "Entropy, Access, Circuity, Inequality, Quality"
    data["type_change_visual_encoding"] = "Scale-change cities use a muted red connector and red city tick label; all point colors still encode dominant morphotype."
    data["source_fields_fallback_notes"] = (
        "Dominant share uses "
        + data["dominant_share_source"].astype(str)
        + "; entropy uses "
        + data["entropy_source"].astype(str)
        + "; strips use Step11 city accessibility/inequality, Step10 city OD circuity, and Step07 quality_score."
    )
    data = add_palette_metadata(
        data,
        palette=MORPHOTYPE_PALETTE_NAME,
        heatmap_cmap=FIG12_HEATMAP_CMAP,
        color_variable="dominant_morphotype",
        color_variable_label="Dominant morphotype",
        palette_note="Dumbbell point colors use the same fixed MT01-MT10 palette as the other morphotype-coded Step12 figures; right strips use one neutral sequential 0-1 normalized colormap.",
    )
    sort_cols = ["sorting_rank", "scale"]
    data["scale_order"] = data["scale"].map({"hex_1km": 1, "hex_2km": 2})
    data = data.sort_values(sort_cols + ["city_id"]).reset_index(drop=True)
    n_cities = int(order["city_id"].nunique())
    data["fig8_y_pos"] = numeric(data["sorting_rank"]) - 1
    data["fig8_dumbbell_y_pos"] = data["fig8_y_pos"]
    data["fig8_strip_row_center_y"] = data["fig8_y_pos"]
    data["fig8_city_block"] = "single"
    data["fig8_city_block_label"] = "Core cities ranked 1-48"
    data["fig8_block_row_pos"] = data["fig8_y_pos"]
    data["fig8_layout_revision"] = (
        "Single continuous 48-city ordered plate with one shared dominant-share x-axis and one aligned metric-strip panel; "
        "the previous two-block stacked split is removed so Fig8 exports as one complete integrated figure."
    )
    data["fig8_shared_ylim_start"] = n_cities - 0.5
    data["fig8_shared_ylim_end"] = -0.5
    data["fig8_strip_extent_y_start"] = n_cities - 0.5
    data["fig8_strip_extent_y_end"] = -0.5
    data["fig8_alignment_note"] = (
        "Within the continuous 48-city plate, left dumbbell and right metric strips use the same row-center y positions; "
        "imshow uses explicit row-edge extent and a separate colorbar axis so strip rows are not vertically compressed."
    )
    return data, "; ".join(sorted(data["dominant_share_fallback_note"].dropna().astype(str).unique()))


def plot_fig12(
    paths: StepPaths,
    city_profiles: pd.DataFrame,
    inequality: pd.DataFrame,
    od_metrics: pd.DataFrame,
    city_quality: pd.DataFrame,
) -> FigureProduct:
    data, fallback_note = build_fig12_data(city_profiles, inequality, od_metrics, city_quality)
    data_path = make_output_path(paths, source_data_name("Fig7", FIG8_CORE_CITY_SLUG))
    write_csv(data, data_path)

    city_rows = (
        data.sort_values(["sorting_rank", "scale_order"])
        .drop_duplicates("city_id")
        .sort_values("sorting_rank")
        .reset_index(drop=True)
    )
    strip_cols = ["entropy_norm", "access_norm", "circuity_norm", "inequality_norm", "quality_norm"]
    strip_labels = ["Entropy", "Access", "Circuity", "Gini", "Quality"]
    fig = plt.figure(figsize=cm_to_in(18.3, FIG8_FIGURE_HEIGHT_CM))
    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[4.85, 1.55],
        left=FIG8_LEFT_MARGIN,
        right=0.970,
        top=0.905,
        bottom=FIG8_MAIN_BOTTOM,
        wspace=0.055,
    )
    cmap = fig12_heatmap_cmap()
    city_rows = city_rows.sort_values("sorting_rank").reset_index(drop=True)
    city_rows["plot_y"] = np.arange(len(city_rows), dtype=float)
    y = city_rows["plot_y"].to_numpy(dtype=float)
    y_start = float(len(city_rows) - 0.5)
    y_end = -0.5
    ax = fig.add_subplot(gs[0, 0])
    ax_strip = fig.add_subplot(gs[0, 1], sharey=ax)

    for _, row in city_rows.iterrows():
        yi = float(row["plot_y"])
        changed = bool(row.get("type_changed_between_scales", False))
        line_color = "#B91C1C" if changed else "#CBD5E1"
        line_alpha = 0.64 if changed else 0.82
        ax.plot(
            [row.get("dominant_share_hex_1km"), row.get("dominant_share_hex_2km")],
            [yi, yi],
            color=line_color,
            lw=0.74 if changed else 0.54,
            alpha=line_alpha,
            zorder=1,
            solid_capstyle="round",
        )
        ax.scatter(
            row.get("dominant_share_hex_1km"),
            yi,
            s=17,
            marker="o",
            color=MORPHOTYPE_COLORS.get(str(row.get("dominant_morphotype_hex_1km")), "#94A3B8"),
            edgecolor="white",
            linewidth=0.32,
            zorder=3,
        )
        ax.scatter(
            row.get("dominant_share_hex_2km"),
            yi,
            s=20,
            marker="^",
            color=MORPHOTYPE_COLORS.get(str(row.get("dominant_morphotype_hex_2km")), "#94A3B8"),
            edgecolor="#334155" if changed else "white",
            linewidth=0.32,
            zorder=4,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(y_start, y_end)
    ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.set_xlabel("Dominant morphotype share")
    ax.set_yticks(y)
    ax.set_yticklabels(city_rows["city_name_en"].astype(str), fontsize=FIG8_CITY_LABEL_FONTSIZE)
    ax.tick_params(axis="y", length=0, pad=2.1)
    ax.tick_params(axis="x", labelsize=FIG8_AXIS_TICK_FONTSIZE)
    ax.grid(axis="x", color="#E2E8F0", lw=0.42)
    ax.set_title(str(city_rows["fig8_city_block_label"].iloc[0]), loc="left", fontsize=FIG8_TITLE_FONTSIZE, fontweight="bold", pad=5)
    for tick, changed in zip(ax.get_yticklabels(), city_rows["type_changed_between_scales"].fillna(False).astype(bool)):
        if changed:
            tick.set_color("#B91C1C")
            tick.set_fontweight("bold")

    matrix = city_rows[strip_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    strip_extent = [-0.5, len(strip_labels) - 0.5, y_start, y_end]
    im = ax_strip.imshow(
        np.ma.masked_invalid(matrix),
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
        origin="upper",
        extent=strip_extent,
    )
    ax_strip.set_xticks(np.arange(len(strip_labels)))
    ax_strip.set_xticklabels(strip_labels, rotation=45, ha="left", rotation_mode="anchor", fontsize=FIG8_STRIP_LABEL_FONTSIZE)
    ax_strip.xaxis.tick_top()
    ax_strip.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False, pad=1.2)
    ax_strip.set_yticks(y)
    ax_strip.tick_params(axis="y", left=False, labelleft=False)
    ax_strip.set_ylim(ax.get_ylim())
    for x in np.arange(-0.5, len(strip_labels) + 0.5, 1):
        ax_strip.axvline(x, color="white", lw=0.75, zorder=2)
    ax_strip.set_xlim(-0.5, len(strip_labels) - 0.5)
    for spine in ax_strip.spines.values():
        spine.set_visible(False)

    if im is None:
        raise ValueError("Fig8 core-city layout has no rows to render.")
    strip_pos = ax_strip.get_position()
    cax = fig.add_axes([strip_pos.x0, FIG8_COLORBAR_Y, strip_pos.width, 0.007])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Normalized within 48 cities", fontsize=FIG8_COLORBAR_LABEL_FONTSIZE, labelpad=1)
    cbar.ax.tick_params(labelsize=FIG8_COLORBAR_TICK_FONTSIZE, length=2)

    shape_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#64748B", markeredgecolor="white", markersize=4.8, label="hex 1 km"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#64748B", markeredgecolor="white", markersize=5.2, label="hex 2 km"),
        Line2D([0, 1], [0, 0], color="#B91C1C", lw=0.8, label="dominant type changes"),
    ]
    morph_handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=TYPE_COLORS[mt], markeredgecolor="none", markersize=4.5, label=mt)
        for mt in MORPHOTYPES
    ]
    fig.legend(
        handles=shape_handles,
        loc="lower left",
        bbox_to_anchor=(FIG8_LEFT_MARGIN, FIG8_SHAPE_LEGEND_Y),
        ncol=3,
        fontsize=FIG8_SHAPE_LEGEND_FONTSIZE,
        handlelength=1.4,
        columnspacing=1.0,
    )
    fig.legend(
        handles=morph_handles,
        loc="lower left",
        bbox_to_anchor=(FIG8_LEFT_MARGIN, FIG8_MORPHOTYPE_LEGEND_Y),
        ncol=10,
        fontsize=FIG8_MORPHOTYPE_LEGEND_FONTSIZE,
        handlelength=0.8,
        handletextpad=0.25,
        columnspacing=0.55,
        title="Dominant morphotype",
        title_fontsize=FIG8_MORPHOTYPE_LEGEND_TITLE_FONTSIZE,
    )

    png, svg, pdf = save_figure(fig, paths.figure_dir / f"Fig7_{FIG8_CORE_CITY_SLUG}", pad_inches=0.12)
    return FigureProduct(
        figure_id="Fig7",
        title="Core-city morphotype stability and performance overview",
        kind="main",
        png_path=png,
        svg_path=svg,
        pdf_path=pdf,
        data_path=data_path,
        source_inputs=[
            "data/09_cluster_morphotypes/city_morphotype_profiles.parquet",
            "data/11_statistical_models_and_robustness_checks/figure_data_inequality.csv",
            "data/10_compute_accessibility_and_detour_validation_metrics/OD_detour_metrics.parquet",
            "data/07_generate_quality_scores_and_type_confidence/quality_scores.parquet",
        ],
        transformation=(
            "Filtered Step09 city profiles to core drive hex_1km/hex_2km rows, sorted cities by hex_1km dominant share, "
            "computed scale-change and share-delta fields, joined city-level access/Gini, OD circuity and quality metrics for aligned normalized strips, "
            "and rendered the 48 cities as one continuous ordered city plate with aligned metric strips."
        ),
        fallback_note=fallback_note,
        palette=MORPHOTYPE_PALETTE_NAME,
        heatmap_cmap=FIG12_HEATMAP_CMAP,
        color_variable="dominant_morphotype",
        palette_note="Dumbbell point colors use MT01-MT10 colors; metric strips use one neutral sequential 0-1 colormap.",
        visual_text_note="All 48 city labels are shown in one continuous ranked panel; dominant-type changes are indicated by muted red connectors and tick labels.",
    )


def products_to_manifest(products: list[FigureProduct], root: Path) -> pd.DataFrame:
    rows = []
    for product in products:
        rows.append(
            {
                "record_role": "canonical",
                "figure_id": product.figure_id,
                "kind": product.kind,
                "title": product.title,
                "png_path": str(product.png_path),
                "svg_path": str(product.svg_path),
                "pdf_path": str(product.pdf_path),
                "data_path": str(product.data_path),
                "png_exists": product.png_path.exists(),
                "svg_exists": product.svg_path.exists(),
                "pdf_exists": product.pdf_path.exists(),
                "data_exists": product.data_path.exists(),
                "source_inputs": "; ".join(product.source_inputs),
                "transformation": product.transformation,
                "fallback_note": product.fallback_note,
                "palette": product.palette,
                "cmap": product.cmap,
                "heatmap_cmap": product.heatmap_cmap,
                "color_variable": product.color_variable,
                "palette_note": product.palette_note,
                "visual_text_note": product.visual_text_note,
                "canonical_png_path": "",
                "canonical_svg_path": "",
                "canonical_pdf_path": "",
                "canonical_data_path": "",
                "alias_note": "",
                "relative_png_path": str(product.png_path.relative_to(root)) if product.png_path.is_relative_to(root) else str(product.png_path),
            }
        )
        for alias in product.aliases:
            alias_png = alias["png_path"]
            alias_svg = alias["svg_path"]
            alias_pdf = alias["pdf_path"]
            alias_data = alias["data_path"]
            rows.append(
                {
                    "record_role": alias.get("record_role", "compatibility_alias"),
                    "figure_id": product.figure_id,
                    "kind": "compatibility_alias",
                    "title": f"{product.title} (compatibility alias)",
                    "png_path": str(alias_png),
                    "svg_path": str(alias_svg),
                    "pdf_path": str(alias_pdf),
                    "data_path": str(alias_data),
                    "png_exists": alias_png.exists(),
                    "svg_exists": alias_svg.exists(),
                    "pdf_exists": alias_pdf.exists(),
                    "data_exists": alias_data.exists(),
                    "source_inputs": str(product.data_path),
                    "transformation": alias.get("hash_relation", "Compatibility alias copied from canonical output."),
                    "fallback_note": product.fallback_note,
                    "palette": product.palette,
                    "cmap": product.cmap,
                    "heatmap_cmap": product.heatmap_cmap,
                    "color_variable": product.color_variable,
                    "palette_note": product.palette_note,
                    "visual_text_note": product.visual_text_note,
                    "canonical_png_path": str(product.png_path),
                    "canonical_svg_path": str(product.svg_path),
                    "canonical_pdf_path": str(product.pdf_path),
                    "canonical_data_path": str(product.data_path),
                    "alias_note": alias.get("alias_note", ""),
                    "relative_png_path": str(alias_png.relative_to(root)) if alias_png.is_relative_to(root) else str(alias_png),
                }
            )
    return pd.DataFrame(rows)


def products_to_lineage(products: list[FigureProduct]) -> pd.DataFrame:
    rows = []
    for product in products:
        for source in product.source_inputs:
            rows.append(
                {
                    "record_role": "canonical",
                    "figure_id": product.figure_id,
                    "figure_title": product.title,
                    "figure_data_path": str(product.data_path),
                    "source_input": source,
                    "transformation": product.transformation,
                    "fallback_note": product.fallback_note,
                    "palette": product.palette,
                    "cmap": product.cmap,
                    "heatmap_cmap": product.heatmap_cmap,
                    "color_variable": product.color_variable,
                    "palette_note": product.palette_note,
                    "visual_text_note": product.visual_text_note,
                    "canonical_figure_data_path": "",
                    "alias_note": "",
                }
            )
        for alias in product.aliases:
            rows.append(
                {
                    "record_role": alias.get("record_role", "compatibility_alias"),
                    "figure_id": product.figure_id,
                    "figure_title": f"{product.title} (compatibility alias)",
                    "figure_data_path": str(alias["data_path"]),
                    "source_input": str(product.data_path),
                    "transformation": (
                        "Compatibility alias copied byte-for-byte from canonical Fig6 source data; "
                        "upstream scientific transformation is unchanged from the canonical Fig6 row."
                    ),
                    "fallback_note": product.fallback_note,
                    "palette": product.palette,
                    "cmap": product.cmap,
                    "heatmap_cmap": product.heatmap_cmap,
                    "color_variable": product.color_variable,
                    "palette_note": product.palette_note,
                    "visual_text_note": product.visual_text_note,
                    "canonical_figure_data_path": str(product.data_path),
                    "alias_note": alias.get("alias_note", ""),
                }
            )
    return pd.DataFrame(rows)


def build_qc(
    paths: StepPaths,
    input_report: dict[str, Any],
    products: list[FigureProduct],
    manifest: pd.DataFrame,
    lineage: pd.DataFrame,
    fig2_data: pd.DataFrame,
    fig3_data: pd.DataFrame,
    fig7_data: pd.DataFrame,
    fig8_data: pd.DataFrame,
    fig9_data: pd.DataFrame,
    step11_qc: pd.DataFrame,
    step13_changed: list[str],
    fig2_alignment: dict[str, Any],
    fig3_layout: dict[str, Any],
    fig1_fig2_preservation: dict[str, Any],
    fig7_vector_preservation: dict[str, Any],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    def add(check: str, status: str, value: Any, expected: Any, detail: str = "") -> None:
        records.append({"check": check, "status": status, "value": value, "expected": expected, "detail": detail})

    def product_by_id(figure_id: str) -> FigureProduct | None:
        return next((p for p in products if p.figure_id == figure_id), None)

    def read_product_data(figure_id: str) -> pd.DataFrame:
        product = product_by_id(figure_id)
        if product is None or not product.data_path.exists():
            return pd.DataFrame()
        return pd.read_csv(product.data_path)

    add(
        "required_inputs_exist",
        "pass" if input_report.get("status") == "pass" else "fail",
        input_report.get("required_failure_count"),
        0,
        "See step12_input_acceptance_report.json for file-level metadata.",
    )
    add(
        "Fig1_run_start_generation_or_protection",
        str(fig1_fig2_preservation.get("status", "fail")),
        json.dumps(
            {
                "mode": fig1_fig2_preservation.get("mode"),
                "protected_file_count": fig1_fig2_preservation.get("protected_file_count"),
                "protected_copy_count": fig1_fig2_preservation.get("protected_copy_count"),
                "run_start_missing": fig1_fig2_preservation.get("run_start_missing", []),
                "preserved_from_run_start": fig1_fig2_preservation.get("preserved_from_run_start"),
                "changed_before_restore": fig1_fig2_preservation.get("changed_before_restore", []),
                "restored": fig1_fig2_preservation.get("restored", []),
                "final_mismatches": fig1_fig2_preservation.get("final_mismatches", []),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "Fig1 PNG/SVG/PDF/source CSV run-start fingerprints are restored when present; first-run missing new Fig1 files must be generated non-empty.",
        fig1_fig2_preservation.get("note", ""),
    )
    reference_mismatches = fig1_fig2_preservation.get("accepted_reference_mismatches", [])
    reference_missing = fig1_fig2_preservation.get("accepted_reference_missing", [])
    add(
        "Fig1_Step13_material_reference",
        "pass" if not reference_mismatches and not reference_missing else "warn",
        json.dumps(
            {
                "accepted_reference_available": fig1_fig2_preservation.get("accepted_reference_available", {}),
                "accepted_reference_matches": fig1_fig2_preservation.get("accepted_reference_matches", {}),
                "mismatches": reference_mismatches,
                "missing": reference_missing,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "Latest Step13 material-list reference hashes match current protected run-start files, or caveat is recorded.",
        fig1_fig2_preservation.get("preservation_caveat", ""),
    )
    add(
        "Fig8_vector_run_start_protection",
        str(fig7_vector_preservation.get("status", "fail")),
        json.dumps(
            {
                "mode": fig7_vector_preservation.get("mode"),
                "protected_file_count": fig7_vector_preservation.get("protected_file_count"),
                "protected_copy_count": fig7_vector_preservation.get("protected_copy_count"),
                "run_start_missing": fig7_vector_preservation.get("run_start_missing", []),
                "preserved_from_run_start": fig7_vector_preservation.get("preserved_from_run_start"),
                "changed_before_restore": fig7_vector_preservation.get("changed_before_restore", []),
                "restored": fig7_vector_preservation.get("restored", []),
                "final_mismatches": fig7_vector_preservation.get("final_mismatches", []),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "Fig8 SVG/PDF run-start SHA-256/size/mtime fingerprints are restored after full Step12 redraw; final_mismatches must be [].",
        fig7_vector_preservation.get("note", ""),
    )
    fig7_reference_mismatches = fig7_vector_preservation.get("accepted_reference_mismatches", [])
    fig7_reference_missing = fig7_vector_preservation.get("accepted_reference_missing", [])
    fig7_reference_warn = bool(fig7_reference_mismatches or fig7_reference_missing)
    add(
        "Fig8_Step13_material_reference_vector_status",
        "warn" if fig7_reference_warn else "pass",
        json.dumps(
            {
                "accepted_reference_available": fig7_vector_preservation.get("accepted_reference_available", {}),
                "accepted_reference_matches": fig7_vector_preservation.get("accepted_reference_matches", {}),
                "mismatches": fig7_reference_mismatches,
                "missing": fig7_reference_missing,
                "accepted_vector_mismatches": fig7_vector_preservation.get("accepted_vector_mismatches", []),
                "accepted_support_mismatches": fig7_vector_preservation.get("accepted_support_mismatches", []),
                "drift_scope": fig7_vector_preservation.get("drift_scope"),
                "non_target_vector_metadata_drift_accepted": fig7_vector_preservation.get(
                    "non_target_vector_metadata_drift_accepted"
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "Fig8 PNG/source CSV should match Step13 material-list hashes; any remaining SVG/PDF mismatch must be recorded as non-target vector metadata/id drift.",
        fig7_vector_preservation.get("preservation_caveat", ""),
    )
    main_products = [p for p in products if p.kind == "main"]
    main_ids = {p.figure_id for p in main_products}
    expected_main_ids = {f"Fig{i}" for i in range(1, 12)}
    add("Fig1_to_Fig11_products_exist", "pass" if main_ids == expected_main_ids else "fail", sorted(main_ids), "Fig1-Fig11")
    supplemental_ids = {p.figure_id for p in products if p.kind == "supplemental"}
    add("no_supplemental_products", "pass" if not supplemental_ids else "fail", sorted(supplemental_ids), "No supplemental canonical products")
    missing_main_files = []
    for p in main_products:
        for suffix, path in [("png", p.png_path), ("svg", p.svg_path), ("pdf", p.pdf_path)]:
            if not path.exists():
                missing_main_files.append(f"{p.figure_id}.{suffix}")
    add("Fig1_to_Fig11_files_exist", "pass" if not missing_main_files else "fail", len(missing_main_files), 0, ";".join(missing_main_files))
    empty_files = []
    for p in products:
        for suffix, path in [("png", p.png_path), ("svg", p.svg_path), ("pdf", p.pdf_path), ("data", p.data_path)]:
            if (not path.exists()) or path.stat().st_size <= 0:
                empty_files.append(f"{p.figure_id}.{suffix}")
    add("figure_outputs_nonempty", "pass" if not empty_files else "fail", len(empty_files), 0, ";".join(empty_files))
    missing_png_svg = [
        p.figure_id
        for p in products
        if not (p.png_path.exists() and p.svg_path.exists())
    ]
    add("each_figure_has_png_and_svg", "pass" if not missing_png_svg else "fail", len(products) - len(missing_png_svg), len(products), ";".join(missing_png_svg))
    missing_data = [p.figure_id for p in products if not p.data_path.exists()]
    add("each_figure_has_source_data", "pass" if not missing_data else "fail", len(products) - len(missing_data), len(products), ";".join(missing_data))

    old_framework_paths = [
        *(paths.figure_dir / f"Fig1_quality_aware_multiscale_typology_framework.{ext}" for ext in ["png", "svg", "pdf", "tiff"]),
        make_output_path(paths, "figure_data_Fig1_quality_aware_multiscale_typology_framework.csv"),
    ]
    old_framework_remaining = [str(path.relative_to(paths.output_dir)) if path.is_relative_to(paths.output_dir) else str(path) for path in old_framework_paths if path.exists()]
    add(
        "old_Fig1_framework_outputs_absent",
        "pass" if not old_framework_remaining else "fail",
        old_framework_remaining,
        [],
        "Old framework stem Fig1_quality_aware_multiscale_typology_framework is no longer a main-flow Step12 output.",
    )

    fig12_data = read_product_data("Fig7")
    fig12_required = {
        "city_id",
        "city_name_en",
        "region",
        "quality_tier",
        "scale",
        "dominant_morphotype",
        "dominant_share",
        "morphotype_entropy",
        "type_changed_between_scales",
        "share_delta",
        "access_metric",
        "circuity_metric",
        "inequality_metric",
        "quality_score",
        "entropy_norm",
        "access_norm",
        "circuity_norm",
        "inequality_norm",
        "quality_norm",
        "sorting_rank",
        "dominant_share_source",
        "source_fields_fallback_notes",
        "fig8_y_pos",
        "fig8_dumbbell_y_pos",
        "fig8_strip_row_center_y",
        "fig8_shared_ylim_start",
        "fig8_shared_ylim_end",
        "fig8_strip_extent_y_start",
        "fig8_strip_extent_y_end",
    }
    fig12_missing = sorted(fig12_required - set(fig12_data.columns))
    fig12_scale_counts = fig12_data.get("scale", pd.Series(dtype=str)).astype(str).value_counts().to_dict()
    fig12_city_count = int(fig12_data.get("city_id", pd.Series(dtype=str)).nunique()) if "city_id" in fig12_data.columns else 0
    if {"city_id", "scale"}.issubset(fig12_data.columns):
        fig12_city_scale_sets = fig12_data.groupby("city_id")["scale"].agg(lambda s: sorted(set(s.astype(str)))).to_dict()
        fig12_missing_city_scales = {
            city_id: scales
            for city_id, scales in fig12_city_scale_sets.items()
            if scales != ["hex_1km", "hex_2km"]
        }
    else:
        fig12_missing_city_scales = {}
    fig12_structure_ok = (
        not fig12_missing
        and len(fig12_data) == 96
        and fig12_city_count == 48
        and fig12_scale_counts.get("hex_1km", 0) == 48
        and fig12_scale_counts.get("hex_2km", 0) == 48
        and not fig12_missing_city_scales
    )
    add(
        "Fig7_core_city_source_structure",
        "pass" if fig12_structure_ok else "fail",
        json.dumps(
            {
                "rows": len(fig12_data),
                "city_count": fig12_city_count,
                "scale_counts": fig12_scale_counts,
                "missing_city_scales": fig12_missing_city_scales,
                "missing_fields": fig12_missing,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "96 rows: 48 core cities x 2 scales, with hex_1km and hex_2km present for every city",
    )
    fig12_metric_cols = ["entropy_norm", "access_norm", "circuity_norm", "inequality_norm", "quality_norm"]
    fig12_metric_nonnull = {
        col: int(numeric(fig12_data.get(col, pd.Series(dtype=float))).notna().sum()) for col in fig12_metric_cols
    }
    fig12_metric_city_nonnull = {}
    if "city_id" in fig12_data.columns:
        for col in ["entropy_metric", "access_metric", "circuity_metric", "inequality_metric", "quality_score"]:
            if col in fig12_data.columns:
                fig12_metric_city_nonnull[col] = int(fig12_data.drop_duplicates("city_id")[col].notna().sum())
    fig12_metrics_ok = bool(fig12_metric_nonnull and all(count >= 80 for count in fig12_metric_nonnull.values()))
    add(
        "Fig7_heatmap_metrics_nonnull",
        "pass" if fig12_metrics_ok else "fail",
        json.dumps({"normalized_nonnull_rows": fig12_metric_nonnull, "raw_nonnull_cities": fig12_metric_city_nonnull}, ensure_ascii=False, sort_keys=True),
        "Each Fig7 normalized strip metric has non-null values for a reasonable city count; fail if all missing",
    )
    fig12_rank_values = sorted(numeric(fig12_data.get("sorting_rank", pd.Series(dtype=float))).dropna().unique().tolist())
    fig12_sort_ok = fig12_rank_values == list(range(1, 49)) and (
        fig12_data.get("sorting_rule", pd.Series(dtype=str)).astype(str).eq(FIG12_SORTING_RULE).all()
        if "sorting_rule" in fig12_data.columns and not fig12_data.empty
        else False
    )
    add(
        "Fig7_sorting_rule_recorded",
        "pass" if fig12_sort_ok else "fail",
        json.dumps({"rank_values": fig12_rank_values[:5] + fig12_rank_values[-5:], "rank_count": len(fig12_rank_values), "sorting_rule": FIG12_SORTING_RULE}, ensure_ascii=False, sort_keys=True),
        "Ranks 1-48 and sorting rule are recorded: dominant hex_1km share descending, city name tie-breaker",
    )
    fig12_alignment_required = {
        "sorting_rank",
        "fig8_y_pos",
        "fig8_dumbbell_y_pos",
        "fig8_strip_row_center_y",
        "fig8_shared_ylim_start",
        "fig8_shared_ylim_end",
        "fig8_strip_extent_y_start",
        "fig8_strip_extent_y_end",
    }
    fig12_alignment_missing = sorted(fig12_alignment_required - set(fig12_data.columns))
    if not fig12_alignment_missing and not fig12_data.empty:
        expected_y = numeric(fig12_data["sorting_rank"]) - 1
        y_pos = numeric(fig12_data["fig8_y_pos"])
        dumbbell_y = numeric(fig12_data["fig8_dumbbell_y_pos"])
        strip_y = numeric(fig12_data["fig8_strip_row_center_y"])
        y_pos_max_abs_diff = float((y_pos - expected_y).abs().max())
        dumbbell_strip_max_abs_diff = float((dumbbell_y - strip_y).abs().max())
        city_y_counts = fig12_data.groupby("city_id")["fig8_y_pos"].nunique(dropna=False).to_dict() if "city_id" in fig12_data.columns else {}
        multi_y_cities = {city_id: int(count) for city_id, count in city_y_counts.items() if int(count) != 1}
        expected_ylim_start = fig12_city_count - 0.5
        expected_ylim_end = -0.5
        ylim_start_ok = bool(numeric(fig12_data["fig8_shared_ylim_start"]).eq(expected_ylim_start).all())
        ylim_end_ok = bool(numeric(fig12_data["fig8_shared_ylim_end"]).eq(expected_ylim_end).all())
        extent_start_ok = bool(numeric(fig12_data["fig8_strip_extent_y_start"]).eq(expected_ylim_start).all())
        extent_end_ok = bool(numeric(fig12_data["fig8_strip_extent_y_end"]).eq(expected_ylim_end).all())
        fig12_alignment_ok = (
            y_pos_max_abs_diff <= 1e-9
            and dumbbell_strip_max_abs_diff <= 1e-9
            and not multi_y_cities
            and ylim_start_ok
            and ylim_end_ok
            and extent_start_ok
            and extent_end_ok
        )
        fig12_alignment_detail = {
            "alignment_columns_missing": fig12_alignment_missing,
            "city_count": fig12_city_count,
            "dumbbell_strip_max_abs_y_diff": dumbbell_strip_max_abs_diff,
            "multi_y_cities": multi_y_cities,
            "shared_ylim": [expected_ylim_start, expected_ylim_end],
            "strip_extent_y": [expected_ylim_start, expected_ylim_end],
            "y_pos_max_abs_diff_from_sorting_rank_minus_1": y_pos_max_abs_diff,
        }
    else:
        fig12_alignment_ok = False
        fig12_alignment_detail = {
            "alignment_columns_missing": fig12_alignment_missing,
            "city_count": fig12_city_count,
        }
    add(
        "Fig7_strip_rows_align_to_city_y_axis",
        "pass" if fig12_alignment_ok else "fail",
        json.dumps(fig12_alignment_detail, ensure_ascii=False, sort_keys=True),
        "Right metric strip row centers equal the left dumbbell/city-label y positions from sorting_rank - 1; shared y-limits and strip extent are recorded.",
    )
    if not fig12_data.empty and {"dominant_morphotype", "dominant_morphotype_color"}.issubset(fig12_data.columns):
        fig12_expected_colors = fig12_data["dominant_morphotype"].map(MORPHOTYPE_COLORS)
        fig12_color_mismatch = int(fig12_data["dominant_morphotype_color"].fillna("").ne(fig12_expected_colors.fillna("")).sum())
    else:
        fig12_color_mismatch = len(fig12_data) if not fig12_data.empty else 1
    add(
        "Fig7_dominant_morphotype_colors_consistent",
        "pass" if fig12_color_mismatch == 0 else "fail",
        fig12_color_mismatch,
        0,
        "Fig7 dumbbell point colors use the fixed MT01-MT10 palette through dominant_morphotype_color.",
    )

    fig5_data = read_product_data("Fig4")
    fig5_score_fields = set(RADAR_SCORE_METRICS)
    fig5_heatmap_ok = (
        not fig5_data.empty
        and fig5_score_fields.issubset(fig5_data.columns)
        and product_by_id("Fig4") is not None
        and "morphotype_atlas_signatures" in str(product_by_id("Fig4").data_path)
    )
    add(
        "Fig4_heatmap_present",
        "pass" if fig5_heatmap_ok else "fail",
        json.dumps(
            {
                "rows": len(fig5_data),
                "score_fields_present": sorted(fig5_score_fields & set(fig5_data.columns)),
                "data_path": str(product_by_id("Fig4").data_path) if product_by_id("Fig4") else "",
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "Fig4 source data contains all morphotype score heatmap fields and uses morphotype_atlas_signatures slug",
    )

    fig6_product = product_by_id("Fig6")
    fig6_data = read_product_data("Fig6")
    fig6_norm = fig6_panel_a_norm_metadata(fig6_data)
    fig6_required = {
        "panel",
        "fig7_panel_label",
        "comparison_type",
        "from_network_type",
        "from_scale",
        "to_network_type",
        "to_scale",
        "category",
        "category_label",
        "morphotype",
        "mean_unit_share_percent",
        "transition_delta_pp",
        FIG7_COLOR_VARIABLE,
        "population_weighted_access_share_percent",
        "category_mean_access_share",
        "value_for_plot",
        "value_units",
        "cell_definition",
        "fig7_panel_heatmap_cmap",
        "fig7_panel_norm",
        "fig7_panel_colorbar_label",
        "fig7_panel_color_variable",
        "fig7_panel_c_source_panel",
        "fig7_panel_c_migration_note",
        "fig7_morphotype_axis_order",
        "fig7_morphotype_axis_order_note",
        "fig7_panel_c_morphotype_axis_order",
        "fig7_panel_c_morphotype_axis_order_note",
        "fig7_panel_c_morphotype_display_order",
        "fig7_figure_level_title_current",
        "fig7_bottom_note_current",
        "fig7_removed_figure_level_title_text",
        "fig7_removed_bottom_note_text",
        "fig7_retained_visual_text_elements",
        "fig7_title_footnote_removal_note",
    }
    fig6_missing = sorted(fig6_required - set(fig6_data.columns))
    fig6_panel_counts = fig6_data.get("panel", pd.Series(dtype=str)).astype(str).value_counts().to_dict()
    manifest_text = " ".join(manifest.astype(str).fillna("").agg(" ".join, axis=1).tolist()) if not manifest.empty else ""
    lineage_text = " ".join(lineage.astype(str).fillna("").agg(" ".join, axis=1).tolist()) if not lineage.empty else ""
    old_manifest_terms = [
        "Fig1_quality_aware_multiscale_typology_framework",
        f"Fig2_{FIG2_CANONICAL_SLUG}",
        "Fig3_osm_quality_confidence",
        f"Fig4_{FIG4_CANONICAL_SLUG}",
        "Fig5_morphotype_atlas_signatures",
        f"Fig6_{FIG6_RADAR_SLUG}",
        f"Fig7_{FIG7_CANONICAL_SLUG}",
        f"Fig8_{FIG8_CORE_CITY_SLUG}",
        f"Fig9_{FIG9_ACCESS_DETOUR_SLUG}",
        f"Fig10_{FIG10_OBSERVED_GROUP_MEANS_SLUG}",
        f"Fig11_{FIG11_CITY_INEQUALITY_SLUG}",
        f"Fig12_{FIG12_MODEL_COEFFICIENT_SLUG}",
        "Fig2_global_sample_city_overview",
        "Fig4_multiscale_metric_curves",
        "Fig6_scale_transition_matrix",
        "Fig6_scale_network_composition_matrix",
        "Fig7_accessibility_detour_performance_by_morphotype",
        "Fig7_accessibility_by_morphotype_category",
        "Fig8_city_level_accessibility_inequality",
        "Fig8_circuity_efficiency_by_morphotype",
        "Fig9_model_coefficient_forest",
        "Fig10_morphotype_radar_signatures",
        "Fig11_observed_weighted_group_means",
        "Fig12_core_city_morphotype_stability_and_performance",
    ]
    old_manifest_hits = [term for term in old_manifest_terms if term in manifest_text or term in lineage_text]
    legacy_supplemental_terms = [f"{'S'}Fig1", f"{'S'}Fig2"]
    legacy_supplemental_hits = [term for term in legacy_supplemental_terms if term in manifest_text or term in lineage_text]
    fig6_true_transition_ok = (
        fig6_product is not None
        and not fig6_product.aliases
        and not fig6_missing
        and fig6_panel_counts.get(FIG6_PANEL_COMPOSITION, 0) == 40
        and fig6_panel_counts.get(FIG6_PANEL_TRANSITION, 0) == 40
        and fig6_panel_counts.get(FIG6_PANEL_PERFORMANCE, 0) == 50
        and not manifest.get("record_role", pd.Series(dtype=str)).astype(str).eq("compatibility_alias").any()
        and "scale_network_transition_matrix" in str(fig6_product.data_path)
    )
    add(
        "Fig6_A_B_C_panels_true_transition_not_alias",
        "pass" if fig6_true_transition_ok else "fail",
        json.dumps(
            {
                "panel_counts": fig6_panel_counts,
                "missing_fields": fig6_missing,
                "product_alias_count": len(fig6_product.aliases) if fig6_product else None,
                "manifest_compatibility_alias_rows": int(manifest.get("record_role", pd.Series(dtype=str)).astype(str).eq("compatibility_alias").sum()) if not manifest.empty else 0,
                "data_path": str(fig6_product.data_path) if fig6_product else "",
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "Fig6 has 40 composition cells, 40 true transition cells, and 50 migrated performance-deviation cells; no compatibility alias; scale_network_transition_matrix slug retained.",
    )
    add(
        "old_pre_reorder_figure_names_absent_from_manifest",
        "pass" if not old_manifest_hits else "fail",
        old_manifest_hits,
        "No pre-reorder compatibility or standalone figure filenames in manifest/lineage",
    )
    fig6_norm_required = {
        "fig7_panel_heatmap_cmap",
        "fig7_panel_norm",
        "fig7_panel_norm_type",
        "fig7_panel_colorbar_label",
        "fig7_panel_color_variable",
        "fig7_panel_a_heatmap_cmap",
        "fig7_panel_a_norm",
        "fig7_panel_a_norm_type",
        "fig7_panel_a_norm_vmin",
        "fig7_panel_a_norm_vmax_formula",
        "fig7_panel_a_norm_vmax_floor",
        "fig7_panel_a_norm_vmax_data_factor",
        "fig7_panel_a_norm_data_max",
        "fig7_panel_a_norm_candidate_vmax",
        "fig7_panel_a_norm_vmax_current",
        "fig7_panel_b_norm",
        "fig7_panel_c_heatmap_cmap",
        "fig7_panel_c_norm",
        "fig7_panel_c_norm_type",
        "fig7_panel_c_norm_vmin",
        "fig7_panel_c_norm_vcenter",
        "fig7_panel_c_norm_vmax",
        "fig7_panel_c_color_limit_abs_pp",
        "fig7_panel_c_color_variable",
        "fig7_panel_c_colorbar_label",
    }
    fig6_norm_missing = sorted(fig6_norm_required - set(fig6_data.columns))
    fig6_norm_summary_values = sorted(fig6_data.get("fig7_panel_a_norm", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    fig6_norm_formula_values = sorted(fig6_data.get("fig7_panel_a_norm_vmax_formula", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    fig6_norm_vmax_values = numeric(fig6_data.get("fig7_panel_a_norm_vmax_current", pd.Series(dtype=float))).dropna().unique().tolist()
    fig6_norm_data_max_values = numeric(fig6_data.get("fig7_panel_a_norm_data_max", pd.Series(dtype=float))).dropna().unique().tolist()
    fig6_panel_a_norm_ok = bool(
        not fig6_data.empty
        and not fig6_norm_missing
        and fig6_data.get("fig7_panel_a_heatmap_cmap", pd.Series(dtype=str)).astype(str).eq(FIG6_FIG7_HEATMAP_CMAP).all()
        and fig6_data.get("fig7_panel_a_norm_type", pd.Series(dtype=str)).astype(str).eq("Normalize").all()
        and len(fig6_norm_summary_values) == 1
        and fig6_norm["panel_a_norm_vmax_formula"] in fig6_norm_summary_values[0]
        and "TwoSlopeNorm" not in fig6_norm_summary_values[0]
        and fig6_norm_formula_values == [fig6_norm["panel_a_norm_vmax_formula"]]
        and len(fig6_norm_vmax_values) == 1
        and math.isclose(float(fig6_norm_vmax_values[0]), float(fig6_norm["panel_a_norm_vmax"]), abs_tol=1e-9)
        and len(fig6_norm_data_max_values) == 1
        and math.isclose(float(fig6_norm_data_max_values[0]), float(fig6_norm["panel_a_norm_data_max"]), abs_tol=1e-9)
    )
    add(
        "Fig6_panel_A_norm_metadata_precise",
        "pass" if fig6_panel_a_norm_ok else "fail",
        json.dumps(
            {
                "missing_fields": fig6_norm_missing,
                "source_norm_summary": fig6_norm_summary_values,
                "source_norm_vmax_formula": fig6_norm_formula_values,
                "source_norm_vmax_current_values": [float(v) for v in fig6_norm_vmax_values],
                "computed_norm": fig6_norm,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "Fig6 panel A records RdBu_r with nonnegative Normalize and vmax=max(25, 1.05*data_max), with current vmax derived from composition data.",
        "Panel A/B share only the palette/cmap; panel A and panel B retain different normalization.",
    )
    fig6_perf_panel = fig6_data[fig6_data.get("panel", pd.Series("", index=fig6_data.index)).astype(str).eq(FIG6_PANEL_PERFORMANCE)].copy()
    if not fig6_perf_panel.empty and {
        "population_weighted_access_share",
        "category_mean_access_share",
        FIG7_COLOR_VARIABLE,
        "category",
        "fig7_panel_c_colorbar_label",
        "fig7_panel_c_migration_note",
        "fig7_panel_norm",
    }.issubset(fig6_perf_panel.columns):
        share_diff = numeric(fig6_perf_panel["population_weighted_access_share"]) - numeric(fig6_perf_panel["category_mean_access_share"])
        pp_diff = share_diff * 100.0
        fig6_c_diff_error = (numeric(fig6_perf_panel[FIG7_COLOR_VARIABLE]) - pp_diff).abs().max()
        fig6_c_category_diff_means = (
            fig6_perf_panel.assign(_diff=numeric(fig6_perf_panel[FIG7_COLOR_VARIABLE])).groupby("category")["_diff"].mean().abs()
        )
        fig6_c_relative_ok = (
            bool(np.isfinite(fig6_c_diff_error))
            and float(fig6_c_diff_error) <= 1e-9
            and fig6_c_category_diff_means.le(1e-9).all()
        )
    else:
        fig6_c_diff_error = np.nan
        fig6_c_category_diff_means = pd.Series(dtype=float)
        fig6_c_relative_ok = False
    fig6_panel_norms_by_panel = (
        fig6_data.groupby("panel")["fig7_panel_norm"].first().to_dict()
        if {"panel", "fig7_panel_norm"}.issubset(fig6_data.columns)
        else {}
    )
    fig6_c_migration_notes = sorted(
        fig6_perf_panel.get("fig7_panel_c_migration_note", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
    )
    fig6_c_colorbar_labels = sorted(
        fig6_perf_panel.get("fig7_panel_c_colorbar_label", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
    )
    fig6_c_axis_orders = sorted(
        fig6_perf_panel.get("fig7_panel_c_morphotype_axis_order", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
    )
    fig6_c_axis_order_notes = sorted(
        fig6_perf_panel.get("fig7_panel_c_morphotype_axis_order_note", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
    )
    fig6_c_display_order = numeric(fig6_perf_panel.get("fig7_panel_c_morphotype_display_order", pd.Series(dtype=float)))
    fig6_c_order_records = (
        fig6_perf_panel.assign(_display_order=fig6_c_display_order)
        .drop_duplicates(["morphotype", "_display_order"])
        .sort_values(["_display_order", "morphotype"])
        if not fig6_perf_panel.empty
        else pd.DataFrame()
    )
    fig6_c_order_sequence = (
        fig6_c_order_records["morphotype"].dropna().astype(str).tolist()
        if "morphotype" in fig6_c_order_records.columns
        else []
    )
    fig6_c_ok = (
        len(fig6_perf_panel) == 50
        and fig6_perf_panel.get("morphotype", pd.Series(dtype=str)).nunique() == 10
        and fig6_perf_panel.get("category", pd.Series(dtype=str)).nunique() == 5
        and fig6_c_relative_ok
        and fig6_c_colorbar_labels == [FIG7_COLORBAR_LABEL]
        and FIG6_PANEL_C_SOURCE_NOTE in fig6_c_migration_notes
        and fig6_c_axis_orders == [FIG6_MORPHOTYPE_AXIS_ORDER_TEXT]
        and fig6_c_axis_order_notes == [FIG6_PANEL_C_MORPHOTYPE_AXIS_ORDER_NOTE]
        and fig6_c_order_sequence == MORPHOTYPES
        and len(set(fig6_panel_norms_by_panel.values())) >= 3
    )
    add(
        "Fig7_panel_C_migrated_access_performance_deviation_valid",
        "pass" if fig6_c_ok else "fail",
        json.dumps(
            {
                "panel_counts": fig6_panel_counts,
                "panel_c_rows": len(fig6_perf_panel),
                "panel_c_morphotype_count": int(fig6_perf_panel.get("morphotype", pd.Series(dtype=str)).nunique()),
                "panel_c_category_count": int(fig6_perf_panel.get("category", pd.Series(dtype=str)).nunique()),
                "max_diff_recalculation_error": float(fig6_c_diff_error) if np.isfinite(fig6_c_diff_error) else None,
                "max_abs_category_mean_diff_pp": float(fig6_c_category_diff_means.max()) if not fig6_c_category_diff_means.empty else None,
                "panel_c_colorbar_labels": fig6_c_colorbar_labels,
                "panel_c_migration_notes": fig6_c_migration_notes,
                "panel_c_morphotype_axis_orders": fig6_c_axis_orders,
                "panel_c_morphotype_axis_order_notes": fig6_c_axis_order_notes,
                "panel_c_morphotype_display_order": fig6_c_order_sequence,
                "panel_norms_by_panel": fig6_panel_norms_by_panel,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "Fig6C contains the migrated 10 morphotypes x 5 facility categories performance-deviation heatmap; access_share_diff_pp is recomputed from raw access share minus category mean; A/B/C norms are panel-specific.",
    )
    fig6_visual_required = {
        "fig7_top_cell_count",
        "fig7_top_cell_flag",
        "fig7_top_cell_rank",
        "fig7_top_cell_metric",
        "fig7_top_cell_metric_value",
        "fig7_top_cell_annotation",
        "fig7_top_cell_outline_color",
        "fig7_top_cell_delta_sign",
    }
    fig6_legacy_side_summary_fields = [col for col in fig6_data.columns if col.startswith("fig7_summary_")]
    fig6_visual_missing = sorted(fig6_visual_required - set(fig6_data.columns))
    fig6_panel = fig6_data.get("panel", pd.Series(dtype=str)).astype(str)
    fig6_top_flag = (
        fig6_data.get("fig7_top_cell_flag", pd.Series(False, index=fig6_data.index))
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
    )
    fig6_comp_mask = fig6_panel.eq(FIG6_PANEL_COMPOSITION)
    fig6_trans_mask = fig6_panel.eq(FIG6_PANEL_TRANSITION)
    fig6_comp_top_count = int((fig6_comp_mask & fig6_top_flag).sum())
    fig6_trans_top_count = int((fig6_trans_mask & fig6_top_flag).sum())
    fig6_comp_top_metrics = sorted(
        fig6_data.loc[fig6_comp_mask & fig6_top_flag, "fig7_top_cell_metric"].dropna().astype(str).unique().tolist()
    ) if "fig7_top_cell_metric" in fig6_data.columns else []
    fig6_trans_top_metrics = sorted(
        fig6_data.loc[fig6_trans_mask & fig6_top_flag, "fig7_top_cell_metric"].dropna().astype(str).unique().tolist()
    ) if "fig7_top_cell_metric" in fig6_data.columns else []
    fig6_b_top = fig6_data.loc[fig6_trans_mask & fig6_top_flag].copy()
    fig6_b_outline_sign_ok = True
    if not fig6_b_top.empty and {"transition_delta_pp", "fig7_top_cell_outline_color"}.issubset(fig6_b_top.columns):
        for _, row in fig6_b_top.iterrows():
            delta = scalar(row.get("transition_delta_pp"))
            expected_color = FIG6_PANEL_B_POSITIVE_OUTLINE_COLOR if pd.notna(delta) and float(delta) >= 0 else FIG6_PANEL_B_NEGATIVE_OUTLINE_COLOR
            if str(row.get("fig7_top_cell_outline_color")) != expected_color:
                fig6_b_outline_sign_ok = False
                break
    fig6_visual_ok = bool(
        not fig6_data.empty
        and not fig6_visual_missing
        and fig6_comp_top_count == FIG6_TOP_CELL_COUNT
        and fig6_trans_top_count == FIG6_TOP_CELL_COUNT
        and fig6_comp_top_metrics == ["mean_unit_share_percent"]
        and fig6_trans_top_metrics == ["absolute_transition_delta_pp"]
        and not fig6_legacy_side_summary_fields
        and fig6_b_outline_sign_ok
    )
    add(
        "Fig7_top_cell_annotations_current_visual_encoding",
        "pass" if fig6_visual_ok else "fail",
        json.dumps(
            json_ready(
                {
                    "missing_fields": fig6_visual_missing,
                    "composition_top_cell_count": fig6_comp_top_count,
                    "transition_top_cell_count": fig6_trans_top_count,
                    "composition_top_metrics": fig6_comp_top_metrics,
                    "transition_top_metrics": fig6_trans_top_metrics,
                    "legacy_side_summary_field_count": len(fig6_legacy_side_summary_fields),
                    "panel_b_outline_sign_ok": fig6_b_outline_sign_ok,
                    "visual_metadata": fig6_enrichment_metadata(fig6_data),
                }
            ),
            ensure_ascii=False,
            sort_keys=True,
        ),
        (
            f"Top {FIG6_TOP_CELL_COUNT} panel A share cells and top {FIG6_TOP_CELL_COUNT} panel B "
            "absolute-delta cells recorded with the current Fig7 visual encoding fields."
        ),
        "Fig7 uses RdBu_r heatmaps plus top-cell outlines with panel-specific normalization.",
    )
    fig6_svg_text = ""
    if fig6_product is not None and fig6_product.svg_path.exists():
        fig6_svg_text = fig6_product.svg_path.read_text(encoding="utf-8", errors="ignore")
    fig6_retained_svg_terms = [
        "A  Scale/network composition",
        "B  Scale/network transitions",
        "C  Performance deviation by facility category",
        "Mean unit share (%)",
        "Delta share (pp)",
        FIG7_COLORBAR_LABEL,
        "Morphotype",
        "Facility category",
    ]
    fig6_c_svg_tick_positions = {mt: fig6_svg_text.rfind(mt) for mt in MORPHOTYPES} if fig6_svg_text else {}
    fig6_c_svg_tick_order_ok = bool(fig6_svg_text) and all(
        fig6_c_svg_tick_positions.get(left, -1) >= 0
        and fig6_c_svg_tick_positions.get(right, -1) >= 0
        and fig6_c_svg_tick_positions[left] < fig6_c_svg_tick_positions[right]
        for left, right in zip(MORPHOTYPES, MORPHOTYPES[1:])
    )
    fig6_c_svg_axis_label_ok = bool(fig6_svg_text) and "Morphotype (ranked by mean access)" not in fig6_svg_text
    fig6_source_title_false = (
        "fig7_figure_level_title_current" in fig6_data.columns
        and fig6_data["fig7_figure_level_title_current"].astype(str).str.strip().str.lower().isin({"false", "0", "no"}).all()
    )
    fig6_source_bottom_false = (
        "fig7_bottom_note_current" in fig6_data.columns
        and fig6_data["fig7_bottom_note_current"].astype(str).str.strip().str.lower().isin({"false", "0", "no"}).all()
    )
    fig6_source_removed_text_ok = (
        "fig7_removed_figure_level_title_text" in fig6_data.columns
        and "fig7_removed_bottom_note_text" in fig6_data.columns
        and sorted(fig6_data["fig7_removed_figure_level_title_text"].dropna().astype(str).unique().tolist()) == [FIG6_REMOVED_SUPTITLE_TEXT]
        and sorted(fig6_data["fig7_removed_bottom_note_text"].dropna().astype(str).unique().tolist()) == [FIG6_REMOVED_BOTTOM_NOTE_TEXT]
    )
    fig6_removed_visual_text_ok = (
        bool(fig6_svg_text)
        and FIG6_REMOVED_SUPTITLE_TEXT not in fig6_svg_text
        and FIG6_REMOVED_BOTTOM_NOTE_TEXT not in fig6_svg_text
        and all(term in fig6_svg_text for term in fig6_retained_svg_terms)
        and fig6_c_svg_axis_label_ok
        and fig6_source_title_false
        and fig6_source_bottom_false
        and fig6_source_removed_text_ok
    )
    add(
        "Fig7_title_and_bottom_note_removed_from_render",
        "pass" if fig6_removed_visual_text_ok else "fail",
        json.dumps(
            {
                "svg_exists": bool(fig6_product is not None and fig6_product.svg_path.exists()),
                "removed_title_absent_in_svg": FIG6_REMOVED_SUPTITLE_TEXT not in fig6_svg_text if fig6_svg_text else False,
                "removed_bottom_note_absent_in_svg": FIG6_REMOVED_BOTTOM_NOTE_TEXT not in fig6_svg_text if fig6_svg_text else False,
                "retained_terms_present": {term: term in fig6_svg_text for term in fig6_retained_svg_terms} if fig6_svg_text else {},
                "panel_c_old_axis_label_absent_in_svg": fig6_c_svg_axis_label_ok,
                "source_figure_level_title_current_false": bool(fig6_source_title_false),
                "source_bottom_note_current_false": bool(fig6_source_bottom_false),
                "source_removed_text_recorded": bool(fig6_source_removed_text_ok),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "Fig7 SVG has no figure-level top title and no bottom footnote/note; panel titles, axis labels and colorbar labels remain.",
        FIG6_TITLE_FOOTNOTE_REMOVAL_NOTE,
    )
    fig6_c_order_metadata_ok = (
        fig6_c_order_sequence == MORPHOTYPES
        and fig6_c_axis_orders == [FIG6_MORPHOTYPE_AXIS_ORDER_TEXT]
        and fig6_c_axis_order_notes == [FIG6_PANEL_C_MORPHOTYPE_AXIS_ORDER_NOTE]
        and fig6_c_svg_tick_order_ok
        and fig6_c_svg_axis_label_ok
        and FIG6_PANEL_C_MORPHOTYPE_AXIS_ORDER_NOTE in manifest_text
        and FIG6_PANEL_C_MORPHOTYPE_AXIS_ORDER_NOTE in lineage_text
    )
    add(
        "Fig7_panel_C_morphotype_order_aligned_with_A_B",
        "pass" if fig6_c_order_metadata_ok else "fail",
        json.dumps(
            {
                "expected_order": MORPHOTYPES,
                "source_display_order": fig6_c_order_sequence,
                "source_axis_orders": fig6_c_axis_orders,
                "source_axis_order_notes": fig6_c_axis_order_notes,
                "svg_last_tick_positions": fig6_c_svg_tick_positions,
                "svg_tick_order_ok": fig6_c_svg_tick_order_ok,
                "svg_old_axis_label_absent": fig6_c_svg_axis_label_ok,
                "manifest_note_present": FIG6_PANEL_C_MORPHOTYPE_AXIS_ORDER_NOTE in manifest_text,
                "lineage_note_present": FIG6_PANEL_C_MORPHOTYPE_AXIS_ORDER_NOTE in lineage_text,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "Fig6C morphotype x-axis and source metadata use MT01->MT10, matching Fig6A/B.",
    )
    add(
        "legacy_supplemental_ids_absent_from_manifest_lineage",
        "pass" if not legacy_supplemental_hits else "fail",
        legacy_supplemental_hits,
        "No legacy supplemental figure ids in manifest/lineage",
    )
    fig2_required = {
        "ucdb_pop_2025",
        "point_size",
        "size_variable",
        "size_variable_label",
        "point_size_units",
        "point_size_scale",
        "quality_tier_color",
        "quality_tier",
        "sample_group",
    }
    fig2_missing = sorted(fig2_required - set(fig2_data.columns))
    add(
        "Fig2_has_population_size_and_group_fields",
        "pass" if not fig2_missing else "fail",
        sorted(fig2_required & set(fig2_data.columns)),
        sorted(fig2_required),
        ";".join(fig2_missing),
    )
    fig2_city_count = int(fig2_data["city_id"].nunique()) if "city_id" in fig2_data.columns else 0
    fig2_core_count = int(fig2_data.get("quality_tier", pd.Series(dtype=str)).astype(str).eq("core").sum())
    add("Fig2_city_count_86_and_core_count_48", "pass" if (fig2_city_count, fig2_core_count) == (86, 48) else "fail", (fig2_city_count, fig2_core_count), (86, 48))
    fig2_size_ok = "point_size" in fig2_data.columns and numeric(fig2_data["point_size"]).gt(0).all()
    add("Fig2_point_sizes_positive", "pass" if fig2_size_ok else "fail", bool(fig2_size_ok), True)
    fig2_number_fields = {"city_number", "city_label", "city_index_label"}
    fig2_number_fields_present = fig2_number_fields.issubset(set(fig2_data.columns))
    fig2_numbers_present = (
        fig2_number_fields_present
        and fig2_data[list(fig2_number_fields)].notna().all(axis=None)
        and fig2_data["city_label"].astype(str).str.strip().ne("").all()
        and fig2_data["city_index_label"].astype(str).str.strip().ne("").all()
    )
    add(
        "Fig2_city_numbers_present",
        "pass" if fig2_numbers_present else "fail",
        sorted(fig2_number_fields & set(fig2_data.columns)),
        sorted(fig2_number_fields),
    )
    if "city_number" in fig2_data.columns:
        fig2_city_numbers = numeric(fig2_data["city_number"]).dropna().astype(int)
    else:
        fig2_city_numbers = pd.Series(dtype=int)
    fig2_city_number_unique = (
        len(fig2_city_numbers) == fig2_city_count
        and fig2_city_numbers.nunique() == fig2_city_count
        and sorted(fig2_city_numbers.tolist()) == list(range(1, fig2_city_count + 1))
    )
    add(
        "Fig2_city_number_unique",
        "pass" if fig2_city_number_unique else "fail",
        {"rows": int(len(fig2_city_numbers)), "unique": int(fig2_city_numbers.nunique())},
        {"rows": 86, "unique": 86, "sequence": "1-86"},
        FIG2_CITY_NUMBER_RULE,
    )
    fig2_city_index_rows = int(fig2_data.get("city_index_label", pd.Series(dtype=str)).astype(str).str.strip().ne("").sum())
    add("Fig2_city_index_rows_86", "pass" if fig2_city_index_rows == 86 else "fail", fig2_city_index_rows, 86)
    fig2_city_index_alignment = fig2_alignment.get("city_index_left_alignment", {})
    add(
        "Fig2_city_index_left_alignment",
        str(fig2_city_index_alignment.get("status", "fail")),
        json.dumps(json_ready(fig2_city_index_alignment), ensure_ascii=False, sort_keys=True),
        f"abs(left_delta) <= {FIG2_ALIGNMENT_THRESHOLD:g}; abs(right_delta) <= {FIG2_ALIGNMENT_THRESHOLD:g}",
        "City index axes left/right edges aligned to the renderer-measured Cities by region visual left edge and legend-panel visual right edge.",
    )
    add(
        "Fig2_bottom_small_panels_preserved",
        "pass" if bool(fig2_alignment.get("bottom_small_panels_preserved")) else "fail",
        json.dumps(
            {
                "bar_panel_top": fig2_alignment.get("bar_panel_top"),
                "legend_panel_top": fig2_alignment.get("legend_panel_top"),
                "city_index_panel_top": fig2_alignment.get("city_index_panel_top"),
                "map_axes_bottom": fig2_alignment.get("map_axes_bottom"),
                "preserved": fig2_alignment.get("bottom_small_panels_preserved"),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "bar and legend panels remain below the map; city index remains below both panels",
    )
    fig2_gap_reduced = bool(fig2_alignment.get("map_bottom_gap_reduced"))
    add(
        "Fig2_map_bottom_gap_reduced",
        "pass" if fig2_gap_reduced else "fail",
        json.dumps(
            {
                "previous_gap": fig2_alignment.get("map_bottom_gap_previous"),
                "current_gap": fig2_alignment.get("map_bottom_gap_current"),
                "reduction": fig2_alignment.get("map_bottom_gap_reduction"),
                "units": fig2_alignment.get("map_bottom_gap_units"),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        f"current gap > 0 and < previous gap ({FIG2_MAP_BOTTOM_GAP_PREVIOUS:g})",
        "Map-to-bottom-row vertical gap is an explicit Fig2 layout parameter.",
    )
    fig2_city_index_note_removed = (
        bool(fig2_alignment.get("city_index_title_present"))
        and not bool(fig2_alignment.get("city_index_right_note_present"))
        and fig2_city_index_rows == 86
    )
    add(
        "Fig2_city_index_right_note_removed",
        "pass" if fig2_city_index_note_removed else "fail",
        json.dumps(
            {
                "removed_note_text": fig2_alignment.get("city_index_right_note_text"),
                "city_index_title_present": fig2_alignment.get("city_index_title_present"),
                "right_note_present": fig2_alignment.get("city_index_right_note_present"),
                "city_index_rows": fig2_city_index_rows,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "City index title remains; the right-side explanatory note is absent; 86 city index rows remain.",
    )
    fig2_no_overlap_payload = fig2_alignment.get("panel_overlap", {})
    add(
        "Fig2_no_overlap",
        "pass" if bool(fig2_alignment.get("no_overlap")) else "fail",
        json.dumps(json_ready(fig2_no_overlap_payload), ensure_ascii=False, sort_keys=True),
        "renderer-measured map, bottom-row and city-index visual blocks have positive vertical gaps",
    )
    fig2_no_axis_labels = (
        int(fig2_alignment.get("visible_map_tick_label_count", -1)) == 0
        and not str(fig2_alignment.get("map_xlabel", "")).strip()
        and not str(fig2_alignment.get("map_ylabel", "")).strip()
    )
    add(
        "Fig2_no_axis_tick_labels",
        "pass" if fig2_no_axis_labels else "fail",
        json.dumps(
            {
                "visible_map_tick_label_count": fig2_alignment.get("visible_map_tick_label_count"),
                "map_xlabel": fig2_alignment.get("map_xlabel"),
                "map_ylabel": fig2_alignment.get("map_ylabel"),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "0 visible map tick labels and empty map x/y axis titles",
    )
    add(
        "Fig2_map_bottom_text_visual_edge_alignment",
        str(fig2_alignment["status"]),
        json.dumps(json_ready(fig2_alignment), ensure_ascii=False, sort_keys=True),
        f"abs(left_delta) <= {FIG2_ALIGNMENT_THRESHOLD:g}; abs(right_delta) <= {FIG2_ALIGNMENT_THRESHOLD:g}",
        "Renderer-measured lower-left bar-panel text bbox and lower-right legend visual edge, in figure-fraction coordinates.",
    )
    fig3_product = next((p for p in products if p.figure_id == "Fig3"), None)
    fig3_missing_outputs = []
    if fig3_product is not None:
        for suffix, path in [
            ("png", fig3_product.png_path),
            ("svg", fig3_product.svg_path),
            ("pdf", fig3_product.pdf_path),
            ("data", fig3_product.data_path),
        ]:
            if (not path.exists()) or path.stat().st_size <= 0:
                fig3_missing_outputs.append(suffix)
    else:
        fig3_missing_outputs.append("product")
    add(
        "Fig3_quality_thumbnail_outputs_exist",
        "pass" if not fig3_missing_outputs else "fail",
        "all present and nonempty" if not fig3_missing_outputs else ";".join(fig3_missing_outputs),
        "png/svg/pdf/csv present and nonempty",
    )
    fig3_required = {
        "quality_score",
        "quality_tier",
        "quality_weight",
        "quality_tier_color",
        "quality_score_plot",
        "quality_tier_plot",
        "color_variable",
        "color_variable_label",
        "component_panel_fields",
        "quality_score_source_step",
        "quality_score_definition",
        "quality_score_not_official_osm_field",
        "quality_score_not_morphology_type_score",
        "road_thumbnail_included",
        "road_thumbnail_display_order",
        "road_thumbnail_rank_group",
        "road_thumbnail_rank",
        "road_thumbnail_source",
        "road_thumbnail_status",
        "road_thumbnail_extent",
        "road_thumbnail_extent_width_km",
        "road_thumbnail_extent_height_km",
        "road_thumbnail_extent_aspect",
        "road_thumbnail_crop_shape",
        "road_thumbnail_coordinate_units",
        "road_thumbnail_background_color",
        "road_thumbnail_road_color",
        "road_thumbnail_border_color",
        "road_thumbnail_style_note",
    }
    fig3_missing_fields = sorted(fig3_required - set(fig3_data.columns))
    add(
        "Fig3_has_quality_thumbnail_source_fields",
        "pass" if not fig3_missing_fields else "fail",
        sorted(fig3_required & set(fig3_data.columns)),
        sorted(fig3_required),
        ";".join(fig3_missing_fields),
    )
    fig3_map_specific_columns = sorted(
        {
            "map_lon",
            "map_lat",
            "map_lon_original",
            "map_lat_original",
            "map_display_offset_lon_deg",
            "map_display_offset_lat_deg",
            "map_display_duplicate_group_size",
            "point_color",
            "point_size",
            "point_size_variable",
            "point_size_units",
            "point_size_scale",
            "point_size_scale_min",
            "point_size_scale_max",
            "point_size_scale_exponent",
            "point_size_transform",
            "point_size_legend_scores",
            "point_size_legend_areas",
            "label_flag",
            "label_reason",
            "label_text",
            "city_label_dx_points",
            "city_label_dy_points",
            "city_label_overlap_score",
        }
        & set(fig3_data.columns)
    )
    if fig3_data.empty:
        add("Fig3_no_world_map_panel", "fail", "no Fig3 source data", "world map panel removed and no map-specific source fields")
        add("Fig3_no_map_point_area_legend", "fail", "no Fig3 source data", "map point-area legend removed")
        add("Fig3_compact_quality_explanation_present", "fail", "no Fig3 source data", "compact quality explanation retained")
        add("Fig3_score_by_tier_panel_present", "fail", "no Fig3 source data", "Quality score by tier panel present")
        add("Fig3_score_by_tier_panel_compact", "fail", "no Fig3 source data", "Quality score by tier panel height compacted")
        add("Fig3_component_legend_summary_RHPLB_present", "fail", "no Fig3 source data", "R/H/P/L/B component definitions present")
        add("Fig3_component_footnote_lines_removed", "fail", "no Fig3 source data", "old component footnote lines absent from the component panel")
        add("Fig3_component_visual_enlarged_and_richer", "fail", "no Fig3 source data", "larger component mini-chart with p10-p90 band, mean dot and Low/High markers")
        add("Fig3_component_legend_labels_attached_to_each_symbol", "fail", "no Fig3 source data", "Each component visual legend symbol has its own adjacent label")
        add("Fig3_quality_to_thumbnail_vertical_gap_reduced", "fail", "no Fig3 source data", f"quality-to-thumbnail gap <= {FIG3_QUALITY_TO_THUMBNAIL_VERTICAL_GAP_MAX:g} and reduced from the previous centered grid")
        add("Fig3_quality_and_thumbnail_blocks_nonoverlap", "fail", "no Fig3 source data", "upper quality block and thumbnail block do not overlap")
        add("Fig3_thumbnails_primary_visual_area", "fail", "no Fig3 source data", "thumbnail axes area > compact quality explanation area")
        add("Fig3_thumbnail_selection_matches_low_high_scores", "fail", "no Fig3 source data", "Low 6 plus High 6 score thumbnails")
        add("Fig3_road_thumbnail_city_count_12", "fail", "no Fig3 source data", 12)
        add("Fig3_road_thumbnail_low_high_counts_6_6", "fail", "no Fig3 source data", "low=6; high=6")
        add("Fig3_road_thumbnail_sources_available", "fail", "no Fig3 source data", "all 12 thumbnails use available road-network sources")
        add("Fig3_road_thumbnail_low_ranks_follow_quality_score", "fail", "no Fig3 source data", "Low ranks ascend by quality_score")
        add("Fig3_road_thumbnail_high_ranks_follow_quality_score", "fail", "no Fig3 source data", "High ranks descend by quality_score")
        add("Fig3_road_thumbnail_tiles_square_aspect_equal", "fail", "no Fig3 source data", "thumbnail axes and crop extents are square")
        add("Fig3_road_thumbnail_white_background_black_roads_border", "fail", "no Fig3 source data", "white background, black roads, black border")
        add("Fig3_road_thumbnail_panel_spans_wide_layout", "fail", "no Fig3 source data", "thumbnail left/right use wide Fig3 layout")
        add("Fig3_road_thumbnail_horizontal_spacing_expanded", "fail", "no Fig3 source data", f"column gap >= {FIG3_ROAD_THUMBNAIL_COLUMN_GAP:g}")
        add("Fig3_quality_component_summary_vertical_spacing", "fail", "no Fig3 source data", f"component summary min vertical gap >= {FIG3_COMPONENT_SUMMARY_MIN_VERTICAL_GAP:g}")
        add("Fig3_quality_summary_title_aligned_with_score_by_tier_title", "fail", "no Fig3 source data", f"title top/anchor deltas <= {FIG3_VISUAL_ALIGNMENT_TOLERANCE:g}")
        add(
            "Fig3_quality_score_definition_documented",
            "fail",
            "no Fig3 source data",
            "Project-derived Step07 OSM data quality/confidence definition recorded; not an official OSM field and not a pure road-morphology score",
        )
    else:
        no_world_map_ok = (
            fig3_layout.get("world_map_panel_present") is False
            and fig3_layout.get("map_axes_created") is False
            and not fig3_map_specific_columns
        )
        add(
            "Fig3_no_world_map_panel",
            "pass" if no_world_map_ok else "fail",
            json.dumps(
                {
                    "layout_world_map_panel_present": fig3_layout.get("world_map_panel_present"),
                    "layout_map_axes_created": fig3_layout.get("map_axes_created"),
                    "map_specific_source_columns": fig3_map_specific_columns,
                    "title": fig3_product.title if fig3_product else "",
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "Fig3 has no world map panel and no map-specific source fields such as map_lon/map_lat/point_size/map labels",
        )
        no_point_area_legend_ok = (
            fig3_layout.get("map_point_area_legend_present") is False
            and fig3_layout.get("map_encoding_present") is False
            and not any(col in fig3_data.columns for col in ["point_size_legend_scores", "point_size_legend_areas", "point_size_scale"])
        )
        add(
            "Fig3_no_map_point_area_legend",
            "pass" if no_point_area_legend_ok else "fail",
            json.dumps(
                {
                    "layout_map_point_area_legend_present": fig3_layout.get("map_point_area_legend_present"),
                    "layout_map_encoding_present": fig3_layout.get("map_encoding_present"),
                    "source_point_area_columns_present": sorted(
                        set(["point_size_legend_scores", "point_size_legend_areas", "point_size_scale"]) & set(fig3_data.columns)
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "Map point-area legend and related point-size encoding are absent from Fig3",
        )
        explanation_ok = (
            bool(fig3_layout.get("quality_explanation_panel_present"))
            and bool(fig3_layout.get("quality_score_definition_note_present"))
            and bool(fig3_layout.get("quality_component_compact_legend_present"))
            and bool(fig3_layout.get("quality_score_by_tier_panel_present"))
        )
        add(
            "Fig3_compact_quality_explanation_present",
            "pass" if explanation_ok else "fail",
            json.dumps(
                {
                    "quality_explanation_panel_present": fig3_layout.get("quality_explanation_panel_present"),
                    "definition_note_present": fig3_layout.get("quality_score_definition_note_present"),
                    "component_compact_legend_present": fig3_layout.get("quality_component_compact_legend_present"),
                    "score_by_tier_panel_present": fig3_layout.get("quality_score_by_tier_panel_present"),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "Fig3 retains a compact quality explanation above the thumbnails",
        )
        fig3_score_values = numeric(fig3_data.get("quality_score_plot", pd.Series(dtype=float))).dropna()
        fig3_axis_min = float(fig3_layout.get("quality_score_axis_min", np.nan))
        fig3_axis_max = float(fig3_layout.get("quality_score_axis_max", np.nan))
        if fig3_score_values.empty:
            fig3_axis_ok = False
            fig3_axis_payload: dict[str, Any] = {"issue": "no Fig3 quality scores"}
        else:
            fig3_data_min = float(fig3_score_values.min())
            fig3_data_max = float(fig3_score_values.max())
            fig3_axis_ok = (
                bool(fig3_layout.get("quality_score_by_tier_panel_present"))
                and np.isfinite(fig3_axis_min)
                and np.isfinite(fig3_axis_max)
                and fig3_axis_min <= fig3_data_min
                and fig3_axis_max >= fig3_data_max
                and fig3_axis_min > 0.0
                and fig3_axis_max < 100.0
                and (fig3_axis_max - fig3_axis_min) < 100.0
            )
            fig3_axis_payload = {
                "axis_range": [fig3_axis_min, fig3_axis_max],
                "data_range": [fig3_data_min, fig3_data_max],
                "method": fig3_layout.get("quality_score_axis_method"),
                "axis_columns_present": sorted(
                    {"quality_score_axis_min", "quality_score_axis_max", "quality_score_axis_method"} & set(fig3_data.columns)
                ),
            }
        add(
            "Fig3_score_by_tier_panel_present",
            "pass" if fig3_axis_ok else "fail",
            json.dumps(fig3_axis_payload, ensure_ascii=False, sort_keys=True),
            "Quality score by tier panel is present and its y-axis spans the observed data range with padding instead of 0-100",
        )
        score_compact_ok = bool(fig3_layout.get("quality_score_by_tier_panel_compact"))
        add(
            "Fig3_score_by_tier_panel_compact",
            "pass" if score_compact_ok else "fail",
            json.dumps(
                {
                    "panel_height": fig3_layout.get("quality_score_by_tier_panel_height"),
                    "previous_height": fig3_layout.get("quality_score_by_tier_panel_previous_height"),
                    "compact_height": fig3_layout.get("quality_score_by_tier_panel_compact_height"),
                    "upper_band_height": fig3_layout.get("quality_upper_band_height"),
                    "height_ratio": fig3_layout.get("quality_score_by_tier_compact_height_ratio"),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "Quality score by tier plot is compacted within the upper band so it reads as auxiliary context",
        )
        component_fields = [field for field, _ in FIG3_QUALITY_COMPONENTS if field in fig3_data.columns]
        component_nonempty = {field: int(numeric(fig3_data[field]).notna().sum()) for field in component_fields}
        expected_component_fields = fig3_component_strip_fields()
        expected_abbr = fig3_component_strip_abbreviations()
        component_defs = fig3_layout.get("quality_component_definitions", {})
        component_legend_ok = (
            bool(fig3_layout.get("quality_component_compact_legend_present"))
            and component_fields == expected_component_fields
            and fig3_layout.get("quality_component_abbreviations") == expected_abbr
            and sorted(component_defs.keys()) == sorted(expected_abbr)
            and all(count > 0 for count in component_nonempty.values())
        )
        add(
            "Fig3_component_legend_summary_RHPLB_present",
            "pass" if component_legend_ok else "fail",
            json.dumps(
                {
                    "layout_component_compact_legend_present": fig3_layout.get("quality_component_compact_legend_present"),
                    "component_fields": component_fields,
                    "expected_component_fields": expected_component_fields,
                    "component_abbreviations": fig3_layout.get("quality_component_abbreviations"),
                    "component_definitions": component_defs,
                    "component_nonempty_counts": component_nonempty,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "Compact component legend/summary defines R/H/P/L/B quality_score components",
        )
        component_footnote_removed_ok = bool(fig3_layout.get("quality_component_footnote_lines_removed"))
        add(
            "Fig3_component_footnote_lines_removed",
            "pass" if component_footnote_removed_ok else "fail",
            json.dumps(
                {
                    "old_component_footnote_line_count": fig3_layout.get("quality_component_removed_footnote_line_count"),
                    "score_definition_note_location": fig3_layout.get("quality_component_score_definition_note_location"),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "The two old small component-panel footnote lines are absent; the score definition note is moved below the score panel",
        )
        component_visual_ok = (
            bool(fig3_layout.get("quality_component_summary_bar_height_enlarged"))
            and bool(fig3_layout.get("quality_component_summary_visual_span_enlarged"))
            and int(fig3_layout.get("quality_component_summary_low_marker_count", 0)) == len(expected_component_fields)
            and int(fig3_layout.get("quality_component_summary_high_marker_count", 0)) == len(expected_component_fields)
            and {"p10_p90_distribution_band", "all_city_mean_dot", "low_6_component_marker", "high_6_component_marker", "0_50_100_reference_ticks"}.issubset(
                set(fig3_layout.get("quality_component_summary_enrichments", []))
            )
        )
        add(
            "Fig3_component_visual_enlarged_and_richer",
            "pass" if component_visual_ok else "fail",
            json.dumps(
                {
                    "bar_height": fig3_layout.get("quality_component_summary_bar_height"),
                    "previous_bar_height": fig3_layout.get("quality_component_summary_previous_bar_height"),
                    "visual_y_span": fig3_layout.get("quality_component_summary_visual_y_span"),
                    "previous_y_span": fig3_layout.get("quality_component_summary_previous_y_span"),
                    "axis_ticks": fig3_layout.get("quality_component_summary_axis_ticks"),
                    "enrichments": fig3_layout.get("quality_component_summary_enrichments"),
                    "low_marker_count": fig3_layout.get("quality_component_summary_low_marker_count"),
                    "high_marker_count": fig3_layout.get("quality_component_summary_high_marker_count"),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "Quality component mini-chart is enlarged and adds distribution bands, all-city means, Low/High markers and 0/50/100 references",
        )
        expected_visual_legend_labels = {key: label for key, _, label in FIG3_COMPONENT_VISUAL_LEGEND_ITEMS}
        visual_legend_items = fig3_layout.get("quality_component_visual_legend_items", [])
        visual_legend_labels = fig3_layout.get("quality_component_visual_legend_attached_labels", {})
        visual_legend_ok = (
            fig3_layout.get("quality_component_visual_legend_label_strategy") == "attached_to_each_symbol"
            and isinstance(visual_legend_items, list)
            and [item.get("key") for item in visual_legend_items] == list(expected_visual_legend_labels.keys())
            and visual_legend_labels == expected_visual_legend_labels
            and bool(fig3_layout.get("quality_component_visual_legend_combined_summary_absent"))
        )
        add(
            "Fig3_component_legend_labels_attached_to_each_symbol",
            "pass" if visual_legend_ok else "fail",
            json.dumps(
                {
                    "label_strategy": fig3_layout.get("quality_component_visual_legend_label_strategy"),
                    "legend_items": visual_legend_items,
                    "attached_labels": visual_legend_labels,
                    "expected_labels": expected_visual_legend_labels,
                    "combined_summary_hits": fig3_layout.get("quality_component_visual_legend_combined_summary_hits"),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "The component visual legend labels p10-p90 band, All-city mean, Low 6 and High 6 are attached directly to their own symbols",
        )
        gap_reduced_ok = bool(fig3_layout.get("quality_to_thumbnail_vertical_gap_reduced"))
        add(
            "Fig3_quality_to_thumbnail_vertical_gap_reduced",
            "pass" if gap_reduced_ok else "fail",
            json.dumps(
                {
                    "previous_centered_gap": fig3_layout.get("quality_to_thumbnail_vertical_gap_previous_centered"),
                    "current_gap": fig3_layout.get("quality_to_thumbnail_vertical_gap"),
                    "gap_reduction": fig3_layout.get("quality_to_thumbnail_vertical_gap_reduction"),
                    "current_gap_max": fig3_layout.get("quality_to_thumbnail_vertical_gap_max"),
                    "min_required_reduction": fig3_layout.get("quality_to_thumbnail_vertical_gap_min_reduction"),
                    "thumbnail_vertical_anchor": fig3_layout.get("road_thumbnail_vertical_anchor"),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "The road-thumbnail grid is top-aligned within its band, reducing the vertical gap under the compact quality panels",
        )
        nonoverlap_ok = bool(fig3_layout.get("quality_and_thumbnail_blocks_nonoverlap"))
        add(
            "Fig3_quality_and_thumbnail_blocks_nonoverlap",
            "pass" if nonoverlap_ok else "fail",
            json.dumps(
                {
                    "quality_visible_block_bottom": fig3_layout.get("quality_visible_block_bottom"),
                    "thumbnail_visible_block_top": fig3_layout.get("thumbnail_visible_block_top"),
                    "visible_gap": fig3_layout.get("quality_thumbnail_visible_gap"),
                    "min_visible_gap": fig3_layout.get("quality_thumbnail_visible_gap_min"),
                    "axis_gap": fig3_layout.get("quality_to_thumbnail_vertical_gap"),
                    "thumbnail_panel_below_quality_explanation": fig3_layout.get("road_thumbnail_panel_below_quality_explanation"),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "The compact quality explanation and road-thumbnail title/tile block remain separated after tightening the gap",
        )
        thumbnail_included = fig3_data.get("road_thumbnail_included", pd.Series(False, index=fig3_data.index)).astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )
        thumbnail_data = fig3_data[thumbnail_included].copy()
        thumbnail_ids = set(thumbnail_data.get("city_id", pd.Series(dtype=str)).astype(str))
        valid_scores = fig3_data[fig3_data["quality_score_plot"].notna()].copy()
        low_expected = valid_scores.sort_values(["quality_score_plot", "city_id"], ascending=[True, True]).head(
            FIG3_KEY_LABEL_BOTTOM_N
        )
        high_expected = valid_scores.sort_values(["quality_score_plot", "city_id"], ascending=[False, True]).head(
            FIG3_KEY_LABEL_TOP_N
        )
        low_ids = set(low_expected["city_id"].astype(str))
        high_ids = set(high_expected["city_id"].astype(str))
        expected_thumbnail_ids = low_ids | high_ids
        selected_excluded_ids = set(
            fig3_data.loc[
                fig3_data["city_id"].astype(str).isin(expected_thumbnail_ids)
                & fig3_data["quality_tier_plot"].astype(str).eq("excluded_candidate"),
                "city_id",
            ].astype(str)
        )
        selection_match_ok = thumbnail_ids == expected_thumbnail_ids
        add(
            "Fig3_thumbnail_selection_matches_low_high_scores",
            "pass" if selection_match_ok else "fail",
            json.dumps(
                {
                    "thumbnail_city_ids": sorted(thumbnail_ids),
                    "expected_low_ids": sorted(low_ids),
                    "expected_high_ids": sorted(high_ids),
                    "missing": sorted(expected_thumbnail_ids - thumbnail_ids),
                    "extra": sorted(thumbnail_ids - expected_thumbnail_ids),
                    "selected_excluded_candidate": sorted(selected_excluded_ids & thumbnail_ids),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "Road thumbnails, rather than map labels, mark exactly the Low 6 and High 6 quality_score cities",
        )
        thumbnail_count_ok = int(thumbnail_included.sum()) == FIG3_ROAD_THUMBNAIL_TARGET_COUNT
        add(
            "Fig3_road_thumbnail_city_count_12",
            "pass" if thumbnail_count_ok else "fail",
            json.dumps(
                {
                    "thumbnail_city_count": int(thumbnail_included.sum()),
                    "thumbnail_city_ids": sorted(thumbnail_ids),
                    "rank_groups": thumbnail_data.get("road_thumbnail_rank_group", pd.Series(dtype=str)).dropna().astype(str).tolist(),
                    "selection_rule": FIG3_ROAD_THUMBNAIL_SELECTION_RULE,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            FIG3_ROAD_THUMBNAIL_TARGET_COUNT,
        )
        thumbnail_groups = thumbnail_data.get("road_thumbnail_rank_group", pd.Series(dtype=str)).dropna().astype(str)
        thumbnail_group_counts = thumbnail_groups.value_counts(dropna=False).to_dict()
        expected_low_count = FIG3_KEY_LABEL_BOTTOM_N
        expected_high_count = FIG3_KEY_LABEL_TOP_N
        low_high_counts_ok = (
            int(thumbnail_group_counts.get("low", 0)) == expected_low_count
            and int(thumbnail_group_counts.get("high", 0)) == expected_high_count
            and set(thumbnail_groups.unique()).issubset({"low", "high"})
        )
        add(
            "Fig3_road_thumbnail_low_high_counts_6_6",
            "pass" if low_high_counts_ok else "fail",
            json.dumps(
                {
                    "group_counts": thumbnail_group_counts,
                    "expected_low_count": expected_low_count,
                    "expected_high_count": expected_high_count,
                    "layout_group_counts": fig3_layout.get("road_thumbnail_group_counts"),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "Low thumbnail count = 6 and High thumbnail count = 6, with no excluded/supplement rank group",
        )
        display_order_data = thumbnail_data.copy()
        display_order_data["_display_order"] = numeric(
            display_order_data.get("road_thumbnail_display_order", pd.Series(np.nan, index=display_order_data.index))
        )
        display_order_records = display_order_data.dropna(subset=["_display_order"]).sort_values("_display_order")
        expected_thumbnail_city_id_order = [
            "main_014",
            "main_015",
            "main_020",
            "main_045",
            "main_046",
            "main_019",
            "chn_006",
            "main_070",
            "chn_005",
            "main_065",
            "main_064",
            "main_008",
        ]
        expected_thumbnail_city_order = [
            "Tokyo",
            "Osaka-Kobe-Kyoto",
            "Paris",
            "Los Angeles",
            "San Francisco",
            "London",
            "Chongqing",
            "Dhaka",
            "Wuhan",
            "Kolkata",
            "Mumbai",
            "Addis Ababa",
        ]
        actual_thumbnail_city_id_order = display_order_records.get("city_id", pd.Series(dtype=str)).astype(str).tolist()
        actual_thumbnail_city_order = display_order_records.get("city_name_en", pd.Series(dtype=str)).astype(str).tolist()
        exact_order_ok = (
            thumbnail_count_ok
            and actual_thumbnail_city_id_order == expected_thumbnail_city_id_order
            and actual_thumbnail_city_order == expected_thumbnail_city_order
        )
        add(
            "Fig3_road_thumbnail_high_low_exact_city_order",
            "pass" if exact_order_ok else "fail",
            json.dumps(
                {
                    "actual_city_ids": actual_thumbnail_city_id_order,
                    "expected_city_ids": expected_thumbnail_city_id_order,
                    "actual_city_names": actual_thumbnail_city_order,
                    "expected_city_names": expected_thumbnail_city_order,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "High 1-6 and Low 1-6 thumbnail city order matches the requested fixed sequence",
        )
        row_grouping_ok = (
            thumbnail_count_ok
            and bool(fig3_layout.get("road_thumbnail_two_group_rows"))
            and bool(fig3_layout.get("road_thumbnail_high_row_above_low"))
            and fig3_layout.get("road_thumbnail_group_row_labels", {}).get("low") == "Low quality cities"
            and fig3_layout.get("road_thumbnail_group_row_labels", {}).get("high") == "High quality cities"
            and fig3_layout.get("road_thumbnail_top_row_group") == "high"
            and fig3_layout.get("road_thumbnail_bottom_row_group") == "low"
        )
        add(
            "Fig3_road_thumbnail_row_order_high_above_low",
            "pass" if row_grouping_ok else "fail",
            json.dumps(
                {
                    "two_group_rows": fig3_layout.get("road_thumbnail_two_group_rows"),
                    "high_row_above_low": fig3_layout.get("road_thumbnail_high_row_above_low"),
                    "top_row_group": fig3_layout.get("road_thumbnail_top_row_group"),
                    "bottom_row_group": fig3_layout.get("road_thumbnail_bottom_row_group"),
                    "group_row_labels": fig3_layout.get("road_thumbnail_group_row_labels"),
                    "group_row_positions": fig3_layout.get("road_thumbnail_group_row_positions"),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "Top row is labeled High quality cities and bottom row is labeled Low quality cities",
        )
        badge_present = thumbnail_data.get(
            "road_thumbnail_score_badge_present", pd.Series(False, index=thumbnail_data.index)
        ).astype(str).str.lower().isin(["true", "1", "yes"])
        badge_text = thumbnail_data.get("road_thumbnail_score_badge_text", pd.Series("", index=thumbnail_data.index)).fillna("").astype(str)
        badge_numeric = pd.to_numeric(badge_text, errors="coerce")
        badge_ok = (
            thumbnail_count_ok
            and int(badge_present.sum()) == FIG3_ROAD_THUMBNAIL_TARGET_COUNT
            and badge_numeric.notna().all()
            and int(fig3_layout.get("road_thumbnail_badge_count", 0)) == FIG3_ROAD_THUMBNAIL_TARGET_COUNT
        )
        add(
            "Fig3_road_thumbnail_score_badges_numeric_12",
            "pass" if badge_ok else "fail",
            json.dumps(
                {
                    "badge_present_count": int(badge_present.sum()),
                    "badge_text": badge_text.tolist(),
                    "badge_numeric_non_null": int(badge_numeric.notna().sum()),
                    "layout_badge_count": fig3_layout.get("road_thumbnail_badge_count"),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "All 12 thumbnails have numeric quality score badges",
        )
        badge_color = (
            thumbnail_data.get("road_thumbnail_score_badge_color", pd.Series("", index=thumbnail_data.index))
            .fillna("")
            .astype(str)
            .str.upper()
        )
        strip_color = (
            thumbnail_data.get("road_thumbnail_component_strip_color", pd.Series("", index=thumbnail_data.index))
            .fillna("")
            .astype(str)
            .str.upper()
        )
        rank_group = thumbnail_data.get("road_thumbnail_rank_group", pd.Series("", index=thumbnail_data.index)).fillna("").astype(str)
        expected_group_colors = {key: value.upper() for key, value in FIG3_ROAD_THUMBNAIL_GROUP_COLORS.items()}
        expected_color_by_row = rank_group.map(expected_group_colors).fillna("")
        layout_group_colors = {
            str(key): str(value).upper()
            for key, value in fig3_layout.get("road_thumbnail_group_colors", {}).items()
        } if isinstance(fig3_layout.get("road_thumbnail_group_colors"), dict) else {}
        thumbnail_group_color_ok = (
            thumbnail_count_ok
            and expected_group_colors.get("low") == "#1D4ED8"
            and expected_group_colors.get("high") == "#B45309"
            and layout_group_colors.get("low") == "#1D4ED8"
            and layout_group_colors.get("high") == "#B45309"
            and badge_color.eq(expected_color_by_row).all()
            and strip_color.eq(expected_color_by_row).all()
        )
        add(
            "Fig3_road_thumbnail_group_colors_low_blue_high_orange",
            "pass" if thumbnail_group_color_ok else "fail",
            json.dumps(
                {
                    "expected_group_colors": expected_group_colors,
                    "layout_group_colors": layout_group_colors,
                    "badge_colors_by_group": {
                        group: sorted(badge_color[rank_group.eq(group)].unique().tolist()) for group in ["low", "high"]
                    },
                    "component_strip_colors_by_group": {
                        group: sorted(strip_color[rank_group.eq(group)].unique().tolist()) for group in ["low", "high"]
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "Bottom thumbnail group colors use Low=blue and High=orange for row labels, badges and component strips",
        )
        strip_present = thumbnail_data.get(
            "road_thumbnail_component_strip_present", pd.Series(False, index=thumbnail_data.index)
        ).astype(str).str.lower().isin(["true", "1", "yes"])
        strip_values_non_null = thumbnail_data.get(
            "road_thumbnail_component_strip_values_non_null", pd.Series(False, index=thumbnail_data.index)
        ).astype(str).str.lower().isin(["true", "1", "yes"])
        expected_component_fields = fig3_component_strip_fields()
        expected_component_field_text = ";".join(expected_component_fields)
        strip_field_text = thumbnail_data.get(
            "road_thumbnail_component_strip_fields", pd.Series("", index=thumbnail_data.index)
        ).fillna("").astype(str)
        component_non_null_by_field = {
            field: int(numeric(thumbnail_data.get(field, pd.Series(np.nan, index=thumbnail_data.index))).notna().sum())
            for field in expected_component_fields
        }
        component_strip_ok = (
            thumbnail_count_ok
            and int(strip_present.sum()) == FIG3_ROAD_THUMBNAIL_TARGET_COUNT
            and strip_values_non_null.all()
            and strip_field_text.eq(expected_component_field_text).all()
            and all(count == FIG3_ROAD_THUMBNAIL_TARGET_COUNT for count in component_non_null_by_field.values())
            and int(fig3_layout.get("road_thumbnail_component_strip_count", 0)) == FIG3_ROAD_THUMBNAIL_TARGET_COUNT
        )
        add(
            "Fig3_road_thumbnail_component_strips_present_non_null",
            "pass" if component_strip_ok else "fail",
            json.dumps(
                {
                    "strip_present_count": int(strip_present.sum()),
                    "strip_values_non_null_count": int(strip_values_non_null.sum()),
                    "expected_component_fields": expected_component_fields,
                    "component_non_null_by_field": component_non_null_by_field,
                    "strip_field_values": sorted(strip_field_text.unique().tolist()),
                    "layout_component_strip_count": fig3_layout.get("road_thumbnail_component_strip_count"),
                    "legend": fig3_layout.get("road_thumbnail_component_strip_legend"),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "All 12 thumbnails have compact R/H/P/L/B component strips with non-null Step07 component values",
        )
        low_thumbnail_data = thumbnail_data[
            thumbnail_data.get("road_thumbnail_rank_group", pd.Series("", index=thumbnail_data.index)).astype(str).eq("low")
        ].copy()
        low_thumbnail_data["_rank"] = numeric(
            low_thumbnail_data.get("road_thumbnail_rank", pd.Series(np.nan, index=low_thumbnail_data.index))
        )
        low_thumbnail_data["_score"] = numeric(
            low_thumbnail_data.get("quality_score_plot", pd.Series(np.nan, index=low_thumbnail_data.index))
        )
        low_rank_records = low_thumbnail_data.dropna(subset=["_rank", "_score"]).sort_values(["_rank", "city_id"])
        low_rank_sequence = low_rank_records["_rank"].astype(int).tolist()
        low_score_monotonic = bool(low_rank_records["_score"].diff().fillna(0).ge(-1e-9).all()) if len(low_rank_records) else False
        low_rank_ok = (
            len(low_rank_records) == len(low_thumbnail_data) == expected_low_count
            and low_rank_sequence == list(range(1, expected_low_count + 1))
            and low_score_monotonic
        )
        add(
            "Fig3_road_thumbnail_low_ranks_follow_quality_score",
            "pass" if low_rank_ok else "fail",
            json.dumps(
                {
                    "low_rank_order": [
                        {
                            "rank": int(row["_rank"]),
                            "city_id": str(row.get("city_id", "")),
                            "city_name_en": str(row.get("city_name_en", "")),
                            "quality_score": float(row["_score"]),
                        }
                        for _, row in low_rank_records.iterrows()
                    ],
                    "expected_low_count": expected_low_count,
                    "low_score_monotonic_by_rank": low_score_monotonic,
                    "selection_rule": FIG3_ROAD_THUMBNAIL_SELECTION_RULE,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "Low 1 is the lowest quality_score city; later Low ranks ascend by score",
        )
        high_thumbnail_data = thumbnail_data[
            thumbnail_data.get("road_thumbnail_rank_group", pd.Series("", index=thumbnail_data.index)).astype(str).eq("high")
        ].copy()
        high_thumbnail_data["_rank"] = numeric(
            high_thumbnail_data.get("road_thumbnail_rank", pd.Series(np.nan, index=high_thumbnail_data.index))
        )
        high_thumbnail_data["_score"] = numeric(
            high_thumbnail_data.get("quality_score_plot", pd.Series(np.nan, index=high_thumbnail_data.index))
        )
        high_rank_records = high_thumbnail_data.dropna(subset=["_rank", "_score"]).sort_values(["_rank", "city_id"])
        high_rank_sequence = high_rank_records["_rank"].astype(int).tolist()
        high_score_monotonic = bool(high_rank_records["_score"].diff().fillna(0).le(1e-9).all()) if len(high_rank_records) else False
        high_rank_ok = (
            len(high_rank_records) == len(high_thumbnail_data) == expected_high_count
            and high_rank_sequence == list(range(1, expected_high_count + 1))
            and high_score_monotonic
        )
        add(
            "Fig3_road_thumbnail_high_ranks_follow_quality_score",
            "pass" if high_rank_ok else "fail",
            json.dumps(
                {
                    "high_rank_order": [
                        {
                            "rank": int(row["_rank"]),
                            "city_id": str(row.get("city_id", "")),
                            "city_name_en": str(row.get("city_name_en", "")),
                            "quality_score": float(row["_score"]),
                        }
                        for _, row in high_rank_records.iterrows()
                    ],
                    "expected_high_count": expected_high_count,
                    "high_score_monotonic_by_rank": high_score_monotonic,
                    "selection_rule": FIG3_ROAD_THUMBNAIL_SELECTION_RULE,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "High 1 is the highest quality_score city; later High ranks descend by score",
        )
        extent_aspect = numeric(
            thumbnail_data.get("road_thumbnail_extent_aspect", pd.Series(np.nan, index=thumbnail_data.index))
        )
        extent_width = numeric(
            thumbnail_data.get("road_thumbnail_extent_width_km", pd.Series(np.nan, index=thumbnail_data.index))
        )
        extent_height = numeric(
            thumbnail_data.get("road_thumbnail_extent_height_km", pd.Series(np.nan, index=thumbnail_data.index))
        )
        square_axes_error = float(fig3_layout.get("road_thumbnail_axes_square_max_abs_aspect_error", np.nan))
        square_extent_error = float(fig3_layout.get("road_thumbnail_extent_square_max_abs_error_km", np.nan))
        square_tiles_ok = (
            thumbnail_count_ok
            and bool(fig3_layout.get("road_thumbnail_axes_square"))
            and bool(fig3_layout.get("road_thumbnail_extent_square"))
            and extent_aspect.notna().all()
            and (extent_aspect.sub(1.0).abs() <= 1e-6).all()
            and (extent_width.sub(extent_height).abs() <= 1e-6).all()
            and thumbnail_data.get("road_thumbnail_crop_shape", pd.Series("", index=thumbnail_data.index))
            .astype(str)
            .eq("square_local_km_window")
            .all()
        )
        add(
            "Fig3_road_thumbnail_tiles_square_aspect_equal",
            "pass" if square_tiles_ok else "fail",
            json.dumps(
                {
                    "layout_axes_square": fig3_layout.get("road_thumbnail_axes_square"),
                    "layout_axes_square_max_abs_aspect_error": square_axes_error,
                    "layout_extent_square": fig3_layout.get("road_thumbnail_extent_square"),
                    "layout_extent_square_max_abs_error_km": square_extent_error,
                    "source_extent_aspect_min": float(extent_aspect.min()) if len(extent_aspect) else np.nan,
                    "source_extent_aspect_max": float(extent_aspect.max()) if len(extent_aspect) else np.nan,
                    "crop_shapes": sorted(
                        thumbnail_data.get("road_thumbnail_crop_shape", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
                    ),
                    "coordinate_units": sorted(
                        thumbnail_data.get("road_thumbnail_coordinate_units", pd.Series(dtype=str))
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "Thumbnail axis boxes are square and road crops are square local-km windows",
        )
        fixed_half = numeric(
            thumbnail_data.get("road_thumbnail_fixed_crop_half_side_km", pd.Series(np.nan, index=thumbnail_data.index))
        )
        fixed_width = numeric(
            thumbnail_data.get("road_thumbnail_fixed_crop_width_km", pd.Series(np.nan, index=thumbnail_data.index))
        )
        fixed_height = numeric(
            thumbnail_data.get("road_thumbnail_fixed_crop_height_km", pd.Series(np.nan, index=thumbnail_data.index))
        )
        actual_half = numeric(thumbnail_data.get("road_thumbnail_half_side_km", pd.Series(np.nan, index=thumbnail_data.index)))
        unified_scale_ok = (
            thumbnail_count_ok
            and fixed_half.notna().all()
            and actual_half.notna().all()
            and fixed_width.notna().all()
            and fixed_height.notna().all()
            and np.isclose(fixed_half, FIG3_ROAD_THUMBNAIL_FIXED_HALF_SIDE_KM).all()
            and np.isclose(actual_half, FIG3_ROAD_THUMBNAIL_FIXED_HALF_SIDE_KM).all()
            and np.isclose(fixed_width, FIG3_ROAD_THUMBNAIL_FIXED_SIDE_KM).all()
            and np.isclose(fixed_height, FIG3_ROAD_THUMBNAIL_FIXED_SIDE_KM).all()
            and np.isclose(extent_width, FIG3_ROAD_THUMBNAIL_FIXED_SIDE_KM).all()
            and np.isclose(extent_height, FIG3_ROAD_THUMBNAIL_FIXED_SIDE_KM).all()
        )
        add(
            "Fig3_road_thumbnail_unified_crop_scale",
            "pass" if unified_scale_ok else "fail",
            json.dumps(
                {
                    "configured_fixed_half_side_km": FIG3_ROAD_THUMBNAIL_FIXED_HALF_SIDE_KM,
                    "configured_fixed_side_km": FIG3_ROAD_THUMBNAIL_FIXED_SIDE_KM,
                    "actual_half_side_values": sorted(actual_half.dropna().round(6).unique().tolist()),
                    "fixed_half_side_values": sorted(fixed_half.dropna().round(6).unique().tolist()),
                    "extent_width_values": sorted(extent_width.dropna().round(6).unique().tolist()),
                    "extent_height_values": sorted(extent_height.dropna().round(6).unique().tolist()),
                    "layout_fixed_half_side_values": fig3_layout.get("road_thumbnail_fixed_crop_half_side_values"),
                    "layout_fixed_width_values": fig3_layout.get("road_thumbnail_fixed_crop_width_values"),
                    "layout_fixed_height_values": fig3_layout.get("road_thumbnail_fixed_crop_height_values"),
                    "recentered_count": fig3_layout.get("road_thumbnail_recentered_count"),
                    "recentered_records": fig3_layout.get("road_thumbnail_recentered_records"),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            f"All 12 thumbnails use the same {FIG3_ROAD_THUMBNAIL_FIXED_SIDE_KM:g} km x {FIG3_ROAD_THUMBNAIL_FIXED_SIDE_KM:g} km local-km crop window",
        )
        scale_bar_present = thumbnail_data.get(
            "road_thumbnail_scale_bar_present", pd.Series(False, index=thumbnail_data.index)
        ).astype(str).str.lower().isin(["true", "1", "yes"])
        scale_bar_km = numeric(thumbnail_data.get("road_thumbnail_scale_bar_km", pd.Series(np.nan, index=thumbnail_data.index)))
        scale_bar_labels = thumbnail_data.get("road_thumbnail_scale_bar_label", pd.Series("", index=thumbnail_data.index)).fillna("").astype(str)
        scale_bar_ok = (
            thumbnail_count_ok
            and int(scale_bar_present.sum()) == FIG3_ROAD_THUMBNAIL_TARGET_COUNT
            and scale_bar_km.notna().all()
            and np.isclose(scale_bar_km, FIG3_ROAD_THUMBNAIL_SCALE_BAR_KM).all()
            and scale_bar_labels.eq(f"{FIG3_ROAD_THUMBNAIL_SCALE_BAR_KM:g} km").all()
            and int(fig3_layout.get("road_thumbnail_scale_bar_count", 0)) == FIG3_ROAD_THUMBNAIL_TARGET_COUNT
        )
        add(
            "Fig3_road_thumbnail_scale_bars_present",
            "pass" if scale_bar_ok else "fail",
            json.dumps(
                {
                    "scale_bar_present_count": int(scale_bar_present.sum()),
                    "scale_bar_km_values": sorted(scale_bar_km.dropna().round(6).unique().tolist()),
                    "scale_bar_labels": sorted(scale_bar_labels.unique().tolist()),
                    "layout_scale_bar_count": fig3_layout.get("road_thumbnail_scale_bar_count"),
                    "layout_scale_bar_km": fig3_layout.get("road_thumbnail_scale_bar_km"),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            f"All 12 thumbnails include a {FIG3_ROAD_THUMBNAIL_SCALE_BAR_KM:g} km scale bar",
        )
        selected_bg = thumbnail_data.get("road_thumbnail_background_color", pd.Series(dtype=str)).fillna("").astype(str)
        selected_roads = thumbnail_data.get("road_thumbnail_road_color", pd.Series(dtype=str)).fillna("").astype(str)
        selected_border = thumbnail_data.get("road_thumbnail_border_color", pd.Series(dtype=str)).fillna("").astype(str)
        thumbnail_style_ok = (
            thumbnail_count_ok
            and selected_bg.str.upper().eq(FIG3_ROAD_THUMBNAIL_BACKGROUND_COLOR).all()
            and selected_roads.str.upper().eq(FIG3_ROAD_THUMBNAIL_ROAD_COLOR).all()
            and selected_border.str.upper().eq(FIG3_ROAD_THUMBNAIL_BORDER_COLOR).all()
            and str(fig3_layout.get("road_thumbnail_background_color", "")).upper() == FIG3_ROAD_THUMBNAIL_BACKGROUND_COLOR
            and str(fig3_layout.get("road_thumbnail_road_color", "")).upper() == FIG3_ROAD_THUMBNAIL_ROAD_COLOR
            and str(fig3_layout.get("road_thumbnail_border_color", "")).upper() == FIG3_ROAD_THUMBNAIL_BORDER_COLOR
        )
        add(
            "Fig3_road_thumbnail_white_background_black_roads_border",
            "pass" if thumbnail_style_ok else "fail",
            json.dumps(
                {
                    "source_background_colors": sorted(selected_bg.unique().tolist()),
                    "source_road_colors": sorted(selected_roads.unique().tolist()),
                    "source_border_colors": sorted(selected_border.unique().tolist()),
                    "layout_background_color": fig3_layout.get("road_thumbnail_background_color"),
                    "layout_road_color": fig3_layout.get("road_thumbnail_road_color"),
                    "layout_border_color": fig3_layout.get("road_thumbnail_border_color"),
                    "style_note": fig3_layout.get("road_thumbnail_style_note"),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "All selected thumbnails use a white tile background, black road-network lines and black tile borders",
        )
        primary_area_ok = bool(fig3_layout.get("road_thumbnail_panel_primary_visual_area"))
        add(
            "Fig3_thumbnails_primary_visual_area",
            "pass" if primary_area_ok else "fail",
            json.dumps(
                {
                    "thumbnail_axes_area_fraction": fig3_layout.get("road_thumbnail_axes_area_fraction"),
                    "quality_explanation_area_fraction": fig3_layout.get("quality_explanation_area_fraction"),
                    "area_ratio": fig3_layout.get("road_thumbnail_to_quality_explanation_area_ratio"),
                    "required_min_ratio": FIG3_THUMBNAIL_PRIMARY_AREA_RATIO_MIN,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "Road-thumbnail axes occupy more figure area than the compact quality explanation band",
        )
        thumbnail_alignment_ok = bool(fig3_layout.get("road_thumbnail_panel_spans_wide_layout"))
        add(
            "Fig3_road_thumbnail_panel_spans_wide_layout",
            "pass" if thumbnail_alignment_ok else "fail",
            json.dumps(
                {
                    "layout_left": fig3_layout.get("road_thumbnail_layout_left"),
                    "layout_right": fig3_layout.get("road_thumbnail_layout_right"),
                    "thumbnail_panel_left": fig3_layout.get("road_thumbnail_panel_left"),
                    "thumbnail_panel_right": fig3_layout.get("road_thumbnail_panel_right"),
                    "left_delta": fig3_layout.get("road_thumbnail_panel_left_layout_delta"),
                    "right_delta": fig3_layout.get("road_thumbnail_panel_right_layout_delta"),
                    "tolerance": FIG3_ROAD_THUMBNAIL_ALIGNMENT_TOLERANCE,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "The 2x6 thumbnail group uses the wide Fig3 layout span after the world-map panel was removed",
        )
        thumbnail_spacing_ok = bool(fig3_layout.get("road_thumbnail_horizontal_spacing_expanded"))
        add(
            "Fig3_road_thumbnail_horizontal_spacing_expanded",
            "pass" if thumbnail_spacing_ok else "fail",
            json.dumps(
                {
                    "column_gap_target": fig3_layout.get("road_thumbnail_column_gap_target"),
                    "column_gap_min": fig3_layout.get("road_thumbnail_column_gap_min"),
                    "column_gap_max": fig3_layout.get("road_thumbnail_column_gap_max"),
                    "grid": fig3_layout.get("road_thumbnail_grid"),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            f"Thumbnail column gap is widened to at least {FIG3_ROAD_THUMBNAIL_COLUMN_GAP:g} figure fraction",
        )
        thumbnail_status = thumbnail_data.get("road_thumbnail_status", pd.Series(dtype=str)).fillna("").astype(str)
        thumbnail_source = thumbnail_data.get("road_thumbnail_source", pd.Series(dtype=str)).fillna("").astype(str)
        thumbnail_edges = numeric(thumbnail_data.get("road_thumbnail_edges_plotted", pd.Series(np.nan, index=thumbnail_data.index)))
        thumbnail_sources_ok = (
            thumbnail_count_ok
            and thumbnail_status.str.startswith("ok_gpkg_edges").all()
            and thumbnail_source.str.contains("Step02 drive GPKG", regex=False).all()
            and thumbnail_edges.gt(0).all()
        )
        add(
            "Fig3_road_thumbnail_sources_available",
            "pass" if thumbnail_sources_ok else "warn" if thumbnail_count_ok else "fail",
            json.dumps(
                {
                    "status_counts": thumbnail_status.value_counts(dropna=False).to_dict(),
                    "edge_count_min": float(thumbnail_edges.min()) if len(thumbnail_edges) else np.nan,
                    "edge_count_max": float(thumbnail_edges.max()) if len(thumbnail_edges) else np.nan,
                    "sources": thumbnail_source.tolist(),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "All 12 thumbnails read Step02 drive GPKG edges and plot at least one road segment",
        )
        definition_values = fig3_data.get("quality_score_definition", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
        source_step_values = fig3_data.get("quality_score_source_step", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
        not_morphology = fig3_data.get("quality_score_not_morphology_type_score", pd.Series(False, index=fig3_data.index)).astype(
            str
        ).str.lower().isin(["true", "1", "yes"]).all()
        not_official_osm = fig3_data.get("quality_score_not_official_osm_field", pd.Series(False, index=fig3_data.index)).astype(
            str
        ).str.lower().isin(["true", "1", "yes"]).all()
        definition_text = " ".join(definition_values)
        definition_ok = (
            bool(definition_values)
            and "project-derived" in definition_text
            and "Step07" in definition_text
            and "OSM" in definition_text
            and "road network integrity" in definition_text
            and "historical maturity" in definition_text
            and "POI completeness" in definition_text
            and "local coverage" in definition_text
            and "population/built-up support" in definition_text
            and "not an official OSM field" in definition_text
            and "not a pure road-morphology score" in definition_text
            and bool(source_step_values)
            and not_official_osm
            and not_morphology
            and "Step07" in str(fig3_layout.get("quality_score_definition", ""))
            and fig3_layout.get("quality_score_not_official_osm_field") is True
        )
        add(
            "Fig3_quality_score_definition_documented",
            "pass" if definition_ok else "fail",
            json.dumps(
                {
                    "definition_values": definition_values[:2],
                    "source_step_values": source_step_values,
                    "not_official_osm_field_all_true": bool(not_official_osm),
                    "not_morphology_type_score_all_true": bool(not_morphology),
                    "layout_definition": fig3_layout.get("quality_score_definition"),
                    "layout_not_official_osm_field": fig3_layout.get("quality_score_not_official_osm_field"),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "quality_score is documented as a project-derived Step07 OSM data quality/confidence score; it is not an official OSM field and not a pure road-morphology score",
        )
    palette_targets = [f"Fig{i}" for i in range(2, 9)]
    palette_missing = [
        figure_id
        for figure_id in palette_targets
        if (product_by_id(figure_id) is None)
        or not (
            str(product_by_id(figure_id).palette).strip()
            or str(product_by_id(figure_id).cmap).strip()
            or str(product_by_id(figure_id).color_variable).strip()
        )
    ]
    add(
        "Unified_palette_applied_to_figures_2_to_8",
        "pass" if not palette_missing else "fail",
        json.dumps(
            {
                p.figure_id: {
                    "palette": p.palette,
                    "cmap": p.cmap,
                    "heatmap_cmap": p.heatmap_cmap,
                    "color_variable": p.color_variable,
                }
                for p in products
                if p.figure_id in palette_targets
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "Fig2 records the categorical quality-tier palette; Fig3-Fig8 record continuous palette/cmap/color-variable metadata",
        ";".join(palette_missing),
    )
    fig6_panel_heatmap_cmaps = (
        fig6_data.groupby("panel")["fig7_panel_heatmap_cmap"].first().to_dict()
        if {"panel", "fig7_panel_heatmap_cmap"}.issubset(fig6_data.columns)
        else {}
    )
    fig6_panel_color_variables = (
        fig6_data.groupby("panel")["fig7_panel_color_variable"].first().to_dict()
        if {"panel", "fig7_panel_color_variable"}.issubset(fig6_data.columns)
        else {}
    )
    fig7_product_for_palette = product_by_id("Fig8")
    fig7_no_heatmap_current = bool(
        fig7_product_for_palette is not None
        and not str(fig7_product_for_palette.cmap).strip()
        and not str(fig7_product_for_palette.heatmap_cmap).strip()
        and fig7_data.get("fig9_panel_a_heatmap_current", pd.Series(False, index=fig7_data.index))
        .astype(str)
        .str.lower()
        .isin({"false", "0"})
        .all()
        and fig7_data.get("fig9_panel_a_colorbar_current", pd.Series(False, index=fig7_data.index))
        .astype(str)
        .str.lower()
        .isin({"false", "0"})
        .all()
    )
    heatmap_state = {
        "Fig4": {
            "cmap": (product_by_id("Fig4").cmap if product_by_id("Fig4") else ""),
            "heatmap_cmap": (product_by_id("Fig4").heatmap_cmap if product_by_id("Fig4") else ""),
            "expected_cmap": FIG5_HEATMAP_CMAP,
            "norm": f"TwoSlopeNorm vmin=-{TYPE_CENTER_SCORE_LIMIT:g}, vcenter=0, vmax={TYPE_CENTER_SCORE_LIMIT:g}",
            "color_variable": (product_by_id("Fig4").color_variable if product_by_id("Fig4") else ""),
        },
        "Fig6": {
            "cmap": (product_by_id("Fig6").cmap if product_by_id("Fig6") else ""),
            "heatmap_cmap": (product_by_id("Fig6").heatmap_cmap if product_by_id("Fig6") else ""),
            "expected_cmap": FIG6_FIG7_HEATMAP_CMAP,
            "norm": (
                f"{fig6_norm['panel_a_norm_summary']} "
                f"Panel B TwoSlopeNorm centered at 0 for signed transition deltas. {fig6_norm['panel_c_norm_summary']}"
            ),
            "panel_heatmap_cmaps": fig6_panel_heatmap_cmaps,
            "panel_color_variables": fig6_panel_color_variables,
            "panel_a_norm_vmax_formula": fig6_norm["panel_a_norm_vmax_formula"],
            "panel_a_norm_vmax_current": fig6_norm["panel_a_norm_vmax"],
            "panel_a_norm_data_max": fig6_norm["panel_a_norm_data_max"],
            "color_variable": (product_by_id("Fig6").color_variable if product_by_id("Fig6") else ""),
            "palette_source": "Fig6 A/B retain their shared RdBu_r palette; C receives the migrated former access-deviation heatmap",
            "colors_low_mid_high": f"A/B={FIG6_FIG7_HEATMAP_CMAP}; C={fig6_norm['panel_c_cmap']}; each panel has its own norm",
        },
        "Fig8": {
            "cmap": (product_by_id("Fig8").cmap if product_by_id("Fig8") else ""),
            "heatmap_cmap": (product_by_id("Fig8").heatmap_cmap if product_by_id("Fig8") else ""),
            "expected_cmap": "",
            "norm": "Current Fig8 has no heatmap; former access heatmap norm is recorded under Fig6C.",
            "color_variable": (product_by_id("Fig8").color_variable if product_by_id("Fig8") else ""),
            "palette_source": "Current Fig8 uses neutral interval styling and fixed morphotype marker colors",
            "colors_low_mid_high": "no current heatmap colorbar",
            "no_heatmap_current": fig7_no_heatmap_current,
        },
    }
    top_journal_heatmap_ok = (
        heatmap_state["Fig4"]["cmap"] == FIG5_HEATMAP_CMAP
        and heatmap_state["Fig4"]["heatmap_cmap"] == FIG5_HEATMAP_CMAP
        and heatmap_state["Fig6"]["cmap"] == FIG6_FIG7_HEATMAP_CMAP
        and fig6_panel_heatmap_cmaps.get(FIG6_PANEL_COMPOSITION) == FIG6_FIG7_HEATMAP_CMAP
        and fig6_panel_heatmap_cmaps.get(FIG6_PANEL_TRANSITION) == FIG6_FIG7_HEATMAP_CMAP
        and fig6_panel_heatmap_cmaps.get(FIG6_PANEL_PERFORMANCE) == fig6_norm["panel_c_cmap"]
        and fig7_no_heatmap_current
    )
    fig6_fig7_match_fig5_style = (
        top_journal_heatmap_ok
        and FIG6_FIG7_HEATMAP_CMAP == FIG5_HEATMAP_CMAP
        and "panel b twoslopenorm centered at 0" in heatmap_state["Fig6"]["norm"].lower()
        and FIG7_COLOR_VARIABLE in str(fig6_panel_color_variables.get(FIG6_PANEL_PERFORMANCE, ""))
        and fig7_no_heatmap_current
    )
    fig6_panel_ab_palette_ok = (
        heatmap_state["Fig6"]["cmap"] == FIG5_HEATMAP_CMAP
        and fig6_panel_heatmap_cmaps.get(FIG6_PANEL_COMPOSITION) == FIG5_HEATMAP_CMAP
        and fig6_panel_heatmap_cmaps.get(FIG6_PANEL_TRANSITION) == FIG5_HEATMAP_CMAP
        and FIG6_FIG7_HEATMAP_CMAP == FIG5_HEATMAP_CMAP
    )
    legacy_colormap_issues: list[str] = []
    for figure_id in ["Fig6", "Fig8"]:
        product = product_by_id(figure_id)
        if product is None:
            legacy_colormap_issues.append(f"{figure_id}:missing_product")
            continue
        product_text = " ".join(
            str(value)
            for value in [product.palette, product.cmap, product.heatmap_cmap, product.palette_note]
        )
        if re.search(r"(viridis|cividis)", product_text, flags=re.IGNORECASE):
            legacy_colormap_issues.append(f"{figure_id}:product_metadata")
        df = read_product_data(figure_id)
        if df.empty:
            legacy_colormap_issues.append(f"{figure_id}:missing_source_data")
            continue
        for col in ["palette", "cmap", "heatmap_cmap", "palette_note"]:
            if col in df.columns and df[col].astype(str).str.contains(r"viridis|cividis", case=False, na=False, regex=True).any():
                legacy_colormap_issues.append(f"{figure_id}:source_data:{col}")
    add(
        "Fig6_Fig8_no_cividis_or_viridis_colormap",
        "pass" if not legacy_colormap_issues else "fail",
        json.dumps({"heatmap_state": heatmap_state, "issues": legacy_colormap_issues}, ensure_ascii=False, sort_keys=True),
        "No Fig6/Fig8 product metadata or source-data palette fields contain cividis or viridis",
        "Fig6 records A/B/C heatmap palettes; current Fig8 records no heatmap palette.",
    )
    add(
        "Fig6_panel_A_B_heatmap_palette_consistent",
        "pass" if fig6_panel_ab_palette_ok else "fail",
        json.dumps(heatmap_state["Fig6"], ensure_ascii=False, sort_keys=True),
        f"Fig6 panel A and panel B both record heatmap_cmap/cmap={FIG5_HEATMAP_CMAP}",
        "Panel A keeps a nonnegative share norm; panel B keeps a zero-centered signed-delta norm.",
    )
    add(
        "Fig6C_migrated_heatmap_and_Fig8_no_heatmap_palette_recorded",
        "pass" if fig6_fig7_match_fig5_style else "fail",
        json.dumps(heatmap_state, ensure_ascii=False, sort_keys=True),
        {
            "Fig4": f"{FIG5_HEATMAP_CMAP} with TwoSlopeNorm centered at 0",
            "Fig6": f"A/B={FIG6_FIG7_HEATMAP_CMAP}; C={fig6_norm['panel_c_cmap']} for {FIG7_COLOR_VARIABLE}",
            "Fig8": "no current heatmap or colorbar",
        },
        "Former access heatmap palette/norm is now recorded as Fig6C; current Fig8 has no heatmap or colorbar.",
    )
    add(
        "Fig4_Fig6_heatmap_palette_and_Fig8_BC_palette_applied",
        "pass" if top_journal_heatmap_ok else "fail",
        json.dumps(heatmap_state, ensure_ascii=False, sort_keys=True),
        {
            "Fig4": FIG5_HEATMAP_CMAP,
            "Fig6": f"A/B={FIG6_FIG7_HEATMAP_CMAP}; C={fig6_norm['panel_c_cmap']}",
            "Fig8": "no current heatmap",
        },
        "Fig4 signed standardized scores and Fig6 A/B/C heatmaps use the expected diverging palettes; Fig8 A/B uses non-heatmap interval/marker encodings.",
    )
    morphotype_color_issues: list[str] = []
    morphotype_color_checked: list[str] = []
    for figure_id in ["Fig4", "Fig5", "Fig6", "Fig8", "Fig9", "Fig11"]:
        product = product_by_id(figure_id)
        if product is None:
            continue
        df = read_product_data(figure_id)
        if df.empty:
            morphotype_color_issues.append(f"{figure_id}:missing_source_data")
            continue
        if "morphotype" not in df.columns or "morphotype_color" not in df.columns:
            morphotype_color_issues.append(f"{figure_id}:missing_morphotype_color_field")
            continue
        rows = df[df["morphotype"].isin(MORPHOTYPES)].copy()
        expected_colors = rows["morphotype"].map(MORPHOTYPE_COLORS)
        mismatch_count = int(rows["morphotype_color"].fillna("").ne(expected_colors.fillna("")).sum())
        if mismatch_count:
            morphotype_color_issues.append(f"{figure_id}:{mismatch_count}_mismatch")
        else:
            morphotype_color_checked.append(figure_id)
    add(
        "Morphotype_colors_consistent",
        "pass" if not morphotype_color_issues else "fail",
        json.dumps({"checked": morphotype_color_checked, "issues": morphotype_color_issues}, ensure_ascii=False, sort_keys=True),
        MORPHOTYPE_COLORS,
        "Fig4/Fig5/Fig6/Fig8/Fig9/Fig11 source data reuse the same MTxx color map. Fig7 core-city dominant colors are checked separately through dominant_morphotype_color; Fig10 is city-level and does not encode morphotype color.",
    )
    fig7_required = {
        "panel",
        "fig9_panel_label",
        "morphotype",
        "category",
        "access_metric",
        "access_metric_name",
        "circuity_metric",
        "circuity_metric_name",
        "quadrant_label",
        "threshold_method",
        "morphotype_mean_access_share",
        "weighted_circuity",
        "p10_circuity",
        "p90_circuity",
        "fig9_panel_a_removed",
        "fig9_panel_a_heatmap_current",
        "fig9_panel_a_colorbar_current",
        "fig9_former_panel_a_migrated_to",
        "fig9_current_visual_panels",
        "fig9_bottom_note_current",
        "fig9_removed_bottom_note_text",
        "fig9_retained_visual_text_elements",
        "fig9_footnote_removal_note",
    }
    fig7_missing_required = sorted(fig7_required - set(fig7_data.columns))
    fig7_panel_counts = fig7_data.get("panel", pd.Series(dtype=str)).astype(str).value_counts().to_dict()
    access_panel = fig7_data[fig7_data.get("panel", pd.Series("", index=fig7_data.index)).astype(str).eq(FIG7_PANEL_ACCESS)].copy()
    detour_panel = fig7_data[fig7_data.get("panel", pd.Series("", index=fig7_data.index)).astype(str).eq(FIG7_PANEL_DETOUR)].copy()
    quadrant_panel = fig7_data[fig7_data.get("panel", pd.Series("", index=fig7_data.index)).astype(str).eq(FIG7_PANEL_QUADRANT)].copy()
    fig7_forbidden_a_fields = [
        field
        for field in [
            "fig9_layout_top_width_ratios_current",
            "fig9_layout_height_ratios_current",
            "fig9_layout_top_wspace",
            "fig9_layout_hspace",
            "fig9_layout_panel_a_transposed",
            "fig9_layout_panel_a_top_horizontal",
            "fig9_layout_panel_a_colorbar_side",
            "fig9_colorbar_label_previous",
            "fig9_colorbar_label_current",
            "fig9_colorbar_labelpad",
            "fig9_colorbar_clipping_fix",
            "fig9_panel_a_display_orientation",
        ]
        if field in fig7_data.columns
    ]
    fig7_panel_a_removed_all = (
        fig7_data.get("fig9_panel_a_removed", pd.Series(False, index=fig7_data.index))
        .astype(str)
        .str.lower()
        .isin({"true", "1", "yes"})
        .all()
    )
    fig7_a_heatmap_false = (
        fig7_data.get("fig9_panel_a_heatmap_current", pd.Series(True, index=fig7_data.index))
        .astype(str)
        .str.lower()
        .isin({"false", "0", "no"})
        .all()
    )
    fig7_a_colorbar_false = (
        fig7_data.get("fig9_panel_a_colorbar_current", pd.Series(True, index=fig7_data.index))
        .astype(str)
        .str.lower()
        .isin({"false", "0", "no"})
        .all()
    )
    fig7_panel_labels = sorted(fig7_data.get("fig9_panel_label", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    fig7_panel_ok = (
        not fig7_missing_required
        and fig7_panel_counts.get(FIG7_PANEL_ACCESS, 0) == 0
        and fig7_panel_counts.get(FIG7_PANEL_DETOUR, 0) == 10
        and fig7_panel_counts.get(FIG7_PANEL_QUADRANT, 0) == 10
        and detour_panel["morphotype"].nunique() == 10
        and quadrant_panel["morphotype"].nunique() == 10
        and quadrant_panel["quadrant_label"].notna().all()
        and quadrant_panel["threshold_method"].astype(str).eq(FIG7_THRESHOLD_METHOD).all()
        and fig7_panel_labels == list(FIG7_CURRENT_VISUAL_PANELS)
        and fig7_panel_a_removed_all
        and fig7_a_heatmap_false
        and fig7_a_colorbar_false
        and not fig7_forbidden_a_fields
    )
    add(
        "Fig8_current_panels_A_B_only_no_access_heatmap_colorbar",
        "pass" if fig7_panel_ok else "fail",
        json.dumps(
            {
                "panel_counts": fig7_panel_counts,
                "missing_fields": fig7_missing_required,
                "access_panel_rows": len(access_panel),
                "detour_panel_rows": len(detour_panel),
                "quadrant_panel_rows": len(quadrant_panel),
                "panel_labels": fig7_panel_labels,
                "panel_a_removed_all": bool(fig7_panel_a_removed_all),
                "panel_a_heatmap_current_all_false": bool(fig7_a_heatmap_false),
                "panel_a_colorbar_current_all_false": bool(fig7_a_colorbar_false),
                "forbidden_a_current_fields_present": fig7_forbidden_a_fields,
                "quadrant_labels": sorted(quadrant_panel.get("quadrant_label", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()),
                "threshold_methods": sorted(quadrant_panel.get("threshold_method", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "Current Fig8 source data has no access_panel rows, no current access heatmap/colorbar fields, and exactly Panels A/B as visual panels.",
    )
    fig7_layout = fig7_layout_metadata()
    fig7_layout_required = {
        "fig9_layout_grid_spec",
        "fig9_layout_width_ratios_previous",
        "fig9_layout_width_ratios_current",
        "fig9_layout_bottom_width_ratios_current",
        "fig9_layout_wspace",
        "fig9_layout_bottom_wspace",
        "fig9_savefig_pad_inches",
        "fig9_current_visual_panels",
        "fig9_panel_a_removed",
        "fig9_panel_a_heatmap_current",
        "fig9_panel_a_colorbar_current",
        "fig9_former_panel_a_migrated_to",
        "fig9_layout_panel_b_c_bottom_side_by_side",
        "fig9_layout_panel_b_y_axis_restored",
        "fig9_panel_b_y_axis_setting",
        "fig9_layout_note",
    }
    fig7_layout_missing = sorted(fig7_layout_required - set(fig7_data.columns))
    if not fig7_data.empty and not fig7_layout_missing:
        first_layout = fig7_data.iloc[0]
        def source_bool(field: str) -> bool:
            value = first_layout[field]
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in {"true", "1", "yes"}

        try:
            source_width_ratios = json.loads(str(first_layout["fig9_layout_width_ratios_current"]))
        except json.JSONDecodeError:
            source_width_ratios = {}
        try:
            source_bottom_width_ratios = json.loads(str(first_layout["fig9_layout_bottom_width_ratios_current"]))
        except json.JSONDecodeError:
            source_bottom_width_ratios = []
        source_wspace = float(first_layout["fig9_layout_wspace"])
        source_bottom_wspace = float(first_layout["fig9_layout_bottom_wspace"])
        source_panel_a_removed = source_bool("fig9_panel_a_removed")
        source_panel_a_heatmap_current = source_bool("fig9_panel_a_heatmap_current")
        source_panel_a_colorbar_current = source_bool("fig9_panel_a_colorbar_current")
        source_panel_b_c_bottom_side_by_side = source_bool("fig9_layout_panel_b_c_bottom_side_by_side")
        source_panel_b_y_axis_restored = source_bool("fig9_layout_panel_b_y_axis_restored")
        source_current_visual_panels = [part.strip() for part in str(first_layout["fig9_current_visual_panels"]).split(",") if part.strip()]
        source_former_panel_a_migrated_to = str(first_layout["fig9_former_panel_a_migrated_to"])
        source_savefig_pad_inches = float(first_layout["fig9_savefig_pad_inches"])
    else:
        source_width_ratios = {}
        source_bottom_width_ratios = []
        source_wspace = np.nan
        source_bottom_wspace = np.nan
        source_panel_a_removed = False
        source_panel_a_heatmap_current = True
        source_panel_a_colorbar_current = True
        source_panel_b_c_bottom_side_by_side = False
        source_panel_b_y_axis_restored = False
        source_current_visual_panels = []
        source_former_panel_a_migrated_to = ""
        source_savefig_pad_inches = np.nan
    fig7_layout_ok = (
        not fig7_layout_missing
        and source_width_ratios == fig7_layout["width_ratios_current"]
        and source_bottom_width_ratios == fig7_layout["bottom_width_ratios_current"]
        and math.isclose(source_wspace, fig7_layout["wspace_current"], rel_tol=0, abs_tol=1e-12)
        and math.isclose(source_bottom_wspace, fig7_layout["bottom_wspace_current"], rel_tol=0, abs_tol=1e-12)
        and source_current_visual_panels == fig7_layout["current_visual_panels"]
        and bool(source_panel_a_removed)
        and not bool(source_panel_a_heatmap_current)
        and not bool(source_panel_a_colorbar_current)
        and source_former_panel_a_migrated_to == fig7_layout["former_panel_a_migrated_to"]
        and math.isclose(source_savefig_pad_inches, fig7_layout["savefig_pad_inches"], rel_tol=0, abs_tol=1e-12)
        and bool(source_panel_b_c_bottom_side_by_side)
        and bool(source_panel_b_y_axis_restored)
        and bool(fig7_layout["panel_a_removed"])
        and not bool(fig7_layout["panel_a_heatmap_current"])
        and not bool(fig7_layout["panel_a_colorbar_current"])
        and bool(fig7_layout["panel_b_c_bottom_side_by_side"])
        and bool(fig7_layout["panel_b_y_axis_restored"])
    )
    add(
        "Fig8_layout_A_B_only_A_y_axis_recorded",
        "pass" if fig7_layout_ok else "fail",
        json.dumps(
            {
                "missing_source_fields": fig7_layout_missing,
                "source_width_ratios_current": source_width_ratios,
                "source_bottom_width_ratios_current": source_bottom_width_ratios,
                "script_width_ratios_previous": fig7_layout["width_ratios_previous"],
                "script_width_ratios_current": fig7_layout["width_ratios_current"],
                "source_wspace": source_wspace if np.isfinite(source_wspace) else None,
                "source_bottom_wspace": source_bottom_wspace if np.isfinite(source_bottom_wspace) else None,
                "current_visual_panels": source_current_visual_panels,
                "panel_a_removed": source_panel_a_removed,
                "panel_a_heatmap_current": source_panel_a_heatmap_current,
                "panel_a_colorbar_current": source_panel_a_colorbar_current,
                "former_panel_a_migrated_to": source_former_panel_a_migrated_to,
                "savefig_pad_inches": source_savefig_pad_inches if np.isfinite(source_savefig_pad_inches) else None,
                "panel_b_c_bottom_side_by_side": fig7_layout["panel_b_c_bottom_side_by_side"],
                "panel_b_y_axis_restored": fig7_layout["panel_b_y_axis_restored"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "Fig8 source data records the current A/B-only layout, no access heatmap/colorbar, and visible Panel A morphotype y-axis.",
        fig7_layout["overlap_controls"],
    )
    fig7_panel_c_required = {
        "fig9_panel_c_marker_size_previous",
        "fig9_panel_c_marker_size_current",
        "fig9_panel_c_marker_size_reduction_percent",
        "fig9_panel_c_marker_size_units",
        "fig9_panel_c_leader_line_labels",
        "fig9_panel_c_leader_line_count",
        "fig9_panel_c_leader_line_style",
        "fig9_panel_c_marker_label_note",
        "fig9_panel_c_label_offset_points",
        "fig9_panel_c_label_offset_revisions",
        "fig9_panel_c_label_adjustment_note",
        "fig9_panel_c_label_has_leader_line",
    }
    fig7_panel_c_missing = sorted(fig7_panel_c_required - set(fig7_data.columns))
    panel_c_source = fig7_data[fig7_data.get("panel", pd.Series(dtype=str)).astype(str).eq(FIG7_PANEL_QUADRANT)].copy()
    expected_leader_labels = sorted(FIG7_QUADRANT_LEADER_LINE_LABELS)
    expected_leader_style = fig7_layout["panel_c_leader_line_style"]
    if not fig7_panel_c_missing and not panel_c_source.empty:
        old_sizes = numeric(panel_c_source["fig9_panel_c_marker_size_previous"]).dropna()
        current_sizes = numeric(panel_c_source["fig9_panel_c_marker_size_current"]).dropna()
        reduction_values = numeric(panel_c_source["fig9_panel_c_marker_size_reduction_percent"]).dropna()
        source_leader_flags = (
            panel_c_source["fig9_panel_c_label_has_leader_line"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        )
        source_leader_labels = sorted(panel_c_source.loc[source_leader_flags, "morphotype"].astype(str).tolist())
        source_leader_count = int(numeric(panel_c_source["fig9_panel_c_leader_line_count"]).dropna().iloc[0])
        try:
            source_leader_style = json.loads(str(panel_c_source["fig9_panel_c_leader_line_style"].dropna().iloc[0]))
        except (IndexError, TypeError, json.JSONDecodeError):
            source_leader_style = {}
        try:
            source_offset_revisions = json.loads(str(panel_c_source["fig9_panel_c_label_offset_revisions"].dropna().iloc[0]))
        except (IndexError, TypeError, json.JSONDecodeError):
            source_offset_revisions = {}
        source_offsets = {}
        for _, source_row in panel_c_source.iterrows():
            try:
                source_offsets[str(source_row["morphotype"])] = json.loads(str(source_row["fig9_panel_c_label_offset_points"]))
            except (TypeError, json.JSONDecodeError):
                source_offsets[str(source_row.get("morphotype", ""))] = None
        source_adjustment_note = str(panel_c_source["fig9_panel_c_label_adjustment_note"].dropna().iloc[0])
        source_units = str(panel_c_source["fig9_panel_c_marker_size_units"].dropna().iloc[0])
        marker_size_ok = (
            len(old_sizes) == len(panel_c_source)
            and len(current_sizes) == len(panel_c_source)
            and len(reduction_values) == len(panel_c_source)
            and np.isclose(old_sizes.to_numpy(dtype=float), FIG7_PANEL_C_MARKER_SIZE_PREVIOUS).all()
            and np.isclose(current_sizes.to_numpy(dtype=float), FIG7_PANEL_C_MARKER_SIZE).all()
            and reduction_values.notna().all()
            and np.isclose(
                reduction_values.to_numpy(dtype=float),
                FIG7_PANEL_C_MARKER_SIZE_REDUCTION_PERCENT,
                rtol=0,
                atol=1e-12,
            ).all()
            and source_units == FIG7_PANEL_C_MARKER_SIZE_UNITS
        )
        leader_line_ok = (
            source_leader_labels == expected_leader_labels
            and source_leader_count == len(FIG7_QUADRANT_LEADER_LINE_LABELS)
            and source_leader_style == expected_leader_style
            and "MT04" in source_leader_labels
        )
        label_adjustment_ok = (
            source_offsets.get("MT03") == list(FIG7_QUADRANT_LABEL_OFFSETS["MT03"])
            and source_offsets.get("MT04") == list(FIG7_QUADRANT_LABEL_OFFSETS["MT04"])
            and source_offset_revisions == json.loads(json.dumps(FIG7_PANEL_C_LABEL_OFFSET_REVISIONS, sort_keys=True))
            and source_adjustment_note == FIG7_PANEL_C_LABEL_ADJUSTMENT_NOTE
        )
    else:
        old_sizes = pd.Series(dtype=float)
        current_sizes = pd.Series(dtype=float)
        reduction_values = pd.Series(dtype=float)
        source_leader_labels = []
        source_leader_count = 0
        source_leader_style = {}
        source_offsets = {}
        source_offset_revisions = {}
        source_adjustment_note = ""
        source_units = ""
        marker_size_ok = False
        leader_line_ok = False
        label_adjustment_ok = False
    add(
        "Fig8_panel_C_marker_size_and_leader_lines_recorded",
        "pass" if marker_size_ok and leader_line_ok else "fail",
        json.dumps(
            {
                "missing_source_fields": fig7_panel_c_missing,
                "marker_size_previous_expected": FIG7_PANEL_C_MARKER_SIZE_PREVIOUS,
                "marker_size_current_expected": FIG7_PANEL_C_MARKER_SIZE,
                "marker_size_reduction_percent_expected": FIG7_PANEL_C_MARKER_SIZE_REDUCTION_PERCENT,
                "marker_size_previous_source_unique": sorted(set(float(v) for v in old_sizes.tolist())),
                "marker_size_current_source_unique": sorted(set(float(v) for v in current_sizes.tolist())),
                "marker_size_reduction_percent_source_unique": sorted(set(round(float(v), 12) for v in reduction_values.tolist())),
                "marker_size_units_source": source_units,
                "leader_line_labels_expected": expected_leader_labels,
                "leader_line_labels_source": source_leader_labels,
                "leader_line_count_source": source_leader_count,
                "leader_line_style_source": source_leader_style,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        (
            f"Current Panel B marker area is reduced from {FIG7_PANEL_C_MARKER_SIZE_PREVIOUS:g} to "
            f"{FIG7_PANEL_C_MARKER_SIZE:g} points^2, and leader lines are recorded for "
            f"{', '.join(FIG7_QUADRANT_LEADER_LINE_LABELS)}."
        ),
        FIG7_PANEL_C_MARKER_LABEL_NOTE,
    )
    add(
        "Fig8_panel_C_MT03_MT04_label_adjustments_recorded",
        "pass" if label_adjustment_ok else "fail",
        json.dumps(
            {
                "missing_source_fields": fig7_panel_c_missing,
                "source_offsets": {key: source_offsets.get(key) for key in ["MT03", "MT04"]},
                "expected_offsets": {
                    key: list(FIG7_QUADRANT_LABEL_OFFSETS[key]) for key in ["MT03", "MT04"]
                },
                "source_offset_revisions": source_offset_revisions,
                "expected_offset_revisions": json.loads(json.dumps(FIG7_PANEL_C_LABEL_OFFSET_REVISIONS, sort_keys=True)),
                "source_adjustment_note": source_adjustment_note,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "Fig8 current Panel B records the MT03 right-side leader adjustment and MT04 shorter/lower leader adjustment.",
        FIG7_PANEL_C_LABEL_ADJUSTMENT_NOTE,
    )
    fig7_product = product_by_id("Fig8")
    fig7_svg_text = ""
    if fig7_product is not None and fig7_product.svg_path.exists():
        fig7_svg_text = fig7_product.svg_path.read_text(encoding="utf-8", errors="ignore")
    fig7_retained_svg_terms = [
        "A  Detour/circuity interval",
        "B  Access-detour quadrants",
        "Weighted OD circuity",
        "Mean access (%)",
        "Morphotype",
        "MT03",
        "MT04",
    ]
    fig7_source_bottom_false = (
        "fig9_bottom_note_current" in fig7_data.columns
        and fig7_data["fig9_bottom_note_current"].astype(str).str.strip().str.lower().isin({"false", "0", "no"}).all()
    )
    fig7_source_removed_text_ok = (
        "fig9_removed_bottom_note_text" in fig7_data.columns
        and sorted(fig7_data["fig9_removed_bottom_note_text"].dropna().astype(str).unique().tolist()) == [FIG7_REMOVED_BOTTOM_NOTE_TEXT]
    )
    fig7_removed_visual_text_ok = (
        bool(fig7_svg_text)
        and FIG7_REMOVED_BOTTOM_NOTE_TEXT not in fig7_svg_text
        and all(term in fig7_svg_text for term in fig7_retained_svg_terms)
        and fig7_source_bottom_false
        and fig7_source_removed_text_ok
    )
    add(
        "Fig8_bottom_note_removed_from_render",
        "pass" if fig7_removed_visual_text_ok else "fail",
        json.dumps(
            {
                "svg_exists": bool(fig7_product is not None and fig7_product.svg_path.exists()),
                "removed_bottom_note_absent_in_svg": FIG7_REMOVED_BOTTOM_NOTE_TEXT not in fig7_svg_text if fig7_svg_text else False,
                "retained_terms_present": {term: term in fig7_svg_text for term in fig7_retained_svg_terms} if fig7_svg_text else {},
                "source_bottom_note_current_false": bool(fig7_source_bottom_false),
                "source_removed_text_recorded": bool(fig7_source_removed_text_ok),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "Fig8 SVG has no bottom footnote/note; panel A/B titles, axes and MT03/MT04 label records remain.",
        FIG7_FOOTNOTE_REMOVAL_NOTE,
    )
    fig8_required = {
        "city_id",
        "category",
        "population_weighted_access_share",
        "accessibility_gini",
        "no_access_population_share",
        "level_note",
        "fig11_row_definition",
    }
    fig8_missing = sorted(fig8_required - set(fig8_data.columns))
    fig8_city_rows_ok = (
        len(fig8_data) == 240
        and not fig8_missing
        and fig8_data["city_id"].nunique() == 48
        and fig8_data["category"].nunique() == 5
        and fig8_data.get("group_level", pd.Series("city", index=fig8_data.index)).astype(str).eq("city").all()
    )
    add(
        "Fig10_city_inequality_rows",
        "pass" if fig8_city_rows_ok else "fail",
        json.dumps(
            {
                "rows": len(fig8_data),
                "city_count": int(fig8_data["city_id"].nunique()) if "city_id" in fig8_data.columns else 0,
                "category_count": int(fig8_data["category"].nunique()) if "category" in fig8_data.columns else 0,
                "missing_fields": fig8_missing,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "240 city-level rows: 48 cities x 5 facility categories",
    )
    fig10_data = read_product_data("Fig5")
    fig10_required_fields = {
        "axis_code",
        "score_metric_label",
        "score_value",
        "radar_radius",
        "score_metric_order",
        "metric_family",
        "family_label",
        "family_color",
        "metric_family_order",
    }
    fig10_rows_ok = (
        len(fig10_data) == 70
        and fig10_data.get("morphotype", pd.Series(dtype=str)).nunique() == 10
        and fig10_data.get("score_metric", pd.Series(dtype=str)).nunique() == 7
        and fig10_required_fields.issubset(fig10_data.columns)
    )
    if {"axis_code", "score_metric", "score_metric_label", "score_metric_order"}.issubset(fig10_data.columns):
        fig10_axis_mapping = (
            fig10_data[["axis_code", "score_metric", "score_metric_label", "score_metric_order"]]
            .drop_duplicates()
            .sort_values("score_metric_order")
        )
    else:
        fig10_axis_mapping = pd.DataFrame(columns=["axis_code", "score_metric", "score_metric_label", "score_metric_order"])
    expected_axis_codes = [RADAR_AXIS_CODES[m] for m in RADAR_SCORE_METRICS]
    observed_axis_codes = fig10_axis_mapping["axis_code"].dropna().astype(str).tolist()
    axis_code_counts = fig10_data.get("axis_code", pd.Series(dtype=str)).dropna().astype(str).value_counts().to_dict()
    fig10_axis_mapping_ok = (
        len(fig10_axis_mapping) == 7
        and observed_axis_codes == expected_axis_codes
        and fig10_axis_mapping["axis_code"].nunique() == 7
        and fig10_axis_mapping["score_metric"].nunique() == 7
        and fig10_axis_mapping["score_metric_label"].nunique() == 7
        and all(axis_code_counts.get(code, 0) == 10 for code in expected_axis_codes)
    )
    add(
        "Fig5_radar_rows",
        "pass" if fig10_rows_ok else "fail",
        json.dumps(
            {
                "rows": len(fig10_data),
                "morphotype_count": int(fig10_data.get("morphotype", pd.Series(dtype=str)).nunique()),
                "score_metric_count": int(fig10_data.get("score_metric", pd.Series(dtype=str)).nunique()),
                "columns": list(fig10_data.columns),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "Fig5 radar source data has 10 morphotypes x 7 Step09 score metrics",
    )
    add(
        "Fig5_axis_code_mapping",
        "pass" if fig10_axis_mapping_ok else "fail",
        json.dumps(
            {
                "mapping": fig10_axis_mapping[["axis_code", "score_metric", "score_metric_label"]].to_dict(orient="records"),
                "axis_code_counts": axis_code_counts,
                "expected_axis_codes": expected_axis_codes,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "Axis codes A-G uniquely map to the seven full Fig5 score metric labels and repeat once per morphotype",
    )
    if {"axis_code", "score_metric", "metric_family", "family_label", "family_color", "metric_family_order"}.issubset(fig10_data.columns):
        fig10_family_mapping = (
            fig10_data[["axis_code", "score_metric", "metric_family", "family_label", "family_color", "metric_family_order", "score_metric_order"]]
            .drop_duplicates()
            .sort_values("score_metric_order")
        )
    else:
        fig10_family_mapping = pd.DataFrame(
            columns=["axis_code", "score_metric", "metric_family", "family_label", "family_color", "metric_family_order", "score_metric_order"]
        )
    expected_family_by_metric = {
        metric: {
            "axis_code": RADAR_AXIS_CODES[metric],
            "metric_family": RADAR_METRIC_FAMILIES[metric]["metric_family"],
            "family_label": RADAR_METRIC_FAMILIES[metric]["family_label"],
            "family_color": RADAR_METRIC_FAMILIES[metric]["family_color"],
            "metric_family_order": RADAR_METRIC_FAMILIES[metric]["metric_family_order"],
        }
        for metric in RADAR_SCORE_METRICS
    }
    observed_family_by_metric = {
        str(row["score_metric"]): {
            "axis_code": str(row["axis_code"]),
            "metric_family": str(row["metric_family"]),
            "family_label": str(row["family_label"]),
            "family_color": str(row["family_color"]),
            "metric_family_order": int(row["metric_family_order"]) if pd.notna(row["metric_family_order"]) else None,
        }
        for _, row in fig10_family_mapping.iterrows()
    }
    family_code_counts = fig10_data.get("metric_family", pd.Series(dtype=str)).dropna().astype(str).value_counts().to_dict()
    fig10_family_mapping_ok = (
        len(fig10_family_mapping) == 7
        and fig10_family_mapping["metric_family"].nunique() == 3
        and set(fig10_family_mapping["metric_family"].dropna().astype(str)) == set(RADAR_FAMILY_ORDER)
        and observed_family_by_metric == expected_family_by_metric
        and family_code_counts.get("intensity_grain", 0) == 20
        and family_code_counts.get("order_hierarchy", 0) == 20
        and family_code_counts.get("friction_access_cost", 0) == 30
    )
    add(
        "Fig5_metric_family_mapping",
        "pass" if fig10_family_mapping_ok else "fail",
        json.dumps(
            {
                "mapping": fig10_family_mapping[
                    ["axis_code", "score_metric", "metric_family", "family_label", "family_color"]
                ].to_dict(orient="records"),
                "family_counts": family_code_counts,
                "family_axis_codes": {family: list(codes) for family, codes in RADAR_FAMILY_AXIS_CODES.items()},
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "Fig5 source data map seven A-G axes into three metric families for the outer domain ring",
    )
    fig10_product = product_by_id("Fig5")
    fig10_product_note = str(fig10_product.palette_note) if fig10_product is not None else ""
    fig10_title = str(fig10_product.title) if fig10_product is not None else ""
    fig10_nature_style_ok = (
        fig10_product is not None
        and fig10_title == "Morphotype radial / structural signatures"
        and "Nature-style radial signature" in fig10_product_note
        and "A-G codes" in fig10_product_note
        and "shared bottom legend" in fig10_product_note
        and "segmented outer domain bands" in fig10_product_note
        and "family-color domain key" in fig10_product_note
        and "title-free at the top" in fig10_product_note
    )
    add(
        "Fig5_nature_style_rendering_recorded",
        "pass" if fig10_nature_style_ok else "fail",
        json.dumps(
            {
                "title": fig10_title,
                "palette_note": fig10_product_note,
                "has_product": fig10_product is not None,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "Fig5 records Nature-style radial signature rendering with A-G axis codes and a shared bottom legend",
    )
    fig10_svg_text = ""
    fig10_svg_path = fig10_product.svg_path if fig10_product is not None else None
    if fig10_svg_path is not None and fig10_svg_path.exists():
        fig10_svg_text = fig10_svg_path.read_text(encoding="utf-8", errors="ignore")
    previous_axis_height = (FIG10_LAYOUT_TOP_PREVIOUS - FIG10_LAYOUT_BOTTOM) / (2 + FIG10_LAYOUT_HSPACE_PREVIOUS)
    current_axis_height = (FIG10_LAYOUT_TOP - FIG10_LAYOUT_BOTTOM) / (2 + FIG10_LAYOUT_HSPACE)
    previous_row_gap = previous_axis_height * FIG10_LAYOUT_HSPACE_PREVIOUS
    current_row_gap = current_axis_height * FIG10_LAYOUT_HSPACE
    fig10_title_layout_ok = (
        bool(fig10_svg_text)
        and FIG10_REMOVED_TOP_TITLE not in fig10_svg_text
        and FIG10_REMOVED_SUBTITLE not in fig10_svg_text
        and FIG10_REMOVED_BOTTOM_NOTE not in fig10_svg_text
        and "A-G metrics" in fig10_svg_text
        and "Domains" in fig10_svg_text
        and FIG10_LAYOUT_TOP > FIG10_LAYOUT_TOP_PREVIOUS
        and FIG10_LAYOUT_HSPACE < FIG10_LAYOUT_HSPACE_PREVIOUS
        and current_row_gap < previous_row_gap
        and FIG10_LEGEND_BOUNDS[3] < 0.132
    )
    add(
        "Fig5_title_area_removed_and_row_gap_tightened",
        "pass" if fig10_title_layout_ok else "fail",
        json.dumps(
            {
                "old_top_title_absent_in_svg": FIG10_REMOVED_TOP_TITLE not in fig10_svg_text if fig10_svg_text else False,
                "old_subtitle_absent_in_svg": FIG10_REMOVED_SUBTITLE not in fig10_svg_text if fig10_svg_text else False,
                "bottom_note_absent_in_svg": FIG10_REMOVED_BOTTOM_NOTE not in fig10_svg_text if fig10_svg_text else False,
                "metric_legend_present_in_svg": "A-G metrics" in fig10_svg_text if fig10_svg_text else False,
                "domain_legend_present_in_svg": "Domains" in fig10_svg_text if fig10_svg_text else False,
                "figure_width_cm": FIG10_FIGURE_WIDTH_CM,
                "figure_height_cm_previous": FIG10_FIGURE_HEIGHT_CM_PREVIOUS,
                "figure_height_cm_current": FIG10_FIGURE_HEIGHT_CM,
                "top_previous": FIG10_LAYOUT_TOP_PREVIOUS,
                "top_current": FIG10_LAYOUT_TOP,
                "top_margin_previous": 1 - FIG10_LAYOUT_TOP_PREVIOUS,
                "top_margin_current": 1 - FIG10_LAYOUT_TOP,
                "hspace_previous": FIG10_LAYOUT_HSPACE_PREVIOUS,
                "hspace_current": FIG10_LAYOUT_HSPACE,
                "row_gap_previous": previous_row_gap,
                "row_gap_current": current_row_gap,
                "row_gap_reduction": previous_row_gap - current_row_gap,
                "legend_bounds": FIG10_LEGEND_BOUNDS,
                "legend_height_previous": 0.132,
                "legend_height_current": FIG10_LEGEND_BOUNDS[3],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "Fig5 SVG omits the former top title/subtitle and the small bottom scale/domain note; the A-G metric legend and Domains legend remain, and the bottom legend panel is tightened.",
    )
    fig11_data = read_product_data("Fig9")
    fig11_policy = fig11_data.get("label_policy", pd.Series(dtype=str)).astype(str)
    fig11_rows_ok = (
        len(fig11_data) == 30
        and fig11_data.get("morphotype", pd.Series(dtype=str)).nunique() == 10
        and fig11_data.get("outcome", pd.Series(dtype=str)).nunique() == 3
        and {"observed_weighted_group_mean", "weight_sum", "label_policy"}.issubset(fig11_data.columns)
        and fig11_policy.str.contains("observed weighted group means", case=False, regex=False, na=False).all()
        and fig11_policy.str.contains("not adjusted marginal effects", case=False, regex=False, na=False).all()
    )
    add(
        "Fig9_observed_weighted_group_means_rows",
        "pass" if fig11_rows_ok else "fail",
        json.dumps(
            {
                "rows": len(fig11_data),
                "morphotype_count": int(fig11_data.get("morphotype", pd.Series(dtype=str)).nunique()),
                "outcome_count": int(fig11_data.get("outcome", pd.Series(dtype=str)).nunique()),
                "columns": list(fig11_data.columns),
                "label_policy_values": sorted(fig11_policy.dropna().unique().tolist()),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "Fig9 source data has 10 morphotypes x 3 outcomes and is labeled as observed weighted group means / group marginal summary, not adjusted marginal effects",
    )
    forbidden_pattern = r"Intercept|C\(city_id\)|fixed_effect|city FE"
    forbidden_rows = fig9_data[
        fig9_data.get("term", pd.Series("", index=fig9_data.index)).astype(str).str.contains(forbidden_pattern, case=False, regex=True, na=False)
        | fig9_data.get("term_kind", pd.Series("", index=fig9_data.index)).astype(str).str.contains("intercept|fixed_effect", case=False, regex=True, na=False)
    ]
    add("Fig11_no_intercept_fixed_effect_or_city_FE", "pass" if forbidden_rows.empty else "fail", len(forbidden_rows), 0)
    add("Fig11_MT01_reference_rows_present", "pass" if int(fig9_data.get("reference_flag", pd.Series(False)).fillna(False).sum()) >= 3 else "warn", int(fig9_data.get("reference_flag", pd.Series(False)).fillna(False).sum()), ">=3")
    canonical_manifest = manifest[manifest.get("record_role", pd.Series(dtype=str)).astype(str).eq("canonical")].copy()
    canonical_ids = set(canonical_manifest.get("figure_id", pd.Series(dtype=str)).astype(str))
    expected_main_sequence = [
        ("Fig1", f"Fig1_{FIG2_CANONICAL_SLUG}"),
        ("Fig2", "Fig2_osm_quality_confidence"),
        ("Fig3", f"Fig3_{FIG4_CANONICAL_SLUG}"),
        ("Fig4", "Fig4_morphotype_atlas_signatures"),
        ("Fig5", f"Fig5_{FIG6_RADAR_SLUG}"),
        ("Fig6", f"Fig6_{FIG7_CANONICAL_SLUG}"),
        ("Fig7", f"Fig7_{FIG8_CORE_CITY_SLUG}"),
        ("Fig8", f"Fig8_{FIG9_ACCESS_DETOUR_SLUG}"),
        ("Fig9", f"Fig9_{FIG10_OBSERVED_GROUP_MEANS_SLUG}"),
        ("Fig10", f"Fig10_{FIG11_CITY_INEQUALITY_SLUG}"),
        ("Fig11", f"Fig11_{FIG12_MODEL_COEFFICIENT_SLUG}"),
    ]
    expected_manifest_ids = [figure_id for figure_id, _ in expected_main_sequence]
    expected_manifest_stems = [stem for _, stem in expected_main_sequence]
    observed_manifest_ids = canonical_manifest.get("figure_id", pd.Series(dtype=str)).astype(str).tolist()
    observed_manifest_stems = [
        Path(str(value)).stem
        for value in canonical_manifest.get("png_path", pd.Series(dtype=str)).astype(str).tolist()
    ]
    manifest_fig1_to_fig11_ok = (
        len(manifest) == 11
        and len(canonical_manifest) == 11
        and canonical_ids == expected_main_ids
        and observed_manifest_ids == expected_manifest_ids
        and observed_manifest_stems == expected_manifest_stems
    )
    add(
        "figure_manifest_fig1_to_fig11_canonical_rows",
        "pass" if manifest_fig1_to_fig11_ok else "fail",
        json.dumps(
            {
                "rows": len(manifest),
                "canonical_rows": len(canonical_manifest),
                "canonical_ids": sorted(canonical_ids),
                "observed_order": observed_manifest_ids,
                "expected_order": expected_manifest_ids,
                "observed_png_stems": observed_manifest_stems,
                "expected_png_stems": expected_manifest_stems,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "11 canonical rows: Fig1-Fig11 in the continuous narrative order and filename slugs",
    )
    add("Step12_did_not_write_Step13_directory", "pass" if not step13_changed else "fail", len(step13_changed), 0, ";".join(step13_changed[:5]))
    if not step11_qc.empty and "status" in step11_qc.columns:
        warn_count = int(step11_qc["status"].eq("warn").sum())
        fail_count = int(step11_qc["status"].eq("fail").sum())
        add("Step11_QC_warnings_nonblocking", "warn" if warn_count else "pass", warn_count, 0, "Expected current Step11 state: one warn about small groups.")
        add("Step11_QC_failures_absent", "pass" if fail_count == 0 else "fail", fail_count, 0)
    return pd.DataFrame(records)


def write_readme(
    paths: StepPaths,
    args: argparse.Namespace,
    products: list[FigureProduct],
    qc: pd.DataFrame,
    fallback_notes: list[str],
    fig1_fig2_preservation: dict[str, Any],
    fig7_vector_preservation: dict[str, Any],
) -> None:
    main = [p for p in products if p.kind == "main"]
    supplemental = [p for p in products if p.kind == "supplemental"]
    qc_counts = qc["status"].value_counts(dropna=False).to_dict()
    protection_caveat = fig1_fig2_preservation.get("preservation_caveat", "")
    accepted_reference_available = fig1_fig2_preservation.get("accepted_reference_available", {})
    accepted_reference_missing = fig1_fig2_preservation.get("accepted_reference_missing", [])
    accepted_reference_mismatches = fig1_fig2_preservation.get("accepted_reference_mismatches", [])
    fig7_reference_available = fig7_vector_preservation.get("accepted_reference_available", {})
    fig7_reference_mismatches = fig7_vector_preservation.get("accepted_reference_mismatches", [])
    fig7_reference_missing = fig7_vector_preservation.get("accepted_reference_missing", [])
    fig7_protection_caveat = fig7_vector_preservation.get("preservation_caveat", "")
    fig6_norm = fig6_norm_metadata_from_products(products)
    fig6_visual = fig6_enrichment_metadata_from_products(products)
    fig7_layout = fig7_layout_metadata()
    lines = [
        f"# {STEP_NAME}",
        "",
        "This directory was generated by `code_upload/12_draw_paper_figures/step12_draw_paper_figures.py`.",
        "",
        "## Scope",
        "",
        "- Step12 generates paper figures, figure source data, QC, lineage, logs and this README.",
        "- Step12 does not generate Step13 paper tables, appendices A-F, variable dictionaries, method records or final Excel packages.",
        "",
        "## Run Command",
        "",
        f"```bash\npython code_upload/12_draw_paper_figures/step12_draw_paper_figures.py --root {args.root} --overwrite\n```",
        "",
        "## Main Figures",
        "",
        *[f"- {p.figure_id}: {p.title} (`{p.png_path.name}`, `{p.svg_path.name}`)" for p in main],
        "",
        "## Supplemental Figures",
        "",
        *([f"- {p.figure_id}: {p.title} (`{p.png_path.name}`, `{p.svg_path.name}`)" for p in supplemental] or ["- None generated."]),
        "",
        "## Fig6 Scale/Network and Performance Heatmaps",
        "",
        f"- Canonical Fig6 outputs are `Fig6_{FIG7_CANONICAL_SLUG}.png/svg/pdf` and `figure_data_Fig6_{FIG7_CANONICAL_SLUG}.csv`.",
        "- No Fig6 compatibility alias is generated in the current Step12 manifest.",
        f"- Fig6 source data contain `{FIG6_PANEL_COMPOSITION}`, `{FIG6_PANEL_TRANSITION}`, and `{FIG6_PANEL_PERFORMANCE}` panels. Composition cells are n_units-weighted mean morphotype shares; transition cells are target mean share minus source mean share in percentage points; performance-deviation cells are migrated access-performance heatmap rows.",
        f"- Fig6 panel A uses `{FIG6_FIG7_HEATMAP_CMAP}` with nonnegative `Normalize(vmin={compact_number(FIG6_PANEL_A_NORM_VMIN)}, {fig6_norm['panel_a_norm_vmax_formula']})`; current `data_max={compact_number(fig6_norm['panel_a_norm_data_max'])}` and current `vmax={compact_number(fig6_norm['panel_a_norm_vmax'])}`.",
        f"- Fig6 panel B uses the same `{FIG6_FIG7_HEATMAP_CMAP}` palette/cmap with a zero-centered signed-delta norm; A/B palette is shared, but A/B norm is not identical.",
        f"- Fig6 panel C uses `{fig6_norm['panel_c_cmap']}` with `{fig6_norm['panel_c_norm_summary']}` and colorbar label `{fig6_norm['panel_c_colorbar_label']}`.",
        f"- Fig6 panel C morphotype x-axis order is aligned with panels A/B as `{FIG6_MORPHOTYPE_AXIS_ORDER_TEXT}`.",
        f"- Fig6 layout choice: {FIG6_LAYOUT_CHOICE_NOTE}",
        f"- Fig6 top-cell annotations outline the top `{fig6_visual['top_cell_count']}` panel A cells by `mean_unit_share_percent` and the top `{fig6_visual['top_cell_count']}` panel B cells by `absolute_transition_delta_pp`; panel B outline color follows delta sign without adding a separate legend.",
        "- Fig6 current visual encodings are heatmap cell color and in-cell numeric labels for A/B/C, with top-cell outlines for A/B.",
        f"- Fig6 rendered canvas has no figure-level top title and no bottom footnote/note; retained text elements are `{fig6_visual['text_cleanup']['retained_visual_text_elements']}`.",
        "",
        "## Source Data",
        "",
        "Each figure has a matching `figure_data_*.csv` file.  `figure_data_lineage.csv` records the upstream input and transformation for each figure-data file.",
        "",
        "## Fig1 Protection",
        "",
        f"- Run-start preservation status: `{fig1_fig2_preservation.get('mode')}`; `preserved_from_run_start={fig1_fig2_preservation.get('preserved_from_run_start')}`.",
        f"- Protected file count: `{fig1_fig2_preservation.get('protected_copy_count')}/{fig1_fig2_preservation.get('protected_file_count')}`.",
        f"- Changed before restore: `{', '.join(fig1_fig2_preservation.get('changed_before_restore', [])) or 'none'}`.",
        f"- Restored from run-start snapshot: `{', '.join(fig1_fig2_preservation.get('restored', [])) or 'none'}`.",
        f"- Final mismatches: `{', '.join(fig1_fig2_preservation.get('final_mismatches', [])) or 'none'}`.",
        "- At Step12 start, existing Fig1 PNG/SVG/PDF/source CSV files are copied to a protected temporary directory; during the first renumbered run, missing run-start Fig1 files are accepted if regenerated as non-empty outputs.",
        f"- Step13 accepted-reference availability by protected file: `{json.dumps(accepted_reference_available, ensure_ascii=False, sort_keys=True)}`.",
        *(
            [
                f"- Preservation caveat: {protection_caveat}",
                f"- Current caveat labels: `{', '.join(accepted_reference_mismatches + accepted_reference_missing)}`.",
            ]
            if protection_caveat
            else ["- No Fig1 accepted-reference caveat was recorded."]
        ),
        "",
        "## Fig8 Vector Drift Guard",
        "",
        f"- Run-start preservation status: `{fig7_vector_preservation.get('mode')}`; `preserved_from_run_start={fig7_vector_preservation.get('preserved_from_run_start')}`.",
        f"- Protected vector file count: `{fig7_vector_preservation.get('protected_copy_count')}/{fig7_vector_preservation.get('protected_file_count')}`.",
        f"- Changed before restore: `{', '.join(fig7_vector_preservation.get('changed_before_restore', [])) or 'none'}`.",
        f"- Restored from run-start snapshot: `{', '.join(fig7_vector_preservation.get('restored', [])) or 'none'}`.",
        f"- Final mismatches: `{', '.join(fig7_vector_preservation.get('final_mismatches', [])) or 'none'}`.",
        f"- Step13 accepted-reference availability by Fig8 material file: `{json.dumps(fig7_reference_available, ensure_ascii=False, sort_keys=True)}`.",
        *(
            [
                f"- Preservation caveat: {fig7_protection_caveat}",
                f"- Current caveat labels: `{', '.join(fig7_reference_mismatches + fig7_reference_missing)}`.",
            ]
            if fig7_protection_caveat
            else ["- No Fig8 accepted-reference caveat was recorded."]
        ),
        f"- Drift scope: `{fig7_vector_preservation.get('drift_scope')}`; `non_target_vector_metadata_drift_accepted={fig7_vector_preservation.get('non_target_vector_metadata_drift_accepted')}`.",
        f"- {FIG7_VECTOR_DRIFT_GUARD_NOTE}",
        "",
        "## Unified Palette",
        "",
        "- Fig1 keeps its accepted design, data semantics and visual encodings.",
        f"- {FIG3_QUALITY_TIER_COMMUNICATION_NOTE}",
        f"- Non-target continuous score/intensity panels keep `{SEQUENTIAL_CMAP}` where a sequential palette was already used.",
        f"- Fig4 records `heatmap_cmap={FIG5_HEATMAP_CMAP}` with a zero-centered diverging norm for signed standardized type-center scores.",
        f"- Fig6 records panel-specific heatmap norms: panel A uses `{fig6_norm['panel_a_norm_vmax_formula']}` with current `vmax={compact_number(fig6_norm['panel_a_norm_vmax'])}`, panel B uses a zero-centered signed-delta norm, and panel C uses `{fig6_norm['panel_c_norm_summary']}`.",
        f"- Fig6 records morphotype heatmap x-axis order as `{FIG6_MORPHOTYPE_AXIS_ORDER_TEXT}` for A/B/C; C is aligned with A/B.",
        "- Current Fig8 records no heatmap cmap/colorbar; migrated access heatmap metadata are recorded under Fig6C.",
        f"- Morphotype classes use `{MORPHOTYPE_PALETTE_NAME}` consistently across Fig4/Fig5/Fig6/Fig8/Fig9/Fig11 source data and morphotype-colored marks; Fig7 uses the same palette for `dominant_morphotype` dumbbell points.",
        f"- Fig7 right-side metric strips use `{FIG12_HEATMAP_CMAP}` with colors `{FIG12_HEATMAP_CMAP_COLORS[0]} -> {FIG12_HEATMAP_CMAP_COLORS[-1]}` after within-Fig7 min-max normalization.",
        "- `figure_manifest.csv`, `figure_data_lineage.csv`, `run_log.json`, and figure source-data CSVs record palette/cmap/heatmap_cmap/color-variable metadata where relevant.",
        "",
        "## Fig8 Accessibility/Detour Encoding",
        "",
        "- Fig8 reads both `figure_data_accessibility_by_type.csv` and `figure_data_morphotype_circuity.csv` from Step11.",
        f"- Current Fig8 source data are a combined long table with `{FIG7_PANEL_DETOUR}` and `{FIG7_PANEL_QUADRANT}` panels only; `{FIG7_PANEL_ACCESS}` is not a current Fig8 visual panel.",
        f"- The migrated access heatmap is now Fig6C, where the access panel color encodes `{FIG7_COLOR_VARIABLE}` and uses colorbar label `{FIG7_COLORBAR_LABEL}`.",
        "- Panel A reports weighted OD circuity with p10-p90 distribution intervals.",
        f"- Panel B uses `{FIG7_THRESHOLD_METHOD}` and records `quadrant_label`, `access_metric`, and `circuity_metric` for each morphotype.",
        (
            f"- Fig8 layout uses `{fig7_layout['grid_spec']}` with A/B width ratios "
            f"`{fig7_layout['bottom_width_ratios_current']}` and `wspace={fig7_layout['bottom_wspace_current']}`."
        ),
        (
            "- Current Fig8 has no access heatmap or heatmap colorbar; Panels A/B are side-by-side, and "
            "Panel A keeps a visible morphotype y-axis with left spine, ticks and MTxx tick labels."
        ),
        (
            f"- Current Panel B marker area is reduced from `{fig7_layout['panel_c_marker_size_previous']:g}` to "
            f"`{fig7_layout['panel_c_marker_size_current']:g}` points^2 "
            f"({fig7_layout['panel_c_marker_size_reduction_percent']:.1f}% smaller), and leader lines are added for "
            f"`{', '.join(fig7_layout['panel_c_leader_line_labels'])}` "
            f"(n={fig7_layout['panel_c_leader_line_count']})."
        ),
        (
            "- Current Panel B leader lines use a thin neutral style: "
            f"`color={fig7_layout['panel_c_leader_line_style']['color']}`, "
            f"`linewidth={fig7_layout['panel_c_leader_line_style']['linewidth']}`, "
            f"`alpha={fig7_layout['panel_c_leader_line_style']['alpha']}`, "
            f"`arrowstyle={fig7_layout['panel_c_leader_line_style']['arrowstyle']}`."
        ),
        f"- {fig7_layout['panel_c_label_adjustment_note']}",
        "- Fig8 source data include `fig9_layout_*`, `fig9_panel_a_removed`, `fig9_panel_a_heatmap_current`, `fig9_panel_a_colorbar_current`, `fig9_panel_b_y_axis_setting`, `fig9_panel_c_marker_size_*`, `fig9_panel_c_leader_line_*`, and `fig9_panel_c_label_*` fields recording the current layout and overlap controls.",
        f"- Fig8 rendered canvas has no bottom footnote/note; retained text elements are `{fig7_layout['retained_visual_text_elements']}`.",
        "",
        "## Fig10 City-Level Inequality",
        "",
        "- Fig10 is promoted from the former supplemental city-level inequality overview and now uses Step11 `figure_data_inequality.csv` city-level rows.",
        "- Fig10 source data contain 240 city x facility-category rows with access share, accessibility Gini and no-access population share.",
        "",
        "## Fig5 Radar Signatures",
        "",
        "- Fig5 uses Step09 type-center score columns as a 10 morphotype x 7 score radar long table.",
        "- Fig5 is rendered as a Nature-style radial signature small-multiple plate with white background, subdued morphotype colors, polygonal grid rings, light area tint, clear outlines, point markers, small `max: A-G` annotations, and segmented outer domain bands.",
        f"- Fig5 has no top title or top subtitle in the rendered panel; the figure height is `{FIG10_FIGURE_HEIGHT_CM}` cm (previous `{FIG10_FIGURE_HEIGHT_CM_PREVIOUS}` cm), and the radar GridSpec uses `top={FIG10_LAYOUT_TOP}` (previous `{FIG10_LAYOUT_TOP_PREVIOUS}`) and `hspace={FIG10_LAYOUT_HSPACE}` (previous `{FIG10_LAYOUT_HSPACE_PREVIOUS}`) to reduce top whitespace and tighten the two radar rows.",
        f"- Fig5 removes the small bottom note (`{FIG10_REMOVED_BOTTOM_NOTE}`) and tightens the compact shared bottom legend panel.",
        "- Fig5 outer labels use short `A`-`G` axis codes only; the shared bottom metric legend maps each code to the full metric label.",
        "- Fig5 panels share a common `0-1` radial scale; the dashed `0.5` middle ring marks the zero-centered Step09 score after clipping/rescaling.",
        "- Fig5 outer domain-ring colors group metrics into three families: "
        + "; ".join(
            f"{RADAR_FAMILY_LABELS[family]} ({','.join(RADAR_FAMILY_AXIS_CODES[family])}, {RADAR_FAMILY_COLORS[family]})"
            for family in RADAR_FAMILY_ORDER
        )
        + ".",
        f"- Fig5 axis-code mapping: {'; '.join(f'{RADAR_AXIS_CODES[m]} = {RADAR_SCORE_LABELS[m]}' for m in RADAR_SCORE_METRICS)}.",
        "- Fig5 source data include `axis_code`, `score_metric`, `score_metric_label`, `score_metric_order`, `metric_family`, `family_label`, `family_color`, and `metric_family_order` for traceable A-G metric and domain-ring mapping.",
        "",
        "## Fig9 Observed Weighted Group Means",
        "",
        "- Fig9 uses Step11 `model_marginal_effects.csv` only as observed weighted group means / group marginal summary, not adjusted marginal effects.",
        "",
        "## Fig7 Core-City Overview",
        "",
        f"- Fig7 source data are `figure_data_Fig7_{FIG8_CORE_CITY_SLUG}.csv` with 96 rows: 48 core drive cities x `hex_1km` and `hex_2km`.",
        f"- Sorting rule: {FIG12_SORTING_RULE}",
        f"- Dominant share priority: {FIG12_DOMINANT_SHARE_PRIORITY}. The current source data record the actual `dominant_share_source` and fallback note per row.",
        "- Fig7 is rendered as one continuous 48-city ordered plate, not as two stacked 24-city blocks.",
        "- Left panel: dumbbell x-axis is dominant morphotype share (0-1); circles show `hex_1km`, triangles show `hex_2km`, and point color is dominant MT01-MT10 at that scale.",
        "- Cities whose dominant MT changes between `hex_1km` and `hex_2km` use a muted red connector and red city tick label; `share_delta` is `dominant_share_hex_2km - dominant_share_hex_1km`.",
        "- Right strips: Entropy uses the hex_1km `morphotype_entropy`; Access and Gini are Step11 city-level means across five facility categories; Circuity is Step10 `OD_detour_metrics.parquet` `group_level=city` `od_circuity_weighted_mean`; Quality is Step07 `quality_score`.",
        "- Strip values are min-max normalized across the 48 Fig7 core cities; higher Circuity and Gini are worse, while higher Access and Quality are better.",
        "",
        "## Fig1 Map and Size Encoding",
        "",
        f"- Fig1 is drawn as a world map with city centroids over a local Natural Earth-compatible basemap when available; otherwise the script uses a built-in coarse continent-outline fallback and records it in this README and `run_log.json`.",
        f"- Fig1 point area represents `{FIG2_SIZE_VARIABLE}` ({FIG2_SIZE_VARIABLE_LABEL}).",
        "- Fig1 source data include `ucdb_pop_2025`, `point_size`, `size_variable`, `size_variable_label`, `point_size_units`, `point_size_scale`, `quality_tier_color`, `city_number`, `city_label`, and `city_index_label`.",
        "- Fig1 color encodes `quality_tier`; marker shape encodes `sample_group`.",
        f"- Fig1 city numbers are assigned by this stable rule: {FIG2_CITY_NUMBER_RULE}",
        f"- Fig1 keeps the original lower panels (`Cities by region` and the encoding legend) and places a {FIG2_CITY_INDEX_COLUMNS}-column city number/name index below them.",
        "",
        "## Fig2 Quality Contrast and Road-Network Examples",
        "",
        f"- {FIG3_QUALITY_TIER_COMMUNICATION_NOTE}",
        f"- {FIG3_NO_MAP_FIELDS_NOTE}",
        "- The compact upper band retains `Quality score by tier` as a smaller auxiliary plot, moves the score-definition note below that plot, and enlarges the R/H/P/L/B component mini-chart with p10-p90 bands, all-city mean dots, Low/High 6 markers and 0/50/100 references. The component visual legend attaches each label directly to its own symbol, and the thumbnail grid is top-aligned to tighten the vertical gap below the quality panels.",
        f"- {FIG3_QUALITY_SCORE_DEFINITION}",
        f"- The road-thumbnail plate uses this selection rule: {FIG3_ROAD_THUMBNAIL_SELECTION_RULE}",
        f"- Road thumbnails are read from local Step02 `data/02_download_road_network_data/city_road_network_gpkg/drive/*_drive.gpkg` `edges` layers where available. All 12 selected thumbnails use the same fixed square {FIG3_ROAD_THUMBNAIL_FIXED_SIDE_KM:g} km x {FIG3_ROAD_THUMBNAIL_FIXED_SIDE_KM:g} km local-km crop window; if the fixed city-center window is sparse or strongly off-center, the crop can be recentered to local road-edge bounds without changing scale, and that decision is recorded in source data/QC.",
        "- Thumbnail axes are square and road segments are plotted in `local_km_from_thumbnail_center` coordinates, so the 2x6 road-network plate uses square tiles rather than rectangular lon/lat stretching.",
        f"- The 2x6 thumbnail group spans the wide Fig2 layout, with the upper row labeled `High quality cities` in orange and the lower row labeled `Low quality cities` in blue. Each tile keeps a white background, black road-network lines and a black tile border, and adds a colored numeric `quality_score` badge, a compact five-component strip (`{FIG3_QUALITY_COMPONENT_STRIP_LEGEND}`), and a {FIG3_ROAD_THUMBNAIL_SCALE_BAR_KM:g} km scale bar.",
        "- If a thumbnail cannot read local drive-road edges, Fig2 records a `road_thumbnail_status` fallback/placeholder reason in the source CSV, QC and run log instead of silently dropping the city.",
        "- Fig2 source data include `quality_score_plot`, `quality_tier_plot`, `quality_tier_color`, `color_variable`, `component_panel_fields`, `quality_score_definition`, `quality_score_not_official_osm_field`, `quality_score_not_morphology_type_score`, `road_thumbnail_included`, `road_thumbnail_display_order`, `road_thumbnail_rank_group`, `road_thumbnail_rank`, `road_thumbnail_source`, `road_thumbnail_status`, `road_thumbnail_extent`, `road_thumbnail_fixed_crop_width_km`, `road_thumbnail_fixed_crop_height_km`, `road_thumbnail_recentered`, `road_thumbnail_scale_bar_km`, `road_thumbnail_score_badge_text`, `road_thumbnail_score_badge_color`, `road_thumbnail_component_strip_color`, `road_thumbnail_component_strip_fields`, `road_thumbnail_component_strip_values`, `road_thumbnail_extent_width_km`, `road_thumbnail_extent_height_km`, `road_thumbnail_extent_aspect`, `road_thumbnail_crop_shape`, `road_thumbnail_coordinate_units`, `road_thumbnail_background_color`, `road_thumbnail_road_color`, and `road_thumbnail_border_color`.",
        "",
        "## Fig11 Model Coefficient Notes",
        "",
        *[f"- {note}" for note in FIG9_README_NOTES],
        "",
        "## QC Summary",
        "",
        f"- QC counts: {qc_counts}",
        "- Detailed checks are in `step12_quality_checks.csv`.",
        "",
        "## Fallbacks",
        "",
        *([f"- {note}" for note in fallback_notes if note] or ["- No figure fallback was required."]),
        "",
        "## Interpretation and Risk Notes",
        "",
        *[f"- {note}" for note in RISK_NOTES],
        "",
        "## Output Files",
        "",
        "- `step12_input_acceptance_report.json` / `step12_input_acceptance_report.md`",
        "- `step12_quality_checks.csv`",
        "- `figure_manifest.csv`",
        "- `figure_data_lineage.csv`",
        "- `run_log.json`",
        "- `reproducibility_manifest.json`",
        "- `figures/*.png`, `figures/*.svg`, `figures/*.pdf`",
        "",
    ]
    (paths.output_dir / OUTPUT_FILES["readme"]).write_text("\n".join(lines), encoding="utf-8")


def write_repro(
    paths: StepPaths,
    args: argparse.Namespace,
    script_path: Path,
    input_report: dict[str, Any],
    products: list[FigureProduct],
    qc: pd.DataFrame,
    fig1_fig2_preservation: dict[str, Any],
    fig7_vector_preservation: dict[str, Any],
) -> None:
    output_paths = [
        paths.output_dir / OUTPUT_FILES["input_report_json"],
        paths.output_dir / OUTPUT_FILES["input_report_md"],
        paths.output_dir / OUTPUT_FILES["qc"],
        paths.output_dir / OUTPUT_FILES["manifest"],
        paths.output_dir / OUTPUT_FILES["lineage"],
        paths.output_dir / OUTPUT_FILES["run_log"],
        paths.output_dir / OUTPUT_FILES["repro"],
        paths.output_dir / OUTPUT_FILES["readme"],
    ]
    output_paths.extend(all_product_output_paths(products))
    output_metadata = {}
    for p in output_paths:
        if not p.exists():
            continue
        meta = file_metadata(p)
        if p.name == OUTPUT_FILES["repro"]:
            meta.pop("top_level_keys", None)
        output_metadata[p.name] = meta
    fig6_norm = fig6_norm_metadata_from_products(products)
    fig6_visual = fig6_enrichment_metadata_from_products(products)
    fig7_layout = fig7_layout_metadata()

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
        "figure_count": len(products),
        "main_figure_count": len([p for p in products if p.kind == "main"]),
        "supplemental_figure_count": len([p for p in products if p.kind == "supplemental"]),
        "figure_aliases": collect_figure_aliases(products),
        "qc_status_counts": qc["status"].value_counts(dropna=False).to_dict(),
        "palette_scheme": palette_scheme_metadata(fig6_norm),
        "fig7_norm_metadata": fig6_norm,
        "fig7_visual_enrichment_metadata": fig6_visual,
        "fig9_layout_metadata": fig7_layout,
        "fig1_fig2_protection": fig1_fig2_preservation,
        "fig9_vector_protection": fig7_vector_preservation,
        "risk_notes": RISK_NOTES,
        "output_metadata": output_metadata,
    }
    write_json(paths.output_dir / OUTPUT_FILES["repro"], repro)


def check_existing_outputs(paths: StepPaths, overwrite: bool) -> None:
    if overwrite:
        return
    expected = [
        paths.output_dir / OUTPUT_FILES["input_report_json"],
        paths.output_dir / OUTPUT_FILES["qc"],
        paths.output_dir / OUTPUT_FILES["manifest"],
        paths.output_dir / OUTPUT_FILES["run_log"],
    ]
    existing = [p for p in expected if p.exists()]
    if existing:
        names = ", ".join(p.name for p in existing)
        raise FileExistsError(f"Step12 outputs already exist ({names}). Re-run with --overwrite.")


def main() -> None:
    start = time.time()
    configure_matplotlib()
    args = parse_args()
    try:
        paths = get_paths(args.root, args.output_dir)
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.figure_dir.mkdir(parents=True, exist_ok=True)
    check_existing_outputs(paths, args.overwrite)
    removed_obsolete_outputs = cleanup_obsolete_step12_outputs(paths)
    script_path = Path(__file__).resolve()
    step13_stability_before = wait_for_stable_directory(paths.step13_dir)
    step13_before = snapshot_dir(paths.step13_dir)
    protection_temp_dir = Path(tempfile.mkdtemp(prefix=".fig1_fig2_protect_", dir=paths.output_dir))
    fig1_fig2_protection = create_fig1_fig2_protection(paths, protection_temp_dir)
    fig7_vector_protection_temp_dir = Path(tempfile.mkdtemp(prefix=".fig7_vector_protect_", dir=paths.output_dir))
    fig7_vector_protection = create_fig7_vector_protection(paths, fig7_vector_protection_temp_dir)
    fig1_fig2_preservation: dict[str, Any] | None = None
    fig7_vector_preservation: dict[str, Any] | None = None
    try:
        specs = collect_input_specs(paths)
        input_records = input_metadata(specs)
        input_report = write_input_report(paths, input_records)
        if input_report["status"] != "pass":
            raise FileNotFoundError("Required Step12 inputs are missing; see step12_input_acceptance_report.json.")

        inputs = load_inputs(paths)
        centroids, map_fallback_note = city_centroids(paths, inputs)
        city_data = build_city_overview(inputs, centroids)

        products: list[FigureProduct] = []
        fig2_product, fig2_alignment = plot_fig2(paths, city_data, map_fallback_note)
        products.append(fig2_product)
        fig3_product, fig3_layout = plot_fig3(paths, city_data, inputs["local_types"], map_fallback_note)
        products.append(fig3_product)
        products.append(plot_fig4(paths, inputs["scale_signature"]))
        products.append(plot_fig5(paths, inputs["type_centers"]))
        radar_product = plot_fig10(paths, inputs["type_centers"])
        products.append(radar_product)
        scale_transition_product = plot_fig6(paths, inputs["city_profiles"], inputs["accessibility"])
        products.append(scale_transition_product)
        core_city_product = plot_fig12(paths, inputs["city_profiles"], inputs["inequality"], inputs["od_metrics"], inputs["city_quality"])
        products.append(core_city_product)
        access_detour_product = plot_fig7(paths, inputs["accessibility"], inputs["circuity"])
        products.append(access_detour_product)
        observed_product = plot_fig11(paths, inputs["group_means"])
        if observed_product is not None:
            products.append(observed_product)
        city_inequality_product = plot_fig8(paths, inputs["inequality"])
        products.append(city_inequality_product)
        model_coefficient_product = plot_fig9(paths, inputs["model"])
        products.append(model_coefficient_product)
        fig1_fig2_preservation = preserve_fig1_fig2_outputs(
            paths,
            fig1_fig2_protection,
            phase="after_generation",
        )
        fig7_vector_preservation = preserve_fig7_vector_outputs(
            paths,
            fig7_vector_protection,
            phase="after_generation",
        )

        manifest = products_to_manifest(products, paths.root)
        lineage = products_to_lineage(products)
        write_csv(manifest, paths.output_dir / OUTPUT_FILES["manifest"])
        write_csv(lineage, paths.output_dir / OUTPUT_FILES["lineage"])

        fig2_product = next(p for p in products if p.figure_id == "Fig1")
        fig2_data = pd.read_csv(fig2_product.data_path)
        fig3_data = pd.read_csv(fig3_product.data_path)
        fig9_access_detour_data = pd.read_csv(access_detour_product.data_path)
        fig11_city_inequality_data = pd.read_csv(city_inequality_product.data_path)
        fig12_model_data = pd.read_csv(model_coefficient_product.data_path)
        step13_after = snapshot_dir(paths.step13_dir)
        step13_changed = compare_snapshot(step13_before, step13_after)
        qc = build_qc(
            paths,
            input_report,
            products,
            manifest,
            lineage,
            fig2_data,
            fig3_data,
            fig9_access_detour_data,
            fig11_city_inequality_data,
            fig12_model_data,
            inputs["step11_qc"],
            step13_changed,
            fig2_alignment,
            fig3_layout,
            fig1_fig2_preservation,
            fig7_vector_preservation,
        )
        write_csv(qc, paths.output_dir / OUTPUT_FILES["qc"])
        fallback_notes = [p.fallback_note for p in products if p.fallback_note]
        write_readme(paths, args, products, qc, fallback_notes, fig1_fig2_preservation, fig7_vector_preservation)

        fig6_norm = fig6_norm_metadata_from_products(products)
        fig6_visual = fig6_enrichment_metadata_from_products(products)
        fig7_layout = fig7_layout_metadata()
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
            "input_report_status": input_report.get("status"),
            "main_figure_count": len([p for p in products if p.kind == "main"]),
            "supplemental_figure_count": len([p for p in products if p.kind == "supplemental"]),
            "figures": [json_ready(product.__dict__) for product in products],
            "figure_aliases": collect_figure_aliases(products),
            "removed_obsolete_outputs": removed_obsolete_outputs,
            "palette_scheme": palette_scheme_metadata(fig6_norm),
            "fig7_norm_metadata": fig6_norm,
            "fig7_visual_enrichment_metadata": fig6_visual,
            "fig9_layout_metadata": fig7_layout,
            "fig1_fig2_protection": fig1_fig2_preservation,
            "fig9_vector_protection": fig7_vector_preservation,
            "fig2_map_bottom_text_visual_edge_alignment": fig2_alignment,
            "fig2_city_index_left_alignment": fig2_alignment.get("city_index_left_alignment", {}),
            "fig3_quality_thumbnail_layout": fig3_layout,
            "fallback_notes": fallback_notes,
            "risk_notes": RISK_NOTES,
            "step11_qc_status_counts": inputs["step11_qc"]["status"].value_counts(dropna=False).to_dict()
            if not inputs["step11_qc"].empty and "status" in inputs["step11_qc"].columns
            else {},
            "qc_status_counts": qc["status"].value_counts(dropna=False).to_dict(),
            "step13_changed_paths": step13_changed,
            "step13_stability_before_snapshot": step13_stability_before,
            "output_metadata": {
                p.name: file_metadata(p)
                for p in [
                    paths.output_dir / OUTPUT_FILES["input_report_json"],
                    paths.output_dir / OUTPUT_FILES["input_report_md"],
                    paths.output_dir / OUTPUT_FILES["qc"],
                    paths.output_dir / OUTPUT_FILES["manifest"],
                    paths.output_dir / OUTPUT_FILES["lineage"],
                    paths.output_dir / OUTPUT_FILES["readme"],
                    *all_product_output_paths(products),
                ]
                if p.exists()
            },
        }
        write_json(paths.output_dir / OUTPUT_FILES["run_log"], run_log)
        write_repro(paths, args, script_path, input_report, products, qc, fig1_fig2_preservation, fig7_vector_preservation)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "output_dir": str(paths.output_dir),
                    "main_figures": len([p for p in products if p.kind == "main"]),
                    "supplemental_figures": len([p for p in products if p.kind == "supplemental"]),
                    "qc_status_counts": qc["status"].value_counts(dropna=False).to_dict(),
                    "duration_seconds": time.time() - start,
                },
                ensure_ascii=False,
            )
        )
    finally:
        if fig1_fig2_preservation is None:
            preserve_fig1_fig2_outputs(paths, fig1_fig2_protection, phase="finally")
        if fig7_vector_preservation is None:
            preserve_fig7_vector_outputs(paths, fig7_vector_protection, phase="finally")
        shutil.rmtree(protection_temp_dir, ignore_errors=True)
        shutil.rmtree(fig7_vector_protection_temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
