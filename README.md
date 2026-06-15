# Quality-Aware Multiscale Street-Network Typologies

Code accompanying the manuscript **"Quality-aware multiscale street-network typologies from OpenStreetMap: Global urban signatures and accessibility validation"**.

The project builds a reproducible workflow for comparing OpenStreetMap-derived street-network morphology across 86 urban centers. It combines fixed urban-center boundaries, cleaned drive and walk networks, population and built-up support data, OSM/Overture facility layers, OSM history summaries, multiscale morphometric indicators, quality tiers, local morphotypes, and accessibility/circuity validation.

## Repository Contents

The workflow is organized as numbered pipeline steps:

| Step | Directory | Purpose |
| --- | --- | --- |
| 00 | `00_environment_versions_and_data_inventory` | Environment notes and source download management. |
| 01 | `01_city_boundaries_and_sample_inventory` | City master table and urban-center boundary preparation. |
| 02 | `02_download_road_network_data` | OSM road-network download/build management. |
| 03 | `03_download_population_built_environment_data` | Population and built-environment raster preparation. |
| 04 | `04_download_facilities_quality_history_data` | Facility, quality, and OSM history data preparation. |
| 05 | `05_build_multiscale_spatial_units` | Full-city, core, and hexagonal spatial unit construction. |
| 06 | `06_clean_road_networks_compute_morphology_metrics` | Road-network cleaning and morphology metric calculation. |
| 07 | `07_generate_quality_scores_type_confidence` | City quality scores and unit-level type-confidence layers. |
| 08 | `08_generate_scale_signatures_city_profiles` | Scale signatures and city profile tables. |
| 09 | `09_cluster_morphotypes` | Local morphotype clustering and projection. |
| 10 | `10_compute_accessibility_detour_validation_metrics` | Facility accessibility and OD detour validation metrics. |
| 11 | `11_statistical_models_robustness_checks` | Explanatory models and robustness checks. |
| 12 | `12_draw_paper_figures` | Manuscript figure generation. |
| 13 | `13_export_paper_tables_appendices` | Manuscript tables, appendices, and export manifests. |

Each step contains a Python script and/or a companion Jupyter notebook. The notebooks document the intended execution sequence, while the Python scripts contain the reproducible processing logic.

## Data

The repository contains code only. Source data and generated outputs are not included.

The workflow expects a project layout with a sibling `data/` directory at the project root. The data inputs referenced by the code include:

- OpenStreetMap road-network extracts and history summaries
- GHS Urban Centre Database boundaries
- GHSL population and built-surface layers
- WorldPop and settlement-support layers where available
- OSM and Overture Places facility layers

Large intermediate files, generated figures, exported manuscript tables, caches, and local notebook checkpoints are intentionally excluded from version control.

## Environment

The scripts are Python-based and use geospatial and scientific-computing packages. Core dependencies include:

- `pandas`
- `geopandas`
- `numpy`
- `shapely`
- `rasterio`
- `pyproj`
- `osmnx`
- `osmium`
- `networkx`
- `scikit-learn`
- `statsmodels`
- `matplotlib`
- `seaborn`
- `pyarrow`
- `openpyxl`
- `requests`

Exact package versions should be recorded with the project environment used to reproduce the manuscript outputs.

## Usage

Run steps from the project root, keeping this repository at `code_upload/` and the data directory at `data/`.

Example:

```bash
python code_upload/09_cluster_morphotypes/step09_cluster_morphotypes.py --root <PROJECT_ROOT>
```

The full workflow is sequential. Earlier steps create the boundary, data, network, spatial-unit, metric, and quality inputs required by later modeling, figure, and appendix steps.

Notebook cells are provided for documentation and interactive inspection. They should be executed only after the corresponding data inputs are available.

## Manuscript Scope

The code supports the manuscript's quality-aware, multiscale design:

- 86 urban centers: 80 global-sample cities and six China pressure-test cities
- city-level OSM quality tiers and unit-level type confidence
- local morphotypes `MT01` through `MT10`
- multiscale street-network signatures at whole-city, core, 1-km, and 2-km scales
- validation against population-weighted 15-minute facility accessibility and OD circuity
- robustness checks for confidence filtering, weighting, and small-group sensitivity

The results should be interpreted as explanatory associations, not causal estimates.

## Citation

If you use this code, cite the associated manuscript:

> Quality-aware multiscale street-network typologies from OpenStreetMap: Global urban signatures and accessibility validation.

## License

No license is declared in this repository yet. Add a license before public reuse or redistribution.
