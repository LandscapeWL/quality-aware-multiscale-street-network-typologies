from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

try:
    from unidecode import unidecode
except ImportError:  # pragma: no cover - current environment has unidecode.
    import unicodedata

    def unidecode(value: str) -> str:
        return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")


ROOT = Path(__file__).resolve().parents[2]

SAMPLE_DIR = ROOT / "data/01_city_boundaries_and_sample_inventory/city_sample"
CANDIDATE_CSV = SAMPLE_DIR / "candidate_city_sample_80_plus_6.csv"
UCDB_GPKG = (
    ROOT
    / "data/01_city_boundaries_and_sample_inventory/GHS_UCDB_official_raw_files/GHS_UCDB_GLOBE_R2024A_V1_1/GHS_UCDB_GLOBE_R2024A.gpkg"
)
BOUNDARY_LAYER = "GHS_UCDB_THEME_GENERAL_CHARACTERISTICS_GLOBE_R2024A"

CITY_MASTER_CSV = SAMPLE_DIR / "city_master.csv"
CITY_MASTER_PARQUET = SAMPLE_DIR / "city_master.parquet"
MATCH_DIAGNOSTIC_CSV = SAMPLE_DIR / "city_match_diagnostics.csv"
MATCH_REVIEW_CSV = SAMPLE_DIR / "city_match_manual_review.csv"
CITY_BOUNDARY_GPKG = SAMPLE_DIR / "city_boundaries.gpkg"
CITY_BOUNDARY_LAYER = "city_boundaries"

ID_FIELD = "ID_UC_G0"
UCDB_FIELD_MAP = {
    "name_main": "GC_UCN_MAI_2025",
    "name_list": "GC_UCN_LIS_2025",
    "country_gad": "GC_CNT_GAD_2025",
    "country_unn": "GC_CNT_UNN_2025",
    "area": "GC_UCA_KM2_2025",
    "population": "GC_POP_TOT_2025",
}

# Standard country-name aliases needed because UCDB uses a mix of GADM and UN names.
COUNTRY_ALIASES_BY_ISO3 = {
    "CIV": ["Côte d'Ivoire", "Cote d Ivoire", "Ivory Coast"],
    "CZE": ["Czechia", "Czech Republic"],
    "KOR": ["South Korea", "Republic of Korea", "Korea Republic of"],
    "TUR": ["Turkey", "Türkiye", "Turkiye"],
    "USA": ["United States", "United States of America"],
    "VNM": ["Vietnam", "Viet Nam"],
}

# City-level manual aliases. These are reported in the diagnostic table.
MANUAL_CITY_ALIASES = {
    ("IND", "Delhi"): ["New Delhi"],
    ("USA", "Washington DC"): ["Washington"],
}


def clean_field_name(value: str) -> str:
    return value.lstrip("\ufeff")


def clean_text(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("\ufeff", "").strip()
    return value


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = unidecode(str(clean_text(value))).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def token_sort_key(value: Any) -> str:
    return " ".join(sorted(normalize_text(value).split()))


def similarity_score(left: Any, right: Any) -> float:
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if not left_norm or not right_norm:
        return 0.0
    direct = SequenceMatcher(None, left_norm, right_norm).ratio()
    token_sorted = SequenceMatcher(None, token_sort_key(left_norm), token_sort_key(right_norm)).ratio()
    return round(max(direct, token_sorted) * 100, 1)


def detect_column(df: pd.DataFrame, preferred: list[str], contains_all: tuple[str, ...] = ()) -> str | None:
    normalized_to_original = {normalize_text(col): col for col in df.columns}
    for name in preferred:
        found = normalized_to_original.get(normalize_text(name))
        if found is not None:
            return found
    if contains_all:
        for col in df.columns:
            norm_col = normalize_text(col)
            if all(token in norm_col for token in contains_all):
                return col
    return None


def detect_candidate_columns(candidates: pd.DataFrame) -> dict[str, str | None]:
    city_col = detect_column(
        candidates,
        ["city_name_en", "city_name", "city", "english_city_name", "name"],
        ("city", "name"),
    )
    country_col = detect_column(candidates, ["country", "country_name", "country_name_en"], ("country",))
    iso3_col = detect_column(candidates, ["iso3", "iso_a3", "country_iso3"], ("iso",))
    city_id_col = detect_column(candidates, ["city_id", "sample_id"], ("id",))

    ucdb_id_col = None
    for col in candidates.columns:
        norm_col = normalize_text(col)
        if ("ucdb" in norm_col or "ghs" in norm_col) and "id" in norm_col:
            ucdb_id_col = col
            break

    missing = [
        label
        for label, col in {"city name": city_col, "country": country_col}.items()
        if col is None
    ]
    if missing:
        raise ValueError(f"Cannot detect required candidate column(s): {', '.join(missing)}")

    return {
        "city": city_col,
        "country": country_col,
        "iso3": iso3_col,
        "city_id": city_id_col,
        "ucdb_id": ucdb_id_col,
    }


def load_candidates() -> tuple[pd.DataFrame, dict[str, str | None]]:
    candidates = pd.read_csv(CANDIDATE_CSV)
    candidates = candidates.copy()
    candidates.columns = [clean_field_name(col) for col in candidates.columns]
    for col in candidates.select_dtypes(include="object").columns:
        candidates[col] = candidates[col].map(clean_text)
    return candidates, detect_candidate_columns(candidates)


def load_ucdb(ignore_geometry: bool = False) -> gpd.GeoDataFrame | pd.DataFrame:
    ucdb = gpd.read_file(UCDB_GPKG, layer=BOUNDARY_LAYER, ignore_geometry=ignore_geometry)
    ucdb = ucdb.rename(columns={col: clean_field_name(col) for col in ucdb.columns})
    for col in ucdb.select_dtypes(include="object").columns:
        ucdb[col] = ucdb[col].map(clean_text)
    required = [ID_FIELD, *UCDB_FIELD_MAP.values()]
    missing = [field for field in required if field not in ucdb.columns]
    if missing:
        raise ValueError(f"UCDB boundary layer missing field(s): {missing}")
    return ucdb


def split_ucdb_names(row: pd.Series) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    main = clean_text(row[UCDB_FIELD_MAP["name_main"]])
    if main:
        entries.append({"name": str(main), "source": "ucdb_main"})
    name_list = clean_text(row[UCDB_FIELD_MAP["name_list"]])
    if name_list:
        for value in str(name_list).split(";"):
            value = value.strip()
            if value:
                entries.append({"name": value, "source": "ucdb_name_list"})

    expanded: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        name = entry["name"]
        variants = [name]
        bracket_values = re.findall(r"\[([^\]]+)\]", name)
        if bracket_values:
            variants.append(re.sub(r"\[[^\]]+\]", " ", name).strip())
            variants.extend(bracket_values)
        for variant in variants:
            key = (normalize_text(variant), entry["source"])
            if key[0] and key not in seen:
                seen.add(key)
                expanded.append({"name": variant, "source": entry["source"]})
    return expanded


def country_aliases(candidate: pd.Series, columns: dict[str, str | None]) -> set[str]:
    aliases: set[str] = set()
    country_col = columns["country"]
    iso3_col = columns["iso3"]
    if country_col:
        aliases.add(str(candidate[country_col]))
    if iso3_col and not pd.isna(candidate[iso3_col]):
        aliases.update(COUNTRY_ALIASES_BY_ISO3.get(str(candidate[iso3_col]).upper(), []))
    return {normalize_text(alias) for alias in aliases if normalize_text(alias)}


def city_variants(candidate: pd.Series, columns: dict[str, str | None]) -> list[dict[str, str]]:
    city_col = columns["city"]
    iso3_col = columns["iso3"]
    city_name = str(candidate[city_col])
    iso3 = "" if iso3_col is None or pd.isna(candidate[iso3_col]) else str(candidate[iso3_col]).upper()

    variants = [{"name": city_name, "source": "original"}]
    if re.search(r"\s*[-/]\s*", city_name):
        for part in re.split(r"\s*[-/]\s*", city_name):
            part = part.strip()
            if part:
                variants.append({"name": part, "source": "derived_split_alias"})

    for alias in MANUAL_CITY_ALIASES.get((iso3, city_name), []):
        variants.append({"name": alias, "source": "manual_city_alias"})

    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for variant in variants:
        key = normalize_text(variant["name"])
        if key and key not in seen:
            seen.add(key)
            deduped.append(variant)
    return deduped


def method_for_exact_match(variant_source: str, name_source: str) -> str:
    if variant_source == "manual_city_alias":
        return "manual_city_alias"
    if variant_source == "derived_split_alias":
        return "derived_split_alias"
    return "exact_main_name" if name_source == "ucdb_main" else "exact_name_list"


def method_priority(method: str) -> int:
    return {
        "candidate_ucdb_id_exact": 0,
        "exact_main_name": 1,
        "exact_name_list": 2,
        "manual_city_alias": 3,
        "derived_split_alias": 4,
        "fuzzy_name": 5,
    }.get(method, 99)


def make_top_alternatives(alternatives: list[dict[str, Any]], limit: int = 5) -> str:
    parts = []
    for alt in alternatives[:limit]:
        parts.append(
            f"{alt['score']}:{alt['ucdb_id']}:{alt['ucdb_name_main']}:{alt['matched_name']}"
        )
    return " | ".join(parts)


def exact_id_match(candidate: pd.Series, columns: dict[str, str | None], ucdb: pd.DataFrame) -> dict[str, Any] | None:
    id_col = columns["ucdb_id"]
    if not id_col or pd.isna(candidate[id_col]):
        return None

    raw_value = str(candidate[id_col]).strip()
    if not raw_value:
        return None

    try:
        ucdb_id = int(float(raw_value))
    except ValueError:
        return {
            "match_status": "unmatched",
            "match_note": f"Candidate UCDB ID is not numeric: {raw_value}",
        }

    matches = ucdb[ucdb[ID_FIELD].astype("int64") == ucdb_id]
    if matches.empty:
        return {
            "match_status": "unmatched",
            "match_note": f"Candidate UCDB ID not found in UCDB: {ucdb_id}",
        }
    row = matches.iloc[0]
    return {
        "ucdb_row": row,
        "match_status": "matched",
        "match_method": "candidate_ucdb_id_exact",
        "match_score": 100.0,
        "matched_name": row[UCDB_FIELD_MAP["name_main"]],
        "matched_alias": raw_value,
        "match_note": "Candidate table UCDB/GHS ID exact match.",
        "country_candidate_count": len(matches),
        "top_alternatives": "",
    }


def match_candidate(candidate: pd.Series, columns: dict[str, str | None], ucdb: pd.DataFrame) -> dict[str, Any]:
    id_result = exact_id_match(candidate, columns, ucdb)
    if id_result is not None and id_result.get("ucdb_row") is not None:
        return id_result

    country_norms = country_aliases(candidate, columns)
    country_aliases_used = "; ".join(sorted(country_norms))
    country_mask = ucdb["_country_norms"].map(lambda values: bool(values & country_norms))
    country_subset = ucdb[country_mask].copy()

    variants = city_variants(candidate, columns)
    exact_matches: list[dict[str, Any]] = []
    fuzzy_matches: list[dict[str, Any]] = []

    for _, ucdb_row in country_subset.iterrows():
        for name_entry in ucdb_row["_name_entries"]:
            name_norm = normalize_text(name_entry["name"])
            for variant in variants:
                variant_norm = normalize_text(variant["name"])
                if not variant_norm:
                    continue
                score = similarity_score(variant["name"], name_entry["name"])
                match_record = {
                    "ucdb_row": ucdb_row,
                    "ucdb_id": int(ucdb_row[ID_FIELD]),
                    "ucdb_name_main": ucdb_row[UCDB_FIELD_MAP["name_main"]],
                    "matched_name": name_entry["name"],
                    "matched_alias": variant["name"],
                    "variant_source": variant["source"],
                    "name_source": name_entry["source"],
                    "score": score,
                }
                if variant_norm == name_norm:
                    method = method_for_exact_match(variant["source"], name_entry["source"])
                    match_record["match_method"] = method
                    exact_matches.append(match_record)
                else:
                    fuzzy_matches.append(match_record)

    if exact_matches:
        exact_matches.sort(
            key=lambda item: (
                method_priority(item["match_method"]),
                -float(item["ucdb_row"][UCDB_FIELD_MAP["population"]]),
            )
        )
        best = exact_matches[0]
        method = best["match_method"]
        note = (
            f"Country-filtered exact match on {best['name_source']} using "
            f"{best['variant_source']}='{best['matched_alias']}'."
        )
        if method == "manual_city_alias":
            city_name = candidate[columns["city"]]
            note = f"Manual city alias applied: {city_name} -> {best['matched_alias']}. " + note
        return {
            "ucdb_row": best["ucdb_row"],
            "match_status": "matched",
            "match_method": method,
            "match_score": 100.0,
            "matched_name": best["matched_name"],
            "matched_alias": best["matched_alias"],
            "match_note": note,
            "country_candidate_count": int(len(country_subset)),
            "country_aliases_used": country_aliases_used,
            "city_variants": "; ".join(f"{v['source']}={v['name']}" for v in variants),
            "top_alternatives": make_top_alternatives(exact_matches),
        }

    fuzzy_matches.sort(key=lambda item: (-item["score"], -float(item["ucdb_row"][UCDB_FIELD_MAP["population"]])))
    best = fuzzy_matches[0] if fuzzy_matches else None
    if best is None:
        return {
            "ucdb_row": None,
            "match_status": "unmatched",
            "match_method": "no_country_or_name_candidate",
            "match_score": 0.0,
            "matched_name": "",
            "matched_alias": "",
            "match_note": f"No UCDB city candidates after country filter: {sorted(country_norms)}",
            "country_candidate_count": int(len(country_subset)),
            "country_aliases_used": country_aliases_used,
            "city_variants": "; ".join(f"{v['source']}={v['name']}" for v in variants),
            "top_alternatives": "",
        }

    status = "matched" if best["score"] >= 90 else "low_confidence" if best["score"] >= 80 else "unmatched"
    return {
        "ucdb_row": best["ucdb_row"] if status != "unmatched" else None,
        "match_status": status,
        "match_method": "fuzzy_name",
        "match_score": best["score"],
        "matched_name": best["matched_name"],
        "matched_alias": best["matched_alias"],
        "match_note": (
            f"Best fuzzy match after country filter. Review if score < 90. "
            f"Matched {best['matched_alias']} to {best['matched_name']}."
        ),
        "country_candidate_count": int(len(country_subset)),
        "country_aliases_used": country_aliases_used,
        "city_variants": "; ".join(f"{v['source']}={v['name']}" for v in variants),
        "top_alternatives": make_top_alternatives(fuzzy_matches),
    }


def build_city_master(candidates: pd.DataFrame, columns: dict[str, str | None], ucdb: pd.DataFrame) -> pd.DataFrame:
    ucdb = ucdb.copy()
    ucdb["_country_norms"] = ucdb.apply(
        lambda row: {
            normalize_text(row[UCDB_FIELD_MAP["country_gad"]]),
            normalize_text(row[UCDB_FIELD_MAP["country_unn"]]),
        },
        axis=1,
    )
    ucdb["_name_entries"] = ucdb.apply(split_ucdb_names, axis=1)

    result_rows: list[dict[str, Any]] = []
    for _, candidate in candidates.iterrows():
        match = match_candidate(candidate, columns, ucdb)
        ucdb_row = match.pop("ucdb_row", None)
        result = candidate.to_dict()
        if ucdb_row is not None:
            result.update(
                {
                    "ucdb_id": int(ucdb_row[ID_FIELD]),
                    "ucdb_name_main": ucdb_row[UCDB_FIELD_MAP["name_main"]],
                    "ucdb_name_list": ucdb_row[UCDB_FIELD_MAP["name_list"]],
                    "ucdb_country": ucdb_row[UCDB_FIELD_MAP["country_gad"]],
                    "ucdb_country_un": ucdb_row[UCDB_FIELD_MAP["country_unn"]],
                    "ucdb_area_km2": ucdb_row[UCDB_FIELD_MAP["area"]],
                    "ucdb_pop_2025": ucdb_row[UCDB_FIELD_MAP["population"]],
                }
            )
        else:
            result.update(
                {
                    "ucdb_id": pd.NA,
                    "ucdb_name_main": "",
                    "ucdb_name_list": "",
                    "ucdb_country": "",
                    "ucdb_country_un": "",
                    "ucdb_area_km2": pd.NA,
                    "ucdb_pop_2025": pd.NA,
                }
            )
        result.update(match)
        result["manual_rule_applied"] = result.get("match_method") == "manual_city_alias"
        result_rows.append(result)

    city_master = pd.DataFrame(result_rows)
    matched = city_master["match_status"].eq("matched")
    duplicate_groups = (
        city_master.loc[matched & city_master["ucdb_id"].duplicated(keep=False)]
        .groupby("ucdb_id", dropna=True)[columns["city"]]
        .apply(lambda values: ", ".join(values.astype(str)))
        .to_dict()
    )
    city_master["duplicate_ucdb_id_with"] = city_master["ucdb_id"].map(duplicate_groups).fillna("")
    duplicate_mask = city_master["duplicate_ucdb_id_with"].ne("")
    city_master.loc[duplicate_mask, "match_note"] = (
        city_master.loc[duplicate_mask, "match_note"].astype(str)
        + "; duplicate UCDB ID with: "
        + city_master.loc[duplicate_mask, "duplicate_ucdb_id_with"].astype(str)
    )

    required_order = [
        *candidates.columns.tolist(),
        "ucdb_id",
        "ucdb_name_main",
        "ucdb_name_list",
        "ucdb_country",
        "ucdb_country_un",
        "ucdb_area_km2",
        "ucdb_pop_2025",
        "match_status",
        "match_method",
        "match_score",
        "match_note",
        "matched_name",
        "matched_alias",
        "country_candidate_count",
        "country_aliases_used",
        "city_variants",
        "top_alternatives",
        "manual_rule_applied",
        "duplicate_ucdb_id_with",
    ]
    existing_order = [col for col in required_order if col in city_master.columns]
    remaining = [col for col in city_master.columns if col not in existing_order]
    return city_master[existing_order + remaining]


def write_outputs(city_master: pd.DataFrame, ucdb_gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    city_master.to_csv(CITY_MASTER_CSV, index=False, encoding="utf-8-sig")
    parquet_written = False
    parquet_error = ""
    try:
        city_master.to_parquet(CITY_MASTER_PARQUET, index=False)
        parquet_written = True
    except Exception as exc:  # pragma: no cover - depends on optional local engines.
        parquet_error = repr(exc)
        if CITY_MASTER_PARQUET.exists():
            CITY_MASTER_PARQUET.unlink()

    diagnostic = city_master.copy()
    diagnostic.to_csv(MATCH_DIAGNOSTIC_CSV, index=False, encoding="utf-8-sig")

    review = city_master[city_master["match_status"].ne("matched")].copy()
    review.to_csv(MATCH_REVIEW_CSV, index=False, encoding="utf-8-sig")

    matched = city_master[city_master["match_status"].eq("matched")].copy()
    matched["ucdb_id"] = matched["ucdb_id"].astype("int64")
    boundaries = matched.merge(
        ucdb_gdf,
        left_on="ucdb_id",
        right_on=ID_FIELD,
        how="left",
        validate="many_to_one",
        suffixes=("", "_ucdb"),
    )
    boundaries = gpd.GeoDataFrame(boundaries, geometry="geometry", crs=ucdb_gdf.crs).to_crs("EPSG:4326")
    if CITY_BOUNDARY_GPKG.exists():
        CITY_BOUNDARY_GPKG.unlink()
    boundaries.to_file(CITY_BOUNDARY_GPKG, layer=CITY_BOUNDARY_LAYER, driver="GPKG")

    return {
        "parquet_written": parquet_written,
        "parquet_error": parquet_error,
        "review_rows": int(len(review)),
        "boundary_rows_written": int(len(boundaries)),
    }


def verify_outputs(city_master: pd.DataFrame, output_status: dict[str, Any]) -> dict[str, Any]:
    success_count = int(city_master["match_status"].eq("matched").sum())
    unmatched_count = int(city_master["match_status"].eq("unmatched").sum())
    low_confidence_count = int(city_master["match_status"].eq("low_confidence").sum())
    duplicate_groups = (
        city_master.loc[
            city_master["match_status"].eq("matched") & city_master["ucdb_id"].duplicated(keep=False),
            ["ucdb_id", "city_name_en" if "city_name_en" in city_master.columns else city_master.columns[0]],
        ]
        .groupby("ucdb_id")
        .agg(lambda values: ", ".join(values.astype(str)))
    )

    output_files = {
        "city_master_csv": CITY_MASTER_CSV,
        "city_master_parquet": CITY_MASTER_PARQUET,
        "match_diagnostic_csv": MATCH_DIAGNOSTIC_CSV,
        "match_review_csv": MATCH_REVIEW_CSV,
        "city_boundary_gpkg": CITY_BOUNDARY_GPKG,
    }
    readable = {
        "city_master_csv": pd.read_csv(CITY_MASTER_CSV).shape,
        "match_diagnostic_csv": pd.read_csv(MATCH_DIAGNOSTIC_CSV).shape,
        "match_review_csv": pd.read_csv(MATCH_REVIEW_CSV).shape,
    }
    if output_status["parquet_written"]:
        readable["city_master_parquet"] = pd.read_parquet(CITY_MASTER_PARQUET).shape

    boundary = gpd.read_file(CITY_BOUNDARY_GPKG, layer=CITY_BOUNDARY_LAYER)
    boundary_crs = boundary.crs.to_string() if boundary.crs else None
    boundary_epsg = boundary.crs.to_epsg() if boundary.crs else None
    geometry_non_empty = bool(boundary.geometry.notna().all() and (~boundary.geometry.is_empty).all())

    assert len(city_master) > 0, "City master is empty."
    assert output_status["boundary_rows_written"] == success_count, "Boundary row count differs from matched count."
    assert len(boundary) == success_count, "Readable boundary row count differs from matched count."
    assert boundary_epsg == 4326, f"Boundary CRS is not EPSG:4326: {boundary_crs}"
    assert geometry_non_empty, "Boundary contains empty geometry."
    for label, path in output_files.items():
        if label == "city_master_parquet" and not output_status["parquet_written"]:
            continue
        assert path.exists(), f"Missing output: {path}"

    return {
        "candidate_rows": int(len(city_master)),
        "matched_rows": success_count,
        "unmatched_rows": unmatched_count,
        "low_confidence_rows": low_confidence_count,
        "manual_rule_rows": int(city_master["manual_rule_applied"].sum()),
        "duplicate_ucdb_id_groups": duplicate_groups.reset_index().to_dict("records"),
        "outputs": {label: str(path) for label, path in output_files.items()},
        "readable_shapes": {key: tuple(value) for key, value in readable.items()},
        "boundary_rows": int(len(boundary)),
        "boundary_crs": boundary_crs,
        "boundary_geometry_non_empty": geometry_non_empty,
        **output_status,
    }


def main() -> dict[str, Any]:
    candidates, columns = load_candidates()
    ucdb_attr = load_ucdb(ignore_geometry=True)
    city_master = build_city_master(candidates, columns, ucdb_attr)
    ucdb_gdf = load_ucdb(ignore_geometry=False)
    output_status = write_outputs(city_master, ucdb_gdf)
    summary = verify_outputs(city_master, output_status)
    summary["detected_columns"] = columns
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return summary


if __name__ == "__main__":
    SUMMARY = main()
