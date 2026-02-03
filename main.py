from pydantic import BaseModel

import os
import csv
import io
import uuid
import json
import re
from time import time
from datetime import datetime
from typing import Any, Dict, List

from fastapi import FastAPI, UploadFile, File, Query, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values

# -----------------------------
# Config
# -----------------------------
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL missing")

SCHEMA = "datasets"
DATASETS_TBL = f"{SCHEMA}.datasets"
ROWS_TBL = f"{SCHEMA}.dataset_rows"

app = FastAPI(title="AIAG-07 Backend (Agentic DataTables)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# SIMPLE QUERY CACHE (Agent)
# =====================================================
QUERY_CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_TTL = 30  # seconds


def get_cache(key: str):
    item = QUERY_CACHE.get(key)
    if not item:
        return None
    if time() - item["ts"] > CACHE_TTL:
        QUERY_CACHE.pop(key, None)
        return None
    return item["data"]


def set_cache(key: str, data: Any):
    QUERY_CACHE[key] = {"data": data, "ts": time()}


# =====================================================
# PERFORMANCE STRATEGY AGENT
# =====================================================
def performance_strategy_agent(
    rows: int,
    columns: int,
    text_columns: int,
    used_fts: bool,
    viewport_width: int
) -> Dict[str, Any]:

    strategy = {
        "processingMode": "server",
        "renderMode": "pagination",
        "pageSize": 25,
        "reason": []
    }

    if rows <= 5000 and columns <= 12:
        strategy["processingMode"] = "client"
        strategy["reason"].append("Small dataset (<=5000 rows, <=12 cols)")

    if rows > 100000:
        strategy["renderMode"] = "infinite_scroll"
        strategy["pageSize"] = 50
        strategy["reason"].append("Very large dataset (>100k rows)")

    # FTS always forces server mode
    if used_fts:
        strategy["processingMode"] = "server"
        strategy["reason"].append("FTS search active -> server mode required")

    if viewport_width and viewport_width < 600:
        strategy["pageSize"] = 25
        strategy["reason"].append("Small viewport (<600px) -> page size 25")

    if viewport_width and viewport_width > 1200 and rows < 100000:
        strategy["pageSize"] = 100
        strategy["reason"].append("Large screen (>1200px) + manageable dataset -> page size 100")

    return strategy


# =====================================================
# RESPONSIVE LAYOUT AGENT
# =====================================================
def responsive_layout_agent(
    columns: List[str],
    col_types: Dict[str, str],
    viewport_width: int
) -> Dict[str, Any]:

    strategy = {
        "layoutMode": "desktop",
        "priorityColumns": [],
        "hiddenColumns": [],
        "reason": []
    }

    priority_keywords = ["id", "name", "title", "status", "category", "created"]

    priority_cols = [
        c for c in columns
        if any(k in c.lower() for k in priority_keywords)
    ]

    if not priority_cols:
        priority_cols = columns[:3]

    if viewport_width and viewport_width < 600:
        strategy["layoutMode"] = "mobile"
        strategy["priorityColumns"] = priority_cols[:2]
        strategy["hiddenColumns"] = [
            c for c in columns if c not in strategy["priorityColumns"]
        ]
        strategy["reason"].append("Mobile viewport (<600px)")

    elif viewport_width and viewport_width < 1024:
        strategy["layoutMode"] = "tablet"
        strategy["priorityColumns"] = priority_cols[:4]
        strategy["hiddenColumns"] = [
            c for c in columns if c not in strategy["priorityColumns"]
        ]
        strategy["reason"].append("Tablet viewport (<1024px)")

    else:
        strategy["layoutMode"] = "desktop"
        strategy["priorityColumns"] = columns
        strategy["hiddenColumns"] = []
        strategy["reason"].append("Desktop viewport (>=1024px)")

    return strategy


# =====================================================
# ADAPTIVE FILTER AGENT  ← NEW
# =====================================================
def analyze_dataset_for_filters(
    cur,
    dataset_id: str,
    columns: List[str],
    col_types: Dict[str, str],
    total_rows: int
) -> List[Dict[str, Any]]:
    """
    Analyse every column and decide what kind of filter widget
    (if any) makes sense.

    Rules:
        datetime                          -> date_range   (two date pickers)
        text     + 2 <= distinct <= 30    -> enum         (dropdown)
        text     + distinct > 30          -> text_search  (free-text input)
        int/float + 2 <= distinct <= 15   -> enum         (dropdown)
        int/float + distinct > 15         -> range        (min / max inputs)
        anything  + distinct < 2          -> skip
    """

    SAMPLE_LIMIT = 50_000          # cap for huge tables
    filters: List[Dict[str, Any]] = []

    for col in columns:
        col_type = col_types.get(col, "text")

        # ── distinct count ──────────────────────────────
        if total_rows > SAMPLE_LIMIT:
            cur.execute(
                f"""
                SELECT COUNT(DISTINCT val) FROM (
                    SELECT row_json->>%s AS val
                    FROM {ROWS_TBL}
                    WHERE dataset_id = %s
                    LIMIT %s
                ) sub
                WHERE val IS NOT NULL AND val <> ''
                """,
                (col, dataset_id, SAMPLE_LIMIT)
            )
        else:
            cur.execute(
                f"""
                SELECT COUNT(DISTINCT row_json->>%s)
                FROM {ROWS_TBL}
                WHERE dataset_id = %s
                  AND row_json->>%s IS NOT NULL
                  AND row_json->>%s <> ''
                """,
                (col, dataset_id, col, col)
            )
        distinct_count: int = cur.fetchone()[0]

        # skip columns that can't filter anything
        if distinct_count < 2:
            continue

        # ── decide filter type ──────────────────────────
        if col_type == "datetime":
            filters.append({
                "column": col,
                "label": col.replace("_", " ").title(),
                "filterType": "date_range",
                "distinctCount": distinct_count
            })

        elif col_type == "text":
            if distinct_count <= 30:
                # fetch actual values for dropdown options
                cur.execute(
                    f"""
                    SELECT DISTINCT row_json->>%s AS val
                    FROM {ROWS_TBL}
                    WHERE dataset_id = %s
                      AND row_json->>%s IS NOT NULL
                      AND row_json->>%s <> ''
                    ORDER BY val
                    LIMIT 30
                    """,
                    (col, dataset_id, col, col)
                )
                options = [r[0] for r in cur.fetchall()]
                filters.append({
                    "column": col,
                    "label": col.replace("_", " ").title(),
                    "filterType": "enum",
                    "options": options,
                    "distinctCount": distinct_count
                })
            else:
                filters.append({
                    "column": col,
                    "label": col.replace("_", " ").title(),
                    "filterType": "text_search",
                    "distinctCount": distinct_count
                })

        elif col_type in ("int", "float"):
            if distinct_count <= 15:
                cur.execute(
                    f"""
                    SELECT DISTINCT row_json->>%s AS val
                    FROM {ROWS_TBL}
                    WHERE dataset_id = %s
                      AND row_json->>%s IS NOT NULL
                      AND row_json->>%s <> ''
                    ORDER BY val
                    LIMIT 15
                    """,
                    (col, dataset_id, col, col)
                )
                options = [r[0] for r in cur.fetchall()]
                filters.append({
                    "column": col,
                    "label": col.replace("_", " ").title(),
                    "filterType": "enum",
                    "options": options,
                    "distinctCount": distinct_count
                })
            else:
                cur.execute(
                    f"""
                    SELECT
                        MIN(NULLIF(row_json->>%s, ''))::numeric,
                        MAX(NULLIF(row_json->>%s, ''))::numeric
                    FROM {ROWS_TBL}
                    WHERE dataset_id = %s
                    """,
                    (col, col, dataset_id)
                )
                row = cur.fetchone()
                min_val = float(row[0]) if row[0] is not None else None
                max_val = float(row[1]) if row[1] is not None else None

                if min_val is not None and max_val is not None and min_val != max_val:
                    filters.append({
                        "column": col,
                        "label": col.replace("_", " ").title(),
                        "filterType": "range",
                        "min": min_val,
                        "max": max_val,
                        "distinctCount": distinct_count
                    })

    return filters


# =====================================================
# DB Helpers
# =====================================================
def get_conn():
    return psycopg2.connect(DATABASE_URL)


def infer_type(values: List[str]) -> str:
    """Infer column type from up to 200 sample values."""
    cleaned = [v for v in values if v and str(v).strip()]
    if not cleaned:
        return "text"

    try:
        for v in cleaned[:200]:
            int(v)
        return "int"
    except (ValueError, TypeError):
        pass

    try:
        for v in cleaned[:200]:
            float(v)
        return "float"
    except (ValueError, TypeError):
        pass

    is_datetime = True
    for v in cleaned[:100]:
        try:
            datetime.fromisoformat(str(v))
        except (ValueError, TypeError):
            is_datetime = False
            break
    if is_datetime:
        return "datetime"

    return "text"


def build_search_text(row: dict, cols: List[str]) -> str:
    return " ".join(str(row.get(c, "")) for c in cols)


def safe_colname(col: str, allowed: List[str]) -> str:
    col = col.strip() if col else ""
    return col if col in allowed else allowed[0]


# =====================================================
# Upload CSV
# =====================================================
@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    content = await file.read()
    reader = csv.DictReader(io.StringIO(content.decode("utf-8", errors="ignore")))
    columns = reader.fieldnames or []

    if not columns:
        raise HTTPException(400, "CSV must have headers")

    rows = []
    samples: Dict[str, List[str]] = {c: [] for c in columns}

    for i, row in enumerate(reader):
        rows.append(row)
        if i < 300:
            for c in columns:
                samples[c].append(row.get(c, ""))

    col_types = {c: infer_type(samples[c]) for c in columns}
    dataset_id = str(uuid.uuid4())

    conn = get_conn()
    try:
        cur = conn.cursor()

        cur.execute(
            f"""
            INSERT INTO {DATASETS_TBL}
            (dataset_id, name, columns_json, column_types)
            VALUES (%s, %s, %s::jsonb, %s::jsonb)
            """,
            (dataset_id, file.filename, json.dumps(columns), json.dumps(col_types))
        )

        values = [
            (
                dataset_id,
                json.dumps(r),
                build_search_text(r, columns),
                build_search_text(r, columns)
            )
            for r in rows
        ]

        execute_values(
            cur,
            f"""
            INSERT INTO {ROWS_TBL}
            (dataset_id, row_json, search_text, search_tsv)
            VALUES %s
            """,
            values,
            template="(%s, %s::jsonb, %s, to_tsvector('simple', %s))"
        )

        conn.commit()

        # ── Adaptive Filter Agent — runs AFTER data is committed ──
        recommended_filters = analyze_dataset_for_filters(
            cur=cur,
            dataset_id=dataset_id,
            columns=columns,
            col_types=col_types,
            total_rows=len(rows)
        )

    finally:
        conn.close()

    return {
        "datasetId": dataset_id,
        "rows": len(rows),
        "columns": columns,
        "columnTypes": col_types,
        "recommendedFilters": recommended_filters   # ← Adaptive Filter Agent output
    }


# =====================================================
# DataTables Endpoint
# =====================================================
@app.get("/api/table")
async def datatable_endpoint(
    request: Request,                                   # ← raw query params for dynamic filters
    datasetId: str = Query(...),
    draw: int = Query(1),
    start: int = Query(0),
    length: int = Query(25),
    viewportWidth: int = Query(0),
    search_value: str = Query("", alias="search[value]"),
    order_col_index: int = Query(0, alias="order[0][column]"),
    order_dir: str = Query("asc", alias="order[0][dir]"),
):
    # ── extract dynamic filter params from the raw query string ──
    raw_filters: Dict[str, str] = {}
    date_from_filters: Dict[str, str] = {}
    date_to_filters:   Dict[str, str] = {}
    range_min_filters: Dict[str, str] = {}
    range_max_filters: Dict[str, str] = {}

    for key, value in request.query_params.items():
        if not value or not value.strip():
            continue
        val = value.strip()

        if key.startswith("filter_col_"):
            raw_filters[key[len("filter_col_"):]] = val
        elif key.startswith("filter_date_from_"):
            date_from_filters[key[len("filter_date_from_"):]] = val
        elif key.startswith("filter_date_to_"):
            date_to_filters[key[len("filter_date_to_"):]] = val
        elif key.startswith("filter_range_min_"):
            range_min_filters[key[len("filter_range_min_"):]] = val
        elif key.startswith("filter_range_max_"):
            range_max_filters[key[len("filter_range_max_"):]] = val

    # ── cache key (includes every filter dimension) ──────────
    cache_key = json.dumps({
        "datasetId": datasetId,
        "start": start,
        "length": length,
        "order": [order_col_index, order_dir],
        "search": search_value,
        "viewportWidth": viewportWidth,
        "filters": raw_filters,
        "dateFrom": date_from_filters,
        "dateTo": date_to_filters,
        "rangeMin": range_min_filters,
        "rangeMax": range_max_filters,
    }, sort_keys=True)

    cached = get_cache(cache_key)
    if cached:
        return cached

    conn = get_conn()
    try:
        cur = conn.cursor()

        # ── load dataset metadata ───────────────────────
        cur.execute(
            f"SELECT columns_json, column_types FROM {DATASETS_TBL} WHERE dataset_id=%s",
            (datasetId,)
        )
        meta = cur.fetchone()
        if not meta:
            raise HTTPException(404, "Invalid datasetId")

        allowed_cols: List[str] = meta[0]
        col_types:   Dict[str, str] = meta[1]

        # ── sort column guard ───────────────────────────
        if order_col_index < 0 or order_col_index >= len(allowed_cols):
            order_col_index = 0
        sort_col = safe_colname(allowed_cols[order_col_index], allowed_cols)

        # ── WHERE clause ────────────────────────────────
        filters_sql  = ["dataset_id=%s"]
        params: list = [datasetId]

        # -- global FTS search --
        if search_value:
            filters_sql.append("search_tsv @@ plainto_tsquery('simple', %s)")
            params.append(search_value)

        # -- dynamic enum / text_search filters --
        for col_name, col_value in raw_filters.items():
            if col_name not in allowed_cols:
                continue                              # whitelist guard
            col_type = col_types.get(col_name, "text")
            if col_type in ("int", "float"):
                filters_sql.append("row_json->>%s = %s")
            else:
                filters_sql.append("row_json->>%s ILIKE %s")
                col_value = f"%{col_value}%"
            params.extend([col_name, col_value])

        # -- dynamic date_range filters --
        all_date_cols = set(date_from_filters.keys()) | set(date_to_filters.keys())
        for col_name in all_date_cols:
            if col_name not in allowed_cols:
                continue
            if col_name in date_from_filters:
                filters_sql.append("(row_json->>%s)::date >= %s")
                params.extend([col_name, date_from_filters[col_name]])
            if col_name in date_to_filters:
                filters_sql.append("(row_json->>%s)::date <= %s")
                params.extend([col_name, date_to_filters[col_name]])

        # -- dynamic numeric range filters --
        all_range_cols = set(range_min_filters.keys()) | set(range_max_filters.keys())
        for col_name in all_range_cols:
            if col_name not in allowed_cols:
                continue
            if col_name in range_min_filters:
                filters_sql.append("NULLIF(row_json->>%s, '')::numeric >= %s")
                params.extend([col_name, range_min_filters[col_name]])
            if col_name in range_max_filters:
                filters_sql.append("NULLIF(row_json->>%s, '')::numeric <= %s")
                params.extend([col_name, range_max_filters[col_name]])

        where_sql = " AND ".join(filters_sql)

        # ── two separate counts ─────────────────────────
        cur.execute(
            f"SELECT COUNT(*) FROM {ROWS_TBL} WHERE dataset_id=%s",
            (datasetId,)
        )
        records_total: int = cur.fetchone()[0]

        cur.execute(
            f"SELECT COUNT(*) FROM {ROWS_TBL} WHERE {where_sql}",
            params
        )
        records_filtered: int = cur.fetchone()[0]

        # ── safe ORDER BY ───────────────────────────────
        sort_expr: str = f"(row_json->>'{sort_col}')"
        if col_types.get(sort_col) in ("int", "float"):
            sort_expr = f"NULLIF((row_json->>'{sort_col}'), '')::numeric"
        elif col_types.get(sort_col) == "datetime":
            sort_expr = f"NULLIF((row_json->>'{sort_col}'), '')::timestamp"

        order_direction = "DESC" if order_dir.strip().lower() == "desc" else "ASC"

        # ── data query ──────────────────────────────────
        query = f"""
            SELECT row_json
            FROM {ROWS_TBL}
            WHERE {where_sql}
            ORDER BY {sort_expr} {order_direction} NULLS LAST
            LIMIT %s OFFSET %s
        """
        cur.execute(query, params + [length, start])
        data = [r[0] for r in cur.fetchall()]

        # ── run agents ──────────────────────────────────
        text_cols = sum(1 for t in col_types.values() if t == "text")

        perf_agent = performance_strategy_agent(
            rows=records_filtered,
            columns=len(allowed_cols),
            text_columns=text_cols,
            used_fts=bool(search_value),
            viewport_width=viewportWidth
        )

        layout_agent = responsive_layout_agent(
            columns=allowed_cols,
            col_types=col_types,
            viewport_width=viewportWidth
        )

        # ── assemble response ───────────────────────────
        response = {
            "draw": draw,
            "recordsTotal": records_total,
            "recordsFiltered": records_filtered,
            "data": data,
            "agentInsights": {
                "agentStrategy": perf_agent,
                "responsiveLayout": layout_agent
            }
        }

        set_cache(cache_key, response)
        return response

    finally:
        conn.close()
