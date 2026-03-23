#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, List, Optional, Dict, Any
import re
from urllib.parse import urlparse

import boto3

# ---------------------------
# Config
# ---------------------------

@dataclass
class AthenaConfig:
    database: str = "healthcare_curated_db"
    workgroup: str = os.environ.get("ATHENA_WORKGROUP", "primary")
    output_location: Optional[str] = os.environ.get(
        "ATHENA_OUTPUT_LOCATION",
        "s3://healthcare-data-lake-gj/athena_query_results/"
    )
    catalog: str = os.environ.get("ATHENA_CATALOG", "AwsDataCatalog")


SQL_ROOT = Path("sql")
ORDERED_DIRS = ["00_bootstrap", "10_fixed", "20_curated", "90_checks"]

PROJECT_AWS_REGION = (
    os.environ.get("PROJECT_AWS_REGION")
    or os.environ.get("AWS_REGION")
    or os.environ.get("AWS_DEFAULT_REGION")
    or "us-east-1"
)

os.environ["AWS_REGION"] = PROJECT_AWS_REGION
os.environ["AWS_DEFAULT_REGION"] = PROJECT_AWS_REGION
PROJECT_AWS_PROFILE = os.environ.get("AWS_PROFILE", "healthcare-dev")

# ---------------------------
# AWS clients
# ---------------------------

_session = None

def aws_session():
    global _session
    if _session is None:
        _session = boto3.Session(
            profile_name=PROJECT_AWS_PROFILE,
            region_name=aws_region()
        )
    return _session

def aws_region() -> str:
    return PROJECT_AWS_REGION

def athena_client():
    return aws_session().client("athena")

def glue_client():
    return aws_session().client("glue")

# ------------------------------------
# Partition helper functions
# ------------------------------------

def partition_location_from_s3_key(s3_key: str) -> str:
    """
    Turn a manifest s3_key like:
      raw/nh_providerinfo_oct2024/ingest_dt=2026-03-14/NH_ProviderInfo_Oct2024.csv
    into:
      s3://healthcare-data-lake-gj/raw/nh_providerinfo_oct2024/ingest_dt=2026-03-14/
    """
    parts = s3_key.split("/")
    if len(parts) < 3:
        raise ValueError(f"Unexpected s3_key shape: {s3_key}")

    # drop filename
    prefix_parts = parts[:-1]
    return f"s3://healthcare-data-lake-gj/{'/'.join(prefix_parts)}/"


def base_location_from_partition_location(partition_location: str) -> str:
    """
    s3://bucket/raw/foo/ingest_dt=2026-03-15/ -> s3://bucket/raw/foo/
    """
    marker = "/ingest_dt="
    if marker not in partition_location:
        raise ValueError(f"Partition marker not found in location: {partition_location}")
    return partition_location.split(marker, 1)[0].rstrip("/") + "/"


def build_add_partition_sql(
    *,
    database_name: str,
    table_name: str,
    ingest_dt: str,
    location: str,
) -> str:
    return f"""
        ALTER TABLE {database_name}.{table_name}
        ADD IF NOT EXISTS
        PARTITION (ingest_dt='{ingest_dt}')
        LOCATION '{location}'
        """.strip()

def generate_partition_add_sql(cfg: AthenaConfig, ingest_dt: str) -> List[str]:
    """
    Build ALTER TABLE ... ADD IF NOT EXISTS PARTITION statements
    by matching manifest s3_key prefixes to Glue table locations.
    """
    inventory_rows = list_tables_with_partitions("healthcare_catalog_db")

    # keep only tables actually partitioned by ingest_dt
    partitioned_tables = [
        r for r in inventory_rows
        if r["partitioned_by_ingest_dt"] and r.get("location")
    ]

    # normalize Glue locations
    tables_by_location: Dict[str, List[str]] = {}
    for row in partitioned_tables:
        loc = normalize_s3_prefix(row["location"])
        tables_by_location.setdefault(loc, []).append(row["table_name"])

    manifest_rows = get_manifest_file_rows(cfg)

    sql_statements: List[str] = []
    seen: set[tuple[str, str, str]] = set()

    for rec in manifest_rows:
        s3_key = rec.get("s3_key", "")
        if not s3_key or f"ingest_dt={ingest_dt}" not in s3_key:
            continue

        partition_location = partition_location_from_s3_key(s3_key)
        base_location = base_location_from_partition_location(partition_location)
        base_location = normalize_s3_prefix(base_location)

        matching_tables = tables_by_location.get(base_location, [])
        for table_name in matching_tables:
            key = (table_name, ingest_dt, partition_location)
            if key in seen:
                continue
            seen.add(key)

            print(f"partition match: table={table_name} ingest_dt={ingest_dt} location={partition_location}")

            sql_statements.append(
                build_add_partition_sql(
                    database_name="healthcare_catalog_db",
                    table_name=table_name,
                    ingest_dt=ingest_dt,
                    location=partition_location,
                )
            )

    return sorted(sql_statements)

# ---------------------------
# S3 helpers (mainly for "aws rm" commands), but also to help automate partitioning
# ---------------------------

def s3_client():
    return aws_session().client("s3")


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    return bucket, prefix


def extract_s3_prefix(sql_text: str) -> Optional[str]:
    """
    Looks for a comment like:
    -- S3_PREFIX: s3://bucket/prefix/
    """
    m = re.search(r"^\s*--\s*S3_PREFIX:\s*(s3://\S+)\s*$", sql_text, flags=re.MULTILINE)
    return m.group(1) if m else None


def s3_delete_prefix(s3_uri: str) -> None:
    bucket, prefix = parse_s3_uri(s3_uri)
    s3 = s3_client()

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    to_delete = []
    for page in pages:
        for obj in page.get("Contents", []):
            to_delete.append({"Key": obj["Key"]})

            if len(to_delete) == 1000:
                s3.delete_objects(Bucket=bucket, Delete={"Objects": to_delete})
                to_delete = []

    if to_delete:
        s3.delete_objects(Bucket=bucket, Delete={"Objects": to_delete})


def normalize_s3_prefix(s: str) -> str:
    return s.rstrip("/") + "/"


def s3_key_exists(bucket: str, key: str) -> bool:
    s3 = s3_client()
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False

# ---------------------------
# Athena helpers
# ---------------------------

def start_query(sql: str, cfg: AthenaConfig) -> str:
    a = athena_client()

    kwargs = {
        "QueryString": sql,
        "QueryExecutionContext": {
            "Database": cfg.database,
            "Catalog": cfg.catalog,
        },
        "WorkGroup": cfg.workgroup,
    }

    if cfg.output_location:
        kwargs["ResultConfiguration"] = {"OutputLocation": cfg.output_location}

    resp = a.start_query_execution(**kwargs)
    return resp["QueryExecutionId"]


def wait_query(qid: str, poll_s: float = 1.5, timeout_s: int = 60 * 20) -> None:
    a = athena_client()
    start = time.time()

    while True:
        resp = a.get_query_execution(QueryExecutionId=qid)
        state = resp["QueryExecution"]["Status"]["State"]

        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            if state != "SUCCEEDED":
                reason = resp["QueryExecution"]["Status"].get("StateChangeReason", "")
                raise RuntimeError(f"Athena query {qid} {state}: {reason}")
            return

        if time.time() - start > timeout_s:
            raise TimeoutError(f"Athena query {qid} timed out after {timeout_s}s")

        time.sleep(poll_s)


def run_sql(sql: str, cfg: AthenaConfig) -> None:
    if not sql.strip():
        return
    qid = start_query(sql, cfg)
    wait_query(qid)


def run_sql_fetch_rows(sql: str, cfg: AthenaConfig) -> List[List[str]]:
    """
    Submit Athena query and return rows as strings.
    First row returned by Athena is usually the header row.
    """
    a = athena_client()
    qid = start_query(sql, cfg)
    wait_query(qid)

    rows: List[List[str]] = []
    paginator = a.get_paginator("get_query_results")
    for page in paginator.paginate(QueryExecutionId=qid):
        for row in page["ResultSet"]["Rows"]:
            vals = [col.get("VarCharValue", "") for col in row.get("Data", [])]
            rows.append(vals)
    return rows

def run_sql_files(
    files: Iterable[Path],
    *,
    ingest_dt: str,
    cfg: AthenaConfig,
    counters: dict
) -> None:
    for sql_file in files:
        sql_text = sql_file.read_text(encoding="utf-8")
        statements = split_sql_statements(sql_text)
        s3_prefix = extract_s3_prefix(sql_text)

        print(f"\n==> Running {sql_file} ({len(statements)} stmt)")
        is_check_file = "90_checks" in str(sql_file)

        for i, stmt in enumerate(statements, 1):
            stmt = stmt.replace("{{INGEST_DT}}", ingest_dt)
            print(f"  -> stmt {i}/{len(statements)}")
            print("----- SQL START -----")
            print(stmt)
            print("----- SQL END -------")

            if is_check_file:
                if len(statements) != 1:
                    raise RuntimeError(f"Check file must contain exactly one statement: {sql_file}")

                expectation = expectation_from_filename(sql_file)
                print(f"     expectation = {expectation}")

                try:
                    run_check_sql(stmt, cfg, expectation=expectation)
                    counters["checks_passed"] += 1
                except Exception:
                    counters["checks_failed"] += 1
                    counters["failed_check_files"].append(sql_file.name)
                    raise

            else:
                run_sql(stmt, cfg)

                stmt_upper = stmt.strip().upper()
                if stmt_upper.startswith("CREATE TABLE"):
                    counters["transformed_tables"] += 1
                    tokens = stmt.strip().split()
                    if len(tokens) >= 3:
                        counters["transformed_table_names"].append(tokens[2])

                if i == 1 and s3_prefix:
                    print(f"  -> deleting S3 prefix {s3_prefix}")
                    s3_delete_prefix(s3_prefix)

# --------------------------------
# Helpers for data validation checks
# --------------------------------

def parse_first_numeric_result(rows: List[List[str]]) -> int:
    """
    Athena returns header row first, then data rows.
    Expect one numeric value in the first data row.
    """
    if len(rows) < 2 or len(rows[1]) < 1:
        raise RuntimeError("Check query returned no data rows.")
    
    raw = rows[1][0].strip() if rows[1][0] is not None else ""
    if raw == "":
        raise RuntimeError("Check query returned blank result.")
    
    return int(raw)


def expectation_from_filename(sql_file: Path) -> str:
    name = sql_file.name.lower()

    if name.endswith("_gt_zero.sql"):
        return "gt_zero"
    if name.endswith("_ge_one.sql"):
        return "ge_one"
    if name.endswith("_zero.sql"):
        return "zero"

    raise RuntimeError(
        f"Could not determine check expectation from filename: {sql_file.name}. "
        f"Use a suffix like _zero.sql, _gt_zero.sql, or _ge_one.sql."
    )


def run_check_sql(
    sql: str,
    cfg: AthenaConfig,
    *,
    expectation: str = "zero"
) -> None:
    rows = run_sql_fetch_rows(sql, cfg)
    value = parse_first_numeric_result(rows)

    print("Check query...")
    print(sql)
    print(f"     check result = {value}")

    if expectation == "zero":
        if value != 0:
            raise RuntimeError(f"Validation failed: expected 0 but got {value}")

    elif expectation == "gt_zero":
        if value <= 0:
            raise RuntimeError(f"Validation failed: expected > 0 but got {value}")

    elif expectation == "ge_one":
        if value < 1:
            raise RuntimeError(f"Validation failed: expected >= 1 but got {value}")

    else:
        raise RuntimeError(f"Unknown check expectation: {expectation}")

# ---------------------------
# SQL file loading
# ---------------------------

def split_sql_statements(text: str) -> List[str]:
    """
    Remove full-line SQL comments, then split on semicolons.
    Assumes you are NOT embedding semicolons inside strings.
    Good enough for this CTAS/DDL workflow.
    """
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        # skip full-line comments like:
        # -- S3_PREFIX: ...
        # -- regular comment
        if stripped.startswith("--"):
            continue
        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)
    parts = [p.strip() for p in cleaned_text.split(";")]
    return [p for p in parts if p.strip()]


# def iter_sql_files() -> Iterable[Path]:
#     for d in ORDERED_DIRS:
#         folder = SQL_ROOT / d
#         if not folder.exists():
#             continue
#         for p in sorted(folder.glob("*.sql")):
#             yield p

def iter_sql_files_for_dirs(dir_names: List[str]) -> Iterable[Path]:
    for d in dir_names:
        folder = SQL_ROOT / d
        if not folder.exists():
            continue
        for p in sorted(folder.glob("*.sql")):
            yield p

# ---------------------------
# Manifest / ingest_dt lookup
# ---------------------------

def get_latest_ingest_dt_from_manifest(cfg: AthenaConfig) -> Optional[str]:
    sql = """
    SELECT max(regexp_extract(s3_key, 'ingest_dt=([0-9]{4}-[0-9]{2}-[0-9]{2})', 1)) AS ingest_dt
    FROM healthcare_catalog_db.manifest_latest_files
    WHERE s3_key IS NOT NULL
    """
    rows = run_sql_fetch_rows(sql, cfg)

    # Expect:
    # rows[0] = header
    # rows[1] = data
    if len(rows) < 2:
        return None

    ingest_dt = rows[1][0].strip() if rows[1] and rows[1][0] else None
    return ingest_dt or None

def get_manifest_file_rows(cfg: AthenaConfig) -> List[Dict[str, str]]:
    sql = """
    SELECT
      ingest_dt,
      filename,
      dataset_key,
      s3_key
    FROM healthcare_catalog_db.manifest_latest_files
    WHERE s3_key IS NOT NULL
    """
    rows = run_sql_fetch_rows(sql, cfg)

    if len(rows) < 2:
        return []

    header = rows[0]
    out: List[Dict[str, str]] = []

    for row in rows[1:]:
        rec = {}
        for i, col in enumerate(header):
            rec[col] = row[i] if i < len(row) else ""
        out.append(rec)

    return out

# ---------------------------
# Glue inventory helpers
# ---------------------------

def list_tables_with_partitions(database_name: str) -> List[Dict[str, Any]]:
    """
    Returns a list of tables with authoritative partition metadata from Glue.
    """
    g = glue_client()
    paginator = g.get_paginator("get_tables")

    out: List[Dict[str, Any]] = []

    for page in paginator.paginate(DatabaseName=database_name):
        for tbl in page.get("TableList", []):
            partition_keys = [pk["Name"] for pk in tbl.get("PartitionKeys", [])]
            storage_desc = tbl.get("StorageDescriptor", {})
            location = storage_desc.get("Location")

            out.append({
                "table_name": tbl["Name"],
                "partition_keys": partition_keys,
                "partitioned": len(partition_keys) > 0,
                "partitioned_by_ingest_dt": "ingest_dt" in partition_keys,
                "location": location,
                "table_type": tbl.get("TableType"),
            })

    return sorted(out, key=lambda x: x["table_name"])


def print_inventory(database_name: str) -> None:
    rows = list_tables_with_partitions(database_name)
    print(f"\nInventory for database: {database_name}")
    for r in rows:
        print(
            f"- {r['table_name']}: "
            f"partition_keys={r['partition_keys']} "
            f"partitioned_by_ingest_dt={r['partitioned_by_ingest_dt']} "
            f"location={r['location']}"
        )


# ---------------------------
# Main runner
# ---------------------------

def main():

    # Fail fast check. If region not correct, end program.
    def validate_project_region() -> None:
        if aws_region() != "us-east-1":
            raise RuntimeError(
                f"Project must run in us-east-1, but aws_region() resolved to {aws_region()}"
            )

    cfg = AthenaConfig()

    validate_project_region()

    print("ATHENA_WORKGROUP =", cfg.workgroup)
    print("ATHENA_OUTPUT_LOCATION =", cfg.output_location)
    print("PROJECT_AWS_REGION =", PROJECT_AWS_REGION)
    print("AWS_REGION =", os.environ.get("AWS_REGION"))
    print("PROJECT_AWS_PROFILE =", PROJECT_AWS_PROFILE)
    print("AWS_DEFAULT_REGION =", os.environ.get("AWS_DEFAULT_REGION"))
    print("boto3 session region =", aws_session().region_name)
    
    print("PROJECT_AWS_PROFILE =", PROJECT_AWS_PROFILE)
    print("boto3 session profile =", aws_session().profile_name)
    print("sts caller identity =", aws_session().client("sts").get_caller_identity()["Arn"])    

    ingest_dt = get_latest_ingest_dt_from_manifest(cfg) or os.environ.get("INGEST_DT")
    if not ingest_dt:
        raise RuntimeError(
            "No INGEST_DT available. "
            "Set env INGEST_DT=YYYY-MM-DD or ensure healthcare_catalog_db.manifest_latest_files is populated."
        )

    print(f"Using ingest_dt={ingest_dt}")

    # Optional visibility: print inventories before running
    print_inventory("healthcare_catalog_db")
    print_inventory("healthcare_curated_db")

    # counters
    counters = {
        "partitions_added": 0,
        "transformed_tables": 0,
        "checks_passed": 0,
        "checks_failed": 0,
        "transformed_table_names": [],
        "failed_check_files": [],
    }

    # Phase 1: bootstrap + fixed table DDL
    run_sql_files(
        iter_sql_files_for_dirs(["00_bootstrap", "10_fixed"]),
        ingest_dt=ingest_dt,
        cfg=cfg,
        counters=counters,
    )

    # Phase 2: auto-add partitions AFTER fixed tables exist
    partition_sql = generate_partition_add_sql(cfg, ingest_dt)

    print(f"\nAuto-generated partition SQL count: {len(partition_sql)}")
    for stmt in partition_sql:
        print("----- PARTITION SQL START -----")
        print(stmt)
        print("----- PARTITION SQL END -------")
        run_sql(stmt, cfg)
        counters["partitions_added"] += 1

    # Phase 3: curated builds + checks
    run_sql_files(
        iter_sql_files_for_dirs(["20_curated", "90_checks"]),
        ingest_dt=ingest_dt,
        cfg=cfg,
        counters=counters,
    )

    print("\nRun summary")
    print(f"  auto-added partitions: {counters['partitions_added']}")
    print(f"  transformed tables:    {counters['transformed_tables']}")
    print(f"  checks passed:         {counters['checks_passed']}")
    print(f"  checks failed:         {counters['checks_failed']}")

    if counters["transformed_table_names"]:
        print("\nTransformed tables:")
        for name in counters["transformed_table_names"]:
            print(f"  - {name}")

    if counters["failed_check_files"]:
        print("\nFailed check files:")
        for name in counters["failed_check_files"]:
            print(f"  - {name}")

    print("\nAll transforms complete.")


if __name__ == "__main__":    
    main()