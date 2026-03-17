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


# ---------------------------
# AWS clients
# ---------------------------

def aws_region() -> str:
    region = (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or boto3.session.Session().region_name
    )
    if not region:
        raise RuntimeError("No AWS region configured.")
    return region

def athena_client():
    return boto3.client("athena", region_name=aws_region())

def glue_client():
    return boto3.client("glue", region_name=aws_region())

# ---------------------------
# S3 helpers for "aws rm" commands
# ---------------------------

def s3_client():
    return boto3.client("s3", region_name=aws_region())


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


def iter_sql_files() -> Iterable[Path]:
    for d in ORDERED_DIRS:
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
    SELECT max(ingest_dt) AS ingest_dt
    FROM healthcare_catalog_db.manifest_latest_files
    """
    rows = run_sql_fetch_rows(sql, cfg)

    # Expect:
    # rows[0] = header
    # rows[1] = data
    if len(rows) < 2:
        return None

    ingest_dt = rows[1][0].strip() if rows[1] and rows[1][0] else None
    return ingest_dt or None


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
    cfg = AthenaConfig()

    print("ATHENA_WORKGROUP =", cfg.workgroup)
    print("ATHENA_OUTPUT_LOCATION =", cfg.output_location)
    print("AWS_REGION =", os.environ.get("AWS_REGION"))
    print("AWS_DEFAULT_REGION =", os.environ.get("AWS_DEFAULT_REGION"))
    print("boto3 session region =", boto3.session.Session().region_name)
    print("athena client region =", athena_client().meta.region_name)

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

    for sql_file in iter_sql_files():
        sql_text = sql_file.read_text(encoding="utf-8")
        statements = split_sql_statements(sql_text)
        s3_prefix = extract_s3_prefix(sql_text)

        print(f"\n==> Running {sql_file} ({len(statements)} stmt)")
        for i, stmt in enumerate(statements, 1):
            stmt = stmt.replace("{{INGEST_DT}}", ingest_dt)
            print(f"  -> stmt {i}/{len(statements)}")
            print("----- SQL START -----")
            print(stmt)
            print("----- SQL END -------")
            run_sql(stmt, cfg)

            # If this file declares an S3 prefix, and we just ran stmt 1 (DROP),
            # clean the target folder before stmt 2 (CREATE)
            if i == 1 and s3_prefix:
                print(f"  -> deleting S3 prefix {s3_prefix}")
                s3_delete_prefix(s3_prefix)

    print("\nAll transforms complete.")


if __name__ == "__main__":    
    main()