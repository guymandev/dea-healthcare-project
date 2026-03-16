#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, List, Optional

import boto3


# ---------------------------
# Config
# ---------------------------

@dataclass
class AthenaConfig:
    database: str = "healthcare_curated_db"
    workgroup: str = os.environ.get("ATHENA_WORKGROUP", "primary")
    output_location: str = os.environ.get(
        "ATHENA_OUTPUT_LOCATION",
        "s3://healthcare-data-lake-gj/athena_query_results/transform/"
    )
    catalog: str = os.environ.get("ATHENA_CATALOG", "AwsDataCatalog")  # case varies in consoles


SQL_ROOT = Path("sql")  # your folder
ORDERED_DIRS = ["00_bootstrap", "10_fixed", "20_curated", "90_checks"]


# ---------------------------
# Athena helpers
# ---------------------------

def athena_client():
    return boto3.client("athena")


def start_query(sql: str, cfg: AthenaConfig) -> str:
    a = athena_client()
    resp = a.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={
            "Database": cfg.database,
            "Catalog": cfg.catalog,
        },
        ResultConfiguration={"OutputLocation": cfg.output_location},
        WorkGroup=cfg.workgroup,
    )
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
    # ignore empty/no-op statements
    if not sql.strip():
        return
    qid = start_query(sql, cfg)
    wait_query(qid)


# ---------------------------
# SQL file loading
# ---------------------------

def split_sql_statements(text: str) -> List[str]:
    """
    Simple splitter: splits on semicolons.
    Assumes you are NOT embedding semicolons inside strings.
    (Good enough for your CTAS/DDL workflow.)
    """
    parts = [p.strip() for p in text.split(";")]
    return [p for p in parts if p.strip()]


def iter_sql_files() -> Iterable[Path]:
    for d in ORDERED_DIRS:
        folder = SQL_ROOT / d
        if not folder.exists():
            continue
        for p in sorted(folder.glob("*.sql")):
            yield p


# ---------------------------
# Manifest / ingest_dt discovery (optional)
# ---------------------------

def get_latest_ingest_dt_from_manifest(cfg: AthenaConfig) -> Optional[str]:
    """
    If you want: query healthcare_catalog_db.manifest_latest_files
    and return max(ingest_dt). Keep it optional so transforms can still run
    even if manifest infra is down.
    """
    # You can implement with get_query_results, but simplest is:
    # - run a small query into Athena results
    # - read first row back
    # Leaving as TODO stub.
    return None


# ---------------------------
# Main runner
# ---------------------------

def main():
    cfg = AthenaConfig()

    # Optional: decide which ingest_dt to use and pass to SQL via templating.
    # ingest_dt = get_latest_ingest_dt_from_manifest(cfg) or os.environ.get("INGEST_DT")
    # if not ingest_dt:
    #     raise RuntimeError("No INGEST_DT available. Set env INGEST_DT=YYYY-MM-DD or implement manifest lookup.")

    for sql_file in iter_sql_files():
        sql_text = sql_file.read_text(encoding="utf-8")
        statements = split_sql_statements(sql_text)

        print(f"\n==> Running {sql_file} ({len(statements)} stmt)")
        for i, stmt in enumerate(statements, 1):
            # If you later want templating:
            # stmt = stmt.replace("{{INGEST_DT}}", ingest_dt)
            print(f"  -> stmt {i}/{len(statements)}")
            run_sql(stmt, cfg)

    print("\nAll transforms complete.")


if __name__ == "__main__":
    main()