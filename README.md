# Healthcare Data Pipeline

## Overview

This project implements an end-to-end healthcare data pipeline on AWS. It ingests raw source files from Google Drive into a partitioned S3 data lake, tracks file-level state through archived and latest manifests, and then runs an Athena-based transformation layer to build curated datasets for analytics and dashboarding. The pipeline is designed to run both locally and in ECS/Fargate using the same Python codebase. 

The pipeline is organized into two main runtime stages:

1. **Ingest stage** (`main_program.py`)  
   Downloads source files, detects changes using SHA-256 checksums, uploads changed files into S3 raw storage, and writes manifest/control artifacts.

2. **Transform stage** (`run_transform.py`)  
   Reads the latest manifest state, determines dataset-specific latest partitions, auto-registers partitions in Athena/Glue, executes ordered SQL transformations, and runs validation checks.

The final solution is containerized, deployed to Amazon ECR, executed as ECS/Fargate tasks, orchestrated with AWS Step Functions, and scheduled with EventBridge Scheduler.

---

## Architecture

### High-level flow

Google Drive source files  
→ ECS/Fargate ingest task  
→ S3 raw zone + control manifests + JSONL inventory  
→ ECS/Fargate transform task  
→ Athena / Glue transformation pipeline  
→ curated healthcare datasets / views  
→ Streamlit analytics dashboard. 

### Core architectural principles

- **Incremental ingestion**  
  Files are hashed with SHA-256 and compared to prior manifest state so unchanged files can be skipped. 

- **Partitioned raw storage**  
  Raw files are stored under dataset-specific prefixes with `ingest_dt=YYYY-MM-DD` partitions in S3. 

- **Manifest-driven processing**  
  The transform layer uses manifest artifacts and a JSONL inventory file instead of relying on hardcoded dates. 

- **Automatic partition registration**  
  The transform runner inspects manifest S3 keys and Glue table metadata to generate partition DDL automatically. 

- **Validation as part of the pipeline**  
  Data checks are executed as a formal final phase, not as an afterthought. 

---

## AWS Services Used

### Amazon S3
S3 serves as both the data lake and the control-store layer. The ingest stage writes:
- raw source data under the raw prefix
- manifest artifacts under the control prefix
- quarantine outputs for files that fail ingest processing. 

### Amazon ECS / AWS Fargate
Both runtime stages are containerized and executed as ECS tasks on Fargate. The code is written to use IAM-role-based credentials automatically in Fargate, while still supporting local AWS profiles during development.

### Amazon ECR
Container images for the ingest and transform modules are stored in separate ECR repositories and pulled by ECS/Fargate at runtime.

### Amazon Athena
Athena is used to execute SQL for:
- table creation
- curated transformations
- partition registration
- validation queries. 

### AWS Glue Data Catalog
Glue is used as the metadata layer for table discovery and partition inspection. The transform runner queries Glue to determine which tables are partitioned and what their underlying S3 locations are. 

### AWS Step Functions
Step Functions orchestrates the full pipeline by running the ingest task first and the transform task second.

### Amazon EventBridge Scheduler
EventBridge Scheduler triggers the Step Functions state machine on a recurring schedule.

### Streamlit Community Cloud
The final analytics dashboard is deployed separately on Streamlit Cloud and queries the curated healthcare analytics layer.

---

## Data Flow

### 1. Source acquisition
The ingest stage downloads:
- a public Google Drive folder containing source files
- a standalone PBJ staffing file from Google Drive. 

### 2. File discovery and hashing
The ingest module recursively scans downloaded `.csv` and `.json` files, computes SHA-256 checksums, and compares each file against prior manifest state keyed by filename. 

### 3. Incremental raw load
A file is uploaded to S3 if:
- it is new
- its checksum changed
- or the prior expected S3 object is missing and must be healed. 

Files are written to S3 under paths like:

```text
raw/<dataset_key>/ingest_dt=<YYYY-MM-DD>/<filename>
```

This preserves history and supports downstream partition-aware querying.

### 4. Manifest generation
The ingest stage writes:
- a per-run archived manifest JSON at ```control/manifests/ingest_dt=<date>/manifest.json```
- a rolling latest-state manifest JSON at ```control/manifests/latest/manifest.json```
- a separate, derived latest-files inventory (JSONL view) at ```control/manifests/latest_files/files.jsonl``` for Athena-friendly querying.

### 5. Dataset-aware transformation
The transform stage reads the latest manifest-derived inventory and builds a ```dataset_ingest_map```, allowing SQL files to bind to the latest partition for each dataset individually.

### 6. Partition registration
The transform stage derives partition locations from manifest S3 keys, matches them to Glue table base locations, and generates ```ALTER TABLE ... ADD IF NOT EXISTS PARTITION``` statements automatically.

### 7. Curated SQL pipeline
SQL files are executed in ordered directories:
- ```00_bootstrap```
- ```10_fixed```
- ```20_curated```
- ```90_checks```

### 8. Validation checks
The final phase runs SQL validation checks whose expectations are inferred from the file suffixes such as:
- ```_zero,sql```
- ```_gt_zero.sql```
- ```_g3_one.sql```
A failed check causes the pipeline to fail.

### Key Design Decisions

#### Manifest-driven state management
Instead of trying to infer the latest files directly from S3, the pipeline writes explicit manifest artifacts. This creates a durable, auditable representation of pipeline state and gives downstream steps a stable control input.

#### Partitioned S3 layout
Using dataset-based prefixes plus ```ingest_dt``` partitions allows the raw zone to retain history while still enabling efficient downstream partition selection.

#### Archive manifest + latest-state manifest
The project keeps both:
- a per-run archived manifest
- a rolling latest-state manifest, along with a separate JSONL version of this for Athena.

This supports both historical traceability and simplified downstream "latest file" logic.

#### JSONL inventory for Athena
The ```files.jsonl``` output flattens manifest metadata into a query-friendly structure, which the transform stage then uses to drive dataset-specific partition logic.

#### Dataset-specific ingest-date substitution
The transform runner supports placeholders like ```{{INGEST_DT:dataset_key}}```, which allows each SQL object to bind to the latest available partition for that particular dataset. This avoids assuming that all source datasets arrive on the same day.

#### Safe handling of CTAS output locations
The transform runner supports ```-- S3_PREFIX:``` comments for CTAS outputs, which enables a preflight step to clean target prefix buckets in S3 before reuse. At the same time, it explicitly blocks that mechanism for ```CREATE EXTERNAL TABLE``` SQL files so source data locations are never accidentally deleted.

#### Quarantine on ingest failure
If a file fails ingest processing, the code attempts a best-effort upload to a quarantine prefix for later debugging.

#### Detailed operational logging
Both stages print detailed logs covering:
- region/profile resolution
- checksum comparisons
- prior-manifest comparisons
- partition matches
- SQL statements
- check expectations
- final run summaries

This is especially useful when debugging through ECS/Fargate and CloudWatch logs.