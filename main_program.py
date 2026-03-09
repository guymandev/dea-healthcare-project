import hashlib
import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

import boto3
import gdown


# ---------------------------
# Config
# ---------------------------
BUCKET = os.environ.get("HEALTHCARE_BUCKET", "healthcare-data-lake-gj")

DRIVE_FOLDER_URL = os.environ.get(
    "DRIVE_FOLDER_URL",
    "https://drive.google.com/drive/folders/15KqJ1MZ7JcgAkOfqcaWcALWkG0dh3jpE",
)

# PBJ staffing file is a standalone Drive file
PBJ_FILE_ID = os.environ.get("PBJ_FILE_ID", "1kZMZFGfTLdcwmdhjDPZh2-XE2_gOBRCz")
PBJ_FILENAME = os.environ.get("PBJ_FILENAME", "PBJ_Daily_Nurse_Staffing_Q2_2024.csv")

RAW_PREFIX = os.environ.get("RAW_PREFIX", "raw")
CONTROL_PREFIX = os.environ.get("CONTROL_PREFIX", "control")
QUARANTINE_PREFIX = os.environ.get("QUARANTINE_PREFIX", "quarantine")

LATEST_MANIFEST_KEY = f"{CONTROL_PREFIX}/manifests/latest/manifest.json"

TMP_DIR = Path("/tmp/data_download") if "AWS_LAMBDA_FUNCTION_NAME" in os.environ else Path("./data_download")


# ---------------------------
# S3 helpers
# ---------------------------
def s3_client():
    return boto3.client("s3")


def s3_get_json(bucket: str, key: str) -> Optional[Dict[str, Any]]:
    s3 = s3_client()
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except s3.exceptions.NoSuchKey:
        return None


def s3_put_json(bucket: str, key: str, payload: Dict[str, Any]) -> None:
    s3 = s3_client()
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def s3_upload_file(local_path: Path, bucket: str, key: str) -> None:
    s3 = s3_client()
    s3.upload_file(Filename=str(local_path), Bucket=bucket, Key=key)


# ---------------------------
# Local helpers
# ---------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_key_from_filename(filename: str) -> str:
    base = filename.lower()
    for ext in (".csv", ".json"):
        if base.endswith(ext):
            base = base[: -len(ext)]
    return base.replace(" ", "_").replace("__", "_")


def iter_data_files(root: Path):
    # recursively find CSV/JSON, skip PDFs
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".csv", ".json"}:
            yield p


# ---------------------------
# Download helpers
# ---------------------------
def download_drive_folder(output_dir: Path) -> None:
    ensure_dir(output_dir)
    gdown.download_folder(DRIVE_FOLDER_URL, output=str(output_dir), quiet=False, use_cookies=False)


def download_pbj_file(output_dir: Path) -> Path:
    ensure_dir(output_dir)
    out_path = output_dir / PBJ_FILENAME
    url = f"https://drive.google.com/uc?id={PBJ_FILE_ID}"
    gdown.download(url, str(out_path), quiet=False)
    return out_path


# ---------------------------
# Ingest
# ---------------------------
def ingest_once() -> Dict[str, Any]:
    ingest_dt = date.today().isoformat()
    run_ts = datetime.now(timezone.utc).isoformat()

    ensure_dir(TMP_DIR)

    # Load prior manifest checksums
    prior = s3_get_json(BUCKET, LATEST_MANIFEST_KEY) or {}
    prior_checksums = {
        f.get("filename"): f.get("sha256")
        for f in prior.get("files", [])
        if f.get("filename") and f.get("sha256")
    }

    manifest: Dict[str, Any] = {
        "ingest_dt": ingest_dt,
        "run_ts_utc": run_ts,
        "source": "google_drive_public",
        "bucket": BUCKET,
        "raw_prefix": RAW_PREFIX,
        "files": [],
        "skipped_unchanged": [],
        "errors": [],
    }

    # 1) Download folder contents
    folder_dir = TMP_DIR / "folder"
    download_drive_folder(folder_dir)

    # 2) Download PBJ standalone file
    pbj_dir = TMP_DIR / "pbj"
    download_pbj_file(pbj_dir)

    # 3) Process all downloaded data files
    all_files = list(iter_data_files(folder_dir)) + list(iter_data_files(pbj_dir))
    print(f"Downloaded data files: {len(all_files)}")
    print([p.name for p in all_files[:5]])

    for file_path in all_files:
        filename = file_path.name
        dataset_key = dataset_key_from_filename(filename)

        try:
            checksum = sha256_file(file_path)
            size_bytes = file_path.stat().st_size

            if prior_checksums.get(filename) == checksum:
                manifest["skipped_unchanged"].append(
                    {"filename": filename, "sha256": checksum, "size_bytes": size_bytes}
                )
                file_path.unlink(missing_ok=True)
                continue

            s3_key = f"{RAW_PREFIX}/{dataset_key}/ingest_dt={ingest_dt}/{filename}"
            s3_upload_file(file_path, BUCKET, s3_key)

            manifest["files"].append(
                {
                    "filename": filename,
                    "dataset_key": dataset_key,
                    "s3_key": s3_key,
                    "size_bytes": size_bytes,
                    "sha256": checksum,
                }
            )

        except Exception as e:
            # quarantine on error
            try:
                err_key = f"{QUARANTINE_PREFIX}/{dataset_key}/ingest_dt={ingest_dt}/{filename}"
                s3_upload_file(file_path, BUCKET, err_key)
            except Exception:
                pass

            manifest["errors"].append({"filename": filename, "dataset_key": dataset_key, "error": repr(e)})

        finally:
            # cleanup local file
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception:
                pass

    # 4) Write manifests
    archive_key = f"{CONTROL_PREFIX}/manifests/ingest_dt={ingest_dt}/manifest.json"
    s3_put_json(BUCKET, archive_key, manifest)
    s3_put_json(BUCKET, LATEST_MANIFEST_KEY, manifest)

    return {
        "status": "ok" if not manifest["errors"] else "partial_failure",
        "uploaded": len(manifest["files"]),
        "skipped_unchanged": len(manifest["skipped_unchanged"]),
        "errors": len(manifest["errors"]),
        "archive_manifest_s3_key": archive_key,
        "latest_manifest_s3_key": LATEST_MANIFEST_KEY,
    }


def handler(event, context):
    return ingest_once()


if __name__ == "__main__":
    print(json.dumps(ingest_once(), indent=2))