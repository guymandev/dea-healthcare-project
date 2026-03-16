import base64
import hashlib
import json
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from copy import deepcopy

import boto3
import gdown
from gdown.exceptions import FileURLRetrievalError

from botocore.config import Config
from botocore.exceptions import ClientError
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
    # More robust retry behavior than boto3 defaults
    cfg = Config(
        retries={"max_attempts": 10, "mode": "adaptive"},
        connect_timeout=10,
        read_timeout=60,
    )
    return boto3.client("s3", config=cfg)


def s3_get_json(bucket: str, key: str) -> Optional[Dict[str, Any]]:
    s3 = s3_client()
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except s3.exceptions.NoSuchKey:
        return None


def s3_put_json(
    bucket: str,
    key: str,
    payload: Dict[str, Any],
    *,
    atomic: bool = True,
    sse: Optional[str] = None,          # e.g. "AES256" or "aws:kms"
    kms_key_id: Optional[str] = None,   # if using aws:kms
) -> None:
    """
    Writes JSON to S3 in a way that is:
      - compact (Athena-friendly)
      - integrity-checked (Content-MD5)
      - retry-hardened
      - optionally "atomic" (temp write + copy)
    """
    s3 = s3_client()

    body_str = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    body_bytes = body_str.encode("utf-8")

    # S3 will validate this and fail if corrupted in transit
    md5_b64 = base64.b64encode(hashlib.md5(body_bytes).digest()).decode("ascii")

    put_kwargs = {
        "Bucket": bucket,
        "Key": key,
        "Body": body_bytes,
        "ContentType": "application/json",
        "CacheControl": "no-store",
        "ContentMD5": md5_b64,
    }

    # Optional SSE
    if sse:
        put_kwargs["ServerSideEncryption"] = sse
        if sse == "aws:kms" and kms_key_id:
            put_kwargs["SSEKMSKeyId"] = kms_key_id

    if not atomic:
        s3.put_object(**put_kwargs)
        return

    # Atomic-ish: write to a temp key first, then copy over final key
    tmp_key = f"{key}.__tmp__{os.urandom(6).hex()}"
    try:
        put_kwargs_tmp = dict(put_kwargs)
        put_kwargs_tmp["Key"] = tmp_key
        s3.put_object(**put_kwargs_tmp)

        copy_kwargs = {
            "Bucket": bucket,
            "CopySource": {"Bucket": bucket, "Key": tmp_key},
            "Key": key,
            "MetadataDirective": "REPLACE",
            "ContentType": "application/json",
            "CacheControl": "no-store",
        }
        if sse:
            copy_kwargs["ServerSideEncryption"] = sse
            if sse == "aws:kms" and kms_key_id:
                copy_kwargs["SSEKMSKeyId"] = kms_key_id

        s3.copy_object(**copy_kwargs)

    except ClientError:
        # Bubble up so your pipeline can quarantine / alert
        raise
    finally:
        # Best-effort cleanup
        try:
            s3.delete_object(Bucket=bucket, Key=tmp_key)
        except Exception:
            pass


def s3_upload_file(local_path: Path, bucket: str, key: str) -> None:
    s3 = s3_client()
    s3.upload_file(Filename=str(local_path), Bucket=bucket, Key=key)

# ---------------------------
# Text file helpers for creating JSONL manifest
# ---------------------------
def s3_put_text(bucket: str, key: str, text: str, content_type: str = "text/plain") -> None:
    s3 = s3_client()
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=text.encode("utf-8"),
        ContentType=content_type,
    )

def manifest_files_to_jsonl(latest_manifest: Dict[str, Any]) -> str:
    """
    Emit one JSON object per line, one per file entry in latest_manifest['files'].
    Each line also includes ingest_dt and run_ts_utc for easy querying.
    """
    ingest_dt = latest_manifest.get("ingest_dt")
    run_ts_utc = latest_manifest.get("run_ts_utc")

    lines = []
    for f in latest_manifest.get("files", []):
        row = {
            "ingest_dt": ingest_dt,
            "run_ts_utc": run_ts_utc,
            # the rest is the file entry
            **f,
        }
        lines.append(json.dumps(row, separators=(",", ":"), ensure_ascii=False))
    return "\n".join(lines) + ("\n" if lines else "")

# Utility function to execute a one-off write of the JSONL file
# without needing to rerun the entire ingest.
def write_latest_files_jsonl_only() -> None:
    latest = s3_get_json(BUCKET, LATEST_MANIFEST_KEY) or {}
    jsonl_text = manifest_files_to_jsonl(latest)
    key = f"{CONTROL_PREFIX}/manifests/latest_files/files.jsonl"
    s3_put_text(BUCKET, key, jsonl_text, content_type="application/x-ndjson")
    print("Wrote:", key)

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
def download_drive_folder(output_dir: Path, *, max_attempts: int = 5) -> None:
    ensure_dir(output_dir)

    delay = 5
    last_err = None

    for attempt in range(1, max_attempts + 1):
        try:
            print(f"[gdown] download_folder attempt {attempt}/{max_attempts} ...")
            gdown.download_folder(
                DRIVE_FOLDER_URL,
                output=str(output_dir),
                quiet=False,
                use_cookies=True,
            )
            return  # success
        except FileURLRetrievalError as e:
            last_err = e
            print(f"[gdown] FileURLRetrievalError: {e}")
            if attempt == max_attempts:
                break
            print(f"[gdown] retrying in {delay}s ...")
            time.sleep(delay)
            delay *= 3  # exponential-ish backoff

    # If we got here, all attempts failed
    raise last_err


def download_pbj_file(output_dir: Path, *, max_attempts: int = 5) -> Path:
    """
    Downloads the PBJ standalone file into output_dir with retries.
    Returns the downloaded file path.
    """
    ensure_dir(output_dir)
    out_path = output_dir / PBJ_FILENAME

    delay = 5
    last_err = None
    url = f"https://drive.google.com/uc?id={PBJ_FILE_ID}"

    for attempt in range(1, max_attempts + 1):
        try:
            print(f"[gdown] PBJ download attempt {attempt}/{max_attempts} ...")
            # use_cookies=True helps with Drive confirmation / throttling flows
            gdown.download(url, str(out_path), quiet=False, use_cookies=True)
            return out_path
        except FileURLRetrievalError as e:
            last_err = e
            print(f"[gdown] PBJ FileURLRetrievalError: {e}")
            if attempt == max_attempts:
                break
            print(f"[gdown] retrying PBJ in {delay}s ...")
            time.sleep(delay)
            delay *= 3

    raise last_err


# ---------------------------
# Ingest
# ---------------------------
def ingest_once() -> Dict[str, Any]:
    ingest_dt = date.today().isoformat()
    run_ts = datetime.now(timezone.utc).isoformat()

    ensure_dir(TMP_DIR)

    print("BUCKET =", BUCKET)
    print("LATEST_MANIFEST_KEY =", LATEST_MANIFEST_KEY)
    print("AWS_PROFILE =", os.environ.get("AWS_PROFILE"))

    # ---------------------------
    # Load prior manifest + build prior_checksums from BOTH lists
    # ---------------------------
    prior = s3_get_json(BUCKET, LATEST_MANIFEST_KEY) or {}
    print("prior loaded? ", prior is not None)

    # Collect prior entries from multiple possible keys (backward compatible)
    prior_entries = []
    for k in ("files", "skipped_unchanged", "files_uploaded", "all_files"):
        v = prior.get(k)
        if isinstance(v, list):
            prior_entries.extend(v)

    # print("prior entries count:", len(prior_entries))

    # Index by basename so path differences never break matching
    prior_by_name: Dict[str, Dict[str, Any]] = {}
    for e in prior_entries:
        fn = e.get("filename")
        sha = e.get("sha256")
        if not fn or not sha:
            continue
        key = Path(fn).name
        # keep the first one; they're usually identical anyway
        prior_by_name.setdefault(key, e)

    prior_checksums = {k: v.get("sha256") for k, v in prior_by_name.items()}
    print("prior unique filenames:", len(prior_checksums))

    # ---------------------------
    # Build this run's manifest (UNCHANGED structure)
    # ---------------------------
    manifest: Dict[str, Any] = {
        "ingest_dt": ingest_dt,
        "run_ts_utc": run_ts,
        "source": "google_drive_public",
        "bucket": BUCKET,
        "raw_prefix": RAW_PREFIX,
        "files": [],              # uploaded this run
        "skipped_unchanged": [],  # skipped this run
        "errors": [],
    }

    # 1) Download folder contents
    folder_dir = TMP_DIR / "folder"
    try:
        download_drive_folder(folder_dir)
    except FileURLRetrievalError as e:
        # record the issue at the run level and continue (maybe PBJ still works)
        manifest["errors"].append({
            "filename": None,
            "dataset_key": None,
            "error": f"Drive folder download failed: {repr(e)}"
        })
        # Continue the run with whatever may have downloaded already

    # 2) Download PBJ standalone file
    pbj_dir = TMP_DIR / "pbj"
    # download_pbj_file(pbj_dir)
    try:
        download_pbj_file(pbj_dir)
    except FileURLRetrievalError as e:
        manifest["errors"].append({
            "filename": PBJ_FILENAME,
            "dataset_key": "pbj_daily_nurse_staffing_q2_2024",
            "error": f"PBJ download failed: {repr(e)}"
        })


    # 3) Process all downloaded data files
    all_files = list(iter_data_files(folder_dir)) + list(iter_data_files(pbj_dir))
    print(f"Downloaded data files: {len(all_files)}")
    print([p.name for p in all_files[:5]])

    for file_path in all_files:
        filename = file_path.name
        filename_key = Path(filename).name  # basename
        dataset_key = dataset_key_from_filename(filename)

        try:
            checksum = sha256_file(file_path)
            size_bytes = file_path.stat().st_size

            if filename_key in prior_checksums:
                print("COMPARE", filename_key, "prior:", prior_checksums[filename_key], "new:", checksum)
            else:
                print("NO PRIOR ENTRY FOR", filename_key)

            if prior_checksums.get(filename_key) == checksum:
                manifest["skipped_unchanged"].append(
                    {"filename": filename_key, "sha256": checksum, "size_bytes": size_bytes}
                )
                file_path.unlink(missing_ok=True)
                continue

            s3_key = f"{RAW_PREFIX}/{dataset_key}/ingest_dt={ingest_dt}/{filename_key}"
            s3_upload_file(file_path, BUCKET, s3_key)

            manifest["files"].append(
                {
                    "filename": filename_key,
                    "dataset_key": dataset_key,
                    "s3_key": s3_key,
                    "size_bytes": size_bytes,
                    "sha256": checksum,
                }
            )

        except Exception as e:
            try:
                err_key = f"{QUARANTINE_PREFIX}/{dataset_key}/ingest_dt={ingest_dt}/{filename_key}"
                s3_upload_file(file_path, BUCKET, err_key)
            except Exception:
                pass

            manifest["errors"].append({"filename": filename_key, "dataset_key": dataset_key, "error": repr(e)})

        finally:
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception:
                pass

    # ---------------------------
    # 4) Write manifests (archive run manifest + durable latest state)
    # ---------------------------
    archive_key = f"{CONTROL_PREFIX}/manifests/ingest_dt={ingest_dt}/manifest.json"

    # Archive manifest: keep exactly as-is (so your current outputs remain correct)
    s3_put_json(BUCKET, archive_key, manifest)

    # Latest manifest: durable state snapshot
    # - keep current fields
    # - but also ensure `files` contains ALL known file entries so future runs don't oscillate
    latest_manifest = deepcopy(manifest)
    latest_manifest["manifest_kind"] = "latest_state"
    latest_manifest["archive_manifest_s3_key"] = archive_key  # breadcrumb

    # Start from prior state (all known files), then overlay updates from this run
    # We'll store "state" entries in latest_manifest["files"].
    state_by_name: Dict[str, Dict[str, Any]] = {}

    # Seed with prior entries
    for name, e in prior_by_name.items():
        state_by_name[name] = dict(e)

    # Apply skipped: ensure they exist in state (keep prior s3_key/dataset_key if present)
    for e in manifest["skipped_unchanged"]:
        name = Path(e["filename"]).name
        prev = state_by_name.get(name, {})
        state_by_name[name] = {
            **prev,
            "filename": name,
            "sha256": e.get("sha256"),
            "size_bytes": e.get("size_bytes"),
        }

    # Apply uploaded: overwrite state with new s3_key etc.
    for e in manifest["files"]:
        name = Path(e["filename"]).name
        state_by_name[name] = dict(e)

    # In latest_state, put the *state* into `files`
    latest_manifest["files"] = list(state_by_name.values())

    # Optional: keep a convenience list too (harmless)
    latest_manifest["all_files"] = latest_manifest["files"]

    # Write latest manifest JSON
    s3_put_json(BUCKET, LATEST_MANIFEST_KEY, latest_manifest)

    # 5) Also write JSONL "latest files" view for Athena
    LATEST_FILES_JSONL_KEY = f"{CONTROL_PREFIX}/manifests/latest_files/files.jsonl"
    jsonl_text = manifest_files_to_jsonl(latest_manifest)  # uses latest_manifest["files"]
    s3_put_text(
        BUCKET,
        LATEST_FILES_JSONL_KEY,
        jsonl_text,
        content_type="application/x-ndjson"
    )

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
    # write_latest_files_jsonl_only()