import hashlib
import json
import os
from datetime import date
from pathlib import Path

import boto3
import gdown


# ---------------------------
# Config
# ---------------------------
BUCKET = "healthcare-data-lake-gj"
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/15KqJ1MZ7JcgAkOfqcaWcALWkG0dh3jpE"
LOCAL_DOWNLOAD_DIR = Path("./data_download")
INGEST_DT = date.today().isoformat()  # e.g. 2026-03-07


# ---------------------------
# Helpers
# ---------------------------
def dataset_key_from_filename(filename: str) -> str:
    """
    Turn a filename into a stable dataset prefix.
    Keep it simple and deterministic.
    """
    base = filename.lower().replace(".csv", "").replace(".json", "")
    base = base.replace("__", "_").replace(" ", "_")
    # Optional: normalize known patterns
    base = base.replace("nh_", "nh_")
    return base


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def upload_to_s3(s3_client, local_path: Path, bucket: str, key: str) -> None:
    """
    Upload file to S3 at s3://bucket/key
    """
    s3_client.upload_file(
        Filename=str(local_path),
        Bucket=bucket,
        Key=key,
    )


# ---------------------------
# Main ingestion flow
# ---------------------------
def main():
    LOCAL_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Download files from Drive folder into LOCAL_DOWNLOAD_DIR
    # NOTE: use_cookies=False works if folder is public/anyone-with-link.
    gdown.download_folder(
        DRIVE_FOLDER_URL,
        output=str(LOCAL_DOWNLOAD_DIR),
        quiet=False,
        use_cookies=False,
    )

    s3 = boto3.client("s3")

    manifest = {
        "ingest_dt": INGEST_DT,
        "source": "google_drive",
        "files": [],
    }

    # 2) Upload each CSV to S3 RAW with the desired key (prefix)
    for file_path in sorted(LOCAL_DOWNLOAD_DIR.glob("*")):
        if not file_path.is_file():
            continue

        # Skip non-data artifacts (like PDFs)
        if file_path.suffix.lower() not in {".csv", ".json"}:
            continue

        filename = file_path.name
        dataset_key = dataset_key_from_filename(filename)

        s3_key = f"raw/{dataset_key}/ingest_dt={INGEST_DT}/{filename}"

        checksum = sha256_file(file_path)
        size_bytes = file_path.stat().st_size

        print(f"Uploading {filename} -> s3://{BUCKET}/{s3_key}")
        upload_to_s3(s3, file_path, BUCKET, s3_key)

        manifest["files"].append(
            {
                "filename": filename,
                "dataset_key": dataset_key,
                "s3_key": s3_key,
                "size_bytes": size_bytes,
                "sha256": checksum,
            }
        )

    # 3) Upload manifest to S3 control/
    manifest_key = f"control/manifests/ingest_dt={INGEST_DT}/manifest.json"
    print(f"Uploading manifest -> s3://{BUCKET}/{manifest_key}")

    s3.put_object(
        Bucket=BUCKET,
        Key=manifest_key,
        Body=json.dumps(manifest, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    print("Done.")


if __name__ == "__main__":
    main()