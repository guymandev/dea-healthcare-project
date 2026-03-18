ALTER TABLE healthcare_catalog_db.raw_nh_providerinfo_oct2024
ADD IF NOT EXISTS
PARTITION (ingest_dt='{{INGEST_DT}}')
LOCATION 's3://healthcare-data-lake-gj/raw/nh_providerinfo_oct2024/ingest_dt={{INGEST_DT}}/';