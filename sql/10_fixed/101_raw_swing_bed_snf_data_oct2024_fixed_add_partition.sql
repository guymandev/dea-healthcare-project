ALTER TABLE healthcare_catalog_db.raw_swing_bed_snf_data_oct2024_fixed
ADD IF NOT EXISTS
PARTITION (ingest_dt='{{INGEST_DT}}')
LOCATION 's3://healthcare-data-lake-gj/raw/swing_bed_snf_data_oct2024/ingest_dt={{INGEST_DT}}/';