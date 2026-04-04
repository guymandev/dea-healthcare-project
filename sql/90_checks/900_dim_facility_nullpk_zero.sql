SELECT coalesce(sum(CASE WHEN ccn_provnum IS NULL THEN 1 ELSE 0 END), 0) AS null_pk
FROM healthcare_curated_db.dim_facility
WHERE ingest_dt = '{{MANIFEST_RUN_INGEST_DT}}';