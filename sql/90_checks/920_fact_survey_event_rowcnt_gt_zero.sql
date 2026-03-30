SELECT count(*) AS row_cnt
FROM healthcare_curated_db.fact_survey_event
WHERE ingest_dt = '{{INGEST_DT}}';