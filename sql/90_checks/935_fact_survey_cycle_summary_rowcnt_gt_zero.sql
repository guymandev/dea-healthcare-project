SELECT count(*)
from healthcare_curated_db.fact_survey_cycle_summary
WHERE ingest_dt = '{{INGEST_DT}}';