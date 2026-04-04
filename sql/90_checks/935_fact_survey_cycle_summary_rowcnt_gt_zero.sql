SELECT count(*)
from healthcare_curated_db.fact_survey_cycle_summary
WHERE ingest_dt = '{{INGEST_DT:nh_surveysummary_oct2024}}';