select
  count(*) - count(distinct ccn_provnum || '|' || cast(inspection_cycle as varchar)) as dupes
from healthcare_curated_db.fact_survey_cycle_summary
WHERE ingest_dt = '{{INGEST_DT}}';