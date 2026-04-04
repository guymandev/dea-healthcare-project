select
  sum(case when ccn_provnum is null then 1 else 0 end) as null_ccn,
  sum(case when inspection_cycle is null then 1 else 0 end) as null_cycle
from healthcare_curated_db.fact_survey_cycle_summary
WHERE ingest_dt = '{{INGEST_DT:nh_surveysummary_oct2024}}';