select 
  sum(case when ccn_provnum is null then 1 else 0 end) + 
  sum(case when penalty_date is null then 1 else 0 end) +
  sum(case when penalty_type is null then 1 else 0 end) + 
  sum(case when penalty_amt is null then 1 else 0 end) 
from healthcare_curated_db.fact_penalties 
WHERE ingest_dt = '{{INGEST_DT}}';