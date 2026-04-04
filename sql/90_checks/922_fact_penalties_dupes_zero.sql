select
  count(*) - count(distinct ccn_provnum || '|' || cast(penalty_date as varchar) || '|' || penalty_type || '|' || cast(penalty_amt as varchar)) as dupes
from healthcare_curated_db.fact_penalties 
WHERE ingest_dt = '{{INGEST_DT:nh_penalties_oct2024}}';