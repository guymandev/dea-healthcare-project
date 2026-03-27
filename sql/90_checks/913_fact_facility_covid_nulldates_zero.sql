select
  sum(case when vax_data_last_update is null then 1 else 0 end) as null_dates
from healthcare_curated_db.fact_facility_covid
WHERE ingest_dt = '{{INGEST_DT}}';