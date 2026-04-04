select
  sum(case when state is null then 1 else 0 end) as null_state
from healthcare_curated_db.fact_state_covid_vax_avgs
WHERE ingest_dt = '{{INGEST_DT:nh_covidvaxaverages_20241027}}';