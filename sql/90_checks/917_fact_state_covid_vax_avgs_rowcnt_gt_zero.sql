select count(*)
from healthcare_curated_db.fact_state_covid_vax_avgs
WHERE ingest_dt = '{{INGEST_DT:nh_covidvaxaverages_20241027}}';