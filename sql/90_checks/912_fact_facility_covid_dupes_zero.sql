select count(*) - count(distinct ccn_provnum) as dupes
from healthcare_curated_db.fact_facility_covid
WHERE ingest_dt = '{{INGEST_DT:nh_covidvaxprovider_20241027}}';