select count(*)
from healthcare_curated_db.fact_facility_covid
WHERE ingest_dt = '{{INGEST_DT}}';