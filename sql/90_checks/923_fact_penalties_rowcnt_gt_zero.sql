select count(*) from healthcare_curated_db.fact_penalties 
WHERE ingest_dt = '{{INGEST_DT}}';