SELECT COUNT(*)
FROM healthcare_curated_db.fact_citations 
WHERE ingest_dt = '{{INGEST_DT:nh_healthcitations_oct2024}}';