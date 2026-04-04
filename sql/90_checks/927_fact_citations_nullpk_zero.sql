SELECT
  sum(CASE WHEN ccn_provnum IS NULL THEN 1 ELSE 0 END) + 
  sum(CASE WHEN survey_date IS NULL THEN 1 ELSE 0 END) + 
  sum(CASE WHEN deficiency_tag_nbr IS NULL THEN 1 ELSE 0 END) + 
  sum(CASE WHEN survey_type IS NULL THEN 1 ELSE 0 END) + 
  sum(CASE WHEN date_id IS NULL THEN 1 ELSE 0 END) 
FROM healthcare_curated_db.fact_citations
WHERE ingest_dt = '{{INGEST_DT:nh_healthcitations_oct2024}}';