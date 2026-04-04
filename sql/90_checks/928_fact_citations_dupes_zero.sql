SELECT
  count(*) - count(
    distinct ccn_provnum || '|' || cast(survey_date as varchar) || '|' || deficiency_tag_nbr || '|' || survey_type
  ) AS dupes
FROM healthcare_curated_db.fact_citations 
WHERE ingest_dt = '{{INGEST_DT:nh_healthcitations_oct2024}}';