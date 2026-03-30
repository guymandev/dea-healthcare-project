SELECT coalesce(count(*), 0) AS dupes
FROM (
  SELECT
    ccn_provnum,
    survey_date,
    survey_type
  FROM healthcare_curated_db.fact_survey_event
  WHERE ingest_dt = '{{INGEST_DT}}'
  GROUP BY ccn_provnum, survey_date, survey_type
  HAVING count(*) > 1
) d;