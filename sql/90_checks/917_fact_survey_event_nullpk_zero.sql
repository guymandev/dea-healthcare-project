SELECT coalesce(
  sum(
    CASE
      WHEN ccn_provnum IS NULL
        OR survey_date IS NULL
        OR survey_type IS NULL
        OR trim(survey_type) = ''
      THEN 1
      ELSE 0
    END
  ),
  0
) AS null_pk_rows
FROM healthcare_curated_db.fact_survey_event
WHERE ingest_dt = '{{INGEST_DT}}';