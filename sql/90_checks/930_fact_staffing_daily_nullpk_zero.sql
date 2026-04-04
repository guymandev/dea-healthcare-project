SELECT coalesce(
  sum(
    CASE
      WHEN ccn_provnum IS NULL
        OR workdate IS NULL
      THEN 1
      ELSE 0
    END
  ),
  0
) AS null_pk_rows
FROM healthcare_curated_db.fact_staffing_daily
WHERE ingest_dt = '{{INGEST_DT:pbj_daily_nurse_staffing_q2_2024}}';