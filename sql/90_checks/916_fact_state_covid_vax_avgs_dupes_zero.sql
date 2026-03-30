SELECT coalesce(count(*), 0) AS dupes
FROM (
  SELECT
    state,
    date_vax_data_last_updated
  FROM healthcare_curated_db.fact_state_covid_vax_avgs
  WHERE ingest_dt = '{{INGEST_DT}}'
  GROUP BY state, date_vax_data_last_updated
  HAVING count(*) > 1
) d;