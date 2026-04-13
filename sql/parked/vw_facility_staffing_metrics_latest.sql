CREATE OR REPLACE VIEW healthcare_curated_db.vw_facility_staffing_metrics_latest AS
WITH latest AS (
  SELECT max(ingest_dt) AS latest_ingest_dt
  FROM healthcare_curated_db.dim_facility
)
SELECT
  f.ccn_provnum,
  f.provider_name,
  f.provider_city,
  f.provider_state,
  f.provider_zip_code,
  f.cms_region,
  f.number_of_certified_beds,
  f.average_number_of_residents_per_day,
  f.adjusted_total_nurse_staffing_hours_per_resident_per_day,
  f.adjusted_rn_staffing_hours_per_resident_per_day,

  -- metric 1: estimated total nurse hours per day
  (
    coalesce(f.average_number_of_residents_per_day, 0.0) *
    coalesce(f.adjusted_total_nurse_staffing_hours_per_resident_per_day, 0.0)
  ) AS estimated_total_nurse_hours_per_day,

  -- metric 2: nurse hours per patient
  f.adjusted_total_nurse_staffing_hours_per_resident_per_day AS nurse_hours_per_patient,

  -- optional helper metric
  CASE
    WHEN coalesce(f.number_of_certified_beds, 0) > 0
    THEN f.average_number_of_residents_per_day / cast(f.number_of_certified_beds AS double)
    ELSE NULL
  END AS occupancy_proxy,

  f.ingest_dt
FROM healthcare_curated_db.dim_facility f
CROSS JOIN latest l
WHERE f.ingest_dt = l.latest_ingest_dt;