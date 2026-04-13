-- S3_PREFIX: s3://healthcare-data-lake-gj/curated/dim_facility/

DROP TABLE IF EXISTS healthcare_curated_db.dim_facility;

CREATE TABLE healthcare_curated_db.dim_facility
WITH (
  format = 'PARQUET',
  external_location = 's3://healthcare-data-lake-gj/curated/dim_facility/',
  parquet_compression = 'SNAPPY',
  partitioned_by = ARRAY['ingest_dt']
) AS
WITH provider AS (
  SELECT
    lpad(
      regexp_replace(trim(p.ccn_provnum), '"', ''),
      6, '0'
    ) AS ccn_provnum,
    regexp_replace(trim(p.provider_name), '"', '')    AS provider_name,
    regexp_replace(trim(p.provider_address), '"', '') AS provider_address,
    regexp_replace(trim(p.provider_city), '"', '')    AS provider_city,
    regexp_replace(trim(p.provider_state), '"', '')   AS provider_state,
    lpad(regexp_replace(trim(p.provider_zip_code), '"', ''), 5, '0') AS provider_zip_code,
    try_cast(p.number_of_certified_beds AS integer)   AS number_of_certified_beds,
    try_cast(p.avg_residents_per_day AS double)       AS avg_residents_per_day,
    regexp_replace(trim(p.avg_residents_per_day_footnote), '"', '') AS avg_residents_per_day_footnote,
    try_cast(p.reported_rn_staffing_hours_per_resident_per_day AS double)
      AS reported_rn_staffing_hours_per_resident_per_day,
    try_cast(p.reported_licensed_staffing_hours_per_resident_per_day AS double)
      AS reported_licensed_staffing_hours_per_resident_per_day,
    try_cast(p.reported_total_nurse_staffing_hours_per_resident_per_day AS double)
      AS reported_total_nurse_staffing_hours_per_resident_per_day,
    try_cast(p.weekend_total_nurse_staffing_hours_per_resident_per_day AS double)
      AS weekend_total_nurse_staffing_hours_per_resident_per_day,
    try_cast(p.weekend_rn_staffing_hours_per_resident_per_day AS double)
      AS weekend_rn_staffing_hours_per_resident_per_day,
    try_cast(p.reported_physical_therapist_staffing_hours_per_resident_per_day AS double)
      AS reported_physical_therapist_staffing_hours_per_resident_per_day,
    try_cast(p.nursing_case_mix_index AS double)      AS nursing_case_mix_index,
    try_cast(p.nursing_case_mix_index_ratio AS double) AS nursing_case_mix_index_ratio,
    try_cast(p.case_mix_nurse_aide_staffing_hours_per_resident_per_day AS double)
      AS case_mix_nurse_aide_staffing_hours_per_resident_per_day,
    try_cast(p.case_mix_lpn_staffing_hours_per_resident_per_day AS double)
      AS case_mix_lpn_staffing_hours_per_resident_per_day,
    try_cast(p.case_mix_rn_staffing_hours_per_resident_per_day AS double)
      AS case_mix_rn_staffing_hours_per_resident_per_day,
    try_cast(p.case_mix_total_nurse_staffing_hours_per_resident_per_day AS double)
      AS case_mix_total_nurse_staffing_hours_per_resident_per_day,
    try_cast(p.case_mix_weekend_total_nurse_staffing_hours_per_resident_per_day AS double)
      AS case_mix_weekend_total_nurse_staffing_hours_per_resident_per_day,
    try_cast(p.adjusted_nurse_aide_staffing_hours_per_resident_per_day AS double)
      AS adjusted_nurse_aide_staffing_hours_per_resident_per_day,
    try_cast(p.adjusted_lpn_staffing_hours_per_resident_per_day AS double)
      AS adjusted_lpn_staffing_hours_per_resident_per_day,
    try_cast(p.adjusted_rn_staffing_hours_per_resident_per_day AS double)
      AS adjusted_rn_staffing_hours_per_resident_per_day,
    try_cast(p.adjusted_total_nurse_staffing_hours_per_resident_per_day AS double)
      AS adjusted_total_nurse_staffing_hours_per_resident_per_day,
    try_cast(p.adjusted_weekend_total_nurse_staffing_hours_per_resident_per_day AS double)
      AS adjusted_weekend_total_nurse_staffing_hours_per_resident_per_day
  FROM healthcare_catalog_db.raw_nh_providerinfo_oct2024_fixed p
  WHERE p.ingest_dt = '{{INGEST_DT:nh_providerinfo_oct2024}}'
    AND p.ccn_provnum IS NOT NULL
),
swing AS (
  SELECT
    lpad(regexp_replace(trim(cms_certification_number_ccn), '"', ''), 6, '0') AS ccn_provnum,
    try_cast(regexp_replace(trim(cms_region), '"', '') AS integer) AS cms_region
  FROM healthcare_catalog_db.raw_swing_bed_snf_data_oct2024_fixed
  WHERE ingest_dt = '{{INGEST_DT:swing_bed_snf_data_oct2024}}'
    AND cms_certification_number_ccn IS NOT NULL
  GROUP BY 1, 2
)
SELECT
  pr.ccn_provnum,
  pr.provider_name,
  pr.provider_address,
  pr.provider_city,
  pr.provider_state,
  pr.provider_zip_code,
  sw.cms_region,
  pr.number_of_certified_beds,
  pr.avg_residents_per_day,
  pr.avg_residents_per_day_footnote,
  pr.reported_rn_staffing_hours_per_resident_per_day,
  pr.reported_licensed_staffing_hours_per_resident_per_day,
  pr.reported_total_nurse_staffing_hours_per_resident_per_day,
  pr.weekend_total_nurse_staffing_hours_per_resident_per_day,
  pr.weekend_rn_staffing_hours_per_resident_per_day,
  pr.reported_physical_therapist_staffing_hours_per_resident_per_day,
  pr.nursing_case_mix_index,
  pr.nursing_case_mix_index_ratio,
  pr.case_mix_nurse_aide_staffing_hours_per_resident_per_day,
  pr.case_mix_lpn_staffing_hours_per_resident_per_day,
  pr.case_mix_rn_staffing_hours_per_resident_per_day,
  pr.case_mix_total_nurse_staffing_hours_per_resident_per_day,
  pr.case_mix_weekend_total_nurse_staffing_hours_per_resident_per_day,
  pr.adjusted_nurse_aide_staffing_hours_per_resident_per_day,
  pr.adjusted_lpn_staffing_hours_per_resident_per_day,
  pr.adjusted_rn_staffing_hours_per_resident_per_day,
  pr.adjusted_total_nurse_staffing_hours_per_resident_per_day,
  pr.adjusted_weekend_total_nurse_staffing_hours_per_resident_per_day,
  CAST('{{MANIFEST_RUN_INGEST_DT}}' AS varchar) AS ingest_dt
FROM provider pr
LEFT JOIN swing sw
  ON pr.ccn_provnum = sw.ccn_provnum;
