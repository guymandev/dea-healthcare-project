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
      regexp_replace(trim(cast(p."cms certification number (ccn)" AS varchar)), '"', ''),
      6, '0'
    ) AS ccn_provnum,

    cast(p."provider name" AS varchar)    AS provider_name,
    cast(p."provider address" AS varchar) AS provider_address,
    cast(p."city/town" AS varchar)        AS provider_city,
    cast(p."state" AS varchar)            AS provider_state,
    lpad(cast(p."zip code" AS varchar), 5, '0') AS provider_zip_code,

    try_cast(p."number of certified beds" AS integer) AS number_of_certified_beds,

    try_cast(p."average number of residents per day" AS double) AS avg_residents_per_day,
    cast(p."average number of residents per day footnote" AS varchar) AS avg_residents_per_day_footnote,

    try_cast(p."reported rn staffing hours per resident per day" AS double)
      AS reported_rn_staffing_hours_per_resident_per_day,

    try_cast(p."reported licensed staffing hours per resident per day" AS double)
      AS reported_licensed_staffing_hours_per_resident_per_day,

    try_cast(p."reported total nurse staffing hours per resident per day" AS double)
      AS reported_total_nurse_staffing_hours_per_resident_per_day,

    try_cast(p."total number of nurse staff hours per resident per day on the weekend" AS double)
      AS weekend_total_nurse_staffing_hours_per_resident_per_day,

    try_cast(p."registered nurse hours per resident per day on the weekend" AS double)
      AS weekend_rn_staffing_hours_per_resident_per_day,

    try_cast(p."reported physical therapist staffing hours per resident per day" AS double)
      AS reported_physical_therapist_staffing_hours_per_resident_per_day,

    try_cast(p."nursing case-mix index" AS double) AS nursing_case_mix_index,
    try_cast(p."nursing case-mix index ratio" AS double) AS nursing_case_mix_index_ratio,

    try_cast(p."case-mix nurse aide staffing hours per resident per day" AS double)
      AS case_mix_nurse_aide_staffing_hours_per_resident_per_day,

    try_cast(p."case-mix lpn staffing hours per resident per day" AS double)
      AS case_mix_lpn_staffing_hours_per_resident_per_day,

    try_cast(p."case-mix rn staffing hours per resident per day" AS double)
      AS case_mix_rn_staffing_hours_per_resident_per_day,

    try_cast(p."case-mix total nurse staffing hours per resident per day" AS double)
      AS case_mix_total_nurse_staffing_hours_per_resident_per_day,

    try_cast(p."case-mix weekend total nurse staffing hours per resident per day" AS double)
      AS case_mix_weekend_total_nurse_staffing_hours_per_resident_per_day,

    try_cast(p."adjusted nurse aide staffing hours per resident per day" AS double)
      AS adjusted_nurse_aide_staffing_hours_per_resident_per_day,

    try_cast(p."adjusted lpn staffing hours per resident per day" AS double)
      AS adjusted_lpn_staffing_hours_per_resident_per_day,

    try_cast(p."adjusted rn staffing hours per resident per day" AS double)
      AS adjusted_rn_staffing_hours_per_resident_per_day,

    try_cast(p."adjusted total nurse staffing hours per resident per day" AS double)
      AS adjusted_total_nurse_staffing_hours_per_resident_per_day,

    try_cast(p."adjusted weekend total nurse staffing hours per resident per day" AS double)
      AS adjusted_weekend_total_nurse_staffing_hours_per_resident_per_day

  FROM healthcare_catalog_db.raw_nh_providerinfo_oct2024 p
  WHERE p.ingest_dt = '{{INGEST_DT:nh_providerinfo_oct2024}}'
    AND p."cms certification number (ccn)" IS NOT NULL
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