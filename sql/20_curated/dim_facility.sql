-- S3_PREFIX: s3://healthcare-data-lake-gj/curated/dim_facility/

DROP TABLE IF EXISTS healthcare_curated_db.dim_facility;

CREATE TABLE healthcare_curated_db.dim_facility
WITH (
  format = 'PARQUET',
  external_location = 's3://healthcare-data-lake-gj/curated/dim_facility/',
  parquet_compression = 'SNAPPY'
) AS
WITH provider AS (
  SELECT
    lpad(
      regexp_replace(trim(cast(p."cms certification number (ccn)" AS varchar)), '"', ''),
      6, '0'
    ) AS ccn_provnum,
    cast(p."provider name" as varchar)    AS provider_name,
    cast(p."provider address" as varchar) AS provider_address,
    cast(p."city/town" as varchar)        AS provider_city,
    cast(p."state" as varchar)            AS provider_state,
    lpad(cast(p."zip code" as varchar), 5, '0') AS provider_zip_code
  FROM healthcare_catalog_db.raw_nh_providerinfo_oct2024 p
  WHERE p.ingest_dt = '{{INGEST_DT}}'
    AND p."cms certification number (ccn)" IS NOT NULL
),
swing AS (
  SELECT
    lpad(regexp_replace(trim(cms_certification_number_ccn), '"', ''), 6, '0') AS ccn_provnum,
    try_cast(regexp_replace(trim(cms_region), '"', '') as integer) AS cms_region
  FROM healthcare_catalog_db.raw_swing_bed_snf_data_oct2024_fixed
  WHERE ingest_dt = '{{INGEST_DT}}'
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
  sw.cms_region
FROM provider pr
LEFT JOIN swing sw
  ON pr.ccn_provnum = sw.ccn_provnum;