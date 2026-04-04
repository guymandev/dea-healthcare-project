DROP TABLE IF EXISTS healthcare_curated_db.fact_citations;

-- S3_PREFIX: s3://healthcare-data-lake-gj/curated/fact_citations/

CREATE TABLE healthcare_curated_db.fact_citations
WITH (
  format = 'PARQUET',
  external_location = 's3://healthcare-data-lake-gj/curated/fact_citations/',
  parquet_compression = 'SNAPPY',
  partitioned_by = ARRAY['ingest_dt']
) AS

WITH fire AS (
  SELECT
    lpad(regexp_replace(trim(cast(cms_certification_number_ccn as varchar)), '"', ''), 6, '0') AS ccn_provnum,
    try(cast(date_parse(trim(survey_date), '%Y-%m-%d') as date)) AS survey_dt,
    cast(deficiency_tag_number as varchar) AS deficiency_tag_nbr,
    cast(survey_type as varchar) AS survey_type,
    
    cast(deficiency_prefix as varchar) AS deficiency_prefix,
    cast(deficiency_category as varchar) AS deficiency_category,
    cast(deficiency_description as varchar) AS deficiency_desc,
    cast(scope_severity_code as varchar) AS scope_severity_code,
    cast(deficiency_corrected as varchar) AS deficiency_corrected,
    try(cast(date_parse(trim(correction_date), '%Y-%m-%d') as date)) AS correction_dt,
    try_cast(inspection_cycle as integer) AS inspection_cycle,
    cast(standard_deficiency as varchar) AS standard_deficiency,
    cast(complaint_deficiency as varchar) AS complaint_deficiency,
    cast(infection_control_inspection_deficiency as varchar) AS infection_control_inspection_deficiency,

    CAST(NULL AS varchar) AS citation_idr,
    CAST(NULL AS varchar) AS citation_iidr,

    ingest_dt
  FROM healthcare_catalog_db.raw_nh_firesafetycitations_oct2024_fixed
  WHERE ingest_dt = '{{INGEST_DT:nh_firesafetycitations_oct2024}}'
),
health AS (
  SELECT
    lpad(regexp_replace(trim(cast(cms_certification_number_ccn as varchar)), '"', ''), 6, '0') AS ccn_provnum,
    try(cast(date_parse(trim(survey_date), '%Y-%m-%d') as date)) AS survey_dt,
    cast(deficiency_tag_number as varchar) AS deficiency_tag_nbr,
    cast(survey_type as varchar) AS survey_type,
    
    cast(deficiency_prefix as varchar) AS deficiency_prefix,
    cast(deficiency_category as varchar) AS deficiency_category,
    cast(deficiency_description as varchar) AS deficiency_desc,
    cast(scope_severity_code as varchar) AS scope_severity_code,
    cast(deficiency_corrected as varchar) AS deficiency_corrected,
    try(cast(date_parse(trim(correction_date), '%Y-%m-%d') as date)) AS correction_dt,
    try_cast(inspection_cycle as integer) AS inspection_cycle,
    cast(standard_deficiency as varchar) AS standard_deficiency,
    cast(complaint_deficiency as varchar) AS complaint_deficiency,
    cast(infection_control_inspection_deficiency as varchar) AS infection_control_inspection_deficiency,

    cast(citation_under_idr as varchar) AS citation_idr,
    CAST(NULL AS varchar) AS citation_iidr,

    ingest_dt
  FROM healthcare_catalog_db.raw_nh_healthcitations_oct2024_fixed
  WHERE ingest_dt = '{{INGEST_DT:nh_healthcitations_oct2024}}'
),
unioned AS (
  SELECT * FROM fire
  UNION ALL
  SELECT * FROM health
),
with_date AS (
  SELECT
    u.*,
    d.date_id
  FROM unioned u
  LEFT JOIN healthcare_curated_db.dim_date d
    ON d.date = u.survey_dt
)
SELECT
  ccn_provnum,
  survey_dt AS survey_date,
  date_id,
  deficiency_tag_nbr,
  survey_type,
  
  deficiency_prefix,
  deficiency_category,
  deficiency_desc,
  scope_severity_code,
  deficiency_corrected,
  correction_dt AS correction_date,
  inspection_cycle,
  standard_deficiency,
  complaint_deficiency,
  infection_control_inspection_deficiency,
  citation_idr,
  citation_iidr,

  ingest_dt
FROM with_date
WHERE ccn_provnum IS NOT NULL
  AND survey_dt IS NOT NULL
  AND deficiency_tag_nbr IS NOT NULL
  AND survey_type IS NOT NULL;