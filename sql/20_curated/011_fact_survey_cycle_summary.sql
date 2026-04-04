DROP TABLE IF EXISTS healthcare_curated_db.fact_survey_cycle_summary;

-- S3_PREFIX: s3://healthcare-data-lake-gj/curated/fact_survey_cycle_summary/

CREATE TABLE healthcare_curated_db.fact_survey_cycle_summary
WITH (
  format = 'PARQUET',
  external_location = 's3://healthcare-data-lake-gj/curated/fact_survey_cycle_summary/',
  parquet_compression = 'SNAPPY',
  partitioned_by = ARRAY['ingest_dt']
) AS
WITH base AS (
  SELECT
    -- keys
    lpad(regexp_replace(trim(cast(cms_certification_number_ccn as varchar)), '"', ''), 6, '0') AS ccn_provnum,
    try_cast(regexp_replace(trim(cast(inspection_cycle as varchar)), '"', '') as integer) AS inspection_cycle,

    -- dates (these files are often YYYY-MM-DD; use try + that format)
    try(cast(date_parse(trim(regexp_replace(cast(health_survey_date as varchar), '"', '')), '%Y-%m-%d') as date)) AS health_survey_date,
    try(cast(date_parse(trim(regexp_replace(cast(fire_safety_survey_date as varchar), '"', '')), '%Y-%m-%d') as date)) AS first_safety_survey_date,

    -- counts
    try_cast(regexp_replace(trim(cast(total_number_of_health_deficiencies as varchar)), '"', '') as integer) AS ttl_nbr_health_deficiences,
    try_cast(regexp_replace(trim(cast(total_number_of_fire_safety_deficiencies as varchar)), '"', '') as integer) AS ttl_nbr_fire_safety_deficiencies,

    try_cast(regexp_replace(trim(cast(count_of_freedom_from_abuse_and_neglect_and_exploitation_deficiencies as varchar)), '"', '') as integer) AS cnt_abuse_neglect_exploitation_deficiencies,
    try_cast(regexp_replace(trim(cast(count_of_quality_of_life_and_care_deficiencies as varchar)), '"', '') as integer) AS cnt_quality_life_care_deficiencies,
    try_cast(regexp_replace(trim(cast(count_of_resident_assessment_and_care_planning_deficiencies as varchar)), '"', '') as integer) AS cnt_resident_assessment_care_planning_deficiencies,
    try_cast(regexp_replace(trim(cast(count_of_nursing_and_physician_services_deficiencies as varchar)), '"', '') as integer) AS cnt_nurse_physician_services_deficiencies,
    try_cast(regexp_replace(trim(cast(count_of_resident_rights_deficiencies as varchar)), '"', '') as integer) AS cnt_resident_rights_deficiencies,
    try_cast(regexp_replace(trim(cast(count_of_nutrition_and_dietary_deficiencies as varchar)), '"', '') as integer) AS cnt_nutrition_dietary_deficiencies,
    try_cast(regexp_replace(trim(cast(count_of_pharmacy_service_deficiencies as varchar)), '"', '') as integer) AS cnt_pharmacy_service_deficiencies,
    try_cast(regexp_replace(trim(cast(count_of_environmental_deficiencies as varchar)), '"', '') as integer) AS cnt_environmental_deficiencies,
    try_cast(regexp_replace(trim(cast(count_of_administration_deficiencies as varchar)), '"', '') as integer) AS cnt_admin_deficiencies,
    try_cast(regexp_replace(trim(cast(count_of_infection_control_deficiencies as varchar)), '"', '') as integer) AS cnt_infection_ctrl_deficiencies,
    try_cast(regexp_replace(trim(cast(count_of_emergency_preparedness_deficiencies as varchar)), '"', '') as integer) AS cnt_emergency_prep_deficiencies,
    try_cast(regexp_replace(trim(cast(count_of_automatic_sprinkler_systems_deficiencies as varchar)), '"', '') as integer) AS cnt_sprinkler_deficiencies,
    try_cast(regexp_replace(trim(cast(count_of_construction_deficiencies as varchar)), '"', '') as integer) AS cnt_construction_deficiencies,
    try_cast(regexp_replace(trim(cast(count_of_services_deficiencies as varchar)), '"', '') as integer) AS cnt_services_deficiencies,
    try_cast(regexp_replace(trim(cast(count_of_corridor_walls_and_doors_deficiencies as varchar)), '"', '') as integer) AS cnt_corridor_walls_doors_deficiencies,
    try_cast(regexp_replace(trim(cast(count_of_egress_deficiencies as varchar)), '"', '') as integer) AS cnt_egress_deficiencies,
    try_cast(regexp_replace(trim(cast(count_of_electrical_deficiencies as varchar)), '"', '') as integer) AS cnt_electrical_deficiences,
    try_cast(regexp_replace(trim(cast(count_of_emergency_plans_and_fire_drills_deficiencies as varchar)), '"', '') as integer) AS cnt_emergency_plans_fire_drills_deficiencies,
    try_cast(regexp_replace(trim(cast(count_of_fire_alarm_systems_deficiencies as varchar)), '"', '') as integer) AS cnt_alarm_system_deficiencies,
    try_cast(regexp_replace(trim(cast(count_of_smoke_deficiencies as varchar)), '"', '') as integer) AS cnt_smoke_deficiencies,
    try_cast(regexp_replace(trim(cast(count_of_interior_deficiencies as varchar)), '"', '') as integer) AS cnt_interior_deficiencies,
    try_cast(regexp_replace(trim(cast(count_of_gas_and_vacuum_and_electrical_systems_deficiencies as varchar)), '"', '') as integer) AS cnt_gas_vacuum_electrical_deficiencies,
    try_cast(regexp_replace(trim(cast(count_of_hazardous_area_deficiencies as varchar)), '"', '') as integer) AS cnt_hazardous_area_deficiencies,
    try_cast(regexp_replace(trim(cast(count_of_illumination_and_emergency_power_deficiencies as varchar)), '"', '') as integer) AS cnt_illumination_emergency_power_deficiencies,
    try_cast(regexp_replace(trim(cast(count_of_laboratories_deficiencies as varchar)), '"', '') as integer) AS cnt_laboratory_deficiencies,
    try_cast(regexp_replace(trim(cast(count_of_medical_gases_and_anaesthetizing_areas_deficiencies as varchar)), '"', '') as integer) AS cnt_medical_gases_anesthetizing_areas_deficiencies,
    try_cast(regexp_replace(trim(cast(count_of_smoking_regulations_deficiencies as varchar)), '"', '') as integer) AS cnt_smoking_regulation_deficiencies,
    try_cast(regexp_replace(trim(cast(count_of_miscellaneous_deficiencies as varchar)), '"', '') as integer) AS cnt_misc_deficiencies,

    ingest_dt
  FROM healthcare_catalog_db.raw_nh_surveysummary_oct2024_fixed
  WHERE ingest_dt = '{{INGEST_DT:nh_surveysummary_oct2024}}'   
    AND cms_certification_number_ccn IS NOT NULL
    AND trim(cast(cms_certification_number_ccn as varchar)) <> ''
    AND inspection_cycle IS NOT NULL
)
SELECT *
FROM base
WHERE ccn_provnum IS NOT NULL
  AND inspection_cycle IS NOT NULL;