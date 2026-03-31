DROP TABLE IF EXISTS healthcare_catalog_db.raw_nh_surveysummary_oct2024_fixed;

CREATE EXTERNAL TABLE healthcare_catalog_db.raw_nh_surveysummary_oct2024_fixed (
  cms_certification_number_ccn string,
  provider_name string,
  provider_address string,
  city_town string,
  state string,
  zip_code string,
  inspection_cycle string,
  health_survey_date string,
  fire_safety_survey_date string,

  total_number_of_health_deficiencies string,
  total_number_of_fire_safety_deficiencies string,

  count_of_freedom_from_abuse_and_neglect_and_exploitation_deficiencies string,
  count_of_quality_of_life_and_care_deficiencies string,
  count_of_resident_assessment_and_care_planning_deficiencies string,
  count_of_nursing_and_physician_services_deficiencies string,
  count_of_resident_rights_deficiencies string,
  count_of_nutrition_and_dietary_deficiencies string,
  count_of_pharmacy_service_deficiencies string,
  count_of_environmental_deficiencies string,
  count_of_administration_deficiencies string,
  count_of_infection_control_deficiencies string,
  count_of_emergency_preparedness_deficiencies string,
  count_of_automatic_sprinkler_systems_deficiencies string,
  count_of_construction_deficiencies string,
  count_of_services_deficiencies string,
  count_of_corridor_walls_and_doors_deficiencies string,
  count_of_egress_deficiencies string,
  count_of_electrical_deficiencies string,
  count_of_emergency_plans_and_fire_drills_deficiencies string,
  count_of_fire_alarm_systems_deficiencies string,
  count_of_smoke_deficiencies string,
  count_of_interior_deficiencies string,
  count_of_gas_and_vacuum_and_electrical_systems_deficiencies string,
  count_of_hazardous_area_deficiencies string,
  count_of_illumination_and_emergency_power_deficiencies string,
  count_of_laboratories_deficiencies string,
  count_of_medical_gases_and_anaesthetizing_areas_deficiencies string,
  count_of_smoking_regulations_deficiencies string,
  count_of_miscellaneous_deficiencies string,

  location string,
  processing_date string
)
PARTITIONED BY (ingest_dt string)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
  'separatorChar' = ',',
  'quoteChar'     = '"',
  'escapeChar'    = '\\'
)
STORED AS TEXTFILE
LOCATION 's3://healthcare-data-lake-gj/raw/nh_surveysummary_oct2024/'
TBLPROPERTIES (
  'skip.header.line.count'='1'
);