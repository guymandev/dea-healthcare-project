DROP TABLE IF EXISTS healthcare_catalog_db.raw_nh_firesafetycitations_oct2024_fixed;

CREATE EXTERNAL TABLE healthcare_catalog_db.raw_nh_firesafetycitations_oct2024_fixed (
  cms_certification_number_ccn          string,
  provider_name                         string,
  provider_address                      string,
  city_town                             string,
  state                                 string,
  zip_code                              string,
  survey_date                           string,
  survey_type                           string,
  deficiency_prefix                     string,
  deficiency_category                   string,
  deficiency_tag_number                 string,
  tag_version                           string,
  deficiency_description                string,
  scope_severity_code                   string,
  deficiency_corrected                  string,
  correction_date                       string,
  inspection_cycle                      string,
  standard_deficiency                   string,
  complaint_deficiency                  string,
  infection_control_inspection_deficiency string
)
PARTITIONED BY (ingest_dt string)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
  'separatorChar' = ',',
  'quoteChar'     = '"',
  'escapeChar'    = '\\'
)
STORED AS TEXTFILE
LOCATION 's3://healthcare-data-lake-gj/raw/nh_firesafetycitations_oct2024/'
TBLPROPERTIES ('skip.header.line.count'='1');