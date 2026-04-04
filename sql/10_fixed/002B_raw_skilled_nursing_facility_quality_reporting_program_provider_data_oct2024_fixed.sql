DROP TABLE IF EXISTS healthcare_catalog_db.raw_skilled_nursing_facility_quality_reporting_program_provider_data_oct2024_fixed;

CREATE EXTERNAL TABLE healthcare_catalog_db.raw_skilled_nursing_facility_quality_reporting_program_provider_data_oct2024_fixed (
  cms_certification_number_ccn  string,
  provider_name                 string,
  address_line_1                string,
  city_town                     string,
  state                         string,
  zip_code                      string,
  county_parish                 string,
  telephone_number              string,
  cms_region                    string,
  measure_code                  string,
  score                         string,
  footnote                      string,
  start_date                    string,
  end_date                      string,
  measure_date_range            string,
  location1                     string
)
PARTITIONED BY (ingest_dt string)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
  'separatorChar' = ',',
  'quoteChar'     = '\"',
  'escapeChar'    = '\\'
)
STORED AS TEXTFILE
LOCATION 's3://healthcare-data-lake-gj/raw/skilled_nursing_facility_quality_reporting_program_provider_data_oct2024/'
TBLPROPERTIES ('skip.header.line.count'='1');