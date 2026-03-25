DROP TABLE IF EXISTS healthcare_catalog_db.raw_nh_surveydates_oct2024_fixed;

CREATE EXTERNAL TABLE healthcare_catalog_db.raw_nh_surveydates_oct2024_fixed (
  cms_certification_number_ccn  string,
  survey_date                   string,
  type_of_survey                string,
  survey_cycle                  string,
  processing_date               string
)
PARTITIONED BY (ingest_dt string)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
  'separatorChar' = ',',
  'quoteChar'     = '"',
  'escapeChar'    = '\\'
)
STORED AS TEXTFILE
LOCATION 's3://healthcare-data-lake-gj/raw/nh_surveydates_oct2024/'
TBLPROPERTIES (
  'skip.header.line.count'='1'
);