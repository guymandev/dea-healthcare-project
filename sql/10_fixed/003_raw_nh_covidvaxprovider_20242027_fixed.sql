DROP TABLE IF EXISTS healthcare_catalog_db.raw_nh_covidvaxprovider_20241027_fixed;

CREATE EXTERNAL TABLE healthcare_catalog_db.raw_nh_covidvaxprovider_20241027_fixed (
  cms_certification_number_ccn                     string,
  state                                            string,
  pct_residents_up_to_date_vaxes                   string,
  pct_staff_up_to_date_vaxes                       string,
  vax_data_last_update                             string
)
PARTITIONED BY (ingest_dt string)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
  'separatorChar' = ',',
  'quoteChar'     = '"',
  'escapeChar'    = '\\'
)
STORED AS TEXTFILE
LOCATION 's3://healthcare-data-lake-gj/raw/nh_covidvaxprovider_20241027/'
TBLPROPERTIES (
  'skip.header.line.count'='1'
);