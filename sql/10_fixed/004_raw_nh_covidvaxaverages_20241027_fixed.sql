DROP TABLE IF EXISTS healthcare_catalog_db.raw_nh_covidvaxaverages_20241027_fixed;

CREATE EXTERNAL TABLE healthcare_catalog_db.raw_nh_covidvaxaverages_20241027_fixed (
  state_raw                                 string,
  pct_residents_up_to_date_raw              string,
  pct_staff_up_to_date_raw                  string,
  vax_data_last_updated_raw                 string
)
PARTITIONED BY (ingest_dt string)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
  'separatorChar' = ',',
  'quoteChar'     = '"',
  'escapeChar'    = '\\'
)
STORED AS TEXTFILE
LOCATION 's3://healthcare-data-lake-gj/raw/nh_covidvaxaverages_20241027/'
TBLPROPERTIES (
  'skip.header.line.count'='1'
);