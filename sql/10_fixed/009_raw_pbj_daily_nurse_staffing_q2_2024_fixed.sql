DROP TABLE IF EXISTS healthcare_catalog_db.raw_pbj_daily_nurse_staffing_q2_2024_fixed;

CREATE EXTERNAL TABLE healthcare_catalog_db.raw_pbj_daily_nurse_staffing_q2_2024_fixed (
  provnum string,
  provname string,
  city string,
  state string,
  county_name string,
  county_fips string,
  cy_qtr string,
  workdate string,
  mdscensus string,

  hrs_rndon string,
  hrs_rndon_emp string,
  hrs_rndon_ctr string,
  hrs_rnadmin string,
  hrs_rnadmin_emp string,
  hrs_rnadmin_ctr string,
  hrs_rn string,
  hrs_rn_emp string,
  hrs_rn_ctr string,
  hrs_lpnadmin string,
  hrs_lpnadmin_emp string,
  hrs_lpnadmin_ctr string,
  hrs_lpn string,
  hrs_lpn_emp string,
  hrs_lpn_ctr string,
  hrs_cna string,
  hrs_cna_emp string,
  hrs_cna_ctr string,
  hrs_natrn string,
  hrs_natrn_emp string,
  hrs_natrn_ctr string,
  hrs_medaide string,
  hrs_medaide_emp string,
  hrs_medaide_ctr string
)
PARTITIONED BY (
  ingest_dt string
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
  'separatorChar' = ',',
  'quoteChar'     = '"',
  'escapeChar'    = '\\'
)
STORED AS TEXTFILE
LOCATION 's3://healthcare-data-lake-gj/raw/pbj_daily_nurse_staffing_q2_2024/'
TBLPROPERTIES ('skip.header.line.count'='1');