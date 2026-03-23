DROP TABLE IF EXISTS healthcare_catalog_db.raw_nh_qualitymsr_claims_oct2024_fixed;

CREATE EXTERNAL TABLE healthcare_catalog_db.raw_nh_qualitymsr_claims_oct2024_fixed (
  cms_certification_number_ccn              string,
  provider_name                             string,
  provider_address                          string,
  city_town                                 string,
  state                                     string,
  zip_code                                  string,
  measure_code                              string,
  measure_description                       string,
  resident_type                             string,
  adjusted_score                            string,
  observed_score                            string,
  expected_score                            string,
  footnote_for_score                        string,
  used_in_quality_measure_five_star_rating  string,
  measure_period                            string,
  location                                  string,
  processing_date                           string
)
PARTITIONED BY (ingest_dt string)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
  'separatorChar' = ',',
  'quoteChar'     = '\"',
  'escapeChar'    = '\\'
)
STORED AS TEXTFILE
LOCATION 's3://healthcare-data-lake-gj/raw/nh_qualitymsr_claims_oct2024/'
TBLPROPERTIES ('skip.header.line.count'='1');