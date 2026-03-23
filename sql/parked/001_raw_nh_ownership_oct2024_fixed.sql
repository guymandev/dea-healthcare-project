DROP TABLE IF EXISTS healthcare_catalog_db.raw_nh_ownership_oct2024_fixed;

-- S3_PREFIX: s3://healthcare-data-lake-gj/raw/nh_ownership_oct2024/

CREATE EXTERNAL TABLE healthcare_catalog_db.raw_nh_ownership_oct2024_fixed (
  cms_certification_number_ccn                 string,
  provider_name                                string,
  provider_address                             string,
  city_town                                    string,
  state                                        string,
  zip_code                                     string,
  role_played_by_owner_or_manager_in_facility  string,
  owner_type                                   string,
  owner_name                                   string,
  ownership_percentage                         string,
  association_date                             string,
  location                                     string,
  processing_date                              string
)
PARTITIONED BY (ingest_dt string)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
  'separatorChar' = ',',
  'quoteChar'     = '"',
  'escapeChar'    = '\\'
)
STORED AS TEXTFILE
LOCATION 's3://healthcare-data-lake-gj/raw/nh_ownership_oct2024/'
TBLPROPERTIES (
  'skip.header.line.count'='1'
);