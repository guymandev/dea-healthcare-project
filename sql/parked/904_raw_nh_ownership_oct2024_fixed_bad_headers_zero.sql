SELECT count(*) AS bad_header_rows
FROM healthcare_catalog_db.raw_nh_ownership_oct2024_fixed
WHERE ingest_dt = '{{INGEST_DT}}'
  AND upper(trim(cms_certification_number_ccn)) = 'CMS CERTIFICATION NUMBER (CCN)';