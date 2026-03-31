SELECT coalesce(count(*), 0) AS dupes
FROM (
    SELECT
        ccn_provnum,
        workdate
    FROM healthcare_curated_db.fact_staffing_daily
    WHERE ingest_dt = '{{INGEST_DT}}'
    GROUP BY ccn_provnum, workdate
    HAVING count(*) > 1
) d;