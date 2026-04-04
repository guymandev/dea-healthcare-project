SELECT coalesce(
  count(*) - count(distinct
    ccn_provnum || '|' ||
    cast(association_date as varchar) || '|' ||
    ownership_role || '|' ||
    owner_type || '|' ||
    ownership_pct || '|' ||
    owner_name
  ),
  0
) AS dupes
FROM healthcare_curated_db.fact_ownership;