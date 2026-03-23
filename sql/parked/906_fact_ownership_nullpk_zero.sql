SELECT coalesce(
  sum(
    CASE
      WHEN ccn_provnum IS NULL
        OR association_date IS NULL
        OR owner_name IS NULL
        OR owner_type IS NULL
        OR ownership_pct IS NULL
        OR ownership_role IS NULL
      THEN 1
      ELSE 0
    END
  ),
  0
) AS null_pk_rows
FROM healthcare_curated_db.fact_ownership;