select
  sum(case when ccn_provnum is null then 1 else 0 end) as null_ccn
from healthcare_curated_db.fact_facility_covid;