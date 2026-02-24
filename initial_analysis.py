import pandas as pd

# This function deals with different encoding, defaults PROVNUM to String, and coerces WorkDate to datetime. 
# It also has flexibility to perform the same coersion on other date fields that are passed into the **kwargs
def read_csv_safely(
    path: str,
    *,
    parse_workdate: bool = False,
    workdate_col: str = "WorkDate",
    workdate_format: str = "%Y%m%d",
    **kwargs
) -> pd.DataFrame:
    # default dtypes we want everywhere
    default_dtype = {"PROVNUM": "string"}

    # allow caller to add/override dtypes
    caller_dtype = kwargs.pop("dtype", None)
    if caller_dtype:
        default_dtype.update(caller_dtype)

    # optionally set up datetime parsing for WorkDate
    if parse_workdate:
        caller_parse_dates = kwargs.pop("parse_dates", None)

        # merge with any user-supplied parse_dates
        if caller_parse_dates is None:
            parse_dates = [workdate_col]
        else:
            # normalize to list
            if isinstance(caller_parse_dates, (str,)):
                parse_dates = [caller_parse_dates]
            else:
                parse_dates = list(caller_parse_dates)
            if workdate_col not in parse_dates:
                parse_dates.append(workdate_col)

        kwargs["parse_dates"] = parse_dates
        kwargs["date_format"] = workdate_format

    def _read(encoding: str) -> pd.DataFrame:
        return pd.read_csv(path, encoding=encoding, dtype=default_dtype, **kwargs)

    try:
        return _read("utf-8")
    except UnicodeDecodeError:
        return _read("cp1252")

def coerce_workdate(df: pd.DataFrame, col="WorkDate") -> pd.DataFrame:
    df[col] = pd.to_datetime(df[col].astype(str), format="%Y%m%d", errors="raise")
    return df

def show_schema(df):
    return df.dtypes.reset_index().rename(columns={"index": "column", 0: "dtype"})

file_name = "PBJ_Daily_Nurse_Staffing_Q2_2024.csv"
df = read_csv_safely(file_name, parse_workdate=True)
#df = coerce_workdate(df)

# print("Pandas version: " + pd.__version__)

#########################
# Intial examination
#########################

print(df.shape)
print(show_schema(df).to_string(index=False))

# null_pct = (df.isna().mean() * 100).sort_values(ascending=False)
# print(null_pct.head(20))

# print(df.nunique().sort_values(ascending=False).head(20))

# Check for duplicate values in PROVNUM
# print(df["PROVNUM"].nunique(), len(df))
# Check for duplicates in the likely combination key: PROVNUM & WorkDate
# key_cols = ["PROVNUM", "WorkDate"]
# dupes = df.duplicated(subset=key_cols).sum()
# print("Duplicate (PROVNUM, WorkDate) rows:", dupes)
# Create datatime version of WorkDate and check for bad values in WorkDate
# df["WorkDate_dt"] = pd.to_datetime(df["WorkDate"].astype(str), format="%Y%m%d", errors="coerce")
# print(df["WorkDate_dt"].isna().sum(), "bad WorkDate values")

# Check for negative values in Hrs columns
# hour_cols = [c for c in df.columns if c.startswith("Hrs_")]
# print((df[hour_cols] < 0).sum().sort_values(ascending=False).head(10))

# Sanity check for "bad" values in PROVNUM
##########################################

# # Checks for non-numeric values
# bad = df[df["PROVNUM"].astype(str).str.contains(r"[^0-9]", regex=True, na=False)]
# print(bad["PROVNUM"].head(20))

# #Checks for leading zeroes
# zeros = df[df["PROVNUM"].astype(str).str.startswith("0", na=False)]
# print(zeros["PROVNUM"].head(20))