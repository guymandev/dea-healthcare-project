import pandas as pd
import gdown
import sys
import re
from pathlib import Path
from datetime import datetime

# Class that will be used to duplicate output to both terminal and a log file.
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()

# This function deals with different encoding, defaults PROVNUM to String, and coerces WorkDate to datetime. 
# It also has flexibility to perform the same coersion on other date fields that are passed into the **kwargs
def read_csv_safely(
    path: str,
    *,
    parse_workdate: bool = False,
    workdate_col: str = "WorkDate",
    workdate_format: str = "%Y%m%d",
    verbose: bool = False,
    **kwargs
) -> pd.DataFrame:
    default_dtype = {
        "PROVNUM": "string",
        "CMS Certification Number (CCN)": "string",
        "CMS Certification Number": "string",
        "CCN": "string",
    }

    caller_dtype = kwargs.pop("dtype", None)
    if caller_dtype:
        default_dtype.update(caller_dtype)

    # Get columns (header-only) with encoding fallback
    try:
        cols = pd.read_csv(path, encoding="utf-8", nrows=0).columns
    except UnicodeDecodeError:
        cols = pd.read_csv(path, encoding="cp1252", nrows=0).columns

    dtype_filtered = {col: typ for col, typ in default_dtype.items() if col in cols}

    # Optional WorkDate parsing (only if column exists)
    if parse_workdate and workdate_col in cols:
        caller_parse_dates = kwargs.pop("parse_dates", None)

        if caller_parse_dates is None:
            parse_dates = [workdate_col]
        else:
            parse_dates = [caller_parse_dates] if isinstance(caller_parse_dates, str) else list(caller_parse_dates)
            if workdate_col not in parse_dates:
                parse_dates.append(workdate_col)

        kwargs["parse_dates"] = parse_dates
        kwargs["date_format"] = workdate_format

    def _read(encoding: str) -> pd.DataFrame:
        if verbose:
            print(f"[read_csv_safely] file={path}")
            print(f"[read_csv_safely] encoding_try={encoding}")
            print(f"[read_csv_safely] columns={len(cols)}")
            print(f"[read_csv_safely] dtype_applied={dtype_filtered}")
            if parse_workdate and workdate_col in cols:
                print(f"[read_csv_safely] parse_dates={kwargs.get('parse_dates')}, date_format={kwargs.get('date_format')}")
        return pd.read_csv(path, encoding=encoding, dtype=dtype_filtered, **kwargs)

    try:
        return _read("utf-8")
    except UnicodeDecodeError:
        return _read("cp1252")

def coerce_workdate(df: pd.DataFrame, col="WorkDate") -> pd.DataFrame:
    df[col] = pd.to_datetime(df[col].astype(str), format="%Y%m%d", errors="raise")
    return df

def show_schema(df):
    return df.dtypes.reset_index().rename(columns={"index": "column", 0: "dtype"})

# file_name = "./data/PBJ_Daily_Nurse_Staffing_Q2_2024.csv"
# df = read_csv_safely("PBJ_Daily_Nurse_Staffing_Q2_2024.csv", parse_workdate=True, verbose=True)
#df = coerce_workdate(df)

# print("Pandas version: " + pd.__version__)

#########################
# Intial examination
#########################

#Check if data folder exists. If not, then download files. 
folder_path = Path("./data/")
if not folder_path.is_dir():
    folder_url = "https://drive.google.com/drive/folders/15KqJ1MZ7JcgAkOfqcaWcALWkG0dh3jpE"
    gdown.download_folder(folder_url, output="data", quiet=False, use_cookies=False)

#Helper functions
def is_dateish_col(col: str) -> bool:
    c = col.lower()
    return any(k in c for k in [
        "date", "dt", "time", "timestamp",
        "qtr", "quarter", "year",
        "period", "processing", "survey", "correction", "association",
        "start", "end", "from", "through", "thru"
    ])

def is_idish_col(col: str) -> bool:
    c = col.lower()
    return any(k in c for k in [
        "id", "num", "key", "prov", "provider", "facility", "ccn", "npi", "fips",
        # event identifiers
        "measure", "code", "tag", "prefix", "category", "cycle", "type", "role"
    ])

BAD_KEYWORDS = [
    "provider name",      # exclude provider display fields
    "provider address",
    "address",
    "location",
    "description",
    "comment",
    "text",
    "telephone",
    "phone",
    "city/town",
    "city",
    "zip",
    "county"
]

def looks_like_bad_key(col: str) -> bool:
    c = col.lower()
    return any(k in c for k in BAD_KEYWORDS)

# A function that rates potential keys based on fields that we know are either keys or
# are likely part of composite keys.
def key_component_score(col: str) -> int:
    c = col.lower()
    score = 0

    # primary hub ids
    if "cms certification number" in c or "(ccn)" in c or c == "ccn":
        score += 100
    if "provnum" in c or "provider id" in c:
        score += 90
    if "npi" in c:
        score += 80
    if "fips" in c:
        score += 70

    # event discriminators
    if "measure code" in c or ("measure" in c and "code" in c):
        score += 60
    if "tag" in c or "prefix" in c or "cycle" in c or "category" in c:
        score += 55
    if "penalty type" in c or ("penalty" in c and "type" in c):
        score += 55
    if "role" in c or "owner type" in c:
        score += 50
    if "owner name" in c or "manager name" in c:
        score += 50

    # time discriminators
    if "start date" in c or "end date" in c or "from date" in c or "through date" in c:
        score += 45
    elif "date" in c or "timestamp" in c or "time" in c:
        score += 25

    # numeric discriminators
    if "amount" in c or "percentage" in c or "percent" in c:
        score += 25
    if "length" in c and "day" in c:
        score += 20

    return score

def discriminator_cols(df):
    cols = []
    for c in df.columns:
        cl = c.lower()
        if "provider name" in cl:
            continue
        if any(k in cl for k in ["owner name", "manager name", "amount", "percent", "percentage",
                                 "length", "days", "measure", "code", "tag", "type", "role",
                                 "start", "end", "from", "through"]):
            cols.append(c)
    return cols

# Even further improved function to guess key fields
def guess_key_greedy_with_steps(df, max_cols=6):
    n = len(df)
    if n == 0:
        return [], None, []

    cols = [c for c in df.columns if not looks_like_bad_key(c)]
    prioritized = [c for c in cols if is_idish_col(c) or is_dateish_col(c)] or cols

    null_rate = df.isna().mean()
    prioritized = [c for c in prioritized if null_rate.get(c, 0) < 0.5]

    # NEW: prioritize likely key components (generic)
    prioritized = sorted(prioritized, key=key_component_score, reverse=True)

    def uniq_ratio(key_cols):
        return df.drop_duplicates(subset=key_cols).shape[0] / n

    best = max(prioritized, key=lambda c: uniq_ratio([c]))
    key = [best]
    steps = [(key.copy(), uniq_ratio(key))]

    while steps[-1][1] < 1.0 and len(key) < max_cols:
        best_next = None
        best_ratio = steps[-1][1]

        for c in prioritized:
            if c in key:
                continue
            r = uniq_ratio(key + [c])
            if r > best_ratio:
                best_ratio = r
                best_next = c

        if not best_next:
            break

        key.append(best_next)
        steps.append((key.copy(), best_ratio))

    # If we didn't reach uniqueness, try one more discriminator column
    if steps[-1][1] < 1.0:
        best_next = None
        best_ratio = steps[-1][1]
        for c in discriminator_cols(df):
            if c in key:
                continue
            r = uniq_ratio(key + [c])
            if r > best_ratio:
                best_ratio = r
                best_next = c
        if best_next:
            key.append(best_next)
            steps.append((key.copy(), best_ratio))

    return key, steps[-1][1], steps

def profile_one_file(csv_path: Path, verbose_print: bool = True) -> dict:
    # parse WorkDate if present
    df = read_csv_safely(csv_path, parse_workdate=True, verbose=False)

    nrows, ncols = df.shape

    # schema + missingness
    schema_df = show_schema(df)
    null_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    top_nulls = null_pct.head(10).to_dict()

    # cardinality (top 10 only)
    nunique_series = df.nunique(dropna=True).sort_values(ascending=False)
    top_cardinality = nunique_series.head(10).to_dict()

    # key guessing + duplicate tests
    key_results = []

    if nrows and nrows > 1:
        #key_cols, best_ratio = guess_key_greedy(df, max_cols=6)
        key_cols, ratio, steps = guess_key_greedy_with_steps(df, max_cols=6)
        print(f"\nSTEPS for {csv_path.name}:")
        for cols, r in steps:
            print(f"{cols} -> {r:.6f}")
        print(f"FINAL KEY: {key_cols}")

        dupes = df.duplicated(subset=key_cols).sum()
        unique_rows = df.drop_duplicates(subset=key_cols).shape[0]

        key_results.append({
            "candidate_key": "|".join(key_cols),
            "dupes": int(dupes),
            "unique_rows": int(unique_rows),
            "row_count": int(nrows),
            "uniqueness_ratio": round(unique_rows / nrows, 6) if nrows else None
        })
    else:
        # 0 or 1 row: key detection isn't meaningful
        key_results.append({
            "candidate_key": "(n/a - <=1 row)",
            "dupes": 0,
            "unique_rows": int(nrows),
            "row_count": int(nrows),
            "uniqueness_ratio": 1.0 if nrows == 1 else None
        })

    if verbose_print:
        print("\n" + "=" * 80)
        print(f"FILE: {csv_path.name}")
        print(f"SHAPE: {nrows:,} rows x {ncols} cols\n")

        print("SCHEMA (col, dtype):")
        print(schema_df.to_string(index=False))

        print("\nTOP NULL % (up to 10):")
        print(null_pct.head(10).round(3).to_string())

        print("\nTOP CARDINALITY (nunique, up to 10):")
        print(nunique_series.head(10).to_string())

        print("\nKEY CANDIDATES:")
        for r in key_results:
            print(f"  - {r['candidate_key']}: dupes={r['dupes']}, "
                  f"unique_rows={r['unique_rows']}, ratio={r['uniqueness_ratio']}")

    # return a compact summary row (for the report)
    best_key = None
    if key_results:
        # prefer highest uniqueness_ratio, then lowest dupes
        best_key = sorted(key_results, key=lambda r: (-r["uniqueness_ratio"], r["dupes"]))[0]

    return {
        "file": csv_path.name,
        "rows": int(nrows),
        "cols": int(ncols),
        "top_null_cols": "; ".join([f"{k}={v:.2f}%" for k, v in list(top_nulls.items())[:5]]),
        "top_cardinality_cols": "; ".join([f"{k}={int(v)}" for k, v in list(top_cardinality.items())[:5]]),
        "best_key_guess": best_key["candidate_key"] if best_key else "",
        "best_key_dupes": best_key["dupes"] if best_key else None,
        "best_key_uniqueness_ratio": best_key["uniqueness_ratio"] if best_key else None,
    }

def profile_all_data_files(data_dir: str = "./data", reports_dir: str = "./reports"):
    data_path = Path(data_dir)
    out_path = Path(reports_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    csv_files = sorted([p for p in data_path.iterdir() if p.is_file() and p.suffix.lower() == ".csv"])

    summary_rows = []
    for csv_path in csv_files:
        try:
            summary_rows.append(profile_one_file(csv_path, verbose_print=True))
        except Exception as e:
            print("\n" + "=" * 80)
            print(f"FILE: {csv_path.name}")
            print(f"ERROR: {type(e).__name__}: {e}")
            summary_rows.append({
                "file": csv_path.name,
                "rows": None,
                "cols": None,
                "top_null_cols": "",
                "top_cardinality_cols": "",
                "best_key_guess": "",
                "best_key_dupes": None,
                "best_key_uniqueness_ratio": None,
                "error": f"{type(e).__name__}: {e}",
            })

    summary_df = pd.DataFrame(summary_rows)

    #Output summary report
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv = out_path / f"data_profile_summary_{ts}.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n✅ Wrote summary report to: {summary_csv.resolve()}")

    #Begin generating suggested join mappings
    def _split_key(best_key_guess: str, max_cols: int = 6):
        if not best_key_guess or best_key_guess.startswith("("):
            return [""] * max_cols
        parts = [p.strip() for p in best_key_guess.split("|")]
        parts = parts[:max_cols] + [""] * (max_cols - len(parts))
        return parts

    def _guess_hub_key(cols):
        """Pick the most likely join hub key from the key columns."""
        joined = " ".join(c.lower() for c in cols if c)
        # prioritize common healthcare identifiers
        for pattern in [
            "cms certification number", "(ccn)", "ccn",
            "provnum", "provnum", "provider id", "npi", "fips"
        ]:
            for c in cols:
                if c and pattern in c.lower():
                    return c
        return cols[0] if cols and cols[0] else ""

    def _guess_date_key(cols):
        for c in cols:
            if not c:
                continue
            cl = c.lower()
            if any(k in cl for k in ["date", "dt", "time", "timestamp", "from", "through", "start", "end"]):
                return c
        return ""

    # ---- build join-key report ----
    max_key_cols = 6
    split_cols = summary_df["best_key_guess"].apply(lambda s: _split_key(s, max_key_cols))
    split_df = pd.DataFrame(split_cols.tolist(), columns=[f"key_col_{i}" for i in range(1, max_key_cols + 1)])

    join_df = pd.concat(
        [summary_df[["file", "rows", "cols", "best_key_guess", "best_key_dupes", "best_key_uniqueness_ratio"]], split_df],
        axis=1
    )

    join_df["hub_key"] = join_df[[f"key_col_{i}" for i in range(1, max_key_cols + 1)]].apply(
        lambda r: _guess_hub_key(list(r.values)), axis=1
    )
    join_df["date_key"] = join_df[[f"key_col_{i}" for i in range(1, max_key_cols + 1)]].apply(
        lambda r: _guess_date_key(list(r.values)), axis=1
    )

    join_csv = out_path / f"join_key_candidates_{ts}.csv"
    join_df.to_csv(join_csv, index=False)
    print(f"✅ Wrote join-key report to: {join_csv.resolve()}")

#Output full log output into a report
reports_dir = Path("./reports")
reports_dir.mkdir(parents=True, exist_ok=True)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = reports_dir / f"profile_run_{ts}.log"

with open(log_path, "w", encoding="utf-8") as f:
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, f)
    try:
        profile_all_data_files()   # <-- your existing function that prints a lot
    finally:
        sys.stdout = original_stdout

print(f"\n✅ Wrote full console log to: {log_path.resolve()}")



# ---------- call it ----------
#profile_all_data_files()


# print(df.shape)
# print(show_schema(df).to_string(index=False))

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