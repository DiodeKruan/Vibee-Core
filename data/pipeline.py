"""Data pipeline for cleaning and transforming Traffy Fondue data."""

import numpy as np
import pandas as pd


# Example ticket text to filter out
EXAMPLE_TICKET_TEXT = "ตัวอย่างการแจ้ง รบกวนมาเก็บขยะด้วยครับ"

# Content columns for duplicate detection (exclude metadata)
CONTENT_COLUMNS = ["ticket_id", "type", "description", "lat", "lon"]

# Outlier thresholds
MAX_ORGANIZATIONS = 10
MAX_TYPES = 5

# Bangkok boundaries
BK_HIGH_LAT = 13.9611
BK_LOW_LAT = 13.4658
BK_RIGHT_LONG = 100.9417
BK_LEFT_LONG = 100.3153


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
  """
  Drop duplicate rows based on content columns.

  Args:
      df: Input DataFrame

  Returns:
      DataFrame with duplicates removed
  """
  existing_cols = [col for col in CONTENT_COLUMNS if col in df.columns]
  if not existing_cols:
    return df
  return df.drop_duplicates(subset=existing_cols, keep="first")


def drop_null_ticket_id(df: pd.DataFrame) -> pd.DataFrame:
  """
  Drop rows where ticket_id is null.

  Args:
      df: Input DataFrame

  Returns:
      DataFrame with null ticket_id rows removed
  """
  if "ticket_id" not in df.columns:
    return df
  return df.dropna(subset=["ticket_id"])


def drop_example_tickets(df: pd.DataFrame) -> pd.DataFrame:
  """
  Drop example/test tickets containing specific Thai text.

  Args:
      df: Input DataFrame

  Returns:
      DataFrame with example tickets removed
  """
  if "description" not in df.columns:
    return df

  mask = df["description"].fillna("").str.contains(EXAMPLE_TICKET_TEXT, regex=False)
  return df[~mask]


def drop_invalid_coordinates(df: pd.DataFrame) -> pd.DataFrame:
  """
  Drop rows with invalid coordinates (0.0, 0.0).

  Args:
      df: Input DataFrame

  Returns:
      DataFrame with invalid coordinates removed
  """
  if "lat" not in df.columns or "lon" not in df.columns:
    return df

  # Remove rows where both lat and lon are exactly 0.0
  mask = (df["lat"] == 0.0) & (df["lon"] == 0.0)
  return df[~mask]


def filter_bangkok_only(df: pd.DataFrame) -> pd.DataFrame:
  """
  Filter for tickets within Bangkok boundaries and province.

  Args:
      df: Input DataFrame

  Returns:
      DataFrame filtered for Bangkok
  """
  if df.empty:
    return df

  # Check required columns
  required_cols = ["lat", "lon", "province"]
  if not all(col in df.columns for col in required_cols):
    return df

  mask = (
      (df["lat"] >= BK_LOW_LAT)
      & (df["lat"] <= BK_HIGH_LAT)
      & (df["lon"] >= BK_LEFT_LONG)
      & (df["lon"] <= BK_RIGHT_LONG)
      & (df["province"] == "กรุงเทพมหานคร")
  )

  return df[mask]


def one_hot_encode_types(df: pd.DataFrame) -> pd.DataFrame:
  """
  One-hot encode the type column (comma-separated values).
  Creates type_* columns and n_types count.

  Optimized: Uses vectorized string operations and MultiLabelBinarizer
  instead of O(n×m) nested loops.

  Args:
      df: Input DataFrame with 'type' column (comma-separated string)

  Returns:
      DataFrame with type_* columns and n_types added
  """
  if "type" not in df.columns:
    df["n_types"] = 0
    return df

  df = df.copy()

  # Vectorized string split using str accessor (faster than apply)
  # Fill NaN with empty string first, then split
  type_series = df["type"].fillna("").astype(str)

  # Split into lists - this is vectorized
  type_lists = type_series.str.split(",")

  # Strip whitespace from each element and filter empty strings
  # Using list comprehension is faster than nested apply for this
  type_lists = type_lists.apply(
      lambda x: [t.strip() for t in x if t.strip()] if x else []
  )

  # Vectorized length calculation
  df["n_types"] = type_lists.str.len()

  # Use MultiLabelBinarizer for O(n) one-hot encoding instead of O(n×m) loop
  from sklearn.preprocessing import MultiLabelBinarizer

  mlb = MultiLabelBinarizer()
  # Transform returns a sparse-friendly numpy array
  one_hot_matrix = mlb.fit_transform(type_lists)

  # Create column names with prefix
  type_columns = [f"type_{t}" for t in mlb.classes_]

  # Assign all one-hot columns at once (much faster than column-by-column)
  one_hot_df = pd.DataFrame(one_hot_matrix, columns=type_columns, index=df.index)
  df = pd.concat([df, one_hot_df], axis=1)

  return df


def add_n_organizations(df: pd.DataFrame) -> pd.DataFrame:
  """
  Add n_organizations column counting organizations per ticket.

  Optimized: Uses vectorized string operations instead of apply().

  Args:
      df: Input DataFrame with 'org' column (comma-separated string)

  Returns:
      DataFrame with n_organizations column added
  """
  if "org" not in df.columns:
    df["n_organizations"] = 0
    return df

  # Vectorized approach: count commas + 1 for non-empty, 0 for empty
  org_series = df["org"].fillna("").astype(str).str.strip()

  # Count non-empty organizations using vectorized string operations
  # Split and count length, but handle empty strings
  df = df.assign(
      n_organizations=np.where(
          org_series == "",
          0,
          org_series.str.split(",").str.len()
      )
  )
  return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
  """
  Remove outlier rows based on n_organizations and n_types.

  Args:
      df: Input DataFrame with n_organizations and n_types columns

  Returns:
      DataFrame with outliers removed
  """
  df = df.copy()

  # Ensure columns exist with defaults
  if "n_organizations" not in df.columns:
    df["n_organizations"] = 0
  if "n_types" not in df.columns:
    df["n_types"] = 0

  # Remove outliers: n_organizations >= 10 OR n_types > 5
  mask = (df["n_organizations"] >= MAX_ORGANIZATIONS) | (df["n_types"] > MAX_TYPES)
  return df[~mask]


def calculate_duration(df: pd.DataFrame) -> pd.DataFrame:
  """
  Calculate duration_hour from last_activity - timestamp.
  Imputes duration for unfinished tickets using max(last_activity).

  Args:
      df: Input DataFrame with timestamp, last_activity, and status columns

  Returns:
      DataFrame with duration_hour column added
  """
  df = df.copy()

  if "timestamp" not in df.columns or "last_activity" not in df.columns:
    df["duration_hour"] = np.nan
    return df

  # Convert to datetime if needed
  timestamp = pd.to_datetime(df["timestamp"], errors="coerce")
  last_activity = pd.to_datetime(df["last_activity"], errors="coerce")

  # Calculate raw duration
  duration_raw = last_activity - timestamp

  # Impute for unfinished tickets if status column exists
  if "status" in df.columns:
    max_last_activity = last_activity.max()
    # Use 'เสร็จสิ้น' (Done) as the completed status
    # If status is NOT 'เสร็จสิ้น', use max_last_activity - timestamp
    if pd.notna(max_last_activity):
      imputed_duration = max_last_activity - timestamp
      duration = duration_raw.where(df["status"] == "เสร็จสิ้น", imputed_duration)
    else:
      duration = duration_raw
  else:
    duration = duration_raw

  # Calculate duration in hours
  df["duration_hour"] = duration.dt.total_seconds() / 3600

  return df


def format_type_display(df: pd.DataFrame) -> pd.DataFrame:
  """
  Format the type column for display in tooltips.
  Replaces commas with bullet points for better readability.
  Shows "Unspecified" for empty/null types.

  Args:
      df: Input DataFrame with 'type' column

  Returns:
      DataFrame with 'type_display' column added
  """
  df = df.copy()

  if "type" not in df.columns:
    df["type_display"] = "ไม่ระบุ"
    return df

  def format_types(x):
    if pd.isna(x) or str(x).strip() == "":
      return "ไม่ระบุ"
    types = [t.strip() for t in str(x).split(",") if t.strip()]
    if not types:
      return "ไม่ระบุ"
    # Join with bullet points for display
    return " • ".join(types)

  df["type_display"] = df["type"].apply(format_types)
  return df


def format_coordinates(df: pd.DataFrame) -> pd.DataFrame:
  """
  Format lat/lon coordinates for tooltip display.

  Args:
      df: Input DataFrame with 'lat' and 'lon' columns

  Returns:
      DataFrame with 'coords_display' column added
  """
  df = df.copy()

  if "lat" not in df.columns or "lon" not in df.columns:
    df["coords_display"] = ""
    return df

  df["coords_display"] = df.apply(
      lambda row: f"{row['lat']:.4f}, {row['lon']:.4f}"
      if pd.notna(row['lat']) and pd.notna(row['lon'])
      else "",
      axis=1
  )
  return df


def process_traffy_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
  """
  Process Traffy Fondue data through the complete pipeline.

  Pipeline steps:
  1. Drop duplicates based on content columns
  2. Drop rows with null ticket_id
  3. Drop example tickets
  4. Drop invalid coordinates (0.0, 0.0)
  5. Filter for Bangkok only (lat/lon bounds + province)
  6. One-hot encode type → type_* columns + n_types
  7. Add n_organizations count
  8. Remove outliers (n_organizations >= 10 OR n_types > 5)
  9. Calculate duration_hour
  10. Format type for display (type_display column)
  11. Format coordinates for display (coords_display column)

  Args:
      df: Raw DataFrame from fetch_traffy_data()
      verbose: If True, print stats after each step

  Returns:
      Cleaned and transformed DataFrame
  """
  if df.empty:
    return df

  initial_count = len(df)

  # Step 1: Drop duplicates
  df = drop_duplicates(df)
  if verbose:
    print(f"After drop_duplicates: {len(df)} rows (removed {initial_count - len(df)})")

  # Step 2: Drop null ticket_id
  count_before = len(df)
  df = drop_null_ticket_id(df)
  if verbose:
    print(f"After drop_null_ticket_id: {len(df)} rows (removed {count_before - len(df)})")

  # Step 3: Drop example tickets
  count_before = len(df)
  df = drop_example_tickets(df)
  if verbose:
    print(f"After drop_example_tickets: {len(df)} rows (removed {count_before - len(df)})")

  # Step 4: Drop invalid coordinates
  count_before = len(df)
  df = drop_invalid_coordinates(df)
  if verbose:
    print(f"After drop_invalid_coordinates: {len(df)} rows (removed {count_before - len(df)})")

  # Step 5: Filter Bangkok only
  count_before = len(df)
  df = filter_bangkok_only(df)
  if verbose:
    print(f"After filter_bangkok_only: {len(df)} rows (removed {count_before - len(df)})")

  # Step 6: One-hot encode types
  df = one_hot_encode_types(df)
  if verbose:
    type_cols = [c for c in df.columns if c.startswith("type_")]
    print(f"After one_hot_encode_types: {len(type_cols)} type columns created")

  # Step 7: Add n_organizations
  df = add_n_organizations(df)
  if verbose:
    print(f"After add_n_organizations: mean={df['n_organizations'].mean():.2f}")

  # Step 8: Remove outliers
  count_before = len(df)
  df = remove_outliers(df)
  if verbose:
    print(f"After remove_outliers: {len(df)} rows (removed {count_before - len(df)})")

  # Step 9: Calculate duration
  df = calculate_duration(df)
  if verbose:
    valid_duration = df["duration_hour"].notna().sum()
    print(f"After calculate_duration: {valid_duration} rows with valid duration")

  # Step 10: Format type for display
  df = format_type_display(df)
  if verbose:
    print(f"After format_type_display: type_display column added")

  # Step 11: Format coordinates for display
  df = format_coordinates(df)
  if verbose:
    print(f"After format_coordinates: coords_display column added")

  # Reset index
  df = df.reset_index(drop=True)

  if verbose:
    print(f"Pipeline complete: {initial_count} → {len(df)} rows")

  return df


def get_pipeline_stats(df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict:
  """
  Get statistics about the pipeline transformation.

  Args:
      df_before: DataFrame before pipeline
      df_after: DataFrame after pipeline

  Returns:
      Dictionary with pipeline statistics
  """
  stats = {
      "rows_before": len(df_before),
      "rows_after": len(df_after),
      "rows_removed": len(df_before) - len(df_after),
      "removal_rate": (len(df_before) - len(df_after)) / max(len(df_before), 1) * 100,
  }

  if "n_types" in df_after.columns:
    stats["unique_types"] = len([c for c in df_after.columns if c.startswith("type_")])
    stats["avg_types_per_ticket"] = df_after["n_types"].mean()

  if "n_organizations" in df_after.columns:
    stats["avg_orgs_per_ticket"] = df_after["n_organizations"].mean()

  if "duration_hour" in df_after.columns:
    valid_duration = df_after["duration_hour"].dropna()
    if len(valid_duration) > 0:
      stats["avg_duration_hours"] = valid_duration.mean()
      stats["median_duration_hours"] = valid_duration.median()

  return stats
