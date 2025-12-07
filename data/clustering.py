import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN


def perform_dbscan_clustering(data: pd.DataFrame, eps_meters: float = 100, min_samples: int = 5) -> pd.DataFrame:
  """
  Performs DBSCAN clustering on geospatial data to identify high-density areas.

  Args:
      data (pd.DataFrame): Input DataFrame containing 'lat' and 'lon' columns.
      eps_meters (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
      min_samples (int): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.

  Returns:
      pd.DataFrame: A DataFrame with columns 'lat', 'lon', and 'count', representing the centroids and sizes of the clusters.
  """
  # 1. Check if lat and lon columns exist and handle empty data
  if data.empty or 'lat' not in data.columns or 'lon' not in data.columns:
    return pd.DataFrame(columns=['lat', 'lon', 'count'])

  # Earth radius in meters (approximate)
  EARTH_RADIUS_METERS = 6371000.0

  # Calculate eps in radians
  eps_radians = eps_meters / EARTH_RADIUS_METERS

  # 2. Convert lat/lon to radians for Haversine metric
  # sklearn's haversine metric expects [lat, lon] in radians
  coords = np.radians(data[['lat', 'lon']])

  # 3. Use sklearn.cluster.DBSCAN
  db = DBSCAN(
      eps=eps_radians,
      min_samples=min_samples,
      algorithm='ball_tree',
      metric='haversine'
  )

  # 4. Fit the model
  db.fit(coords)

  # Assign cluster labels to the original data
  # We work on a copy to avoid SettingWithCopyWarning on the input DF
  clustered_data = data.copy()
  clustered_data['cluster_label'] = db.labels_

  return clustered_data
