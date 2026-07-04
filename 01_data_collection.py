# -*- coding: utf-8 -*-
"""
================================================================================
Istanbul as a Layered Gradient — Data Collection Pipeline
================================================================================

Reproducibility script for the manuscript:
    "Istanbul as a Layered Gradient: A Structural Analysis of Street-Scale
     Urban Fabric in a Sample Guided by Classical Urban Safety Theories"
    Submitted to Environment and Planning B: Urban Analytics and City Science
    (under peer review).

This script collects 360-degree panoramic images from the Google Street View
Static API. For each of the 85 initial candidate streets, 12 evenly-spaced
sampling points are interpolated between manually selected start and end
coordinates, and at each point 12 headings (0°, 30°, 60°, ..., 330°) are queried.
A single facade-facing heading per point is retained.

Requirements:
    - Python 3.9+
    - googlemaps
    - requests
    - Pillow

Usage:
    1. Set your Google Maps Static API key as an environment variable:
        export GOOGLE_MAPS_API_KEY=your_key_here

    2. Prepare a coordinates Excel file (`street_coordinates.xlsx`) with columns:
        - street_name (e.g., "Selanikliler_Street")
        - neighborhood
        - district
        - start_lat, start_lng, end_lat, end_lng
        - facade_side (either "left" or "right")

    3. Run:
        python 01_data_collection.py

Output:
    ./street_view_data/<street_name>/<street_name>_<point>_heading<h>.jpg
"""

import os
import sys
import time
import requests
import pandas as pd
import googlemaps
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
# API key is read from environment variable — never hardcode it.
API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
if not API_KEY:
    print("ERROR: Set GOOGLE_MAPS_API_KEY environment variable before running.")
    sys.exit(1)

# Input / output paths (change to match your local structure)
COORDS_FILE = "street_coordinates.xlsx"
OUTPUT_DIR = Path("street_view_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Collection parameters — DO NOT CHANGE these to reproduce the paper's dataset
N_POINTS_PER_STREET = 12       # 12 evenly-spaced sampling points per street
HEADINGS = list(range(0, 360, 30))  # 12 headings: 0, 30, 60, ..., 330
IMAGE_SIZE = "640x640"         # Static API returns 640x640 by default
FOV = 90                       # Field of view (degrees)
PITCH = 0                      # Pitch angle (degrees)

# Rate-limit safety
REQUEST_DELAY_SEC = 0.1        # 100 ms between requests

# =============================================================================
# GEODESIC INTERPOLATION
# =============================================================================
def interpolate_points(lat1, lng1, lat2, lng2, n_points):
    """
    Linearly interpolate n_points between (lat1, lng1) and (lat2, lng2).
    Returns a list of (lat, lng) tuples of length n_points.
    """
    points = []
    for i in range(n_points):
        frac = i / (n_points - 1) if n_points > 1 else 0.0
        lat = lat1 + frac * (lat2 - lat1)
        lng = lng1 + frac * (lng2 - lng1)
        points.append((lat, lng))
    return points


# =============================================================================
# STREET VIEW STATIC API DOWNLOAD
# =============================================================================
def download_streetview_image(lat, lng, heading, output_path):
    """
    Downloads one Street View image for the given (lat, lng, heading).
    Returns True if the image was successfully saved, False otherwise.
    """
    base_url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "location": f"{lat},{lng}",
        "size": IMAGE_SIZE,
        "heading": heading,
        "fov": FOV,
        "pitch": PITCH,
        "return_error_code": "true",
        "key": API_KEY,
    }
    try:
        r = requests.get(base_url, params=params, timeout=30)
        if r.status_code == 200 and r.headers.get("Content-Type", "").startswith("image/"):
            with open(output_path, "wb") as f:
                f.write(r.content)
            return True
        return False
    except requests.RequestException as e:
        print(f"  Request failed for {lat},{lng} heading {heading}: {e}")
        return False


# =============================================================================
# MAIN COLLECTION LOOP
# =============================================================================
def collect_street(row):
    """
    Collects Street View images for one street.
    Returns the number of successfully saved images.
    """
    name = row["street_name"]
    street_dir = OUTPUT_DIR / name
    street_dir.mkdir(exist_ok=True)

    points = interpolate_points(
        row["start_lat"], row["start_lng"],
        row["end_lat"], row["end_lng"],
        N_POINTS_PER_STREET
    )

    saved = 0
    for pt_idx, (lat, lng) in enumerate(points):
        for h in HEADINGS:
            out_path = street_dir / f"{name}_{pt_idx}_heading{h}.jpg"
            if out_path.exists():
                saved += 1
                continue
            ok = download_streetview_image(lat, lng, h, out_path)
            if ok:
                saved += 1
            time.sleep(REQUEST_DELAY_SEC)
    return saved


def main():
    if not os.path.exists(COORDS_FILE):
        print(f"ERROR: Coordinates file not found at {COORDS_FILE}")
        print("Expected columns: street_name, neighborhood, district, "
              "start_lat, start_lng, end_lat, end_lng, facade_side")
        sys.exit(1)

    df = pd.read_excel(COORDS_FILE)
    print(f"Loaded {len(df)} streets from {COORDS_FILE}")
    print("=" * 60)

    total_saved = 0
    for i, row in df.iterrows():
        print(f"[{i+1}/{len(df)}] {row['street_name']} "
              f"({row['neighborhood']}, {row['district']})")
        n = collect_street(row)
        total_saved += n
        print(f"  Saved: {n} images")

    print("=" * 60)
    print(f"Total images downloaded: {total_saved}")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print("\nNext step: run the semantic segmentation pipeline in "
          "02_analysis_pipeline.py")


if __name__ == "__main__":
    main()
