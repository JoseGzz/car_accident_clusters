from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from datetime import datetime, timedelta

app = Flask(__name__)

# Load data once (in a real app, you might use a DB)
try:
    df = pd.read_csv('accidents.csv')
    df['date'] = pd.to_datetime(df['date'])
except FileNotFoundError:
    print("Error: accidents.csv not found. Please run generate_data.py first.")
    df = pd.DataFrame(columns=['latitude', 'longitude', 'date'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    params = request.json
    
    # Parameters
    epsilon_meters = float(params.get('epsilon', 50))
    min_samples = int(params.get('minPoints', 10))
    
    # New Date Range Logic
    # Expecting dates in format 'YYYY-MM-DD'
    start_str = params.get('startDate')
    end_str = params.get('endDate')

    # Default to full 2025 if missing (though frontend sends it)
    if start_str and end_str:
        start_date = pd.to_datetime(start_str)
        # Add one day to end_date to make it inclusive if time is 00:00:00, 
        # or simply ensure logic covers the full day.
        # Simplest is making end_date end of that day:
        end_date = pd.to_datetime(end_str) + timedelta(days=1) - timedelta(seconds=1)
    else:
        start_date = datetime(2025, 1, 1)
        end_date = datetime(2025, 12, 31, 23, 59, 59)
    
    # Filter by date range
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    filtered_df = df[mask].copy()
    
    if len(filtered_df) == 0:
        return jsonify({
            'points': [],
            'stats': {'total': 0, 'n_clusters': 0, 'noise': 0}
        })

    # Prepare coordinates for DBSCAN
    # Haversine metric requires radians
    coords = filtered_df[['latitude', 'longitude']].values
    coords_rad = np.radians(coords)
    
    # Earth's radius in meters approx 6371000
    # Epsilon in radians = epsilon_meters / Earth_radius
    kms_per_radian = 6371.0088
    epsilon_rad = (epsilon_meters / 1000.0) / kms_per_radian
    
    # Run DBSCAN
    db = DBSCAN(eps=epsilon_rad, min_samples=min_samples, algorithm='ball_tree', metric='haversine')
    db.fit(coords_rad)
    
    # Assign labels to dataframe
    filtered_df['cluster'] = db.labels_
    
    # Format results
    results = []
    clusters_found = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    noise_points = list(db.labels_).count(-1)
    
    # Group by clusters for the frontend logic (optional, but sending raw points is fine too)
    # We will send a flat list of points with their cluster ID
    for _, row in filtered_df.iterrows():
        results.append({
            'lat': row['latitude'],
            'lng': row['longitude'],
            'cluster': int(row['cluster']) # -1 is noise
        })
        
    return jsonify({
        'points': results,
        'stats': {
            'total': len(filtered_df),
            'n_clusters': clusters_found,
            'noise': noise_points
        }
    })

if __name__ == '__main__':
    # host='0.0.0.0' allows the server to be accessible externally 
    # (required for Docker, some cloud IDEs, and local network sharing)
    app.run(debug=True, host='0.0.0.0', port=8080)