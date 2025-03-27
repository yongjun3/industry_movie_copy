import pandas as pd
import requests
import time
import re
import os
import json
import logging
from data_quality import data_quality_monitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='log/data_cleaning.log'
)
logger = logging.getLogger('data_cleaning')

# Create output directories
os.makedirs("data", exist_ok=True)
os.makedirs("data/drift_reports", exist_ok=True)

print("Starting data cleaning and drift detection process...")
logger.info("Starting data cleaning and drift detection process")

# convert Kafka log to watch and rate dataframes
watch_data = []
rate_data = []

# Track data for drift detection
monthly_batches = {}

print("Parsing log entries from data/output.txt...")
with open('data/output.txt', "r") as file:
    for line in file:
        parts = line.strip().split(",")
        if len(parts) < 3:
            continue
        timestamp, user_id, request = parts[0], parts[1], parts[2]

        # Extract month for drift detection batching
        try:
            month = timestamp[:7]  # format: YYYY-MM
        except:
            month = "unknown"

        if "GET /data/m/" in request:
            match = re.search(r"GET /data/m/([^/]+)/(\d+).mpg", request)
            if match:
                movie_id = match.group(1)
                minute = int(match.group(2))
                watch_data.append([timestamp, user_id, movie_id, minute])

        elif "GET /rate/" in request:
            match = re.search(r"GET /rate/([^=]+)=(\d+)", request)
            if match:
                movie_id = match.group(1)
                rating = int(match.group(2))
                rate_data.append([timestamp, user_id, movie_id, rating])
                
                # Add to monthly batch for drift detection
                entry = {
                    'timestamp': timestamp,
                    'user_id': user_id,
                    'movie_id': movie_id,
                    'rating': rating
                }
                
                if month not in monthly_batches:
                    monthly_batches[month] = []
                monthly_batches[month].append(entry)

print(f"Parsed {len(watch_data)} watch events and {len(rate_data)} rating events")
logger.info(f"Parsed {len(watch_data)} watch events and {len(rate_data)} rating events")

# Create DataFrames
watch_df = pd.DataFrame(watch_data, columns=["timestamp", "user_id", "movie_id", "minute_watched"])
rate_df = pd.DataFrame(rate_data, columns=["timestamp", "user_id", "movie_id", "rating"])

# get only watch and rating after February 2025
watch_df_feb = watch_df[watch_df["timestamp"] >= "2025-02-01"]
rate_df_feb = rate_df[rate_df["timestamp"] >= "2025-02-01"]

print(f"Filtered to {len(watch_df_feb)} watch events and {len(rate_df_feb)} rating events after Feb 2025")

# Extract unique movie and user IDs from watch_df and rate_df
unique_movie_ids = set(watch_df_feb["movie_id"]).intersection(set(rate_df_feb["movie_id"]))
unique_user_ids = set(watch_df_feb["user_id"]).intersection(set(rate_df_feb["user_id"]))

# keep only the movies and users that are in both dataframes
unique_rate_df = rate_df_feb[rate_df_feb["movie_id"].isin(unique_movie_ids) & rate_df_feb["user_id"].isin(unique_user_ids)]
unique_watch_df = watch_df_feb[watch_df_feb["movie_id"].isin(unique_movie_ids) & watch_df_feb["user_id"].isin(unique_user_ids)]

print(f"Filtered to {len(unique_rate_df)} ratings and {len(unique_watch_df)} watches with users and movies in both sets")

# Save filtered data
unique_rate_df.to_csv("data/unique_rate_df.csv", index=False)
unique_watch_df.to_csv("data/unique_watch_df.csv", index=False)

print("Saved filtered data to unique_rate_df.csv and unique_watch_df.csv")

# Get unique movie IDs for API fetching
movie_ids = unique_rate_df["movie_id"].unique()

print(f"Fetching data for {len(movie_ids)} unique movies from API...")

# Base URL for the API
base_url = "http://128.2.204.215:8080/movie/"

# Open CSV file for writing
with open("data/movie_data.csv", "w", encoding="utf-8") as file:
    # Write header
    file.write("id,tmdb_id,imdb_id,title,original_title,adult,belongs_to_collection,budget,genres,homepage,"
               "original_language,overview,popularity,poster_path,production_companies,production_countries,"
               "release_date,revenue,runtime,spoken_languages,status,vote_average,vote_count\n")

    # Fetch movie information from API
    api_success = 0
    api_errors = 0
    
    for movie_id in movie_ids:
        url = f"{base_url}{movie_id}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                movie_info = response.json()
                
                # Validate API response
                expected_fields = ["id", "title", "genres"]
                if not data_quality_monitor.validate_api_response(movie_info, expected_fields):
                    logger.warning(f"Invalid API response for movie {movie_id}")
                    api_errors += 1
                    continue

                # Function to safely retrieve values, replacing missing values with "NAN"
                def get_value(data, key, default="NAN"):
                    value = data.get(key, default)
                    return str(value).replace(",", " ") if value is not None else default  # Avoid CSV issues

                # Parse movie data with proper handling of missing values
                parsed_data = {
                    "id": get_value(movie_info, "id"),
                    "tmdb_id": get_value(movie_info, "tmdb_id"),
                    "imdb_id": get_value(movie_info, "imdb_id"),
                    "title": get_value(movie_info, "title"),
                    "original_title": get_value(movie_info, "original_title"),
                    "adult": get_value(movie_info, "adult"),
                    "belongs_to_collection": get_value(movie_info.get("belongs_to_collection", {}), "name"),
                    "budget": get_value(movie_info, "budget"),
                    "genres": "|".join([genre["name"] for genre in movie_info.get("genres", [])]) if movie_info.get("genres") else "NAN",
                    "homepage": get_value(movie_info, "homepage"),
                    "original_language": get_value(movie_info, "original_language"),
                    "overview": get_value(movie_info, "overview"),
                    "popularity": get_value(movie_info, "popularity"),
                    "poster_path": get_value(movie_info, "poster_path"),
                    "production_companies": "|".join([company["name"] for company in movie_info.get("production_companies", [])]) if movie_info.get("production_companies") else "NAN",
                    "production_countries": "|".join([country["name"] for country in movie_info.get("production_countries", [])]) if movie_info.get("production_countries") else "NAN",
                    "release_date": get_value(movie_info, "release_date"),
                    "revenue": get_value(movie_info, "revenue"),
                    "runtime": get_value(movie_info, "runtime"),
                    "spoken_languages": "|".join([lang["name"] for lang in movie_info.get("spoken_languages", [])]) if movie_info.get("spoken_languages") else "NAN",
                    "status": get_value(movie_info, "status"),
                    "vote_average": get_value(movie_info, "vote_average"),
                    "vote_count": get_value(movie_info, "vote_count"),
                }

                # Write parsed data to CSV
                file.write(",".join(parsed_data.values()) + "\n")
                api_success += 1
            else:
                logger.warning(f"API error for movie {movie_id}: Status code {response.status_code}")
                api_errors += 1

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data for movie ID {movie_id}: {e}")
            api_errors += 1
            print(f"Error fetching data for movie ID {movie_id}: {e}")

print(f"Movie data API requests: {api_success} successful, {api_errors} errors")
print("Movie data successfully saved to data/movie_data.csv")

# Clean movie data
print("Cleaning movie data...")
df = pd.read_csv("data/movie_data.csv", on_bad_lines="skip")

cleaned_df = df[["title", "id", "imdb_id", "tmdb_id", "adult", "genres", "release_date", "revenue", "runtime", "vote_average", "vote_count"]]

# Generate one-hot encoded genre columns
genres_one_hot = cleaned_df['genres'].str.get_dummies(sep='|')
cleaned_df = pd.concat([cleaned_df, genres_one_hot], axis=1)
cleaned_df.drop(columns=['genres', 'NAN'], inplace=True, errors='ignore')
cleaned_df.dropna(axis=1, how='all', inplace=True)

# Convert datetime and boolean columns
cleaned_df["release_date"] = pd.to_datetime(cleaned_df["release_date"], errors='coerce')
cleaned_df['adult'] = cleaned_df['adult'].astype(int)

# Save cleaned movie data
cleaned_df.to_csv("data/cleaned_movie_data.csv", index=False)
print("Cleaned movie data saved to data/cleaned_movie_data.csv")

# Fetch user data
print("Fetching user data...")
server_ip = "128.2.204.215"
base_url = f"http://{server_ip}:8080/user/"

unique_user_ids = pd.read_csv("data/unique_rate_df.csv")["user_id"].unique()

user_data_success = 0
user_data_errors = 0

with open("data/user_data.csv", "w") as file:
    file.write("user_id,age,occupation,gender\n")

    # Fetch user information from API
    for user_id in unique_user_ids:
        url = f"{base_url}{user_id}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                user_info = response.json()
                
                # Validate user API response
                expected_fields = ["user_id"]
                if not data_quality_monitor.validate_api_response(user_info, expected_fields):
                    logger.warning(f"Invalid API response for user {user_id}")
                    user_data_errors += 1
                    continue
                    
                file.write(
                    f"{user_info['user_id']},{user_info.get('age', 'N/A')},{user_info.get('occupation', 'N/A')},{user_info.get('gender', 'N/A')}\n"
                )
                user_data_success += 1
            else:
                print(f"Failed to fetch data for {user_id}: Status Code {response.status_code}")
                user_data_errors += 1
                
        except Exception as e:
            logger.error(f"Error fetching data for user {user_id}: {e}")
            user_data_errors += 1
            print(f"Error fetching data for {user_id}: {e}")

        time.sleep(0.5)  # Prevent excessive API requests

print(f"User data API requests: {user_data_success} successful, {user_data_errors} errors")
print("User data saved to data/user_data.csv")

# Load and merge all data
print("Merging all datasets...")
user_data_df = pd.read_csv("data/user_data.csv")
movie_data_df = pd.read_csv("data/cleaned_movie_data.csv")
movie_data_df = movie_data_df.rename(columns={'id': 'movie_id'})
unique_rate_df = pd.read_csv("data/unique_rate_df.csv")

# Convert ID columns to same type for merging
unique_rate_df['user_id'] = unique_rate_df['user_id'].astype(int)
user_data_df['user_id'] = user_data_df['user_id'].astype(int)

# Merge data
merged_df = pd.merge(unique_rate_df, user_data_df, on="user_id")
merged_df = pd.merge(merged_df, movie_data_df, on='movie_id')

# Save merged data
merged_df.to_csv("data/merged_data.csv", index=False)
print("Merged data saved to data/merged_data.csv")


# ========== DATA DRIFT DETECTION ==========
print("\n===== DATA DRIFT DETECTION =====")
logger.info("Starting data drift detection")

# First establish baseline from the first month of data
months = sorted(monthly_batches.keys())

if len(months) >= 2:
    print(f"Found data from {len(months)} months: {', '.join(months)}")
    
    # Use first month as baseline
    baseline_month = months[0]
    baseline_df = pd.DataFrame(monthly_batches[baseline_month])
    print(f"Using {baseline_month} as baseline with {len(baseline_df)} entries")
    
    # Establish baseline
    data_quality_monitor.establish_baseline(baseline_df)
    
    # Check drift for subsequent months
    drift_found = False
    
    for month in months[1:]:
        current_df = pd.DataFrame(monthly_batches[month])
        print(f"Checking drift for {month} with {len(current_df)} entries")
        
        if len(current_df) < 100:
            print(f"Skipping {month} - insufficient data ({len(current_df)} entries)")
            continue
            
        # Detect drift
        drift_result = data_quality_monitor.detect_data_drift(current_df)
        
        # Save drift detection results
        drift_file = f"data/drift_reports/drift_{baseline_month}_vs_{month}.json"
        with open(drift_file, "w") as f:
            json.dump(drift_result, f, indent=2)
        
        if drift_result['drift_detected']:
            drift_found = True
            print(f"DATA DRIFT DETECTED between {baseline_month} and {month}!")
            print(f"Drift metrics: {json.dumps(drift_result['drift_metrics'], indent=2)}")
            logger.warning(f"Data drift detected between {baseline_month} and {month}: {drift_result['drift_metrics']}")
        else:
            print(f"No significant drift detected between {baseline_month} and {month}")
    
    if not drift_found:
        print("No significant data drift detected across all months")
else:
    print(f"Insufficient monthly data for drift detection. Found {len(months)} months.")
    logger.warning("Insufficient monthly data for drift detection")

print("\nData cleaning and drift detection process completed!")