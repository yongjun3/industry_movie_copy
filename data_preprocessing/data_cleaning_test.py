import pandas as pd
import requests
import time
import re
import os

def create_directories():
    """Create necessary directories for data storage."""
    # Create data directory if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Created data directory")

def parse_log_data(log_file='output.txt'):
    """Parse Kafka log data from a file into watch and rate dataframes.
    
    Args:
        log_file (str): Path to the log file
        
    Returns:
        tuple: (watch_df, rate_df) containing the parsed data
    """
    watch_data = []
    rate_data = []
    
    try:
        with open(log_file, "r") as file:
            for line in file:
                parts = line.strip().split(",")
                if len(parts) < 3:
                    continue
                timestamp, user_id, request = parts[0], parts[1], parts[2]

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
    except FileNotFoundError:
        print(f"Error: Log file '{log_file}' not found.")
        return None, None
    
    watch_df = pd.DataFrame(watch_data, columns=["timestamp", "user_id", "movie_id", "minute_watched"])
    rate_df = pd.DataFrame(rate_data, columns=["timestamp", "user_id", "movie_id", "rating"])
    
    return watch_df, rate_df

def filter_recent_data(watch_df, rate_df, cutoff_date="2025-02-01"):
    """Filter data to only include records after the cutoff date.
    
    Args:
        watch_df (DataFrame): Watch data
        rate_df (DataFrame): Rate data
        cutoff_date (str): Cutoff date in YYYY-MM-DD format
        
    Returns:
        tuple: (watch_df_filtered, rate_df_filtered)
    """
    if watch_df is None or rate_df is None:
        return None, None
        
    watch_df_filtered = watch_df[watch_df["timestamp"] >= cutoff_date]
    rate_df_filtered = rate_df[rate_df["timestamp"] >= cutoff_date]
    
    return watch_df_filtered, rate_df_filtered

def extract_common_entities(watch_df, rate_df):
    """Extract entities (users, movies) that appear in both watch and rate dataframes.
    
    Args:
        watch_df (DataFrame): Watch data
        rate_df (DataFrame): Rate data
        
    Returns:
        tuple: (unique_rate_df, unique_watch_df)
    """
    if watch_df is None or rate_df is None:
        return None, None
        
    # Extract unique movie and user IDs from watch_df and rate_df
    unique_movie_ids = set(watch_df["movie_id"]).intersection(set(rate_df["movie_id"]))
    unique_user_ids = set(watch_df["user_id"]).intersection(set(rate_df["user_id"]))

    # Keep only the movies and users that are in both dataframes
    unique_rate_df = rate_df[rate_df["movie_id"].isin(unique_movie_ids) & 
                            rate_df["user_id"].isin(unique_user_ids)]
    unique_watch_df = watch_df[watch_df["movie_id"].isin(unique_movie_ids) & 
                              watch_df["user_id"].isin(unique_user_ids)]
                              
    return unique_rate_df, unique_watch_df

def fetch_movie_data(movie_ids, api_base_url="http://128.2.204.215:8080/movie/"):
    """Fetch movie data from the API for the given movie IDs.
    
    Args:
        movie_ids (list): List of movie IDs
        api_base_url (str): Base URL for the movie API
        
    Returns:
        DataFrame: Movie data
    """
    create_directories()
    
    # Open CSV file for writing
    with open("data/movie_data.csv", "w", encoding="utf-8") as file:
        # Write header
        file.write("id,tmdb_id,imdb_id,title,original_title,adult,belongs_to_collection,budget,genres,homepage,"
                   "original_language,overview,popularity,poster_path,production_companies,production_countries,"
                   "release_date,revenue,runtime,spoken_languages,status,vote_average,vote_count\n")

        # Fetch movie information from API
        for movie_id in movie_ids:
            url = f"{api_base_url}{movie_id}"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    movie_info = response.json()

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

            except requests.exceptions.RequestException as e:
                print(f"Error fetching data for movie ID {movie_id}: {e}")

    print("Movie data successfully saved to data/movie_data.csv")
    
    try:
        return pd.read_csv("data/movie_data.csv", on_bad_lines="skip")
    except:
        print("Error reading movie data CSV.")
        return None

def clean_movie_data(movie_df):
    """Clean and prepare movie data.
    
    Args:
        movie_df (DataFrame): Raw movie data
        
    Returns:
        DataFrame: Cleaned movie data
    """
    if movie_df is None:
        return None
        
    create_directories()
    
    try:
        # Select relevant columns
        cleaned_df = movie_df[["title", "id", "imdb_id", "tmdb_id", "adult", "genres", 
                              "release_date", "revenue", "runtime", "vote_average", "vote_count"]]
        
        # One-hot encode genres
        genres_one_hot = cleaned_df['genres'].str.get_dummies(sep='|')
        cleaned_df = pd.concat([cleaned_df, genres_one_hot], axis=1)
        
        # Clean up dataframe
        cleaned_df.drop(columns=['genres', 'NAN'], inplace=True, errors='ignore')
        cleaned_df.dropna(axis=1, how='all', inplace=True)
        
        # Convert data types
        cleaned_df["release_date"] = pd.to_datetime(cleaned_df["release_date"], errors='coerce')
        cleaned_df['adult'] = cleaned_df['adult'].astype(int)
        
        # Save cleaned data
        cleaned_df.to_csv("data/cleaned_movie_data.csv", index=False)
        print("Cleaned movie data saved to data/cleaned_movie_data.csv")
        
        return cleaned_df
    except Exception as e:
        print(f"Error cleaning movie data: {e}")
        return None

def fetch_user_data(user_ids, api_base_url="http://128.2.204.215:8080/user/"):
    """Fetch user data from the API for the given user IDs.
    
    Args:
        user_ids (list): List of user IDs
        api_base_url (str): Base URL for the user API
        
    Returns:
        DataFrame: User data
    """
    create_directories()
    
    with open("data/user_data.csv", "w") as file:
        file.write("user_id,age,occupation,gender\n")

        # Fetch user information from API
        for user_id in user_ids:
            url = f"{api_base_url}{user_id}"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    user_info = response.json()
                    file.write(
                        f"{user_info['user_id']},{user_info.get('age', 'N/A')},{user_info.get('occupation', 'N/A')},{user_info.get('gender', 'N/A')}\n"
                    )
                else:
                    print(f"Failed to fetch data for {user_id}: Status Code {response.status_code}")
            except Exception as e:
                print(f"Error fetching data for {user_id}: {e}")

            

    print("User data saved to data/user_data.csv")
    
    try:
        return pd.read_csv("data/user_data.csv")
    except:
        print("Error reading user data CSV.")
        return None

def merge_all_data(rate_df, user_df, movie_df):
    """Merge rate, user, and movie data into a single DataFrame.
    
    Args:
        rate_df (DataFrame): Rating data
        user_df (DataFrame): User data
        movie_df (DataFrame): Movie data
        
    Returns:
        DataFrame: Merged data
    """
    if rate_df is None or user_df is None or movie_df is None:
        return None
        
    create_directories()
    
    try:
        # Ensure user_id is of the same type in both dataframes
        rate_df['user_id'] = rate_df['user_id'].astype(int)
        user_df['user_id'] = user_df['user_id'].astype(int)
        
        # Rename movie ID column for consistency
        movie_df = movie_df.rename(columns={'id': 'movie_id'})
        
        # Merge dataframes
        merged_df = pd.merge(rate_df, user_df, on="user_id")
        merged_df = pd.merge(merged_df, movie_df, on='movie_id')
        
        # Save merged data
        merged_df.to_csv("data/merged_data.csv", index=False)
        print("Merged data saved to data/merged_data.csv")
        
        return merged_df
    except Exception as e:
        print(f"Error merging data: {e}")
        return None

def run_data_cleaning_pipeline(log_file='output.txt', cutoff_date="2025-02-01"):
    """Run the complete data cleaning pipeline.
    
    Args:
        log_file (str): Path to the log file
        cutoff_date (str): Cutoff date in YYYY-MM-DD format
        
    Returns:
        tuple: (merged_df, rate_df, watch_df, user_df, movie_df)
    """
    # Create necessary directories
    create_directories()
    
    # Step 1: Parse log data
    print("Step 1: Parsing log data...")
    watch_df, rate_df = parse_log_data(log_file)
    if watch_df is None or rate_df is None:
        print("Failed to parse log data. Pipeline aborted.")
        return None, None, None, None, None
    
    # Step 2: Filter recent data
    print("Step 2: Filtering data by date...")
    watch_df_recent, rate_df_recent = filter_recent_data(watch_df, rate_df, cutoff_date)
    
    # Step 3: Extract common entities
    print("Step 3: Extracting common entities...")
    unique_rate_df, unique_watch_df = extract_common_entities(watch_df_recent, rate_df_recent)
    
    # Save intermediate data
    unique_rate_df.to_csv("unique_rate_df.csv", index=False)
    unique_watch_df.to_csv("unique_watch_df.csv", index=False)
    print("Intermediate data saved to unique_rate_df.csv and unique_watch_df.csv")
    
    # Step 4: Fetch movie data
    print("Step 4: Fetching movie data...")
    movie_ids = unique_rate_df["movie_id"].unique()
    movie_df = fetch_movie_data(movie_ids)
    
    # Step 5: Clean movie data
    print("Step 5: Cleaning movie data...")
    cleaned_movie_df = clean_movie_data(movie_df)
    
    # Step 6: Fetch user data
    print("Step 6: Fetching user data...")
    user_ids = unique_rate_df["user_id"].unique()
    user_df = fetch_user_data(user_ids)
    
    # Step 7: Merge all data
    print("Step 7: Merging all data...")
    merged_df = merge_all_data(unique_rate_df, user_df, cleaned_movie_df)
    
    print("Data cleaning pipeline completed successfully!")
    return merged_df, unique_rate_df, unique_watch_df, user_df, cleaned_movie_df

if __name__ == "__main__":
    # Run the complete pipeline when script is executed directly
    merged_df, rate_df, watch_df, user_df, movie_df = run_data_cleaning_pipeline()
    
    # Display some basic information about the data
    if merged_df is not None:
        print(f"\nDataset Summary:")
        print(f"Total merged records: {len(merged_df)}")
        print(f"Unique users: {len(merged_df['user_id'].unique())}")
        print(f"Unique movies: {len(merged_df['movie_id'].unique())}")
        print(f"Average rating: {merged_df['rating'].mean():.2f}")
