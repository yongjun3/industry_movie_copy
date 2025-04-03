"""
Configuration settings for the recommendation system.
"""

import os

# Data paths
DATA_DIR = "data"
DATA_PATH = "data/merged_data.csv"
EVAL_DATA_PATH = "data/unique_watch_df.csv"
TRAIN_DATA_PATH = "data/train_data.csv"
VAL_DATA_PATH = "data/val_data.csv"
TEST_DATA_PATH = "data/test_data.csv"

# Results path
RESULTS_PATH = 'data/offline_eval_results.pkl'

# Model paths
USER_MODEL_PATH = "user_movie_similarity.pkl"
MOVIE_MODEL_PATH = "movie_genre_similarity.pkl"

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8082

# Recommendation settings
DEFAULT_K = 20  # Number of recommendations to generate

# Movie themes/genres
GENRE_COLUMNS = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
    "Drama", "Family", "Fantasy", "Foreign", "History", "Horror", "Music",
    "Mystery", "Romance", "Science Fiction", "TV Movie", "Thriller", "War", "Western"
]

# Function to override settings for testing
def get_test_config(temp_dir=None):
    """
    Returns a configuration with test paths for unit testing.
    
    Args:
        temp_dir: Optional temporary directory to use
        
    Returns:
        Dict of configuration settings for testing
    """
    test_dir = temp_dir or "test_data"
    return {
        "DATA_DIR": test_dir,
        "DATA_PATH": os.path.join(test_dir, "merged_data.csv"),
        "EVAL_DATA_PATH": os.path.join(test_dir, "unique_watch_df.csv"),
        "TRAIN_DATA_PATH": os.path.join(test_dir, "train_data.csv"),
        "VAL_DATA_PATH": os.path.join(test_dir, "val_data.csv"),
        "TEST_DATA_PATH": os.path.join(test_dir, "test_data.csv"),
        "RESULTS_PATH": os.path.join(test_dir, "results.pkl"),
        "USER_MODEL_PATH": os.path.join(test_dir, "user_movie_similarity.pkl"),
        "MOVIE_MODEL_PATH": os.path.join(test_dir, "movie_genre_similarity.pkl"),
        "DEFAULT_K": 5  # Use smaller k for testing
    }

