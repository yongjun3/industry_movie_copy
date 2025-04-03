
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
import time
import sys
import os
import pickle

from config import *



def load_data(csv_file_path=DATA_PATH):

    return pd.read_csv(csv_file_path)


def creating_needed_matrix(csv_data=DATA_PATH):
  
    merged_df = load_data(csv_data)
    
    # Movie themes
    theme_cols = GENRE_COLUMNS
    
    # Create the movie-theme matrix 
    # Assuming if there are multiple rows for the same movie, genre distribution will be the same
    movie_genre_matrix = merged_df[theme_cols].groupby(merged_df['movie_id']).last()
    
    # For duplicate rating of movie_user instances, collect the last rating instance
    user_movie_temp = merged_df.groupby(['user_id', 'movie_id'], as_index=False)['rating'].last()

    # Create user-movie matrix, filling missing ratings with 0
    user_movie_matrix = user_movie_temp.pivot(index="user_id", columns="movie_id", values="rating").fillna(0)
    
    # Returning both matrices
    return user_movie_matrix, movie_genre_matrix


def get_watched_movies(user_id, user_movie_matrix):

    # Retrieves all movies a user has watched, including those they rated.
    if user_id not in user_movie_matrix.index:
        return set()
    return set(user_movie_matrix.loc[user_id][user_movie_matrix.loc[user_id] > 0].index)


def get_top_rated_movies(df=None, n=20):
    """
    When user does not exist
    """
    if df is None:
        df = load_data()
        
    sorted_df = df.sort_values(by=['vote_average', 'vote_count'], ascending=False)
    top_movies = list(sorted_df['movie_id'].unique()[:n])
    np.random.shuffle(top_movies)
    return top_movies


