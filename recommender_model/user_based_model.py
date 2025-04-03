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
from recommender_model.data_loader import creating_needed_matrix, get_top_rated_movies
from recommender_model.train import load_similarity_matrices

# Global df reference for fallback recommendations
df = None

def recommend_user_based(user_id, user_movie_similarity=None, user_movie_matrix=None, top_n=20):
    """
    Recommend movies based on user similarity.
    
    Args:
        user_id (int): ID of the user to recommend for.
        user_movie_similarity (numpy.ndarray, optional): User-movie similarity matrix.
        user_movie_matrix (pandas.DataFrame, optional): User-movie matrix.
        top_n (int): Number of recommendations to return.
        
    Returns:
        list: List of recommended movie IDs.
    """
    global df
    
    # If matrices not provided, load them
    if user_movie_matrix is None or user_movie_similarity is None:
        user_movie_matrix, _ = creating_needed_matrix()
        user_movie_similarity, _ = load_similarity_matrices()
    
    # Load data if needed for fallback recommendations
    if df is None:
        df = pd.read_csv(DATA_PATH)
    
    # Check if user exists, if not return popular movies
    if user_id not in user_movie_matrix.index:
        return get_top_rated_movies(df, top_n)
    
    # Get index of the target user
    user_index = user_movie_matrix.index.get_loc(user_id)
    
    # Get similarity scores for the user
    similarities = user_movie_similarity[user_index, :]

    # Get user indices sorted by similarity (descending order)
    most_similar_user_indices = np.argsort(similarities)[::-1]

    # Set to track already recommended movies
    recommended_dict = {}

    # List to store final recommendations
    recommended_movies = []

    # Track processed users
    processed_users = 0
    
    while len(recommended_dict) < top_n and processed_users < len(similarities):
        
        # Get the most similar user's index
        similar_user_index = most_similar_user_indices[processed_users]
        
        # Skip if it's the same user or if the user has already been processed
        if similar_user_index == user_index:
            processed_users += 1
            continue
        
        # Get the user ID of the most similar user
        similar_user_id = user_movie_matrix.index[similar_user_index]
        
        # Get movies rated 5 by this similar user
        rated_5_movies = user_movie_matrix.loc[similar_user_id][user_movie_matrix.loc[similar_user_id] == 5].index.tolist()
        
        # Get movies already rated by the target user, condition greater than 0 because 0 indiciates missing value
        already_rated_movies = user_movie_matrix.loc[user_id][user_movie_matrix.loc[user_id] > 0].index.tolist()

        # Filter out movies that are already rated by the target user and avoid duplicates
        new_recommendations = [movie for movie in rated_5_movies if movie not in already_rated_movies]

        #update dictionary assign higher values for more similar user mentions
        for movie in new_recommendations:
            decay_factor = 1 / (1 + processed_users)
            if movie in recommended_dict:
                recommended_dict[movie] += decay_factor
            else:
                recommended_dict[movie] = decay_factor

        
        # Stop if we have collected enough recommendations
        processed_users += 1

    sorted_recommend = dict(sorted(recommended_dict.items(), key=lambda item: item[1], reverse=True))
    for key in sorted_recommend:
        recommended_movies.append(key)
    
    return recommended_movies


def recommend_user_based_with_watched(user_id, user_movie_similarity, user_movie_matrix, top_n=20):
    """
    Recommends movies for a user without filtering out already watched/rated movies.
    Used for evaluation purposes.
    
    Args:
        user_id (int): ID of the user to recommend for.
        user_movie_similarity (numpy.ndarray): User-movie similarity matrix.
        user_movie_matrix (pandas.DataFrame): User-movie matrix.
        top_n (int): Number of recommendations to return.
        
    Returns:
        list: List of recommended movie IDs.
    """
    # Check if user exists
    if user_id not in user_movie_matrix.index:
        return []
        
    user_index = user_movie_matrix.index.get_loc(user_id)
    similarities = user_movie_similarity[user_index, :]
    most_similar_user_indices = np.argsort(similarities)[::-1]
    recommended_movies = []
    
    processed_users = 0
    while len(recommended_movies) < top_n and processed_users < len(similarities):
        similar_user_index = most_similar_user_indices[processed_users]
        if similar_user_index == user_index:
            processed_users += 1
            continue
        
        similar_user_id = user_movie_matrix.index[similar_user_index]
        rated_movies = user_movie_matrix.loc[similar_user_id][user_movie_matrix.loc[similar_user_id] > 0].index.tolist()
        recommended_movies.extend(rated_movies)
        recommended_movies = list(dict.fromkeys(recommended_movies))  # Remove duplicates while keeping order
        processed_users += 1
    
    return recommended_movies[:top_n]