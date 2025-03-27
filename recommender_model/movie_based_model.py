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

def recommend_movie_based(user_id, movie_genre_matrix=None, movie_genre_similarity=None, user_movie_matrix=None, top_n=20):
    """
    Recommend movies based on movie genre similarity.
    
    Args:
        user_id (int): ID of the user to recommend for.
        movie_genre_matrix (pandas.DataFrame, optional): Movie-genre matrix.
        movie_genre_similarity (numpy.ndarray, optional): Movie-genre similarity matrix.
        user_movie_matrix (pandas.DataFrame, optional): User-movie matrix.
        top_n (int): Number of recommendations to return.
        
    Returns:
        list: List of recommended movie IDs.
    """
    global df
    
    # If matrices not provided, load them
    if user_movie_matrix is None or movie_genre_matrix is None or movie_genre_similarity is None:
        user_movie_matrix, movie_genre_matrix = creating_needed_matrix()
        _, movie_genre_similarity = load_similarity_matrices()
    
    # Load data if needed for fallback recommendations
    if df is None:
        df = pd.read_csv(DATA_PATH)

    # Check if user exists, if not return popular movies
    if user_id not in user_movie_matrix.index:
        return get_top_rated_movies(df, top_n)

    user_movies = user_movie_matrix.loc[user_id]
    high_rated_movies = user_movies[user_movies >= 4].index.tolist()

    # Set to store unique recommended movies
    recommended_movies = set()

    # Get all movies the user has already rated (to avoid duplicate recommendations)
    already_rated_movies = set(user_movie_matrix.loc[user_id][user_movie_matrix.loc[user_id] > 0].index)

    for movie in high_rated_movies:
        if movie not in movie_genre_matrix.index:
            continue  # Skip movies not in genre matrix
        
        # Get numerical index of the movie in the genre matrix
        movie_index = movie_genre_matrix.index.get_loc(movie)

        # Get similarity scores for this movie
        movie_similarities = movie_genre_similarity[movie_index, :]

        # Get indices of similar movies sorted by similarity (highest first)
        similar_movie_indices = np.argsort(movie_similarities)[::-1]

        for similar_index in similar_movie_indices:
            similar_movie = movie_genre_matrix.index[similar_index]
            
            # Ensure we do not recommend already-rated movies
            if similar_movie not in already_rated_movies:
                recommended_movies.add(similar_movie)

            # Stop if we reach top_n recommendations
            if len(recommended_movies) >= top_n:
                break
        
        if len(recommended_movies) >= top_n:
            break  # Stop iterating over rated movies if we have enough recommendations
        
    return list(recommended_movies)[:top_n]



def recommend_movie_based_with_watched(user_id, movie_genre_similarity, user_movie_matrix, top_n=20):
    """
    Recommends movies for a user without filtering out already watched/rated movies.
    Used for evaluation purposes.
    
    Args:
        user_id (int): ID of the user to recommend for.
        movie_genre_similarity (numpy.ndarray): Movie-genre similarity matrix.
        user_movie_matrix (pandas.DataFrame): User-movie matrix.
        top_n (int): Number of recommendations to return.
        
    Returns:
        list: List of recommended movie IDs.
    """
    # Check if user exists
    if user_id not in user_movie_matrix.index:
        return []
    
    user_movies = user_movie_matrix.loc[user_id]
    rated_movies = user_movies[user_movies > 0].index.tolist()
    
    recommended_movies = []
    for movie in rated_movies:
        if movie not in user_movie_matrix.columns:
            continue
        
        movie_index = list(user_movie_matrix.columns).index(movie)
        movie_similarities = movie_genre_similarity[movie_index, :]
        similar_movie_indices = np.argsort(movie_similarities)[::-1]
        
        for similar_index in similar_movie_indices:
            similar_movie = user_movie_matrix.columns[similar_index]
            
            # Add to recommendations (without filtering already watched)
            if similar_movie not in recommended_movies:
                recommended_movies.append(similar_movie)
                
            if len(recommended_movies) >= top_n:
                break
                
        if len(recommended_movies) >= top_n:
            break
    
    return recommended_movies[:top_n]