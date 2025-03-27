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
from recommender_model.data_loader import creating_needed_matrix

def compute_similarity_matrices(user_movie_matrix, movie_genre_matrix):
    
    # Compute user-movie similarity
    start_time = time.time()
    user_movie_similarity = cosine_similarity(user_movie_matrix)
    user_movie_training_time = time.time() - start_time
    print(f"User-Movie Similarity Matrix Training Time: {user_movie_training_time:.2f} seconds")
    
    # Compute movie-genre similarity
    start_time = time.time()
    movie_genre_similarity = cosine_similarity(movie_genre_matrix)
    movie_genre_training_time = time.time() - start_time
    print(f"Movie-Genre Similarity Matrix Training Time: {movie_genre_training_time:.2f} seconds")
    
    # Print memory usage information
    user_model_size = sys.getsizeof(user_movie_similarity)
    movie_model_size = sys.getsizeof(movie_genre_similarity)
    print(f"User-Based Model Size: {user_model_size / (1024 * 1024):.2f} MB")
    print(f"Movie-Based Model Size: {movie_model_size / (1024 * 1024):.2f} MB")
    
    return user_movie_similarity, movie_genre_similarity


def save_similarity_matrices(user_movie_similarity, movie_genre_similarity):
   
    with open(USER_MODEL_PATH, "wb") as f:
        pickle.dump(user_movie_similarity, f)

    with open(MOVIE_MODEL_PATH, "wb") as f:
        pickle.dump(movie_genre_similarity, f)

    print(f"Saved Model File Sizes: User-Based: {os.path.getsize(USER_MODEL_PATH) / (1024 * 1024):.2f} MB, "
          f"Movie-Based: {os.path.getsize(MOVIE_MODEL_PATH) / (1024 * 1024):.2f} MB")

def load_similarity_matrices():
    
    with open(USER_MODEL_PATH, "rb") as f:
        user_movie_similarity = pickle.load(f)

    with open(MOVIE_MODEL_PATH, "rb") as f:
        movie_genre_similarity = pickle.load(f)
    
    return user_movie_similarity, movie_genre_similarity



def train(csv_file_path):
  
    user_movie_matrix, movie_genre_matrix = creating_needed_matrix(csv_file_path)
    user_movie_similarity, movie_genre_similarity = compute_similarity_matrices(user_movie_matrix, movie_genre_matrix)
    save_similarity_matrices(user_movie_similarity, movie_genre_similarity)
    
    return user_movie_similarity, movie_genre_similarity
