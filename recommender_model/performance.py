import time
import numpy as np
from recommender_model.user_based_model import recommend_user_based
from recommender_model.movie_based_model import recommend_movie_based

def measure_inference_time(user_ids, model_type, user_movie_similarity, movie_genre_similarity, user_movie_matrix, movie_genre_matrix, top_n=20):
    """
    Measure the average inference time for a recommendation model.
    
    Args:
        user_ids (list): List of user IDs to test.
        model_type (str): Type of recommendation model ('user' or 'movie').
        user_movie_similarity (numpy.ndarray): User-movie similarity matrix.
        movie_genre_similarity (numpy.ndarray): Movie-genre similarity matrix.
        user_movie_matrix (pandas.DataFrame): User-movie matrix.
        movie_genre_matrix (pandas.DataFrame): Movie-genre matrix.
        top_n (int): Number of recommendations to generate.
        
    Returns:
        float: Average inference time in milliseconds.
    """
    total_time = 0
    num_users = len(user_ids)

    for user_id in user_ids:
        start_time = time.time()
        if model_type == "user":
            recommend_user_based(user_id, user_movie_similarity, user_movie_matrix, top_n)
        elif model_type == "movie":
            recommend_movie_based(user_id, movie_genre_matrix, movie_genre_similarity, user_movie_matrix, top_n)

        total_time += (time.time() - start_time) * 1000  

    avg_inference_time = total_time / num_users
    print(f"Average Inference Time per User ({model_type}-based model): {avg_inference_time:.2f} ms")

    return avg_inference_time