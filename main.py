"""Main module for the movie recommendation system."""

import numpy as np
from config import DATA_PATH
from recommender_model.data_loader import load_data, creating_needed_matrix
from recommender_model.train import train
from recommender_model.user_based_model import recommend_user_based
from recommender_model.movie_based_model import recommend_movie_based
from recommender_model.performance import measure_inference_time

def main():
    """Main function to demonstrate the recommendation system."""
    print("Loading data and training models...")
    
    # Load original dataframe
    df = load_data(DATA_PATH)
    
    # Create matrices
    user_movie_matrix, movie_genre_matrix = creating_needed_matrix(DATA_PATH)
    
    # Train models (compute similarity matrices)
    user_movie_similarity, movie_genre_similarity = train(DATA_PATH)
    
    print("\nUser-Movie Matrix Shape:")
    print(user_movie_matrix.shape) 
    print("\nMovie-Genre Matrix Shape:")
    print(movie_genre_matrix.shape)
    print("\nUser-Movie Similarity Matrix Shape:")
    print(user_movie_similarity.shape)
    print("\nMovie-Genre Similarity Matrix Shape:")
    print(movie_genre_similarity.shape)
    print()
    
    # Test with existing user
    existing_user_id = 6249
    
    user_based_recs = recommend_user_based(
        existing_user_id, 
        user_movie_similarity, 
        user_movie_matrix, 
        top_n=20
    )
    print(f"User-based Recommendations for user {existing_user_id}:")
    print(user_based_recs)

    movie_based_recs = recommend_movie_based(
        existing_user_id, 
        movie_genre_matrix, 
        movie_genre_similarity,
        user_movie_matrix, 
        top_n=20
    )
    print(f"Movie-based Recommendations for user {existing_user_id}:")
    print(movie_based_recs)
    print()

    # Test with non-existing user
    not_existing_user_id = 6248
    user_based_recs = recommend_user_based(
        not_existing_user_id, 
        user_movie_similarity, 
        user_movie_matrix, 
        top_n=20
    )
    print(f"User-based Recommendations for user {not_existing_user_id}:")
    print(user_based_recs)

    movie_based_recs = recommend_movie_based(
        not_existing_user_id, 
        movie_genre_matrix, 
        movie_genre_similarity,
        user_movie_matrix, 
        top_n=20
    )
    print(f"Movie-based Recommendations for user {not_existing_user_id}:")
    print(movie_based_recs)

    # Measure inference time
    print("\nMeasuring inference time...")
    user_ids = np.random.choice(user_movie_matrix.index, 100, replace=False)
    
    user_based_avg_time = measure_inference_time(
        user_ids, 
        "user", 
        user_movie_similarity, 
        movie_genre_similarity, 
        user_movie_matrix, 
        movie_genre_matrix
    )
    
    movie_based_avg_time = measure_inference_time(
        user_ids, 
        "movie", 
        user_movie_similarity, 
        movie_genre_similarity, 
        user_movie_matrix, 
        movie_genre_matrix
    )
    
    print(f"Inference cost: User-Based: {user_based_avg_time:.2f} ms, Movie-Based: {movie_based_avg_time:.2f} ms")

if __name__ == '__main__':
    main()