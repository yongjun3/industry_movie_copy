import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
from config import *
from sklearn.metrics.pairwise import cosine_similarity
from recommender_model.data_loader import creating_needed_matrix, get_watched_movies
from recommender_model.train import train
from recommender_model.user_based_model import recommend_user_based_with_watched
from recommender_model.movie_based_model import recommend_movie_based_with_watched
from evaluation_metrics import *



user_movie_matrix, movie_genre_matrix = creating_needed_matrix(DATA_PATH)

# Similarity matrix itself
user_movie_similarity, movie_genre_similarity = train(DATA_PATH)


def get_watched_movies(user_id, user_movie_matrix):
    # Retrieves all movies a user has watched, including those they rated.
    if user_id not in user_movie_matrix.index:
        return set()
    return set(user_movie_matrix.loc[user_id][user_movie_matrix.loc[user_id] > 0].index)

def recommend_user_based_with_watched(user_id, user_movie_similarity, user_movie_matrix, top_n=20):
    # Recommends movies for a user without filtering out already watched/rated movies.
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

def recommend_movie_based_with_watched(user_id, movie_genre_similarity, user_movie_matrix, top_n=20):
    # Recommends movies for a user without filtering out already watched/rated movies (Movie-Based).
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
            recommended_movies.append(similar_movie)
            if len(recommended_movies) >= top_n:
                break
        if len(recommended_movies) >= top_n:
            break
    
    return recommended_movies[:top_n]

def evaluate_single_user(recommendation_func, user_movie_matrix, watched_data, user_id, k):
    # Evaluates recommendation performance for a single user using Precision@K, Recall@K, and F1@K.
    if user_id not in user_movie_matrix.index:
        return {'Precision@K': 0, 'Recall@K': 0, 'F1@K': 0}
    
    watched_movies = get_watched_movies(user_id, user_movie_matrix)
    recommended_movies = set(recommendation_func(user_id)[:k])
    
    if not watched_movies:
        return {'Precision@K': 0, 'Recall@K': 0, 'F1@K': 0}
    
    true_positives = len(recommended_movies & watched_movies)
    precision = true_positives / k
    recall = true_positives / len(watched_movies)
    
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    
    return {'Precision@K': precision, 'Recall@K': recall, 'F1@K': f1}

def evaluate_all_users(recommendation_func, user_movie_matrix, watched_data, k=10):
    # Evaluates the recommendation model for all users using Precision@K, Recall@K, and F1@K.
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    unique_users = watched_data['user_id'].unique()
    
    for user_id in unique_users:
        user_eval = evaluate_single_user(recommendation_func, user_movie_matrix, watched_data, user_id, k)
        precision_scores.append(user_eval['Precision@K'])
        recall_scores.append(user_eval['Recall@K'])
        f1_scores.append(user_eval['F1@K'])
    
    return {
        'Precision@K': np.mean(precision_scores),
        'Recall@K': np.mean(recall_scores),
        'F1@K': np.mean(f1_scores)
    }

watched_data = pd.read_csv("data/unique_watch_df.csv")

# Evaluate on all users using User-Based Recommendation
all_users_evaluation_user_based = evaluate_all_users(lambda user_id: recommend_user_based_with_watched(user_id, user_movie_similarity, user_movie_matrix),
                                                     user_movie_matrix, watched_data, k=20)
print("Holistic Evaluation for User-Based Recommendation:")
print(all_users_evaluation_user_based)

# Evaluate on all users using Movie-Based Recommendation
all_users_evaluation_movie_based = evaluate_all_users(lambda user_id: recommend_movie_based_with_watched(user_id, movie_genre_similarity, user_movie_matrix),
                                                      user_movie_matrix, watched_data, k=20)
print("Holistic Evaluation for Movie-Based Recommendation:")
print(all_users_evaluation_movie_based)