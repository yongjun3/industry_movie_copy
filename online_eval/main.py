import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity


'''
Explanation of logic regarding code below 

1. User-Movie Matrix: (11299 rows x 5676 columns) (type=pandas dataframe)
   A matrix representing user ratings for movies, with users as rows and movies as columns.

2. Movie Genre Matrix: (5676 rows x 20 columns) (type=pandas dataframe)
   A matrix representing movie genres, with movies as rows and genres as columns.

3. User-Movie Similarity Matrix: (11299 x 11299) (type=numpy array)
   A matrix of cosine similarities between users, showing how similar each user is to every other user.

4. Movie-Genre Similarity Matrix: (5676 x 5676) (type=numpy array)
   A matrix of cosine similarities between movies, showing how similar each movie is to every other movie based on genre.
'''

'''
Plan Breakdown:

1. User-Movie Matrix Creation:
	•	The creating_needed_matrix function loads data from a CSV file containing movie ratings.
	•	It constructs:
	•	Movie Genre Matrix: This matrix represents the genres of each movie. Each movie is associated with specific genres (e.g., Action, Comedy, etc.). If there are multiple rows for the same movie, the last genre instance is used.
	•	User-Movie Matrix: This matrix represents ratings given by users to movies. Missing ratings are filled with zero.

2. Training Phase (Cosine Similarity Calculation):
	•	In the train function, we use the User-Movie Matrix to calculate user-user similarity using cosine similarity. The result is a User-Movie Similarity Matrix.
	•	Similarly, the Movie Genre Matrix is used to compute Movie-Genre Similarity using cosine similarity.
	•	These matrices allow you to compare how similar users and movies are to each other.

3. User-Movie and Movie-Genre Similarity Matrices:
	•	User-Movie Similarity Matrix: It calculates the similarity between users based on their movie ratings.
	•	Movie-Genre Similarity Matrix: It calculates how similar movies are based on their genre distribution.

Recommendation Strategies:

1. User-Based Recommendation:
	1.	Identify User Index: Given a user ID, find the corresponding index in the User-Movie Matrix.
	2.	Access Similarity: Using the user index, access the User-Movie Similarity Matrix to find similar users.
	3.	Find Movies Watched by Similar Users: For each similar user, find movies they have rated highly but the current user hasn’t watched.(denoted by 0 in user-movie matrix)
	4.	Prioritize Movies: Sort these movies by the rating of similar users (e.g., prioritize movies rated 4 or 5).
	5.	Return Top N Movies: Continue this process until you have the desired number of recommendations.

2. Movie-Based Recommendation:
	1.	Identify Rated Movies: Find movies the user has rated highly (e.g., 5).
    • Have to use User-movie matrix here
	2.	Movie-Genre Matrix Matching: Identify the genres of these movies using the Movie Genre Matrix.
	3.	Find Similar Movies: From the Movie-Movie Similarity Matrix (based on genre), recommend similar movies that the user hasn’t rated yet.
	4.	Prioritize Movies: Rank them by their similarity score.
	5.	Return Top N Movies: Continue until you have the top N recommended movies.

'''

df = pd.read_csv("data/merged_data.csv")

def creating_needed_matrix(csv_data):
  
    merged_df = pd.read_csv(csv_data)
    
    # Movie themes
    theme_cols = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
                  "Drama", "Family", "Fantasy", "Foreign", "History", "Horror", "Music",
                  "Mystery", "Romance", "Science Fiction", "TV Movie", "Thriller", "War", "Western"]
    
    # Create the movie-theme matrix 
    # Assuming if there are multiple rows for the same movie, genre distribution will be the same
    movie_genre_matrix = merged_df[theme_cols].groupby(merged_df['movie_id']).last()
    
    # For duplicate rating of movie_user instances, collect the last rating instance
    user_movie_temp = merged_df.groupby(['user_id', 'movie_id'], as_index=False)['rating'].last()

    # Create user-movie matrix, filling missing ratings with 0
    user_movie_matrix = user_movie_temp.pivot(index="user_id", columns="movie_id", values="rating").fillna(0)
    
    # Returning both matrices
    return user_movie_matrix, movie_genre_matrix


def train(csv_data):
    user_movie_matrix, movie_genre_matrix = creating_needed_matrix(csv_data)
    user_movie_similarity = cosine_similarity(user_movie_matrix)
    
    # Compute cosine similarity for the Movie-Genre Matrix
    movie_genre_similarity = cosine_similarity(movie_genre_matrix)
    # Example of how to access and print cosine similarity matrices
   
    return user_movie_similarity, movie_genre_similarity


def recommend_user_based_header(user_id, user_movie_similarity, user_movie_matrix,  top_n=20):
 
    # Create deep copies of the matrices to avoid modifying the originals
    user_movie_matrix_copy = user_movie_matrix.copy()
    user_movie_similarity_copy = user_movie_similarity.copy()
    
    # Call the function to perform the actual recommendation process
    return recommend_user_based(user_id, user_movie_similarity_copy, user_movie_matrix_copy, top_n)


def recommend_user_based(user_id, user_movie_similarity, user_movie_matrix, top_n=20):

    if user_id not in user_movie_matrix.index:
        sorted_df = df.sort_values(by=['vote_average', 'vote_count'], ascending=False)
        top50 = list(sorted_df['movie_id'].unique()[:50])
        np.random.shuffle(top50)
        return top50[:top_n]
    
    # Get index of the target user
    user_index = user_movie_matrix.index.get_loc(user_id)
    
    # Get similarity scores for the user
    similarities = user_movie_similarity[user_index, :]

    # Get user indices sorted by similarity (descending order)
    most_similar_user_indices = np.argsort(similarities)[::-1]

    # Set to track already recommended movies
    recommended_dict=dict()

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
                recommended_dict[movie]+=decay_factor
            else:
                recommended_dict[movie]=decay_factor

        
        # Stop if we have collected enough recommendations
        processed_users += 1
    
    sorted_recommend = dict(sorted(recommended_dict.items(), key=lambda item: item[1], reverse=True))
    for key in sorted_recommend:
        recommended_movies.append(key)
    
    
    return recommended_movies
        


def recommend_movie_based_header(user_id, movie_genre_matrix, movie_genre_similarity, 
                                 user_movie_matrix,user_movie_similarity,top_n=20):
    # Create deep copies of the matrices to avoid modifying the originals
    movie_genre_matrix_copy = movie_genre_matrix.copy()
    movie_genre_similarity_copy = movie_genre_similarity.copy()
    user_movie_matrix_copy=user_movie_matrix.copy()
    user_movie_similarity_copy=user_movie_similarity.copy()
    
    # Call the function to perform the actual recommendation process
    return recommend_movie_based(user_id, movie_genre_matrix_copy, movie_genre_similarity_copy,
                                 user_movie_matrix_copy,top_n)


def recommend_movie_based(user_id, movie_genre_matrix, movie_genre_similarity, 
                          user_movie_matrix, top_n=20):
  
    # Step 1: Get movies that the user rated 5
    if user_id not in user_movie_matrix.index:
        sorted_df = df.sort_values(by=['vote_average', 'vote_count'], ascending=False)
        top50 = list(sorted_df['movie_id'].unique()[:50])
        np.random.shuffle(top50)
        return top50[:top_n]

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


def main():
    # Might need this for indexing later
    user_movie_matrix, movie_genre_matrix = creating_needed_matrix("data/merged_data.csv")

    # Similarity matrix itself
    user_movie_similarity, movie_genre_similarity = train("data/merged_data.csv")

    print("\nUser-Movie Matrix:")
    print(user_movie_matrix) 
    print("\nMovie-Genre Matrix:")
    print(movie_genre_matrix)
    print("\nUser-Movie Similarity Matrix:")
    print(user_movie_similarity)
    print(user_movie_similarity.shape)
    print("\nMovie-Genre Similarity Matrix:")
    print(movie_genre_similarity.shape)
    print()

    existing_user_id = 6249
    
    user_based_recs = recommend_user_based_header(existing_user_id, user_movie_similarity, user_movie_matrix, top_n=20)
    print(f"User-based Recommendations for user {existing_user_id}:")
    print(user_based_recs)

    movie_based_recs = recommend_movie_based_header(existing_user_id, movie_genre_matrix, movie_genre_similarity,
                                                     user_movie_matrix, user_movie_similarity, top_n=20)
    print(f"Movie-based Recommendations for user {existing_user_id}:")
    print(movie_based_recs)
    print()

    not_existing_user_id = 6248
    user_based_recs = recommend_user_based_header(not_existing_user_id, user_movie_similarity, user_movie_matrix, top_n=20)
    print(f"User-based Recommendations for user {not_existing_user_id}:")
    print(user_based_recs)

    movie_based_recs = recommend_movie_based_header(not_existing_user_id, movie_genre_matrix, movie_genre_similarity,
                                                     user_movie_matrix, user_movie_similarity, top_n=20)
    print(f"Movie-based Recommendations for user {not_existing_user_id}:")
    print(movie_based_recs)

if __name__ == '__main__':
    main()