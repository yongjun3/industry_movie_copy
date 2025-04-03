import pandas as pd
import numpy as np

def get_user_recommendations(merged_df, user_id, top_n=10, min_rating=5, debug=False):
    """
    Get movie recommendations for a specific user based on their highly rated movies,
    prioritizing exact genre matches and using demographic similarity as tiebreaker
    
    Parameters:
    merged_df (pandas.DataFrame): DataFrame containing merged user ratings and movie data
    user_id (int): ID of the user to get recommendations for
    top_n (int): Number of recommendations to return
    min_rating (int): Minimum rating threshold to consider as "highly rated" (default: 5)
    debug (bool): Whether to print debug information
    
    Returns:
    pandas.DataFrame: Recommended movies sorted by similarity score
    """
    if debug:
        print(f"Looking for recommendations for user {user_id} with minimum rating {min_rating}")
    
    # Identify genre columns
    genre_cols = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 
                  'Documentary', 'Drama', 'Family', 'Fantasy', 'Foreign', 
                  'History', 'Horror', 'Music', 'Mystery', 'Romance', 
                  'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western']
    
    # Get user demographics
    user_info = merged_df[merged_df['user_id'] == user_id]
    if user_info.empty:
        print(f"User {user_id} not found in the dataset")
        return pd.DataFrame()
        
    user_demographics = user_info.iloc[0]
    
    # Safely check if demographic columns exist
    has_age = 'age' in merged_df.columns
    has_gender = 'gender' in merged_df.columns
    
    user_age = user_demographics['age'] if has_age else None
    user_gender = user_demographics['gender'] if has_gender else None
    
    if debug:
        demo_info = []
        if user_age is not None:
            demo_info.append(f"Age: {user_age}")
        if user_gender is not None:
            demo_info.append(f"Gender: {user_gender}")
        
        if demo_info:
            print(f"User demographics - {', '.join(demo_info)}")
        else:
            print("No demographic information available for this user")
    
    # Find movies highly rated by this user
    user_movies = merged_df[(merged_df['user_id'] == user_id) & 
                           (merged_df['rating'] >= min_rating)]
    
    if user_movies.empty:
        print(f"User {user_id} has not rated any movies {min_rating} stars or higher")
        return pd.DataFrame()

    if debug:
        print(f"Found {len(user_movies)} movies rated {min_rating}+ stars by user {user_id}")

    # Get tmdb_ids of user's highly rated movies
    user_movie_ids = user_movies['tmdb_id'].tolist()
    
    # Extract the movies the user has rated highly
    target_movies = merged_df[merged_df['tmdb_id'].isin(user_movie_ids)].drop_duplicates(subset='tmdb_id')
    
    if debug:
        print(f"Found {len(target_movies)} unique movies rated highly by user {user_id}")
    
    # Display the user's highly rated movies
    if debug:
        print("\nUser's highly rated movies:")
        for _, movie in target_movies.iterrows():
            movie_genres = [col for col in genre_cols if movie[col] == 1]
            print(f"- {movie['title']} (ID: {movie['tmdb_id']}): {', '.join(movie_genres)}")
    else:
        # Still need to process genres even if not debugging
        for _, movie in target_movies.iterrows():
            movie_genres = [col for col in genre_cols if movie[col] == 1]
        
    # Create a profile combining genres from all target movies
    genre_profile = np.zeros(len(genre_cols))
    
    for _, movie in target_movies.iterrows():
        for i, genre in enumerate(genre_cols):
            if movie[genre] == 1:
                genre_profile[i] += 1
    
    # Get active genres (those with non-zero weight)
    active_genres = [genre for i, genre in enumerate(genre_cols) if genre_profile[i] > 0]
    
    if not active_genres:
        print("The user's favorite movies don't have any genres assigned")
        return pd.DataFrame()
    
    print(f"User's favorite genres: {', '.join(active_genres)}")
    
    # Create a mask to exclude movies the user has already rated
    all_user_rated_movie_ids = merged_df[merged_df['user_id'] == user_id]['tmdb_id'].tolist()
    
    # Explicitly exclude the movies user has already seen
    other_movies = merged_df[~merged_df['tmdb_id'].isin(all_user_rated_movie_ids)].copy()
    other_movies = other_movies.drop_duplicates(subset='tmdb_id')
    
    if debug:
        print(f"Found {len(other_movies)} unwatched movies for recommendation")
    
    if other_movies.empty:
        print("No unwatched movies found for recommendation")
        return pd.DataFrame()
    
    # First, identify movies that match ALL the user's favorite genres
    exact_matches = []
    partial_matches = []
   
    # Process all unwatched movies in a single loop for better performance
    for idx, movie in other_movies.iterrows():
        # Check if movie has all active genres
        has_all_active_genres = True
        genre_match_count = 0
        
        for genre in active_genres:
            if movie[genre] == 1:
                genre_match_count += 1
            else:
                has_all_active_genres = False
        
        # Skip movies with no matching genres
        if genre_match_count == 0:
            continue
            
        # Calculate demographic similarity score for tiebreaking
        demo_similarity = 0
        
        # For exact matches, use movie's own demographics
        if has_all_active_genres:
            if has_age and user_age is not None:
                # Check if the row has age value (for movies with user demographic info)
                if 'age' in movie and not pd.isna(movie['age']):
                    age_diff = abs(user_age - movie['age'])
                    if age_diff <= 5:  # Within 5 years
                        demo_similarity += 2
                    elif age_diff <= 10:  # Within 10 years
                        demo_similarity += 1
                        
            if has_gender and user_gender is not None:
                # Check if the row has gender value
                if 'gender' in movie and not pd.isna(movie['gender']):
                    if user_gender == movie['gender']:
                        demo_similarity += 3
                
            exact_matches.append({
                'tmdb_id': movie['tmdb_id'],
                'title': movie['title'],
                'movie_id': movie['movie_id'],
                'genres': '|'.join([col for col in genre_cols if movie[col] == 1]),
                'demo_similarity': demo_similarity,
                'exact_match': True,
                'genre_match_count': len(active_genres)
            })
        else:
            # For partial matches, use demographics of users who rated this movie
            movie_id = movie['tmdb_id']
            
            if has_age and user_age is not None:
                # Get the average age of users who rated this movie
                movie_ratings = merged_df[merged_df['tmdb_id'] == movie_id]
                if not movie_ratings.empty and 'age' in movie_ratings.columns:
                    avg_age = movie_ratings['age'].mean()
                    if not pd.isna(avg_age):
                        age_diff = abs(user_age - avg_age)
                        if age_diff <= 5:  # Within 5 years
                            demo_similarity += 2
                        elif age_diff <= 10:  # Within 10 years
                            demo_similarity += 1
                        
            if has_gender and user_gender is not None:
                # Get the percentage of same-gender users who rated this movie
                movie_ratings = merged_df[merged_df['tmdb_id'] == movie_id]
                if not movie_ratings.empty and 'gender' in movie_ratings.columns:
                    gender_matches = movie_ratings[movie_ratings['gender'] == user_gender]
                    if not movie_ratings.empty:
                        same_gender_pct = len(gender_matches) / len(movie_ratings)
                        demo_similarity += same_gender_pct * 3  # Scale to match exact match scoring
            
            partial_matches.append({
                'tmdb_id': movie_id,
                'title': movie['title'],
                'movie_id': movie['movie_id'],
                'genres': '|'.join([col for col in genre_cols if movie[col] == 1]),
                'demo_similarity': demo_similarity,
                'exact_match': False,
                'genre_match_count': genre_match_count
            })
    
    if debug:
        print(f"Found {len(exact_matches)} exact genre matches and {len(partial_matches)} partial matches")
    
    # If we have enough exact matches, use those with demographic tiebreaking
    if len(exact_matches) >= top_n:
        if debug:
            print(f"Using {min(len(exact_matches), top_n)} exact matches for recommendations")
            
        exact_match_df = pd.DataFrame(exact_matches)
        exact_match_df = exact_match_df.sort_values('demo_similarity', ascending=False)
        
        final_df = exact_match_df.head(top_n)[['movie_id', 'title', 'demo_similarity']]
        final_df = final_df.rename(columns={'demo_similarity': 'demo_score'})
        
        return final_df.reset_index(drop=True)
    
    # If we don't have enough exact matches, combine with partial matches
    if debug and exact_matches:
        print(f"Not enough exact matches, including partial matches as well")
    
    # Combine exact and partial matches
    all_matches = exact_matches + partial_matches
    
    # Create DataFrame and sort by:
    # 1. Whether it's an exact match
    # 2. Number of matching genres
    # 3. Demographic similarity for tiebreaking
    if all_matches:
        result_df = pd.DataFrame(all_matches)
        result_df = result_df.sort_values(['exact_match', 'genre_match_count', 'demo_similarity'], 
                                        ascending=[False, False, False])
        
        # Rename columns for final output
        result_df = result_df.rename(columns={
            'genre_match_count': 'match_score',
            'demo_similarity': 'demo_score'
        })
        
        return result_df.head(top_n)[['movie_id', 'title', 'match_score', 'demo_score', 'exact_match']].reset_index(drop=True)
    else:
        print("No matching movies found")
        return pd.DataFrame()


def recommendation_func(merged_df, user_id, min_rating=5):
    """
    Wrapper function to get movie recommendations for a specific user
    
    Parameters:
    merged_df (pandas.DataFrame): DataFrame containing merged user ratings and movie data
    user_id (int): ID of the user to get recommendations for
    top_n (int): Number of recommendations to return
    min_rating (int): Minimum rating threshold to consider as "highly rated" (default: 5)
    debug (bool): Whether to print debug information
    
    Returns:
    List: Recommended movies id sorted by similarity score in list
    """
    movies_df = get_user_recommendations(merged_df, user_id, 20, min_rating, False)
    if movies_df.empty:
        sorted_df = merged_df.sort_values(by=['vote_average', 'vote_count'], ascending=False)
        top_movies = list(sorted_df['movie_id'].unique()[:20])
        np.random.shuffle(top_movies)
        return top_movies
    else:
        # Return the top 20 recommended movies
        return list(movies_df['movie_id'].unique()[:20])
    

# Example usage:
if __name__ == "__main__":
    # Load merged data (containing both user ratings and movie data)
    merged_df = pd.read_csv("data/merged_data.csv")
    
    # Get recommendations for a specific user
    user_id = 18429  # Example user ID
    recommendations = recommendation_func(
        merged_df, 
        user_id=user_id,
        min_rating=5
    )
    # get_user_recommendations(
    #     merged_df, 
    #     user_id=user_id,
    #     top_n=20,
    #     min_rating=5,
    #     debug=False
    # )
    
    print(recommendations)



