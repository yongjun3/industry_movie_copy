from recommender_model.train import *
from recommender_model.data_loader import *
from recommender_model.user_based_model import *
from recommender_model.movie_based_model import *
from config import *
from data_preprocessing.data_cleaning_test import *
import numpy as np
import pandas as pd
import pytest
import pickle
import os
from unittest.mock import patch, MagicMock
import sys
from offline_eval import *





@pytest.fixture
def normal_sample(tmp_path):
    # Load the full dataset
    full_data = load_data(DATA_PATH)
    
    # Get number of rows
    row_num = np.shape(full_data)[0]
    
    # Select random rows
    random_rows = np.random.choice(range(1, row_num), size=50, replace=False)
    sample = full_data.iloc[random_rows]
    
    # Create a filepath in the temporary directory
    filepath = tmp_path / "normal_sample.csv"
    
    # Save the sample data to CSV
    sample.to_csv(filepath, index=False)
    
    # Return the filepath
    return str(filepath)


class TestCreateNeededMatrix:
    
    def test_load_data(self, normal_sample):
        data = load_data(normal_sample)
        assert data is not None
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert all(col in data.columns for col in ['user_id', 'movie_id', 'rating'])
    
    def test_creating_needed_matrix_basic(self, normal_sample):
        data = load_data(normal_sample)
        user_movie_matrix, movie_genre_matrix = creating_needed_matrix(normal_sample)
        
        # Basic assertions - matrices exist and have correct structure
        assert user_movie_matrix is not None 
        assert movie_genre_matrix is not None
        assert user_movie_matrix.index.name == 'user_id'
        assert user_movie_matrix.columns.name == 'movie_id'
        assert movie_genre_matrix.index.name == 'movie_id'
    
    def test_matrix_indices_and_columns(self, normal_sample):
        data = load_data(normal_sample)
        user_movie_matrix, movie_genre_matrix = creating_needed_matrix(normal_sample)
        
        # Testing user matrix indices and columns
        sample_user_ids = set(data['user_id'].unique())
        matrix_user_ids = set(user_movie_matrix.index)
        assert sample_user_ids.issubset(matrix_user_ids), "Some users missing in the user_movie matrix"
        
        sample_movie_ids = set(data['movie_id'].unique())
        matrix_movie_ids = set(user_movie_matrix.columns)
        assert sample_movie_ids.issubset(matrix_movie_ids), "Some movie IDs from sample data are missing in the user_movie matrix"
        
        # Testing movie genre matrix indices and columns
        assert sample_movie_ids.issubset(set(movie_genre_matrix.index)), "Some movie IDs from sample data are missing in the movie_genre matrix"
        assert set(GENRE_COLUMNS) == set(movie_genre_matrix.columns), "Some themes are missing in the movie_genre matrix"
    
    def test_matrix_values(self, normal_sample):
        """Test that matrix values are within expected ranges."""
        user_movie_matrix, movie_genre_matrix = creating_needed_matrix(normal_sample)
        
        # Test ratings range
        assert user_movie_matrix.min().min() >= 0, "Negative ratings found"
        assert user_movie_matrix.max().max() <= 5, "Ratings above 5 found"
        
        # Check genre values are binary (0 or 1)
        assert movie_genre_matrix.min().min() >= 0, "Negative genre values found"
        assert movie_genre_matrix.max().max() <= 1, "Genre values above 1 found"
        assert set(movie_genre_matrix.values.flatten()).issubset({0, 1}), "Non-binary genre values found"
        
        # Check for missing values
        assert not user_movie_matrix.isna().any().any(), "NaN values found in user_movie_matrix"
        assert not movie_genre_matrix.isna().any().any(), "NaN values found in movie_genre_matrix"
    
    def test_matrix_shapes(self, normal_sample):
        """Test that matrices have the correct shapes."""
        data = load_data(normal_sample)
        user_movie_matrix, movie_genre_matrix = creating_needed_matrix(normal_sample)
        
        unique_users = data['user_id'].nunique()
        unique_movies = data['movie_id'].nunique()
        
        assert user_movie_matrix.shape[0] == unique_users, "Wrong number of users in matrix"
        assert user_movie_matrix.shape[1] == unique_movies, "Wrong number of movies in user_movie_matrix"
        assert movie_genre_matrix.shape[0] == unique_movies, "Wrong number of movies in movie_genre_matrix"
        assert movie_genre_matrix.shape[1] == len(GENRE_COLUMNS), "Wrong number of genres"
    
    def test_empty_input(self, tmp_path):
        # Create empty dataset with correct columns
        empty_df = pd.DataFrame(columns=['user_id', 'movie_id', 'rating'] + GENRE_COLUMNS)
        
        # Save to CSV
        filepath = tmp_path / "empty_test.csv"
        empty_df.to_csv(filepath, index=False)
        
        # Process with function - should handle gracefully
        user_movie_matrix, movie_genre_matrix = creating_needed_matrix(str(filepath))
        
        # Check matrices are empty but properly structured
        assert user_movie_matrix.shape[0] == 0, "User matrix should be empty"
        assert movie_genre_matrix.shape[0] == 0, "Genre matrix should be empty"
        assert movie_genre_matrix.shape[1] == len(GENRE_COLUMNS), "Genre matrix should have all genre columns"


class TestGetWatchedMovies:
    """Tests for the get_watched_movies function."""

    @pytest.fixture
    def sample_user_movie_matrix(self):
        """Create a sample user-movie matrix for testing."""
        # Create sample data with various rating scenarios
        data = {
            'movie_1': [5.0, 0.0, 0.0, 1.0],
            'movie_2': [0.0, 4.0, 0.0, 0.0],
            'movie_3': [2.0, 0.0, 0.0, 3.0],
            'movie_4': [0.0, 0.0, 0.0, 0.0],
            'movie_5': [4.0, 1.0, 0.0, 1.0],  
            'movie_6': [0.0, 3.0, 0.0, 5.0],  
        }
        return pd.DataFrame(data, index=['user_1', 'user_2', 'user_3', 'user_4'])
    
    def test_user_with_multiple_movies(self, sample_user_movie_matrix):
        result = get_watched_movies('user_1', sample_user_movie_matrix)
        expected = {'movie_1', 'movie_3', 'movie_5'}
        assert result == expected
        
    def test_user_with_one_movie(self, sample_user_movie_matrix):
        # Create a user with just one positive rating
        sample_user_movie_matrix.loc['user_5'] = [0.0, 0.0, 4.0, 0.0, 0.0, 0.0]
        result = get_watched_movies('user_5', sample_user_movie_matrix)
        expected = {'movie_3'}
        assert result == expected
    
    def test_exist_user_without_rating(self, sample_user_movie_matrix):
        sample_user_movie_matrix.loc['user_2'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        result = get_watched_movies('user_2', sample_user_movie_matrix)
        expected = set()
        assert result == expected

    def test_nonexistent_user(self, sample_user_movie_matrix):
        result = get_watched_movies('9239jf', sample_user_movie_matrix)
        assert result == set()


class TestGetTopRatedMovies:
    """Tests for the get_top_rated_movies function."""

    @pytest.fixture
    def sample_movie_df(self):
        data = {
            'movie_id': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'vote_average': [8.5, 7.6, 9.0, 6.8, 8.9, 9.0, 7.2, 8.0, 6.5, 7.8],
            'vote_count': [1000, 800, 500, 300, 700, 600, 400, 900, 200, 350],
            'title': [f'Movie {i}' for i in range(1, 11)]
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def duplicate_movie_df(self):
        data = {
            'movie_id': [101, 101, 102, 102, 103, 104, 105],
            'vote_average': [8.5, 7.5, 7.6, 8.6, 9.0, 6.8, 8.9],
            'vote_count': [1000, 900, 800, 700, 500, 300, 700],
            'title': ['Movie 1', 'Movie 1b', 'Movie 2', 'Movie 2b', 'Movie 3', 'Movie 4', 'Movie 5']
        }
        return pd.DataFrame(data)
    
    def test_default_n_value(self, sample_movie_df):
        result = get_top_rated_movies(sample_movie_df)
        assert len(result) == len(sample_movie_df) if len(sample_movie_df) < 20 else 20
        
    def test_custom_n_value(self, sample_movie_df):
        n = 5
        result = get_top_rated_movies(sample_movie_df, n)
        assert len(result) == n

    def test_with_real_data(self, normal_sample):
        data = load_data(normal_sample)
        # Sort the data the same way the function does
        sample_sorted = data.sort_values(by=['vote_average', 'vote_count'], ascending=False)
        expected = set(sample_sorted['movie_id'].unique()[:20])
        
        top_movies = get_top_rated_movies(data)
        assert expected.issubset(set(top_movies))
        assert set(top_movies).issubset(expected)


class TestComputeSimilarityMatrices:
    """Tests for compute_similarity_matrices function."""
    
    @pytest.fixture
    def input_data(self, normal_sample):
        user_movie_sam, movie_genre_sam = creating_needed_matrix(normal_sample)
        return (user_movie_sam, movie_genre_sam)
    
    def test_dimensions(self, input_data):
        user_movie_sam = input_data[0]
        movie_genre_sam = input_data[1]
        user_num = user_movie_sam.shape[0]
        movie_num = movie_genre_sam.shape[0]
        user_sim, movie_sim = compute_similarity_matrices(user_movie_sam, movie_genre_sam)
        assert user_sim is not None
        assert movie_sim is not None
        assert user_sim.shape[0] == user_sim.shape[1]
        assert movie_sim.shape[0] == movie_sim.shape[1]
        assert user_sim.shape[0] == user_num
        assert movie_sim.shape[0] == movie_num


class TestTrain:
    """Tests for the train function."""
    
    @pytest.fixture
    def input_data(self, normal_sample):
        user_movie_sam, movie_genre_sam = creating_needed_matrix(normal_sample)
        return (user_movie_sam, movie_genre_sam)
    
    def test_dimensions(self, input_data, normal_sample):
        user_movie_sam = input_data[0]
        movie_genre_sam = input_data[1]
        user_num = user_movie_sam.shape[0]
        movie_num = movie_genre_sam.shape[0]
        user_sim, movie_sim = train(normal_sample)
        assert user_sim is not None
        assert movie_sim is not None
        assert user_sim.shape[0] == user_sim.shape[1]
        assert movie_sim.shape[0] == movie_sim.shape[1]
        assert user_sim.shape[0] == user_num
        assert movie_sim.shape[0] == movie_num


class TestRecommendUserBased_and_Watchted:
    @pytest.fixture
    def input_data(self,normal_sample):
        user_movie_sam, movie_genre_sam = creating_needed_matrix(normal_sample)
        user_sim, movie_sim = train(normal_sample)
        return (user_movie_sam,user_sim)
    
    def test_existing_user(self, input_data):
        user_movie_sam = input_data[0]
        user_movie_sim = input_data[1]
        assert len(user_movie_sam.index) > 5, "Not enough users in test data"
        exist_user_1 = user_movie_sam.index[0]
        exist_user_2 = user_movie_sam.index[5]
        movie_recommendation_1 = recommend_user_based(exist_user_1,user_movie_sim,user_movie_sam)
        movie_eval_1 = recommend_user_based_with_watched(exist_user_1,user_movie_sim,user_movie_sam)
        assert isinstance(movie_recommendation_1, list), "Result should be a list"
        assert isinstance(movie_eval_1, list), "Result should be a list"
        assert len(movie_recommendation_1) <= 20, "Should return at most 20 recommendations"
        assert len(movie_eval_1) <= 20
        
        movie_recommendation_2 = recommend_user_based(exist_user_2,user_movie_sim,user_movie_sam, top_n=0)
        movie_eval_2 = recommend_user_based_with_watched(exist_user_2,user_movie_sim,user_movie_sam, top_n=0)
        assert isinstance(movie_recommendation_2, list), "Result should be a list"
        assert isinstance(movie_eval_2, list), "Result should be a list"
        assert len(movie_recommendation_2) == 0, "Should return empty list with top_n=0"
        assert len(movie_eval_2) == 0, "Should return empty list with top_n=0"

        
        
        try:
            movie_recommendation_3 = recommend_user_based(exist_user_2,user_movie_sim,user_movie_sam, top_n=60)
            movie_eval_3 = recommend_user_based_with_watched(exist_user_2,user_movie_sim,user_movie_sam, top_n=60)
            assert isinstance(movie_recommendation_3, list), "Result should be a list"
            assert len(movie_recommendation_3)>0
            assert isinstance(movie_eval_3, list), "Result should be a list"
            assert len(movie_eval_3)>0
        except Exception as e:
            pytest.fail(f"recommend_user_based raised an exception with large top_n: {e}")
    

    def test_nonexisting_user(self, input_data):
        user_movie_sam = input_data[0]
        user_movie_sim = input_data[1]
        non_exist_user1 = 0.0928
        non_exist_user2 = "Hello!"
        
        movie_recommendation_4 = recommend_user_based(non_exist_user1, user_movie_sim, user_movie_sam)
        assert isinstance(movie_recommendation_4, list)
        
        movie_eval_4 = recommend_user_based_with_watched(non_exist_user1, user_movie_sim, user_movie_sam)
        assert isinstance(movie_eval_4, list)
        assert len(movie_eval_4) == 0
        
        movie_recommendation_5 = recommend_user_based(non_exist_user2, user_movie_sim, user_movie_sam, top_n=20)
        assert isinstance(movie_recommendation_5, list)
        
        movie_eval_5 = recommend_user_based_with_watched(non_exist_user2, user_movie_sim, user_movie_sam, top_n=20)
        assert isinstance(movie_eval_5, list)
        assert len(movie_eval_5) == 0
    
class TestMovieRecommendContentBased:
    @pytest.fixture
    def input_data(self, normal_sample):
        user_movie_sam, movie_genre_sam = creating_needed_matrix(normal_sample)
        _, movie_sim = train(normal_sample)
        return (movie_genre_sam, movie_sim, user_movie_sam)
        
    def test_existing_user(self, input_data):
        movie_genre_sam = input_data[0]
        movie_sim = input_data[1]
        user_movie_sam = input_data[2]
        
        assert len(user_movie_sam.index) > 5, "Not enough users in test data"
        
        exist_user_1 = user_movie_sam.index[0]
        exist_user_2 = user_movie_sam.index[5]
        
        movie_recommendation_1 = recommend_movie_based(exist_user_1, movie_genre_sam, movie_sim, user_movie_sam)
        movie_eval_1 = recommend_movie_based_with_watched(exist_user_1, movie_sim, user_movie_sam)
        
        assert isinstance(movie_recommendation_1, list), "Result should be a list"
        assert isinstance(movie_eval_1, list), "Result should be a list"
        assert len(movie_recommendation_1) <= 20, "Should return at most 20 recommendations"
        assert len(movie_eval_1) <= 20
        
        movie_recommendation_2 = recommend_movie_based(exist_user_2, movie_genre_sam, movie_sim, user_movie_sam, top_n=0)
        movie_eval_2 = recommend_movie_based_with_watched(exist_user_2, movie_sim, user_movie_sam, top_n=0)
        
        assert isinstance(movie_recommendation_2, list), "Result should be a list"
        assert isinstance(movie_eval_2, list), "Result should be a list"
        assert len(movie_recommendation_2) == 0, "Should return empty list with top_n=0"
        assert len(movie_eval_2) == 0, "Should return empty list with top_n=0"
        
        try:
            movie_recommendation_3 = recommend_movie_based(exist_user_2, movie_genre_sam, movie_sim, user_movie_sam, top_n=60)
            movie_eval_3 = recommend_movie_based_with_watched(exist_user_2, movie_sim, user_movie_sam, top_n=60)
            
            assert isinstance(movie_recommendation_3, list), "Result should be a list"
            assert len(movie_recommendation_3) >= 0
            assert isinstance(movie_eval_3, list), "Result should be a list"
            assert len(movie_eval_3) >=0
        except Exception as e:
            pytest.fail(f"recommend_movie_based raised an exception with large top_n: {e}")
    
    def test_nonexisting_user(self, input_data):
        movie_genre_sam = input_data[0]
        movie_sim = input_data[1]
        user_movie_sam = input_data[2]
        
        non_exist_user1 = 0.0928
        non_exist_user2 = "Hello!"
        
        try:
            movie_recommendation_4 = recommend_movie_based(non_exist_user1, movie_genre_sam, movie_sim, user_movie_sam)
            movie_eval_4 = recommend_movie_based_with_watched(non_exist_user1, movie_sim, user_movie_sam)
            
            assert isinstance(movie_recommendation_4, list)
            assert isinstance(movie_eval_4, list)
            assert len(movie_eval_4) == 0
        except Exception as e:
            pytest.fail(f"Doesn't work for non-existent user: {e}")
        
        try:
            movie_recommendation_5 = recommend_movie_based(non_exist_user2, movie_genre_sam, movie_sim, user_movie_sam, top_n=20)
            movie_eval_5 = recommend_movie_based_with_watched(non_exist_user2, movie_sim, user_movie_sam, top_n=20)
            
            assert isinstance(movie_recommendation_5, list)
            # This should check against actual fallback behavior for movie-based recommendations
            assert len(movie_recommendation_5) > 0
            assert isinstance(movie_eval_5, list)
            assert len(movie_eval_5) == 0
        except Exception as e:
            pytest.fail(f"Doesn't work for non-existent user when using top n parameter: {e}")



class TestCoreDataClean:
    @pytest.fixture
    def mock_log_file(self, tmpdir):
        """Create a temporary file with mock log data for testing."""
        log_content = """2025-01-03T08:12:34,45621,GET /data/m/titanic/1.mpg
2020-01-03T08:13:42,45621,GET /data/m/titanic/2.mpg
2021-01-03T08:15:22,45621,GET /rate/titanic=4
2022-01-03T08:16:33,78901,GET /data/m/avatar/1.mpg
2023-01-03T08:17:45,78901,GET /data/m/avatar/2.mpg
2024-01-03T08:21:08,78901,GET /rate/avatar=5
2025-01-03T08:32:27,23456,GET /data/m/godfather/1.mpg
2026-01-03T08:35:49,23456,GET /rate/godfather=4"""
        
        log_file = tmpdir.join("test_log.txt")
        log_file.write(log_content)
        return os.path.join(tmpdir, "test_log.txt")
    
    
    
    def test_parse_log_data(self, mock_log_file):
        """Test the log parsing function with smaller dataset."""
        import re  
        
        watch_df, rate_df = parse_log_data(mock_log_file)
        

        assert len(watch_df) == 5
        assert set(watch_df['movie_id']) == {'titanic', 'avatar', 'godfather'}
        

        assert len(rate_df) == 3
        assert set(rate_df['rating']) == {4, 5}
        

        assert watch_df[watch_df['movie_id'] == 'titanic']['minute_watched'].tolist() == [1, 2]
        assert rate_df[rate_df['movie_id'] == 'avatar']['rating'].iloc[0] == 5
    
    def test_filter_recent_data(self, mock_log_file):
        import re
        watch_df, rate_df = parse_log_data(mock_log_file)
        watch_df_filtered, rate_df_filtered = filter_recent_data(watch_df, rate_df, "2023-01-01")
        
        assert len(watch_df_filtered) == 3  
        assert len(rate_df_filtered) == 2  
        
        years_in_watch = [timestamp[:4] for timestamp in watch_df_filtered['timestamp']]
        assert set(years_in_watch) == {'2023', '2025'}  
        
        years_in_rate = [timestamp[:4] for timestamp in rate_df_filtered['timestamp']]
        assert set(years_in_rate) == {'2024', '2026'}  
    
    def test_extract_common_entities(self,mock_log_file):
        watch_df, rate_df = parse_log_data(mock_log_file)
        watch_df_filtered, rate_df_filtered = filter_recent_data(watch_df, rate_df, "2023-01-01")
        unique_rate_df,unique_watch_df=extract_common_entities(watch_df_filtered,rate_df_filtered)
        assert set(unique_rate_df["user_id"])==set(unique_watch_df["user_id"])
        assert set(unique_rate_df["movie_id"])==set(unique_watch_df["movie_id"])
    
    
    
    def test_clean_movie_data(self):
        sample_movie_data=pd.DataFrame({
            "title": ["The Matrix", "Inception"],
            "id": [1, 2],
            "imdb_id": ["tt0133093", "tt1375666"],
            "tmdb_id": [603, 27205],
            "adult": [0, 0],
            "genres": ["Action|Sci-Fi", "Action|Thriller|Sci-Fi"],
            "release_date": ["1999-03-31", "2010-07-16"],
              "revenue": [463517383, 825532764],
              "runtime": [136, 148],
              "vote_average": [8.2, 8.4],
              "vote_count": [18680, 29145]
              })
        result = clean_movie_data(sample_movie_data)
        assert result is not None
        assert "Action" in result.columns
        assert "Sci-Fi" in result.columns
        assert len(result) == 2




class TestMergeAllData:
    """Tests for the merge_all_data function focused on data transformation for model training."""
    
    @pytest.fixture
    def sample_dataframes(self):
        """Create sample dataframes for testing."""
        # Sample rating data
        rate_data = {
            'user_id': [1, 2, 3, 1, 2],
            'movie_id': [101, 102, 103, 104, 101],
            'rating': [4.5, 3.0, 5.0, 3.5, 4.0],
            'timestamp': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']
        }
        rate_df = pd.DataFrame(rate_data)
        
        # Sample user data
        user_data = {
            'user_id': [1, 2, 3, 4],
            'age': [25, 30, 45, 22],
            'gender': ['M', 'F', 'M', 'F']
        }
        user_df = pd.DataFrame(user_data)
        
        # Sample movie data - including genres for model training
        movie_data = {
            'id': [101, 102, 103, 104, 105],
            'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
            'genres': ['Action|Thriller', 'Comedy', 'Drama|Romance', 'Sci-Fi', 'Horror']
        }
        movie_df = pd.DataFrame(movie_data)
        
        return rate_df, user_df, movie_df
    
    def test_merge_data_for_model_training(self, sample_dataframes, tmpdir):
        """Test that merged data is properly formatted for model training."""
        rate_df, user_df, movie_df = sample_dataframes
        
        # Store original directory for cleanup
        original_dir = os.getcwd()
        
        # Create a temporary directory for saving output files
        os.chdir(tmpdir)
        
        try:
            # Create required directories manually instead of using the function
            if not os.path.exists('data'):
                os.makedirs('data')
            
            # Create deep copies to avoid modifying fixture data
            rate_df_copy = rate_df.copy()
            user_df_copy = user_df.copy()
            movie_df_copy = movie_df.copy()
            
            # Run the function
            result = merge_all_data(rate_df_copy, user_df_copy, movie_df_copy)
            
            # Verify the data structure is correct for model training
            assert result is not None
            assert isinstance(result, pd.DataFrame)
            
            # Check essential columns for model training are present
            essential_columns = ['user_id', 'movie_id', 'rating', 'genres']
            for col in essential_columns:
                assert col in result.columns
            
            # Verify user_id and movie_id are correctly formatted as integers
            assert pd.api.types.is_integer_dtype(result['user_id'])
            assert pd.api.types.is_integer_dtype(result['movie_id'])
            
            # Verify the merge maintained all valid ratings
            assert len(result) == 5  # Should match number of ratings
            assert set(result['user_id'].unique()) == {1, 2, 3}
            assert set(result['movie_id'].unique()) == {101, 102, 103, 104}
            
            # Verify movie metadata was correctly joined
            for idx, row in result.iterrows():
                movie_id = row['movie_id']
                movie_row = movie_df_copy[movie_df_copy['id'] == movie_id].iloc[0]
                assert row['title'] == movie_row['title']
                assert row['genres'] == movie_row['genres']
                
            # Verify file was created
            assert os.path.exists('data/merged_data.csv')
            
        finally:
            # Clean up by going back to original directory
            os.chdir(original_dir)
    
    def test_missing_input_handling(self, tmpdir):
        """Test that function handles missing inputs appropriately."""
        # Store original directory for cleanup
        original_dir = os.getcwd()
        
        # Create a temporary directory for saving output files
        os.chdir(tmpdir)
        
        try:
            # Create required directories manually
            if not os.path.exists('data'):
                os.makedirs('data')
                
            # Test with None inputs
            result = merge_all_data(None, None, None)
            assert result is None
        finally:
            # Clean up
            os.chdir(original_dir)
    
    def test_type_conversion_for_model_compatibility(self, sample_dataframes, tmpdir):
        """Test that type conversion ensures model compatibility."""
        rate_df, user_df, movie_df = sample_dataframes
        
        # Store original directory for cleanup
        original_dir = os.getcwd()
        
        # Create a temporary directory for saving output files
        os.chdir(tmpdir)
        
        try:
            # Create required directories manually
            if not os.path.exists('data'):
                os.makedirs('data')
                
            # Make a deep copy of the data
            rate_df_copy = rate_df.copy()
            user_df_copy = user_df.copy()
            movie_df_copy = movie_df.copy()
            
            # Change types to test conversion
            rate_df_copy['user_id'] = rate_df_copy['user_id'].astype(str)
            user_df_copy['user_id'] = user_df_copy['user_id'].astype(float)
            
            # Run function
            result = merge_all_data(rate_df_copy, user_df_copy, movie_df_copy)
            
            # Verify user_id is converted to int for model compatibility
            assert pd.api.types.is_integer_dtype(result['user_id'])
        finally:
            # Clean up
            os.chdir(original_dir)

# If none of the recommended items were watched

class TestCalculateMRR:
    
    @pytest.fixture
    def mrr_test_data(self):
        """Create minimal test data for testing the calculate_mrr function."""
        return [
            # Basic test case - match at different positions
            {
                "name": "first_position_match",
                "recommended": [101, 102, 103, 104, 105],
                "watched": {101, 201},
                "expected": 1.0  # 1/1 = 1.0
            },
            {
                "name": "third_position_match",
                "recommended": [201, 301, 101, 104, 105],
                "watched": {101},
                "expected": 0.3333333333333333  
            },
            
            # Edge cases
            {
                "name": "empty_watched",
                "recommended": [101, 102, 103],
                "watched": set(),
                "expected": 0  # No relevant items
            },
            {
                "name": "no_matches",
                "recommended": [101, 102, 103],
                "watched": {201, 301},
                "expected": 0  # No matches
            }
        ]
    
    def test_calculate_mrr(self, mrr_test_data):
        """Test the calculate_mrr function with minimal test cases."""
        for test_case in mrr_test_data:
            result = calculate_mrr(test_case["recommended"], test_case["watched"])
            assert result == pytest.approx(test_case["expected"])


class test_compute_accuracy:
    def test_basic(self):
        recommend=["Movie1","Movie2","Movie3"]
        watched=[]
        test=compute_accuracy(recommend,watched)
        assert len(test)==4
        for key in compute_accuracy:
            assert test[key]==0
    
    def test_basic_present(self):
        recommend=["Movie1","Movie2","Movie3"]
        watched=["Movie1"]
        test=compute_accuracy(recommend,watched)
        assert len(test)==4


class TestEvaluateByDemographics:
    
    def test_evaluate_infrastructure(self):
        
        merged_df = pd.DataFrame({
            'user_id': [1, 2],
            'movie_id': [101, 102],
            'rating': [5, 4],
            'timestamp': pd.date_range(start='1/1/2022', periods=2),
            'gender': ['M', 'F'],
            'age': [25, 30],
            'occupation': ['student', 'artist']
        })
        
        train_df = merged_df.copy()
        val_df = pd.DataFrame(columns=merged_df.columns)
        test_df = merged_df.copy()
        
        # Create minimal mock functions that return expected types
        def mock_creating_matrix(path):
            # Return a small user-movie matrix
            matrix = pd.DataFrame({
                1: {101: 5, 102: 0},
                2: {101: 0, 102: 4}
            }).T
            return matrix, None
            
        def mock_train(path):
            # Return simple similarity matrices
            user_sim = pd.DataFrame({
                1: {1: 1.0, 2: 0.5},
                2: {1: 0.5, 2: 1.0}
            })
            movie_sim = pd.DataFrame({
                101: {101: 1.0, 102: 0.5},
                102: {101: 0.5, 102: 1.0}
            })
            return user_sim, movie_sim
            
        def mock_get_watched(user_id, matrix):
            # Return a set of watched movies
            if user_id not in matrix.index:
                return set()
            return {col for col, val in matrix.loc[user_id].items() if val > 0}
            
        def mock_recommend(user_id, sim, matrix, k):
            # Return a list of movie recommendations
            return [101, 102]
        
        # Call the function
        result = evaluate_by_demographics(
            merged_df, train_df, val_df, test_df,
            train_path="train.csv",
            val_path="val.csv",
            test_path="test.csv",
            creating_matrix_fn=mock_creating_matrix,
            train_fn=mock_train,
            get_watched_fn=mock_get_watched,
            recommend_user_fn=mock_recommend,
            recommend_movie_fn=mock_recommend
        )
        
        # Check the basic structure of the result
        assert isinstance(result, dict)
        assert 'gender' in result
        assert 'age_bracket' in result
        assert 'occupation' in result
        
        # Check that we have results for each demographic group
        assert 'M' in result['gender']
        assert 'F' in result['gender']
        
        # Check that each group has the expected metrics
        expected_metrics = ['user_count', 'avg_precision', 'avg_recall', 'avg_f1', 'avg_mrr']
        for metric in expected_metrics:
            assert metric in result['gender']['M']
            assert metric in result['gender']['F']
        
        # Verify the user counts
        assert result['gender']['M']['user_count'] > 0
        assert result['gender']['F']['user_count'] > 0
        
        # Verify that metrics are numeric values
        for gender in ['M', 'F']:
            for metric in ['avg_precision', 'avg_recall', 'avg_f1', 'avg_mrr']:
                assert isinstance(result['gender'][gender][metric], (int, float))



import tempfile


class TestOfflineEvaluation:
    
    def test_offline_evaluation_infrastructure(self):
        temp_dir = tempfile.mkdtemp()
        data_path = os.path.join(temp_dir, "data.csv")
        train_path = os.path.join(temp_dir, "train.csv")
        val_path = os.path.join(temp_dir, "val.csv")
        test_path = os.path.join(temp_dir, "test.csv")
        
        # Create minimal test data with timestamps
        data = {
            'user_id': [1, 1, 2, 2, 3, 3],
            'movie_id': [101, 102, 102, 103, 103, 104],
            'rating': [5, 4, 4, 5, 3, 5],
            'timestamp': pd.date_range(start='1/1/2022', periods=6),
            'gender': ['M', 'M', 'F', 'F', 'M', 'M'],
            'age': [25, 25, 30, 30, 35, 35],
            'occupation': ['student', 'student', 'artist', 'artist', 'engineer', 'engineer']
        }
        test_df = pd.DataFrame(data)
        
        # Save the test data
        test_df.to_csv(data_path, index=False)
        
        # Create simple mock functions
        def mock_creating_matrix(file_path):
            matrix = pd.DataFrame({
                1: {101: 5, 102: 4, 103: 0, 104: 0},
                2: {101: 0, 102: 4, 103: 5, 104: 0},
                3: {101: 0, 102: 0, 103: 3, 104: 5}
            }).T
            return matrix, None
        
        def mock_train(file_path):
            user_sim = np.array([
                [1.0, 0.5, 0.3],
                [0.5, 1.0, 0.4],
                [0.3, 0.4, 1.0]
            ])
            movie_sim = np.array([
                [1.0, 0.5, 0.2, 0.1],
                [0.5, 1.0, 0.3, 0.2],
                [0.2, 0.3, 1.0, 0.5],
                [0.1, 0.2, 0.5, 1.0]
            ])
            return user_sim, movie_sim
        
        def mock_get_watched(user_id, matrix):
            if user_id not in matrix.index:
                return set()
            return {col for col, val in matrix.loc[user_id].items() if val > 0}
        
        def mock_recommend(user_id, sim, matrix, top_n=10):
            if user_id == 1:
                return [103, 104]
            elif user_id == 2:
                return [101, 104]
            elif user_id == 3:
                return [101, 102]
            return []
        
        # Try calling the function with all dependencies explicitly provided
        try:
            # Call the function with our dependencies
            result = offline_evaluation(
                method='user_based',
                k=5,
                data_path=data_path,
                train_path=train_path,
                val_path=val_path,
                test_path=test_path,
                creating_matrix_fn=mock_creating_matrix,
                train_fn=mock_train,
                get_watched_fn=mock_get_watched,
                recommend_user_fn=mock_recommend,
                recommend_movie_fn=mock_recommend
            )
            
            # Check that the result has the expected structure
            assert isinstance(result, dict)
            assert 'validation' in result
            assert 'test' in result
            assert 'demographics' in result
            
        except Exception as e:
            pytest.fail(f"offline_evaluation failed with error: {str(e)}")
        
        finally:
            # Clean up temp files
            try:
                os.remove(data_path)
                os.remove(train_path)
                os.remove(val_path)
                os.remove(test_path)
                os.rmdir(temp_dir)
            except:
                pass

class TestEvaluateSplit:
    
    def test_evaluate_split_infrastructure(self):
        
    
        train_matrix = pd.DataFrame({
            1: {101: 5, 102: 4, 103: 0},
            2: {101: 0, 102: 4, 103: 5}
        }).T
        
        eval_matrix = pd.DataFrame({
            1: {104: 4, 105: 5},
            2: {104: 3, 106: 4}
        }).T
        
        # Create similarity matrices
        user_sim = np.array([
            [1.0, 0.5],
            [0.5, 1.0]
        ])
        
        movie_sim = np.array([
            [1.0, 0.5, 0.3, 0.2, 0.1, 0.1],
            [0.5, 1.0, 0.4, 0.3, 0.2, 0.2],
            [0.3, 0.4, 1.0, 0.5, 0.3, 0.3],
            [0.2, 0.3, 0.5, 1.0, 0.6, 0.4],
            [0.1, 0.2, 0.3, 0.6, 1.0, 0.5],
            [0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
        ])
        
        # User set to evaluate
        users = {1, 2}
        
        # Mock functions
        def mock_get_watched(user_id, matrix):
            if user_id not in matrix.index:
                return set()
            return {col for col, val in matrix.loc[user_id].items() if val > 0}
        
        def mock_recommend_user(user_id, sim, matrix, top_n=10):
            if user_id == 1:
                return [103, 104, 105, 106]
            elif user_id == 2:
                return [101, 104, 105, 106]
            return []
        
        def mock_recommend_movie(user_id, sim, matrix, top_n=10):
            # Return same recommendations for simplicity
            return mock_recommend_user(user_id, sim, matrix, top_n)
        
        # Call the function
        try:
            result = evaluate_split(
                users,
                train_matrix,
                eval_matrix,
                user_sim,
                movie_sim,
                method='user_based',
                k=3,
                get_watched_fn=mock_get_watched,
                recommend_user_fn=mock_recommend_user,
                recommend_movie_fn=mock_recommend_movie
            )
            
            # Check that result has expected structure
            assert isinstance(result, dict)
            assert "avg_precision" in result
            assert "avg_recall" in result
            assert "avg_f1" in result
            assert "avg_mrr" in result
            assert "total_evaluated_users" in result
            
            # Check data types
            assert isinstance(result["avg_precision"], float)
            assert isinstance(result["avg_recall"], float)
            assert isinstance(result["avg_f1"], float)
            assert isinstance(result["avg_mrr"], float)
            assert isinstance(result["total_evaluated_users"], int)
            
            # Make sure we evaluated at least some users
            assert result["total_evaluated_users"] > 0
            
        except Exception as e:
            pytest.fail(f"evaluate_split failed with error: {str(e)}")
    
    def test_evaluate_split_empty_users(self):
        """Test that evaluate_split handles empty user set properly."""

        

        train_matrix = pd.DataFrame({1: {101: 5}}).T
        eval_matrix = pd.DataFrame({1: {102: 4}}).T
        user_sim = np.array([[1.0]])
        movie_sim = np.array([[1.0, 0.5], [0.5, 1.0]])
        

        users = set()
        

        def mock_get_watched(user_id, matrix):
            return set()
        
        def mock_recommend(user_id, sim, matrix, top_n=10):
            return []
        
        # Call the function with empty users
        result = evaluate_split(
            users,
            train_matrix,
            eval_matrix,
            user_sim,
            movie_sim,
            method='user_based',
            k=3,
            get_watched_fn=mock_get_watched,
            recommend_user_fn=mock_recommend,
            recommend_movie_fn=mock_recommend
        )
        
        # Check default values for empty set
        assert result["avg_precision"] == 0
        assert result["avg_recall"] == 0
        assert result["avg_f1"] == 0
        assert result["avg_mrr"] == 0
        assert result["total_evaluated_users"] == 0
