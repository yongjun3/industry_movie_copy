import pandas as pd
import numpy as np
import os
import pickle
from recommender_model.data_loader import creating_needed_matrix, get_watched_movies
from recommender_model.train import train
from recommender_model.user_based_model import recommend_user_based_with_watched
from recommender_model.movie_based_model import recommend_movie_based_with_watched

# Import configuration
import config

def calculate_mrr(recommended, watched):
    if not watched:
        return 0
    
    for rank, movie in enumerate(recommended, 1):
        if movie in watched:
            return 1.0 / rank
    return 0  # If none of the recommended items were watched

def compute_accuracy(recommended, watched, k=20):
    """
    Returns:
    - dict of accuracy metrics
    """
    # Use at most k recommendations
    recommended_items = recommended[:k]
    recommended_set = set(recommended_items)
    
    if not watched:
        return {"Precision@K": 0, "Recall@K": 0, "F1@K": 0, "MRR": 0}
    
    true_positives = len(recommended_set & watched)
    precision = true_positives / len(recommended_items) if recommended_items else 0
    recall = true_positives / len(watched)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    mrr = calculate_mrr(recommended_items, watched)
    
    return {
        "Precision@K": precision, 
        "Recall@K": recall, 
        "F1@K": f1,
        "MRR": mrr
    }

def evaluate_by_demographics(merged_df, train_df, val_df, test_df, method='user_based', k=None, 
                             train_path=None, val_path=None, test_path=None,
                             creating_matrix_fn=None, train_fn=None, 
                             get_watched_fn=None, recommend_user_fn=None, recommend_movie_fn=None):
    """
    Perform subpopulation analysis across different demographic groups
    Returns:
    - dict of demographic analysis results
    """
    # Use configuration or default values
    k = k or config.DEFAULT_K
    train_path = train_path or config.TRAIN_DATA_PATH
    val_path = val_path or config.VAL_DATA_PATH
    test_path = test_path or config.TEST_DATA_PATH
    
    # Use provided functions or default imports
    creating_matrix_fn = creating_matrix_fn or creating_needed_matrix
    train_fn = train_fn or train
    get_watched_fn = get_watched_fn or get_watched_movies
    recommend_user_fn = recommend_user_fn or recommend_user_based_with_watched
    recommend_movie_fn = recommend_movie_fn or recommend_movie_based_with_watched
    
    # Build model using train data
    user_movie_matrix, _ = creating_matrix_fn(train_path)
    user_movie_similarity, movie_genre_similarity = train_fn(train_path)
    
    # Create test matrices for validation and testing
    val_user_movie_matrix, _ = creating_matrix_fn(val_path) 
    test_user_movie_matrix, _ = creating_matrix_fn(test_path)
    
    # Create user demographics dataframe
    user_demographics = merged_df.groupby('user_id').agg({
        'age': 'first',
        'gender': 'first',
        'occupation': 'first'
    }).reset_index()
    
    # Create age brackets
    def age_to_bracket(age):
        if pd.isna(age) or age < 0:
            return 'unknown'
        elif age < 18:
            return 'under_18'
        elif age < 25:
            return '18_to_25'
        elif age < 35:
            return '25_to_35'
        elif age < 50:
            return '35_to_50'
        else:
            return 'over_50'
    
    # Add age bracket column if age exists
    if 'age' in user_demographics.columns:
        user_demographics['age_bracket'] = user_demographics['age'].apply(age_to_bracket)
    
    # Analysis results
    demographic_results = {'gender': {}, 'age_bracket': {}, 'occupation': {}}
    
    # For each demographic attribute
    for attribute in ['gender', 'age_bracket', 'occupation']:
        if attribute not in user_demographics.columns and attribute != 'age_bracket':
            continue
            
        # Group users by demographic attribute
        user_groups = {}
        for _, row in user_demographics.iterrows():
            user_id = row['user_id']
            if attribute in row and not pd.isna(row[attribute]):
                group = row[attribute]
                if group not in user_groups:
                    user_groups[group] = []
                user_groups[group].append(user_id)
        
        # Analyze each demographic group
        for group, users in user_groups.items():
            precision_list = []
            recall_list = []
            f1_list = []
            mrr_list = []
            test_users_count = 0
            
            # Use test data for evaluation
            test_users = set(test_df['user_id'].unique()) & set(users)
            
            for user_id in test_users:
                if user_id not in user_movie_matrix.index:
                    continue
                    
                # Get future watched movies for this user
                watched_movies = get_watched_fn(user_id, test_user_movie_matrix)
                if not watched_movies:
                    continue
                
                # Get already watched movies (training data) to avoid recommending them
                user_watched_train = get_watched_fn(user_id, user_movie_matrix)
                
                # Generate recommendations
                if method == 'user_based':
                    all_recommendations = recommend_user_fn(user_id, user_movie_similarity, user_movie_matrix, k*2)
                    # Filter out already watched movies
                    recommendations = [movie for movie in all_recommendations if movie not in user_watched_train][:k]
                else:
                    all_recommendations = recommend_movie_fn(user_id, movie_genre_similarity, user_movie_matrix, k*2)
                    # Filter out already watched movies
                    recommendations = [movie for movie in all_recommendations if movie not in user_watched_train][:k]
                
                if not recommendations:
                    continue
                
                # Compute accuracy metrics
                eval_metrics = compute_accuracy(recommendations, watched_movies, k)
                
                precision_list.append(eval_metrics["Precision@K"])
                recall_list.append(eval_metrics["Recall@K"])
                f1_list.append(eval_metrics["F1@K"])
                mrr_list.append(eval_metrics["MRR"])
                test_users_count += 1
            
            # Calculate average metrics for this group
            if test_users_count > 0:
                demographic_results[attribute][group] = {
                    'user_count': test_users_count,
                    'avg_precision': np.mean(precision_list),
                    'avg_recall': np.mean(recall_list),
                    'avg_f1': np.mean(f1_list),
                    'avg_mrr': np.mean(mrr_list)
                }
    
    return demographic_results

def offline_evaluation(method='user_based', k=None, data_path=None, train_path=None, val_path=None, test_path=None,
                      creating_matrix_fn=None, train_fn=None, 
                      get_watched_fn=None, recommend_user_fn=None, recommend_movie_fn=None):
    """
    Returns:
    - dict of evaluation metrics
    """
    # Use configuration or default values
    k = k or config.DEFAULT_K
    data_path = data_path or config.DATA_PATH
    train_path = train_path or config.TRAIN_DATA_PATH
    val_path = val_path or config.VAL_DATA_PATH
    test_path = test_path or config.TEST_DATA_PATH
    
    # Use provided functions or default imports
    creating_matrix_fn = creating_matrix_fn or creating_needed_matrix
    train_fn = train_fn or train
    get_watched_fn = get_watched_fn or get_watched_movies
    recommend_user_fn = recommend_user_fn or recommend_user_based_with_watched
    recommend_movie_fn = recommend_movie_fn or recommend_movie_based_with_watched
    
    # Read the full merged data
    merged_df = pd.read_csv(data_path)
    
    # Convert timestamp to datetime if it's string
    if isinstance(merged_df['timestamp'].iloc[0], str):
        merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'], format="mixed", errors="coerce")
    
    # Sort by timestamp
    merged_df = merged_df.sort_values('timestamp')
    
    # Determine the timestamp thresholds for train-val-test split (60% train, 20% val, 20% test)
    val_split_time = merged_df['timestamp'].quantile(0.6)
    test_split_time = merged_df['timestamp'].quantile(0.8)
    
    # Split the data
    train_df = merged_df[merged_df['timestamp'] < val_split_time]
    val_df = merged_df[(merged_df['timestamp'] >= val_split_time) & 
                        (merged_df['timestamp'] < test_split_time)]
    test_df = merged_df[merged_df['timestamp'] >= test_split_time]
    
    print(f"Training data: {len(train_df)} entries from {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    print(f"Validation data: {len(val_df)} entries from {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
    print(f"Testing data: {len(test_df)} entries from {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    
    # Save temporary CSVs for train, val, and test
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # Build Model Using Train Data
    user_movie_matrix, _ = creating_matrix_fn(train_path)
    user_movie_similarity, movie_genre_similarity = train_fn(train_path)
    
    # Create test matrices for validation and testing
    val_user_movie_matrix, _ = creating_matrix_fn(val_path) 
    test_user_movie_matrix, _ = creating_matrix_fn(test_path)
    
    # Evaluate on validation data
    val_users = set(val_df['user_id'].unique())
    val_metrics = evaluate_split(val_users, user_movie_matrix, val_user_movie_matrix,
                                user_movie_similarity, movie_genre_similarity, method, k,
                                get_watched_fn, recommend_user_fn, recommend_movie_fn)
    
    # Evaluate on test data
    test_users = set(test_df['user_id'].unique())
    test_metrics = evaluate_split(test_users, user_movie_matrix, test_user_movie_matrix,
                                 user_movie_similarity, movie_genre_similarity, method, k,
                                 get_watched_fn, recommend_user_fn, recommend_movie_fn)
    
    # Perform demographic analysis
    demographic_results = evaluate_by_demographics(
        merged_df, train_df, val_df, test_df, method, k, 
        train_path, val_path, test_path,
        creating_matrix_fn, train_fn, 
        get_watched_fn, recommend_user_fn, recommend_movie_fn
    )
    
    return {
        'validation': val_metrics,
        'test': test_metrics,
        'demographics': demographic_results
    }

def evaluate_split(users, train_matrix, eval_matrix, user_sim, movie_sim, method, k=20,
                  get_watched_fn=None, recommend_user_fn=None, recommend_movie_fn=None):
    """
    Evaluate recommendations on a set of users
    """
    # Use provided functions or default imports
    get_watched_fn = get_watched_fn or get_watched_movies
    recommend_user_fn = recommend_user_fn or recommend_user_based_with_watched
    recommend_movie_fn = recommend_movie_fn or recommend_movie_based_with_watched
    
    precision_list = []
    recall_list = []
    f1_list = []
    mrr_list = []
    total_evaluated = 0
    
    for user_id in users:
        if user_id not in train_matrix.index:
            continue
        
        # Get future watched movies for this user
        watched_movies = get_watched_fn(user_id, eval_matrix)
        if not watched_movies:
            continue
            
        # Get already watched movies (training data) to avoid recommending them
        user_watched_train = get_watched_fn(user_id, train_matrix)
        
        # Generate recommendations
        if method == 'user_based':
            all_recommendations = recommend_user_fn(user_id, user_sim, train_matrix, k*2)
            # Filter out already watched movies
            recommendations = [movie for movie in all_recommendations if movie not in user_watched_train][:k]
        else:
            all_recommendations = recommend_movie_fn(user_id, movie_sim, train_matrix, k*2)
            # Filter out already watched movies
            recommendations = [movie for movie in all_recommendations if movie not in user_watched_train][:k]
        
        if not recommendations:
            continue
            
        # Compute accuracy metrics
        eval_metrics = compute_accuracy(recommendations, watched_movies, k)
        
        precision_list.append(eval_metrics["Precision@K"])
        recall_list.append(eval_metrics["Recall@K"])
        f1_list.append(eval_metrics["F1@K"])
        mrr_list.append(eval_metrics["MRR"])
        total_evaluated += 1
    
    # Calculate average metrics
    return {
        "avg_precision": np.mean(precision_list) if precision_list else 0,
        "avg_recall": np.mean(recall_list) if recall_list else 0,
        "avg_f1": np.mean(f1_list) if f1_list else 0,
        "avg_mrr": np.mean(mrr_list) if mrr_list else 0,
        "total_evaluated_users": total_evaluated
    }

def print_demographic_analysis(demographic_results, k=None):
    """Print formatted demographic analysis results"""
    k = k or config.DEFAULT_K
    
    print("\n=== Demographic Analysis ===")
    
    for attribute, groups in demographic_results.items():
        if not groups:
            continue
            
        print(f"\nAnalysis by {attribute.replace('_', ' ').title()}:")
        
        # Sort groups for consistent display
        sorted_groups = sorted(groups.items())
        
        for group, metrics in sorted_groups:
            print(f"\n  {group.replace('_', ' ').title()}")
            print(f"    Users evaluated: {metrics['user_count']}")
            print(f"    Recall@{k}: {metrics['avg_recall']:.4f}")
            print(f"    Precision@{k}: {metrics['avg_precision']:.4f}")
            print(f"    F1@{k}: {metrics['avg_f1']:.4f}")
            print(f"    MRR: {metrics['avg_mrr']:.4f}")

if __name__ == "__main__":
    k = config.DEFAULT_K
    results_path = config.RESULTS_PATH
    
    # Evaluate user-based collaborative filtering
    print("Evaluating User-Based Collaborative Filtering...")
    user_based_results = offline_evaluation(method='user_based', k=k)
    
    # Evaluate movie-based collaborative filtering
    print("\nEvaluating Movie-Based Collaborative Filtering...")
    movie_based_results = offline_evaluation(method='movie_based', k=k)
    
    # Print validation results
    print("\n=== Validation Results ===")
    print("User-Based Collaborative Filtering:")
    for metric, value in user_based_results['validation'].items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    print("\nMovie-Based Collaborative Filtering:")
    for metric, value in movie_based_results['validation'].items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    # Print test results
    print("\n=== Test Results ===")
    print("User-Based Collaborative Filtering:")
    for metric, value in user_based_results['test'].items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    print("\nMovie-Based Collaborative Filtering:")
    for metric, value in movie_based_results['test'].items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    # Print demographic analysis
    print_demographic_analysis(user_based_results['demographics'], k)
    
    # Save full results to file
    try:
        results = {
            'user_based': user_based_results,
            'movie_based': movie_based_results
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nResults saved to '{results_path}'")
    except Exception as e:
        print(f"\nError saving results: {e}")
