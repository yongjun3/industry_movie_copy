from flask import Flask, jsonify, Response
import time
import numpy as np
import json
import pandas as pd
from main import creating_needed_matrix, train, recommend_user_based, recommend_movie_based
# from evaluation import recommend_user_based_with_watched, recommend_movie_based_with_watched, get_watched_movies

app = Flask(__name__)

# --- Data Splitting ---
# Read the full merged data and split into train and test using timestamp
merged_df = pd.read_csv("data/merged_data.csv")

threshold = "2025-02-08T00:00:00"

# Split the data: Training data (past) and Test data (future)
train_df = merged_df[merged_df["timestamp"] < threshold]
test_df = merged_df[merged_df["timestamp"] >= threshold]

# Save temporary CSVs for train and test, so our existing functions can use them.
train_df.to_csv("data/train_data.csv", index=False)
test_df.to_csv("data/test_data.csv", index=False)

# --- Build Model Using Train Data ---
_, movie_genre_matrix = creating_needed_matrix("data/merged_data.csv")
user_movie_matrix, _ = creating_needed_matrix("data/train_data.csv")
user_movie_similarity, movie_genre_similarity = train("data/train_data.csv")

# --- Build Test Matrix for Evaluation: Future Watched Movies ---
test_user_movie_matrix, _ = creating_needed_matrix("data/test_data.csv")

TELEMETRY = []

def get_watched_movies(user_id, user_movie_matrix):
    # Retrieves all movies a user has watched, including those they rated.
    if user_id not in user_movie_matrix.index:
        return set()
    return set(user_movie_matrix.loc[user_id][user_movie_matrix.loc[user_id] > 0].index)

def compute_accuracy(recommended, watched, k=20):
    recommended_set = set(recommended[:k])
    if not watched:
        return {"Precision@K": 0, "Recall@K": 0, "F1@K": 0}
    true_positives = len(recommended_set & watched)
    precision = true_positives / k if k > 0 else 0
    recall = true_positives / len(watched)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    return {"Precision@K": precision, "Recall@K": recall, "F1@K": f1}

@app.route('/recommend/<int:user_id>', methods=['GET'])
def recommend(user_id):
    start_time = time.time()
    # Use the train-based recommendation function.
    recommendations = recommend_user_based(user_id, user_movie_similarity, user_movie_matrix)
    if not recommendations:
        recommendations = []
    recommendations_str = ",".join(recommendations)
    response_time = (time.time() - start_time) * 1000  # in ms

    # For evaluation, get the movies the user watched in the future (from test set).
    watched_movies = get_watched_movies(user_id, test_user_movie_matrix)
    # print("--" * 20)
    # print(f"Future watched movies for user {user_id}: {watched_movies}")
    # print(f"Recommended movies for user {user_id}: {recommendations}")
    # print("--" * 20)
    eval_metrics = compute_accuracy(recommendations, watched_movies)

    TELEMETRY.append({
        "user_id": user_id,
        "response_time_ms": response_time,
        "num_recommendations": len(recommendations),
        "eval_metrics": eval_metrics
    })

    # Write current request's eval metrics to file (one JSON object per line)
    with open("request_eval_metrics.jsonl", "a") as f:
        json.dump({"user_id": user_id, "eval_metrics": eval_metrics}, f)
        f.write("\n")

    print(f"User {user_id}, Recommendations: {recommendations_str}, Response time: {response_time:.2f}ms")
    return Response(recommendations_str, status=200, mimetype='text/plain')

@app.route('/evaluation_metrics', methods=['GET'])
def evaluation_metrics():
    if not TELEMETRY:
        return jsonify({"message": "No telemetry data available."})
    
    avg_response_time = sum(d["response_time_ms"] for d in TELEMETRY) / len(TELEMETRY)
    precision_list = [d["eval_metrics"].get("Precision@K", 0) for d in TELEMETRY if d["eval_metrics"]]
    recall_list = [d["eval_metrics"].get("Recall@K", 0) for d in TELEMETRY if d["eval_metrics"]]
    f1_list = [d["eval_metrics"].get("F1@K", 0) for d in TELEMETRY if d["eval_metrics"]]
    
    avg_precision = np.mean(precision_list) if precision_list else None
    avg_recall = np.mean(recall_list) if recall_list else None
    avg_f1 = np.mean(f1_list) if f1_list else None
    
    result = {
        "avg_response_time_ms": avg_response_time,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_f1": avg_f1,
        "total_requests": len(TELEMETRY)
    }
    
    # Write the aggregated evaluation metrics to a local JSON file.
    with open("online_eval_output.json", "w") as f:
        json.dump(result, f)
    
    return jsonify(result)

if __name__ == '__main__':
    # Load data and train when starting the server
    app.run(host='0.0.0.0', port=8082)