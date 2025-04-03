from flask import Flask, jsonify, Response
import time
from recommender_model.data_loader import creating_needed_matrix
from recommender_model.train import train
from recommender_model.user_based_model import recommend_user_based
from recommender_model.movie_based_model import recommend_movie_based
from config import *

app = Flask(__name__)

user_movie_matrix, movie_genre_matrix = creating_needed_matrix(DATA_PATH)
user_movie_similarity, movie_genre_similarity = train(DATA_PATH)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/recommend/<int:user_id>', methods=['GET'])
def recommend(user_id):
    start_time = time.time()
    recommendations = recommend_user_based(user_id, user_movie_similarity, user_movie_matrix)

    if not recommendations:
        recommendations = []

    # Convert movie titles to indices or IDs if necessary
    recommendations_str = ",".join(recommendations)

    response_time = (time.time() - start_time) * 1000

    print(f"User {user_id}, Recommendations: {recommendations_str}, Response time: {response_time:.2f}ms")

    return Response(recommendations_str, status=200, mimetype='text/plain')

if __name__ == '__main__':
    # Load data and train when starting the server
    app.run(host=API_HOST, port=API_PORT)