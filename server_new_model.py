from flask import Flask, jsonify, Response
import time
from movie_user_mixed import *
from config import *
import pandas

app = Flask(__name__)

merged_df=pd.read_csv(DATA_PATH)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/recommend/<int:user_id>', methods=['GET'])
def recommend(user_id):
    start_time = time.time()
    recommendations = recommendation_func(merged_df, user_id)

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