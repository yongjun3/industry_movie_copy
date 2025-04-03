import time
import random
from flask import Flask, Response, jsonify
import requests
import os

app = Flask(__name__)

# Configure model servers and A/B split percentages
servers_list = [s.strip() for s in os.environ.get('MODEL_SERVERS', 'http://user_model:8082,http://new_movie_model:8082').split(',')]

# Example: 70% traffic to user_model, 30% to new_movie_model
traffic_split = [0.7, 0.3]

# Request counters
request_counts = {server: 0 for server in servers_list}

def wait_for_backends():
    for url in servers_list:
        healthy = False
        retries = 0
        while not healthy and retries < 30:
            try:
                response = requests.get(f"{url}/health", timeout=2)
                if response.status_code == 200:
                    print(f"Backend {url} is healthy!")
                    healthy = True
                else:
                    print(f"Backend {url} not ready, retrying... ({retries}/30)")
            except Exception:
                print(f"Backend {url} not ready, retrying... ({retries}/30)")
            if not healthy:
                retries += 1
                time.sleep(5)

@app.route('/recommend/<int:user_id>', methods=['GET'])
def load_balance(user_id):
    # A/B selection based on configured split
    backend_url = random.choices(servers_list, weights=traffic_split, k=1)[0]
    try:
        response = requests.get(f"{backend_url}/recommend/{user_id}", timeout=3)
        request_counts[backend_url] += 1
        flask_response = Response(response.text, status=response.status_code, mimetype='text/plain')
        flask_response.headers['X-Backend'] = backend_url  
        return flask_response
    except Exception as e:
        return Response(f"Error contacting backend: {str(e)}", status=500)

@app.route('/stats', methods=['GET'])
def get_stats():
    return jsonify(request_counts)

@app.route('/reset-stats', methods=['POST'])
def reset_stats():
    for key in request_counts.keys():
        request_counts[key] = 0
    return jsonify({"message": "Request stats have been reset."})

if __name__ == '__main__':
    wait_for_backends()
    app.run(host='0.0.0.0', port=8081)