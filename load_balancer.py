import time
from flask import Flask, Response, jsonify
import requests
import itertools
import os

app = Flask(__name__)
# Configure model servers - can be updated without restarting container
servers = itertools.cycle([s.strip() for s in os.environ.get('MODEL_SERVERS', 'http://user_model:8082,http://new_movie_model:8082').split(',')])

def wait_for_backends():
    backend_urls = os.environ.get('MODEL_SERVERS', 'http://user_model:8082,http://new_movie_model:8082').split(',')
    for url in backend_urls:
        url = url.strip()
        healthy = False
        max_retries = 30
        retries = 0
        while not healthy and retries < max_retries:
            try:
                response = requests.get(f"{url}/health", timeout=2)
                if response.status_code == 200:
                    print(f"Backend {url} is healthy!")
                    healthy = True
                else:
                    print(f"Backend {url} not ready, retrying... ({retries}/{max_retries})")
            except Exception as e:
                print(f"Backend {url} not ready, retrying... ({retries}/{max_retries})")
            
            if not healthy:
                retries += 1
                time.sleep(5)  # Wait 5 seconds before retry

@app.route('/recommend/<int:user_id>', methods=['GET'])
def load_balance(user_id):
    backend_url = next(servers)
    response = requests.get(f"{backend_url}/recommend/{user_id}")
    return Response(response.text, status=response.status_code, mimetype='text/plain')

if __name__ == '__main__':
    wait_for_backends()
    app.run(host='0.0.0.0', port=8082)