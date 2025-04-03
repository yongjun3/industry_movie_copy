import os
import time
import sqlite3
import atexit
import threading
from kafka import KafkaConsumer
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define database file (it will be created in a persistent volume)
DB_FILE = "db/recommendation.db"

# Global connection variable
db_conn = None

def get_db_conn():
    global db_conn
    if db_conn is None:
        db_conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    return db_conn

def close_db_conn():
    global db_conn
    if db_conn:
        db_conn.close()
        db_conn = None

atexit.register(close_db_conn)

# Initialize the database and create tables if not exists
def init_db():
    os.makedirs("db", exist_ok=True)  # ensure folder exists
    conn = get_db_conn()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            recommended_movies TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS watch_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            movie_id TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()

# Simple function to clean up old data
def cleanup_database():
    print("Cleaning up database...")
    conn = get_db_conn()
    c = conn.cursor()
    try:
        # Delete all data from tables
        c.execute("DELETE FROM recommendations")
        c.execute("DELETE FROM watch_events")
        # Vacuum to reclaim space
        c.execute("VACUUM")
        conn.commit()
        print("Database cleaned up successfully")
    except Exception as e:
        conn.rollback()
        print(f"Error cleaning up database: {e}")

# Insert a recommendation record
def insert_recommendation(user_id, recommended_movies):
    # print(f"Inserting recommendation for user {user_id}: {recommended_movies}")
    conn = get_db_conn()
    c = conn.cursor()
    try:
        c.execute('''
            INSERT INTO recommendations (user_id, recommended_movies) VALUES (?, ?)
        ''', (user_id, recommended_movies))
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error inserting recommendation for user {user_id}: {e}")

# Insert a watch event record
def insert_watch_event(user_id, movie_id):
    # print(f"Inserting watch event for user {user_id}: {movie_id}")
    conn = get_db_conn()
    c = conn.cursor()
    try:
        c.execute('''
            INSERT INTO watch_events (user_id, movie_id) VALUES (?, ?)
        ''', (user_id, movie_id))
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error inserting watch event for user {user_id}: {e}")

# Function to compute recall for a given user
def compute_recall_for_user(user_id):
    conn = get_db_conn()
    c = conn.cursor()
    c.execute('''
        SELECT recommended_movies FROM recommendations 
        WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1
    ''', (user_id,))
    row = c.fetchone()
    recommended = set(row[0].split(',')) if row and row[0] else set()
    c.execute('''
        SELECT movie_id FROM watch_events WHERE user_id = ?
    ''', (user_id,))
    watched = {r[0] for r in c.fetchall()}
    if not watched:
        return 0.0
    true_positive = len(recommended.intersection(watched))
    recall = true_positive / len(watched)
    return recall

topic = 'movielog27'

# Prometheus metrics
REQUEST_COUNT = Counter('request_count', 'Recommendation Request Count', ['http_status'])
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency')
RECALL_SCORE = Gauge('recall_score', 'Recall score for recommendations per user', ['user_id'])

def main():
    print("Starting Kafka monitoring service...")
    init_db()
    
    start_http_server(8765)
    print(f"HTTP server started on port 8765")
    
    # Set up database cleanup every 3 days
    cleanup_interval = 3 * 24 * 60 * 60  # 3 days in seconds
    last_cleanup_time = time.time()
    
    try:
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers='localhost:9092',
            auto_offset_reset='latest',
            group_id=topic,
            enable_auto_commit=True,
            auto_commit_interval_ms=1000
        )
        
        print(f"Waiting for messages on topic: {topic}")
        
        for message in consumer:
            # Check if it's time to clean up the database
            current_time = time.time()
            if current_time - last_cleanup_time >= cleanup_interval:
                print("Performing database cleanup...")
                cleanup_database()
                last_cleanup_time = current_time
            
            event = message.value.decode('utf-8')
            values = event.split(',')
            
            if len(values) >= 3:
                if 'recommendation request' in values[2]:
                    status = values[3].strip().split()[1]
                    REQUEST_COUNT.labels(status).inc()
                    try:
                        time_taken = float(values[-1].strip().split(" ")[0])
                    except Exception as e:
                        print(f"Error parsing time: {e}")
                        time_taken = 0
                    REQUEST_LATENCY.observe(time_taken / 1000)
                    
                    if status == "200":
                        try:
                            rec_part = event.split("result:")[1]
                            rec_movies = rec_part.rsplit(",", 1)[0].strip()
                        except Exception as e:
                            print(f"Error parsing recommendation: {e}")
                            rec_movies = ""
                        user = int(values[1].strip())
                        insert_recommendation(user, rec_movies)
                
                elif '/data/' in values[2]:
                    # Process a watch event – extract movie id from URL
                    request_url = values[2].strip()
                    parts = request_url.split('/')
                    movie_id = parts[3] if len(parts) >= 3 else ""
                    try:
                        user = int(values[1].strip())
                        insert_watch_event(user, movie_id)
                        
                        # Recompute recall for the user and update Prometheus gauge
                        recall_value = compute_recall_for_user(user)
                        RECALL_SCORE.labels(user_id=user).set(recall_value)
                    except Exception as e:
                        print(f"Error processing watch event: {e}")
    
    except Exception as e:
        print(f"Error in Kafka consumer: {e}")

if __name__ == "__main__":
    main()