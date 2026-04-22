import os
import time
import psycopg2
import redis
from flask import Flask, jsonify

app = Flask(__name__)

database_url = os.environ.get("DATABASE_URL")
redis_url = os.environ.get("REDIS_URL")

if not database_url:
    raise ValueError("DATABASE_URL environment variable not set")
if not redis_url:
    raise ValueError("REDIS_URL environment variable not set")

def get_db_connection():
    return psycopg2.connect(database_url)

def get_redis_client():
    return redis.from_url(redis_url)

@app.route("/health")
def health():
    return jsonify({"status": "ok", "timestamp": time.time()})

@app.route("/api/orders/<email>")
def get_orders(email):
    # Intentional lack of index on user_email creates sequential scan
    try:
        start_time = time.time()
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM orders WHERE user_email = %s", (email,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        elapsed = time.time() - start_time
        return jsonify({
            "count": len(rows), 
            "elapsed_ms": round(elapsed*1000, 2),
            "orders": rows[:10]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/stats")
def stats():
    # Demonstrates Redis caching pattern. If cache misses, heavy aggregator runs on Postgres
    try:
        r = get_redis_client()
        cached = r.get("daily_stats")
        if cached:
            return jsonify({"source": "cache", "data": cached.decode()})
        
        # Cache miss, hit the DB.
        start_time = time.time()
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT status, COUNT(*), SUM(amount) FROM orders GROUP BY status")
        result = str(cur.fetchall())
        cur.close()
        conn.close()
        
        # Set cache
        r.setex("daily_stats", 300, result) # 5 min TTL
        
        elapsed = time.time() - start_time
        return jsonify({"source": "database", "data": result, "elapsed_ms": round(elapsed*1000, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
