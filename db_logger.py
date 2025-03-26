import os
import psycopg2
from datetime import datetime

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "db"),
        database=os.getenv("POSTGRES_DB", "mnist_db"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
    )

## To run the code in local machine, use the below code. Thin include change the postgres_host, the password of the database and the port number.
#     return psycopg2.connect(
#         host=os.getenv("POSTGRES_HOST", "localhost"),
#         database=os.getenv("POSTGRES_DB", "mnist_db"),
#         user=os.getenv("POSTGRES_USER", "postgres"),
#         password=os.getenv("POSTGRES_PASSWORD", "Password@23"),
#         port=os.getenv("POSTGRES_PORT", "5432")
#     )
    
def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP,
            predicted_digit INTEGER,
            true_label INTEGER,
            confidence FLOAT
        )
    ''')
    
    conn.commit()
    cur.close()
    conn.close()

def log_prediction(predicted_digit, true_label=None, confidence=None):
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute(
        "INSERT INTO predictions (timestamp, predicted_digit, true_label, confidence) "
        "VALUES (%s, %s, %s, %s)",
        (datetime.now(), predicted_digit, true_label, confidence)
    )
    
    conn.commit()
    cur.close()
    conn.close()

def get_prediction_history():
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("""
        SELECT timestamp, predicted_digit, true_label 
        FROM predictions 
        ORDER BY timestamp DESC
    """)
    
    history = cur.fetchall()
    
    cur.close()
    conn.close()
    
    return history
