"""
============================================================
 AQI Monitor — Python ML Auto-Retrain Microservice (SQLite)
 Student : Anantharajan Vel Murugan | 294FAVZE | UoH
 
 PURPOSE:
 Separate Python service that:
 1. Stores sensor readings in local SQLite (no external DB needed)
 2. Auto-retrains Random Forest every N new readings
 3. Saves trained model + accuracy
 4. Exposes /predict and /status endpoints for Node backend

 NOTE: Render free tier wipes disk on restart. For continuous
 uptime between restarts, this works perfectly. CSV backup
 of your original 1,440 readings protects against data loss.
============================================================
"""

import os
import time
import sqlite3
import threading
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

app = Flask(__name__)
CORS(app)

# ── Config ────────────────────────────────────────────────
RETRAIN_EVERY = int(os.environ.get("RETRAIN_EVERY", "100"))
DB_PATH = "/tmp/aqi_data.db"
MODEL_PATH = "/tmp/rf_model.pkl"
ENCODER_PATH = "/tmp/label_encoder.pkl"

FEATURES = ['temperature', 'humidity', 'pm25', 'pm10', 'mq135', 'mq7']

# ════════════════════════════════════════════════════════════
#  SQLITE SETUP
# ════════════════════════════════════════════════════════════
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            temperature REAL,
            humidity REAL,
            pm25 REAL,
            pm10 REAL,
            mq135 REAL,
            mq7 REAL,
            aqi REAL,
            aqi_label TEXT,
            received_at TEXT
        )
    """)
    conn.commit()
    conn.close()
    print("[SQLite] Database initialised")

def get_db():
    return sqlite3.connect(DB_PATH)

def count_readings():
    conn = get_db()
    count = conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
    conn.close()
    return count

def insert_reading(data):
    conn = get_db()
    conn.execute("""
        INSERT INTO readings
        (temperature, humidity, pm25, pm10, mq135, mq7, aqi, aqi_label, received_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data['temperature'], data['humidity'], data['pm25'], data['pm10'],
        data['mq135'], data['mq7'], data['aqi'], data['aqi_label'],
        datetime.utcnow().isoformat() + "Z"
    ))
    conn.commit()
    conn.close()

def fetch_all_readings():
    conn = get_db()
    df = pd.read_sql_query("SELECT * FROM readings", conn)
    conn.close()
    return df

def seed_from_csv_if_empty(csv_path="seed_data.csv"):
    """Optional: pre-load your original 1,440 readings if DB starts empty"""
    if count_readings() > 0:
        return
    if not os.path.exists(csv_path):
        print("[Seed] No seed_data.csv found — starting with empty DB")
        return
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.lower().str.strip()
        conn = get_db()
        for _, row in df.iterrows():
            try:
                conn.execute("""
                    INSERT INTO readings
                    (temperature, humidity, pm25, pm10, mq135, mq7, aqi, aqi_label, received_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    float(row['temperature']), float(row['humidity']),
                    float(row['pm25']), float(row['pm10']),
                    float(row['mq135']), float(row['mq7']),
                    float(row['aqi']), str(row['aqi_label']).upper(),
                    datetime.utcnow().isoformat() + "Z"
                ))
            except Exception:
                continue
        conn.commit()
        conn.close()
        print(f"[Seed] Loaded {count_readings()} rows from {csv_path}")
    except Exception as e:
        print(f"[Seed] Error loading CSV: {e}")


# ── Global state ──────────────────────────────────────────
state = {
    "model": None,
    "encoder": None,
    "accuracy": None,
    "cv_accuracy": None,
    "trained_at": None,
    "training_rows": 0,
    "last_count_checked": 0,
    "is_training": False,
}

# ════════════════════════════════════════════════════════════
#  TRAINING FUNCTION
# ════════════════════════════════════════════════════════════
def train_model():
    if state["is_training"]:
        return False
    state["is_training"] = True
    print(f"[Train] Starting at {datetime.now()}")

    try:
        df = fetch_all_readings()
        if len(df) < 30:
            print(f"[Train] Not enough data yet ({len(df)} rows) — need 30+")
            state["is_training"] = False
            return False

        for c in FEATURES + ['aqi']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df['aqi_label'] = df['aqi_label'].astype(str).str.upper().str.strip()
        df.dropna(subset=FEATURES + ['aqi', 'aqi_label'], inplace=True)
        df = df[df['temperature'] > 0]
        df = df[df['humidity'] > 0]
        df.drop_duplicates(subset=FEATURES, inplace=True)

        if len(df) < 30:
            print(f"[Train] Not enough clean data ({len(df)} rows)")
            state["is_training"] = False
            return False

        X = df[FEATURES]
        y = df['aqi_label']

        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        if len(le.classes_) < 2:
            print("[Train] Only 1 class present — skipping training")
            state["is_training"] = False
            return False

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.2, random_state=42,
            stratify=y_enc if len(np.unique(y_enc)) > 1 else None
        )

        clf = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)

        cv_folds = min(5, len(np.unique(y_train)))
        cv_scores = cross_val_score(clf, X, y_enc, cv=max(2, cv_folds))

        joblib.dump(clf, MODEL_PATH)
        joblib.dump(le, ENCODER_PATH)

        state["model"] = clf
        state["encoder"] = le
        state["accuracy"] = round(acc * 100, 2)
        state["cv_accuracy"] = round(cv_scores.mean() * 100, 2)
        state["trained_at"] = datetime.utcnow().isoformat() + "Z"
        state["training_rows"] = len(df)
        state["last_count_checked"] = count_readings()

        print(f"[Train] Done! Accuracy={state['accuracy']}% CV={state['cv_accuracy']}% Rows={len(df)}")
        state["is_training"] = False
        return True

    except Exception as e:
        print(f"[Train] ERROR: {e}")
        state["is_training"] = False
        return False


def load_existing_model():
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
            state["model"] = joblib.load(MODEL_PATH)
            state["encoder"] = joblib.load(ENCODER_PATH)
            print("[Startup] Loaded existing model from disk")
    except Exception as e:
        print(f"[Startup] Could not load existing model: {e}")


# ════════════════════════════════════════════════════════════
#  BACKGROUND THREAD
# ════════════════════════════════════════════════════════════
def background_retrain_checker():
    while True:
        try:
            current_count = count_readings()
            if current_count - state["last_count_checked"] >= RETRAIN_EVERY:
                print(f"[Auto-retrain] Trigger: {current_count} readings")
                train_model()
            elif state["trained_at"] is None and current_count >= 30:
                print(f"[Auto-retrain] First training with {current_count} readings")
                train_model()
        except Exception as e:
            print(f"[Auto-retrain] Checker error: {e}")
        time.sleep(30)


# ════════════════════════════════════════════════════════════
#  API ENDPOINTS
# ════════════════════════════════════════════════════════════

@app.route("/")
def home():
    return jsonify({
        "service": "AQI ML Auto-Retrain Microservice (SQLite)",
        "student": "Anantharajan Vel Murugan | 294FAVZE | UoH",
        "routes": ["/status", "/predict (POST)", "/retrain-now (POST)", "/ingest (POST)"],
    })


@app.route("/status")
def status():
    total = count_readings()
    return jsonify({
        "model_trained": state["model"] is not None,
        "accuracy": state["accuracy"],
        "cv_accuracy": state["cv_accuracy"],
        "trained_at": state["trained_at"],
        "training_rows": state["training_rows"],
        "total_readings_in_db": total,
        "readings_until_next_retrain": max(0, RETRAIN_EVERY - (total - state["last_count_checked"])),
        "is_training": state["is_training"],
    })


@app.route("/predict", methods=["POST"])
def predict():
    if state["model"] is None:
        return jsonify({"error": "Model not trained yet. Need at least 30 readings."}), 400

    data = request.get_json()
    try:
        features = [[
            float(data["temperature"]),
            float(data["humidity"]),
            float(data["pm25"]),
            float(data["pm10"]),
            float(data["mq135"]),
            float(data["mq7"]),
        ]]
        pred_idx = state["model"].predict(features)[0]
        pred_label = state["encoder"].inverse_transform([pred_idx])[0]
        pred_proba = state["model"].predict_proba(features)[0]
        confidence = round(max(pred_proba) * 100, 1)

        return jsonify({
            "predicted_label": pred_label,
            "confidence_pct": confidence,
            "model_accuracy": state["accuracy"],
            "trained_at": state["trained_at"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/retrain-now", methods=["POST"])
def retrain_now():
    success = train_model()
    return jsonify({
        "success": success,
        "accuracy": state["accuracy"],
        "cv_accuracy": state["cv_accuracy"],
        "trained_at": state["trained_at"],
    })


@app.route("/ingest", methods=["POST"])
def ingest():
    """Node backend calls this for every ESP32 reading"""
    data = request.get_json()
    required = ['temperature', 'humidity', 'pm25', 'pm10', 'mq135', 'mq7', 'aqi', 'aqi_label']
    if not all(k in data for k in required):
        return jsonify({"error": "Missing fields", "required": required}), 400

    insert_reading(data)
    count = count_readings()
    return jsonify({"status": "ok", "total_readings": count})


# ════════════════════════════════════════════════════════════
#  STARTUP
# ════════════════════════════════════════════════════════════
init_db()
seed_from_csv_if_empty()
load_existing_model()

bg_thread = threading.Thread(target=background_retrain_checker, daemon=True)
bg_thread.start()
print("[Startup] Background auto-retrain checker started")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
