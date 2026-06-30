"""
============================================================
 AQI Monitor — Python ML Auto-Retrain Microservice
 Student : Anantharajan Vel Murugan | 294FAVZE | UoH
 Version : No-pandas (avoids C-compile issues on Render)
============================================================
"""

import os
import csv
import time
import sqlite3
import threading
import numpy as np
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

RETRAIN_EVERY = int(os.environ.get("RETRAIN_EVERY", "100"))
DB_PATH = "/tmp/aqi_data.db"
MODEL_PATH = "/tmp/rf_model.pkl"
ENCODER_PATH = "/tmp/label_encoder.pkl"

FEATURES = ['temperature', 'humidity', 'pm25', 'pm10', 'mq135', 'mq7']

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            temperature REAL, humidity REAL, pm25 REAL, pm10 REAL,
            mq135 REAL, mq7 REAL, aqi REAL, aqi_label TEXT,
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
    c = conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
    conn.close()
    return c

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
    """Returns numpy arrays X, y without pandas"""
    conn = get_db()
    cur = conn.execute(
        "SELECT temperature, humidity, pm25, pm10, mq135, mq7, aqi_label FROM readings"
    )
    rows = cur.fetchall()
    conn.close()
    return rows

def seed_from_csv_if_empty(csv_path="seed_data.csv"):
    if count_readings() > 0:
        return
    if not os.path.exists(csv_path):
        print("[Seed] No seed_data.csv found — starting with empty DB")
        return
    try:
        conn = get_db()
        loaded = 0
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    keys = {k.lower().strip(): v for k, v in row.items()}
                    conn.execute("""
                        INSERT INTO readings
                        (temperature, humidity, pm25, pm10, mq135, mq7, aqi, aqi_label, received_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        float(keys['temperature']), float(keys['humidity']),
                        float(keys['pm25']), float(keys['pm10']),
                        float(keys['mq135']), float(keys['mq7']),
                        float(keys['aqi']), str(keys['aqi_label']).upper(),
                        datetime.utcnow().isoformat() + "Z"
                    ))
                    loaded += 1
                except Exception:
                    continue
        conn.commit()
        conn.close()
        print(f"[Seed] Loaded {loaded} rows from {csv_path}")
    except Exception as e:
        print(f"[Seed] Error loading CSV: {e}")


state = {
    "model": None, "encoder": None, "accuracy": None, "cv_accuracy": None,
    "trained_at": None, "training_rows": 0, "last_count_checked": 0,
    "is_training": False,
}

def train_model():
    if state["is_training"]:
        return False
    state["is_training"] = True
    print(f"[Train] Starting at {datetime.now()}")

    try:
        rows = fetch_all_readings()
        if len(rows) < 30:
            print(f"[Train] Not enough data yet ({len(rows)} rows) — need 30+")
            state["is_training"] = False
            return False

        X_list, y_list = [], []
        seen = set()
        for r in rows:
            try:
                temp, hum, pm25, pm10, mq135, mq7, label = r
                temp, hum, pm25, pm10, mq135, mq7 = map(float, [temp, hum, pm25, pm10, mq135, mq7])
                if temp <= 0 or hum <= 0:
                    continue
                key = (temp, hum, pm25, pm10, mq135, mq7)
                if key in seen:
                    continue
                seen.add(key)
                X_list.append([temp, hum, pm25, pm10, mq135, mq7])
                y_list.append(str(label).upper().strip())
            except Exception:
                continue

        if len(X_list) < 30:
            print(f"[Train] Not enough clean data ({len(X_list)} rows)")
            state["is_training"] = False
            return False

        X = np.array(X_list)
        y = np.array(y_list)

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
        state["training_rows"] = len(X_list)
        state["last_count_checked"] = count_readings()

        print(f"[Train] Done! Accuracy={state['accuracy']}% CV={state['cv_accuracy']}% Rows={len(X_list)}")
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


@app.route("/")
def home():
    return jsonify({
        "service": "AQI ML Auto-Retrain Microservice",
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
            float(data["temperature"]), float(data["humidity"]),
            float(data["pm25"]), float(data["pm10"]),
            float(data["mq135"]), float(data["mq7"]),
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
        "success": success, "accuracy": state["accuracy"],
        "cv_accuracy": state["cv_accuracy"], "trained_at": state["trained_at"],
    })


@app.route("/ingest", methods=["POST"])
def ingest():
    data = request.get_json()
    required = ['temperature', 'humidity', 'pm25', 'pm10', 'mq135', 'mq7', 'aqi', 'aqi_label']
    if not all(k in data for k in required):
        return jsonify({"error": "Missing fields", "required": required}), 400
    insert_reading(data)
    return jsonify({"status": "ok", "total_readings": count_readings()})


init_db()
seed_from_csv_if_empty()
load_existing_model()

bg_thread = threading.Thread(target=background_retrain_checker, daemon=True)
bg_thread.start()
print("[Startup] Background auto-retrain checker started")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
