"""
============================================================
 AQI Monitor — Python ML Auto-Retrain Microservice
 Student : Anantharajan Vel Murugan | 294FAVZE | UoH
 
 PURPOSE:
 Separate Python service that:
 1. Reads sensor readings from MongoDB
 2. Auto-retrains Random Forest every N new readings
 3. Saves trained model + accuracy
 4. Exposes /predict and /status endpoints for Node backend
============================================================
"""

import os
import time
import threading
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

app = Flask(__name__)
CORS(app)

# ── Config ────────────────────────────────────────────────
MONGO_URI = os.environ.get("MONGO_URI", "")
RETRAIN_EVERY = int(os.environ.get("RETRAIN_EVERY", "100"))  # retrain every N new readings
MODEL_PATH = "/tmp/rf_model.pkl"
ENCODER_PATH = "/tmp/label_encoder.pkl"

mongo_client = MongoClient(MONGO_URI) if MONGO_URI else None
db = mongo_client["aqi_monitor"] if mongo_client else None
readings_col = db["readings"] if db is not None else None

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

FEATURES = ['temperature', 'humidity', 'pm25', 'pm10', 'mq135', 'mq7']

# ════════════════════════════════════════════════════════════
#  TRAINING FUNCTION
# ════════════════════════════════════════════════════════════
def train_model():
    if state["is_training"]:
        return False
    state["is_training"] = True
    print(f"[Train] Starting at {datetime.now()}")

    try:
        # Pull all readings from MongoDB
        docs = list(readings_col.find({}, {"_id": 0}))
        if len(docs) < 30:
            print(f"[Train] Not enough data yet ({len(docs)} rows) — need 30+")
            state["is_training"] = False
            return False

        df = pd.DataFrame(docs)

        # Clean
        for c in FEATURES + ['aqi']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df['aqi_label'] = df['aqi_label'].astype(str).str.upper().str.strip()
        df.dropna(subset=FEATURES + ['aqi', 'aqi_label'], inplace=True)
        df = df[df['temperature'] > 0]
        df = df[df['humidity'] > 0]
        df.drop_duplicates(inplace=True)

        if len(df) < 30:
            print(f"[Train] Not enough clean data ({len(df)} rows)")
            state["is_training"] = False
            return False

        X = df[FEATURES]
        y = df['aqi_label']

        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        # Guard: need at least 2 classes
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

        # Save model + encoder
        joblib.dump(clf, MODEL_PATH)
        joblib.dump(le, ENCODER_PATH)

        # Update state
        state["model"] = clf
        state["encoder"] = le
        state["accuracy"] = round(acc * 100, 2)
        state["cv_accuracy"] = round(cv_scores.mean() * 100, 2)
        state["trained_at"] = datetime.utcnow().isoformat() + "Z"
        state["training_rows"] = len(df)
        state["last_count_checked"] = len(docs)

        print(f"[Train] Done! Accuracy={state['accuracy']}% CV={state['cv_accuracy']}% Rows={len(df)}")
        state["is_training"] = False
        return True

    except Exception as e:
        print(f"[Train] ERROR: {e}")
        state["is_training"] = False
        return False


def load_existing_model():
    """Load previously saved model on startup if it exists"""
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
            state["model"] = joblib.load(MODEL_PATH)
            state["encoder"] = joblib.load(ENCODER_PATH)
            print("[Startup] Loaded existing model from disk")
    except Exception as e:
        print(f"[Startup] Could not load existing model: {e}")


# ════════════════════════════════════════════════════════════
#  BACKGROUND THREAD — checks every 30s if retrain needed
# ════════════════════════════════════════════════════════════
def background_retrain_checker():
    while True:
        try:
            if readings_col is not None:
                current_count = readings_col.count_documents({})
                if current_count - state["last_count_checked"] >= RETRAIN_EVERY:
                    print(f"[Auto-retrain] Trigger: {current_count} readings "
                          f"(last trained at {state['last_count_checked']})")
                    train_model()
                elif state["trained_at"] is None and current_count >= 30:
                    # First-time training
                    print(f"[Auto-retrain] First training with {current_count} readings")
                    train_model()
        except Exception as e:
            print(f"[Auto-retrain] Checker error: {e}")
        time.sleep(30)  # check every 30 seconds


# ════════════════════════════════════════════════════════════
#  API ENDPOINTS
# ════════════════════════════════════════════════════════════

@app.route("/")
def home():
    return jsonify({
        "service": "AQI ML Auto-Retrain Microservice",
        "student": "Anantharajan Vel Murugan | 294FAVZE | UoH",
        "routes": ["/status", "/predict (POST)", "/retrain-now (POST)"],
    })


@app.route("/status")
def status():
    total_readings = readings_col.count_documents({}) if readings_col is not None else 0
    return jsonify({
        "model_trained": state["model"] is not None,
        "accuracy": state["accuracy"],
        "cv_accuracy": state["cv_accuracy"],
        "trained_at": state["trained_at"],
        "training_rows": state["training_rows"],
        "total_readings_in_db": total_readings,
        "readings_until_next_retrain": max(0, RETRAIN_EVERY - (total_readings - state["last_count_checked"])),
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
    """Manual trigger for testing"""
    success = train_model()
    return jsonify({
        "success": success,
        "accuracy": state["accuracy"],
        "cv_accuracy": state["cv_accuracy"],
        "trained_at": state["trained_at"],
    })


@app.route("/ingest", methods=["POST"])
def ingest():
    """Node backend calls this to forward each reading into MongoDB"""
    if readings_col is None:
        return jsonify({"error": "MongoDB not configured"}), 500

    data = request.get_json()
    required = ['temperature', 'humidity', 'pm25', 'pm10', 'mq135', 'mq7', 'aqi', 'aqi_label']
    if not all(k in data for k in required):
        return jsonify({"error": "Missing fields"}), 400

    data['received_at'] = datetime.utcnow().isoformat() + "Z"
    readings_col.insert_one(data)

    count = readings_col.count_documents({})
    return jsonify({"status": "ok", "total_readings": count})


# ════════════════════════════════════════════════════════════
#  STARTUP
# ════════════════════════════════════════════════════════════
load_existing_model()

if readings_col is not None:
    bg_thread = threading.Thread(target=background_retrain_checker, daemon=True)
    bg_thread.start()
    print("[Startup] Background auto-retrain checker started")
else:
    print("[Startup] WARNING: MONGO_URI not set — auto-retrain disabled")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
