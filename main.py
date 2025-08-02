from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import create_engine, Session
from inputs import attack_description
import pandas as pd
import numpy as np
import joblib
import asyncio
import re
from typing import List
from contextlib import asynccontextmanager
import os

# XAMPP Apache log path
APACHE_LOG_PATH = r"C:/xampp/apache/logs/access.log"

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        print("connection open")
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        print("connection closed")
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        print("Broadcasting:", message)
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# Load models safely
def get_model(prefix: str) -> dict:
    model = joblib.load("objects.pkl")
    if prefix == 'attack_detector':
        model.pop('attack_classifer', None)
    elif prefix == 'attack_classifier':
        model.pop('attack_detector', None)
    else:
        raise ValueError("Invalid model prefix")
    return model

# Feature preparation for the ML models
def prepare_features_attack(data_dict, tfidf, scaler):
    df = pd.DataFrame([data_dict])
    df['timestamp_parsed'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['hour'] = df['timestamp_parsed'].dt.hour
    df['minute'] = df['timestamp_parsed'].dt.minute
    df['second'] = df['timestamp_parsed'].dt.second
    df['day_of_week'] = df['timestamp_parsed'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['is_night'] = df['hour'].apply(lambda h: 1 if h < 6 or h >= 22 else 0)
    df['method'] = df['method'].apply(lambda m: 1 if m == 'POST' else 0)

    numeric_cols = ['status', 'size', 'hour', 'minute', 'second',
                    'day_of_week', 'is_weekend', 'is_night', 'method']
    X_numeric = df[numeric_cols].fillna(0)

    X_scaled = scaler.transform(X_numeric[['status', 'size']].values)
    other_features = X_numeric.drop(['status', 'size'], axis=1).values
    url_vec = tfidf.transform(df['url'].fillna('')).toarray()

    return np.hstack([X_scaled, other_features, url_vec])

# Tail and process Apache logs
async def tail_apache_logs():
    if not os.path.exists(APACHE_LOG_PATH):
        print(f"Log file {APACHE_LOG_PATH} does not exist.")
        return

    with open(APACHE_LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                await asyncio.sleep(1)
                continue

            log_parts = line.split()
            if len(log_parts) < 10:
                continue

            try:
                ip = log_parts[0]
                method = log_parts[5].strip('"')
                url = log_parts[6]
                status = int(log_parts[8]) if log_parts[8].isdigit() else 200
                size = int(log_parts[9]) if log_parts[9].isdigit() else 0
                timestamp = pd.Timestamp.now().isoformat()

                data_dict = {
                    "timestamp": timestamp,
                    "method": method,
                    "status": status,
                    "size": size,
                    "url": url
                }

                model_objects = get_model('attack_detector')
                input_x = prepare_features_attack(data_dict, model_objects['vectorizer'], model_objects['scaler'])
                is_attack = model_objects['attack_detector'].predict(input_x)[0]

                if is_attack == 1:
                    model_objects = get_model('attack_classifier')
                    input_x = prepare_features_attack(data_dict, model_objects['vectorizer'], model_objects['scaler'])
                    attack_type = model_objects['attack_classifer'].predict(input_x)[0]
                    message = f"⚠️ Attack detected from {ip} | Type: {attack_type}"
                else:
                    message = f"✅ Normal traffic from {ip}"

                await manager.broadcast(message)

            except Exception as e:
                await manager.broadcast(f"🚫 Error: {str(e)}")

# Lifespan lifecycle hook
@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(tail_apache_logs())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        print("Log tailing task cancelled")

# App setup
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

@app.websocket("/ws/attack_logs")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post('/detect_attack')
def attack_detector(data_dict: attack_description):
    try:
        model_objects = get_model('attack_detector')
        input_x = prepare_features_attack(data_dict.model_dump(), model_objects['vectorizer'], model_objects['scaler'])
        y_pred = model_objects['attack_detector'].predict(input_x)[0]
        return {'status': 'success', 'attack': 'Yes' if y_pred == 1 else 'No'}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

@app.post('/classify_attack')
def attack_classifier(data_dict: attack_description):
    try:
        model_objects = get_model('attack_classifier')
        input_x = prepare_features_attack(data_dict.model_dump(), model_objects['vectorizer'], model_objects['scaler'])
        y_pred = model_objects['attack_classifer'].predict(input_x)[0]
        return {'status': 'success', 'attack_type': y_pred}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}
    
    
