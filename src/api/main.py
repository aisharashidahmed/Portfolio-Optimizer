# src/api/main.py

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from src.tasks import run_optimization
from pydantic import BaseModel
from typing import List, Optional
import yfinance as yf
import pandas as pd
import numpy as np
import redis
import json

# Import our Engine Room machines (the robots we built)
from src.algorithms.hrp import HRPOptimizer
from src.algorithms.mvo import MVOOptimizer


# --- 1. The Mailbox (Pydantic Models) ---
class OptimizationRequest(BaseModel):
    tickers: List[str]
    start_date: str
    end_date: str
    algorithm: str
    max_weight: Optional[float] = 0.20

class OptimizationResponse(BaseModel):
    tickers: List[str]
    weights: List[float]
    algorithm: str
    status: str

# --- 2. The Front Door (FastAPI App) - THIS IS THE "app" THE ELECTRICIAN IS LOOKING FOR! ---
app = FastAPI(title="Portfolio Optimizer Engine", version="1.0.0")
# --- CORS: Allow requests from browsers ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# --- 3. Helper: Data Fetcher ---
def fetch_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)['Adj Close']
    daily_returns = data.pct_change().dropna()
    cov_matrix = daily_returns.cov() * 252
    return daily_returns, cov_matrix

# --- 4. The Dispatcher (The Endpoint) ---

@app.post("/optimize", response_model=dict)
async def optimize_portfolio(request: OptimizationRequest):
    """
    Submits an optimization request to the kitchen.
    Returns a task_id immediately so the user can check status later.
    """
    # 1. Validate the algorithm
    if request.algorithm not in ["hrp", "mvo"]:
        raise HTTPException(status_code=400, detail="Algorithm must be 'hrp' or 'mvo'")

    # 2. Submit the task to Celery (put it on the Redis rack)
    task = run_optimization.delay(
        tickers=request.tickers,
        start_date=request.start_date,
        end_date=request.end_date,
        algorithm=request.algorithm,
        max_weight=request.max_weight
    )

    # 3. Hand the customer their receipt (task_id)
    return {
        "task_id": task.id,
        "status": "Processing started",
        "message": "Use GET /status/{task_id} to check the result."
    }

import redis
import json

redis_client = redis.Redis(host='localhost', port=6379, db=0)

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """
    Checks the status of a submitted task by reading directly from Redis.
    """
    # Try to get the result from Redis
    result_data = redis_client.get(f"task:{task_id}")
    
    if result_data is None:
        # Check if the task is still pending or failed via Celery
        task = run_optimization.AsyncResult(task_id)
        if task.pending:
            return {"task_id": task_id, "status": "PENDING", "result": None}
        elif task.failed():
            return {"task_id": task_id, "status": "FAILED", "result": str(task.info)}
        else:
            return {"task_id": task_id, "status": "UNKNOWN", "result": None}
    
    # Decode the result
    try:
        result = json.loads(result_data.decode('utf-8'))
        return {
            "task_id": task_id,
            "status": "SUCCESS",
            "result": result
        }
    except:
        return {"task_id": task_id, "status": "ERROR", "result": "Failed to parse result"}

# --- 5. Health Check ---
@app.get("/health")
async def health_check():
    return {"status": "Control Room is operational!"}