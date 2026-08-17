# src/tasks.py

from celery import Celery
import yfinance as yf
import pandas as pd
import numpy as np
import redis
import json

from src.algorithms.hrp import HRPOptimizer
from src.algorithms.mvo import MVOOptimizer
from src.database import SessionLocal, OptimizationRun

# Connect to Redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)

app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
)

@app.task(bind=True)
def run_optimization(self, tickers, start_date, end_date, algorithm, max_weight, user_id):
    task_id = self.request.id
    try:
        # A. Fetch data
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)['Adj Close']
        daily_returns = data.pct_change().dropna()

        if not isinstance(daily_returns, pd.DataFrame):
            daily_returns = pd.DataFrame(daily_returns)

        cov_matrix = daily_returns.cov() * 252

        # B. Pick the right machine
        if algorithm == "hrp":
            optimizer = HRPOptimizer()
        else:
            optimizer = MVOOptimizer(risk_free_rate=0.045, max_weight=max_weight)

        # C. Run the optimization
        weights = optimizer.optimize(daily_returns, cov_matrix)

        # D. Store in Redis (for immediate status checks)
        result = {
            "status": "success",
            "weights": weights.tolist(),
            "tickers": tickers,
            "algorithm": algorithm
        }
        redis_client.setex(f"task:{task_id}", 3600, json.dumps(result))

        # E. PERMANENT ARCHIVE: Save to PostgreSQL
        try:
            db = SessionLocal()
            db_record = OptimizationRun(
                user_id=user_id,  # <-- Now linked to the user!
                tickers=tickers,
                algorithm=algorithm,
                start_date=start_date,
                end_date=end_date,
                weights=weights.tolist()
            )
            db.add(db_record)
            db.commit()
            print(f"📁 [Archive] Successfully saved optimization to PostgreSQL!")
        except Exception as db_error:
            print(f"⚠️ [Archive] Failed to save to DB: {db_error}")
        finally:
            db.close()

        return result

    except Exception as e:
        error_result = {
            "status": "failure",
            "error": str(e)
        }
        redis_client.setex(f"task:{task_id}", 3600, json.dumps(error_result))
        return error_result