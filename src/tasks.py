"""Celery worker.

Optimizations run out-of-band because they hit the network for price history
and, for long date ranges, spend real CPU time in SLSQP. The task itself is a
thin wrapper -- all the logic lives in ``src.services.optimization`` so it can
be tested without a broker.
"""

import logging

from celery import Celery

from src.config import get_settings
from src.database import OptimizationRun, SessionLocal
from src.services.optimization import run_optimization as run_optimization_service

logger = logging.getLogger(__name__)
settings = get_settings()

celery_app = Celery(
    "portfolio",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=settings.result_ttl_seconds,
    task_track_started=True,
    # Fetch + optimize should never run for minutes; fail loudly instead of
    # occupying a worker forever.
    task_soft_time_limit=120,
    task_time_limit=180,
)


def _persist(task_id: str, user_id: int, request: dict, result: dict) -> None:
    """Archive a completed run. A persistence failure must not fail the task."""
    db = SessionLocal()
    try:
        db.add(
            OptimizationRun(
                task_id=task_id,
                user_id=user_id,
                tickers=result["tickers"],
                algorithm=result["algorithm"],
                start_date=request["start_date"],
                end_date=request["end_date"],
                weights=result["weights"],
                metrics=result["metrics"],
            )
        )
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Failed to archive optimization %s", task_id)
    finally:
        db.close()


@celery_app.task(bind=True, name="portfolio.optimize")
def optimize_task(
    self,
    tickers: list[str],
    start_date: str,
    end_date: str,
    algorithm: str,
    max_weight: float,
    user_id: int,
) -> dict:
    """Run one optimization and archive it.

    Returns the result dict. Celery's result backend stores it -- we do not
    write to Redis by hand, so there is one source of truth for task state.
    """
    result = run_optimization_service(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        algorithm=algorithm,
        risk_free_rate=settings.risk_free_rate,
        max_weight=max_weight,
    ).to_dict()

    _persist(
        self.request.id, user_id, {"start_date": start_date, "end_date": end_date}, result
    )

    return result
