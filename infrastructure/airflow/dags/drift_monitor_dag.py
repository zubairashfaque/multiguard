"""Airflow DAG for drift monitoring."""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "multiguard",
    "depends_on_past": False,
    "email_on_failure": True,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "multiguard_drift_monitor",
    default_args=default_args,
    description="Monitor data and prediction drift",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["multiguard", "monitoring"],
) as dag:
    check_drift = BashOperator(
        task_id="check_drift",
        bash_command="cd /app && python -c 'from src.monitoring.drift_detector import DriftDetector; print(\"Drift check placeholder\")'",
    )
