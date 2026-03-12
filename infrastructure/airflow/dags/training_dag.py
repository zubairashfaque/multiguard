"""Airflow DAG for model training pipeline."""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "multiguard",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    "multiguard_training",
    default_args=default_args,
    description="Train multimodal classification model",
    schedule_interval=None,  # Triggered manually or by drift detection
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["multiguard", "training"],
) as dag:
    train_baseline = BashOperator(
        task_id="train_baseline",
        bash_command="cd /app && python scripts/train.py --config configs/train/baseline.yaml",
    )

    evaluate = BashOperator(
        task_id="evaluate",
        bash_command="cd /app && python scripts/evaluate.py --config configs/experiment/baseline.yaml",
    )

    export = BashOperator(
        task_id="export_model",
        bash_command="cd /app && python scripts/export_model.py --format onnx",
    )

    train_baseline >> evaluate >> export
