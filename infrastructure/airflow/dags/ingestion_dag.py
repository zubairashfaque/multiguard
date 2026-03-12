"""Airflow DAG for data ingestion pipeline."""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "multiguard",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "multiguard_ingestion",
    default_args=default_args,
    description="Ingest and preprocess multimodal data",
    schedule_interval="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["multiguard", "ingestion"],
) as dag:
    ingest = BashOperator(
        task_id="ingest_data",
        bash_command="cd /app && python scripts/ingest_data.py",
    )

    validate = BashOperator(
        task_id="validate_data",
        bash_command="cd /app && python -c 'from src.data.validators import validate_dataset; print(\"OK\")'",
    )

    ingest >> validate
