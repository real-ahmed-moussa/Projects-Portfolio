# Importe the required libraries
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

# [2] Define Airflow Default Args
default_args = {
    'owner': 'admin',
    'depends_on_past': False,
    'start_date': datetime(2024, 12, 1),
    'retries': 1,
}

# [3] Define the Airflow DAG
with DAG(
    'tfx_pipeline_dag',
    default_args=default_args,
    description='Run TFX pipeline using Airflow',
    schedule_interval=None,
) as dag:
    run_tfx_pipeline = BashOperator(
        task_id='run_tfx_pipeline',
        bash_command='source /path/to/virtual/environment/bin/activate && python3 path/to/pipeline_run.py',
    )
    
    run_tfx_pipeline
