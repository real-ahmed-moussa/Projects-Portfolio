# Import the required libraries
import os
from typing import Text

from tfx.orchestration import metadata, pipeline
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from base_pipeline import init_components

# [1] Pipeline Name
pipeline_name = "charges_pipeline_local"

# [2] Pipeline Inputs
airflow_dir = "path to airflow directory"
pipeline_dir = "path to pipeline directory"
data_dir = "path to data directory"
module_file = "path to module.py file"

# [3] Pipeline Outputs
pipeline_root = "path to pipeline root directory"
metadata_path = "path to metadata.sqlite directory"
serving_model_dir = "path to serving model directory"

# [4] Pipeline Instantiation Function
def init_pipeline(components, pipeline_root: Text):
    
    p = pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(metadata_path),
    )
    return p

# [5] Retrieve Pipeline Components from base_pipeline.py file
components = init_components(
    data_dir,
    module_file,
    training_steps=1000,
    eval_steps=100,
    serving_model_dir=serving_model_dir
)

# [6] Create the Pipeline
pipeline = init_pipeline(components, pipeline_root)


# [7] Run the Pipeline Locally
if __name__ == '__main__':
    print("Running pipeline locally...")
    LocalDagRunner().run(pipeline)

