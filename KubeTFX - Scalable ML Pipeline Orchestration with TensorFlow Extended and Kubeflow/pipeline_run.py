# Import Libraries and Functions
import os
from typing import Text
from absl import logging

from kfp import onprem

from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.orchestration import pipeline
from base_pipeline import init_components


# [1] Pipeline Name and Specs
pipeline_name = "tfx_pipeline_kubeflow"

persistent_volume_claim = "tfx-pvc"                                                                     # Name of the PVC
persistent_volume = "tfx-pv"                                                                            # Name of the volume
persistent_volume_mount = "/home/ahmedmoussa/kf_tfx/pl_comps"                                           # This path must exist in your Minikube container

#  .yaml file for Kubeflow Pipelines
output_filename = f"{pipeline_name}.yaml"                                                               # Name of the output .yaml file that will be loaded into Kubeflow Pipelines
output_dir = "/home/ahmedmoussa/kf_tfx/pl_yaml_output"                                                  # This path must exist in your Minikube container

# [2] Pipeline Inputs
data_dir = os.path.join(persistent_volume_mount, "data")                                                # Path to the data directory
module_file = os.path.join(persistent_volume_mount, 'module.py')                                        # Path to the module file

# [3] Pipeline Outputs
output_base = os.path.join(persistent_volume_mount, 'output')                                           # Path to the output directory
serving_model_dir = os.path.join(output_base, "serving_model_dir")                                      # Path to the serving model directory (for serving model)

# [4] Kubeflow Pipeline Instantiation Function
def init_kubeflow_pipeline(components, pipeline_root: Text, direct_num_workers: int):
    
    logging.info(f"Pipeline root was set to {pipeline_root}")
    beam_arg = [
        f"--direct_num_workers={direct_num_workers}",
        "--direct_running_mode=multi_processing",
    ]
    p = pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        beam_pipeline_args=beam_arg
    )
    return p

# [5] Main Pipeline Function
if __name__ == "__main__":
    
    # 1. configure the logging
    logging.set_verbosity(logging.INFO)
    
    # 2. set module path
    module_path = os.getcwd()
    
    # 3. get default metadata configuration
    metadata_config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()
    
    # 4. load pipeline components
    components = init_components(
        data_dir=data_dir,
        module_file=module_file,
        serving_model_dir=serving_model_dir,
        training_steps=500,
        eval_steps=100
    )
    
    # 5. pipeline runner configuration
    runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
        kubeflow_metadata_config=metadata_config,
        pipeline_operator_funcs=(
            # Add default pipeline operator functions for GKE if applicable,
            # and include mounting of the Persistent Volume Claim (PVC).
            kubeflow_dag_runner.get_default_pipeline_operator_funcs() + [
                onprem.mount_pvc(
                    persistent_volume_claim,                                                            # Name of the PVC
                    persistent_volume,                                                                  # Name of the volume
                    persistent_volume_mount                                                             # Mount path inside the container
                )
            ]
        ),
    )

    # 6. kubeflow pipeline
    p = init_kubeflow_pipeline(components, output_base, direct_num_workers=0)
    output_filename = f"{pipeline_name}.yaml"
    kubeflow_dag_runner.KubeflowDagRunner(
        config = runner_config,
        output_dir = output_dir,
        output_filename = output_filename
    ).run(p)