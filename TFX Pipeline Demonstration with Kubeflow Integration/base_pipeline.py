import tensorflow_transform as tft
import tensorflow_data_validation as tfdv
import tensorflow_model_analysis as tfma

from tfx.components import (StatisticsGen, SchemaGen, ExampleValidator, CsvExampleGen,
                           StatisticsGen, Transform, Tuner, Trainer, Evaluator, Pusher)
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing



# Identify Preset Parameters
TRAIN_STEPS = 1000
EVAL_STEPS = 100

# Identify Pipeline Components' Instantiation
def init_components(
    data_dir,
    module_file,
    training_steps = TRAIN_STEPS,
    eval_steps = EVAL_STEPS,
    serving_model_dir = None,
):

    # define the input configuration of the pre-existing splits
    
    
    input_config = example_gen_pb2.Input(splits=[
        example_gen_pb2.Input.Split(name='train', pattern='span-{SPAN}/train/*.csv'),
        example_gen_pb2.Input.Split(name='eval', pattern='span-{SPAN}/eval/*.csv')
    ])

    # Create the CsvExampleGen component
    example_gen = CsvExampleGen(input_base=data_dir, 
                                input_config=input_config)

    # Create a StatisticsGen Component
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'],
                                   exclude_splits=['eval'])

    # Create a SchemaGen Component
    schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'], 
                           infer_feature_shape=True, exclude_splits=['eval'])

    # Create an ExampleValidator Component
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
        )

    # Create a Transform Component
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema = schema_gen.outputs['schema'],
        module_file=module_file
    )

    # Create a Tuner Component
    tuner = Tuner(
        module_file = module_file,
        examples = transform.outputs['transformed_examples'],
        transform_graph = transform.outputs['transform_graph'],
        train_args = trainer_pb2.TrainArgs(num_steps=20),
        eval_args = trainer_pb2.EvalArgs(num_steps=5))

    # Create a Trainer Component
    trainer = Trainer(
        module_file = module_file,
        examples = transform.outputs['transformed_examples'],
        transform_graph = transform.outputs["transform_graph"],
        schema = schema_gen.outputs["schema"],
        
        hyperparameters = tuner.outputs["best_hyperparameters"],
        train_args = trainer_pb2.TrainArgs(num_steps=training_steps),
        eval_args = trainer_pb2.EvalArgs(num_steps=eval_steps)
    )


    # Get the latest blessed model
    model_resolver = Resolver(
        strategy_class = latest_blessed_model_resolver.LatestBlessedModelResolver,
        model = Channel(type=Model),
        model_blessing = Channel(type=ModelBlessing)
        ) #.with_id('latest_blessed_model_resolver')


    # Create an Evaluator Component
    # 1. define evaluation configuration
    eval_config = tfma.EvalConfig(
        model_specs = [tfma.ModelSpec(label_key="charges")],
        slicing_specs = [tfma.SlicingSpec()],
        metrics_specs = [
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(class_name="MeanAbsoluteError"),
                    tfma.MetricConfig(class_name="MeanSquaredError"),
                    tfma.MetricConfig(class_name="RootMeanSquaredError"),
                    tfma.MetricConfig(class_name="ExampleCount")
                        ],
                thresholds = {
                    "mean_absolute_error": tfma.MetricThreshold(
                                                value_threshold = tfma.GenericValueThreshold(lower_bound={'value': 50}),
                                                change_threshold = tfma.GenericChangeThreshold(
                                                                        direction = tfma.MetricDirection.LOWER_IS_BETTER,
                                                                        absolute = {'value': 0.00}
                                                                                            )
                                                            )
                            }
                            )
                        ]
                                )

    # 2. define Evaluator component
    evaluator = Evaluator(
        examples = example_gen.outputs["examples"],
        model = trainer.outputs["model"],
        baseline_model = model_resolver.outputs["model"],
        eval_config = eval_config
    )

    # Create a Pusher Component
    pusher = Pusher(
        model = trainer.outputs["model"],
        model_blessing = evaluator.outputs["blessing"],
        push_destination = pusher_pb2.PushDestination(
            filesystem = pusher_pb2.PushDestination.Filesystem(base_directory = serving_model_dir)
                                                    )
    )

    components = [example_gen,
                  statistics_gen,
                  schema_gen,
                  example_validator,
                  transform,
                  tuner,
                  trainer,
                  model_resolver,
                  evaluator,
                  pusher
                  ]
    
    return components