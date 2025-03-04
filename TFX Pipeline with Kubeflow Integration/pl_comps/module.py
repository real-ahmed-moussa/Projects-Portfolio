import os
from typing import NamedTuple, Dict, Any
from typing import Text

import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs
from keras_tuner.engine import base_tuner
from keras_tuner import HyperParameters, RandomSearch
import kerastuner


label_key = "charges"

#####################
# Transformation Code
#####################

# Categorical Features
categorical_features = {
    "region": 4,
    "sex": 2,
    "smoker": 2
}
# Numerical Features
numerical_features = {
    "age": None,
    "bmi": None,
    "children": None
}


# [1] Function to Define Transformed Features
def transformed_name(key):
    return key + '_xf'

# [2] Function to Transform Sparse to Dense Features and to Fill-in Missing Values
def fill_in_missing(x):
    
    if x is None:
        return tf.constant([], dtype=tf.float32)  # or a suitable default constant value
    
    elif isinstance(x, tf.sparse.SparseTensor):
        # Define a default value based on dtype
        default_value = "" if x.dtype==tf.string else 0
        
        # Convert the SparseTensor to Dense with default values for missing entries
        x = tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
            default_value
        )
    
    return tf.squeeze(x, axis=1)


# [3] Function to Perform One-hot Encoding
def convert_num_to_one_hot(label_tensor, num_labels=2):
    one_hot_tensor = tf.one_hot(label_tensor, num_labels)
    return tf.reshape(one_hot_tensor, [-1, num_labels])

# [4] Function to Perform Scaling
def scaler_func(value_tensor):
    return tft.scale_to_0_1(value_tensor)


# [5] Define the Preprocessing Function
def preprocessing_fn(inputs: tf.Tensor) -> tf.Tensor:
    outputs = {}
    
    # Categorical Features
    for key in categorical_features.keys():
        dim = categorical_features[key]
        
        # Convert Categorical Features to Integer IDs
        int_value = tft.compute_and_apply_vocabulary(
            fill_in_missing(inputs[key]), 
            top_k=dim)
        # Apply One-hot Encoding
        outputs[transformed_name(key)] = convert_num_to_one_hot(int_value, num_labels=dim)
        
    # Numerical Features
    for key in numerical_features.keys():
        # Apply Scaling to Numerical Features Directly
        outputs[transformed_name(key)] = scaler_func(fill_in_missing(inputs[key]))
    
    # Output Label
    outputs[transformed_name(label_key)] = fill_in_missing(inputs[label_key])
    
    # Return the values
    return outputs




############
# Model Code
############

# [1] function to define model architecture
def get_model(hp: HyperParameters, show_summary: bool=True) -> tf.keras.models.Model:
    
    # one-hot categorical features
    categorical_inputs = []
    for key, dim in categorical_features.items():
        categorical_inputs.append(
            tf.keras.Input(shape=(dim,), name=transformed_name(key))
        )
    
    # normalized numerical features
    numerical_inputs = []
    for key in numerical_features.keys():
        numerical_inputs.append(
            tf.keras.Input(shape=(1,), name=transformed_name(key))
        )
    
    # define deep network for categorical features
    deep = tf.keras.layers.concatenate(categorical_inputs)
    deep = tf.keras.layers.Dense(
        units = hp.Int("deep_units_1", min_value=64, max_value=256, step=64),
        activation = "relu"
        )(deep)
    deep = tf.keras.layers.Dense(
        units = hp.Int("deep_units_2", min_value=32, max_value=128, step=32),
        activation = "relu")(deep)
    deep = tf.keras.layers.Dense(
        units = hp.Int("deep_units_3", min_value=16, max_value=64, step=16),
        activation = "relu"
        )(deep)
    
    # define wide network for numerical features
    wide = tf.keras.layers.concatenate(numerical_inputs)
    wide = tf.keras.layers.Dense(16, activation="relu")(wide)
    
    # combine deep and wide parts
    both = tf.keras.layers.concatenate([deep, wide])
    
    # output regression layer
    output = tf.keras.layers.Dense(1, activation=None)(both)
    
    # combine all inputs
    inputs = categorical_inputs + numerical_inputs
    
    # define and compile the model
    keras_model = tf.keras.models.Model(inputs, output)
    keras_model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")),
        loss = "mean_squared_error",
        metrics = [tf.keras.metrics.MeanAbsoluteError(),
                   tf.keras.metrics.MeanSquaredError(),
                   tf.keras.metrics.RootMeanSquaredError()]
    )
    
    # display summary - if specified
    if show_summary:
        keras_model.summary()
    
    return keras_model

# [2] function to parse serialized tf.Example
def get_serve_tf_examples_fn(model, tf_transform_output):
    
    model.tft_layer = tf_transform_output.transform_features_layer()
    
    # [3] function to return the output used in serving signature
    @tf.function 
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(label_key)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        
        transformed_features = model.tft_layer(parsed_features)
        
        outputs = model(transformed_features)
        return {"outputs": outputs}
    
    return serve_tf_examples_fn

# [3] function that creates a record reader that can create gzip'ed files
def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")

# [4] function to generate features and labels for tuning/training
def input_fn(file_pattern, tf_transform_output, batch_size=64):
    
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=gzip_reader_fn,
        label_key=transformed_name(label_key)
    )
    return dataset

# [5] TFX tuner component
TunerFnResult = NamedTuple("TunerFnResult", [("tuner", base_tuner.BaseTuner),
                                             ("fit_kwargs", Dict[Text, Any])
                            ])
def tuner_fn(fn_args) -> TunerFnResult:
    
    # Load the Transform Graph
    tft_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    
    # Use Existing Input Function to Load Training and Evaluation Data
    train_data = input_fn(
        file_pattern = fn_args.train_files,
        tf_transform_output = tft_output,
        batch_size = 64
    )
    eval_data = input_fn(
        file_pattern = fn_args.eval_files,
        tf_transform_output = tft_output,
        batch_size = 64
    )
    
    
    # Create Tuner Instance
    tuner = RandomSearch(
        hypermodel = get_model,
        objective = "val_mean_absolute_error",
        max_trials = 10,
        executions_per_trial = 2,
        directory = fn_args.working_dir,
        project_name = "model_tuning"
    )
    
    # Arguments for the Tuner's Fitting
    fit_kwargs = {
        "x": train_data,
        "validation_data": eval_data,
        "steps_per_epoch": fn_args.train_steps,
        "validation_steps": fn_args.eval_steps,
        "epochs": 10
    }
    
    return TunerFnResult(tuner=tuner, fit_kwargs=fit_kwargs)
    



# [6] TFX model training function
def run_fn(fn_args: FnArgs):
    
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    
    train_dataset = input_fn(fn_args.train_files, tf_transform_output, 64)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, 64)
    
    # Load the Hyperparameters from the Tuner
    hp = kerastuner.HyperParameters.from_config(fn_args.hyperparameters)
    
    model = get_model(hp=hp)
    
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq="batch"
    )
    
    model.fit(
        train_dataset,
        epochs = 1,
        steps_per_epoch = fn_args.train_steps,
        validation_data = eval_dataset,
        validation_steps = fn_args.eval_steps,
        callbacks = [tensorboard_callback]
    )
    
    signatures = {
        "serving_default": get_serve_tf_examples_fn
        (
            model, tf_transform_output
        ).get_concrete_function
        (
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        )
    }
    
    model.save(fn_args.serving_model_dir, save_format="tf", signatures=signatures)