from utils import data_utils

import tensorflow as tf
def create_train_input_fn(
        train_config, train_input_config, model_config):
    def _train_input_fn(params=None):
        x = None
        y = None

        dataset = tf.estimator.inputs.numpy_input_fn(
            x,
            y,
            batch_size=params.batch_size,
            num_threads=params.num_threads
        )
        return dataset

    return _train_input_fn