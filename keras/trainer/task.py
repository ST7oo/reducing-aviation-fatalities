import argparse
import os
import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam

from . import util
from . import model


def get_args():
    """Argument parser.

    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='local or GCS location for writing checkpoints and exporting models')
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=10,
        help='number of times to go through the data, default=20')
    parser.add_argument(
        '--batch-size',
        default=128,
        type=int,
        help='number of records to read during each training step, default=128')
    parser.add_argument(
        '--learning-rate',
        default=.01,
        type=float,
        help='learning rate for gradient descent, default=.01')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')
    parser.add_argument(
        '--input-dir',
        default='../../dataset',
        help='local or GCS location for reading the training file'
    )
    parser.add_argument(
        '--train-file',
        default='train2.csv',
        help='name of the training file'
    )
    return parser.parse_args()


def train_and_evaluate(hparams):
    """Trains and evaluates the Keras model.

    Uses the Keras model defined in model.py and trains on data loaded and
    preprocessed in util.py. Saves the trained model in TensorFlow SavedModel
    format to the path defined in part by the --job-dir argument.

    Args:
      hparams: dictionary of hyperparameters - see get_args() for details
    """

    train_x, train_y = util.load_data(hparams.input_dir, hparams.train_file)

    # dimensions
    num_train_examples, input_dim = train_x.shape

    # Create the Keras Model
    keras_model = model.create_keras_model(input_dim=input_dim, learning_rate=hparams.learning_rate)

    # Pass a numpy array by passing DataFrame.values
    training_dataset, validation_dataset = model.input_fn(
        features=train_x.values,
        labels=train_y,
        num_epochs=hparams.num_epochs,
        batch_size=hparams.batch_size)

    # Setup Learning Rate decay.
    lr_decay = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: hparams.learning_rate + 0.02 * (0.5 ** (1 + epoch)),
        verbose=True)

    # Train model
    keras_model.fit(
        training_dataset,
        steps_per_epoch=int((0.8 * num_train_examples) / hparams.batch_size),
        epochs=hparams.num_epochs,
        validation_data=validation_dataset,
        validation_steps=1,
        verbose=1,
        # callbacks=[lr_decay]
    )

    export_path = tf.contrib.saved_model.save_keras_model(keras_model, os.path.join(hparams.job_dir, 'keras_export'))
    export_path = export_path.decode('utf-8')
    print('Model exported to: ', export_path)


if __name__ == '__main__':
    args = get_args()
    tf.logging.set_verbosity(args.verbosity)
    hyperparams = hparam.HParams(**args.__dict__)
    train_and_evaluate(hyperparams)