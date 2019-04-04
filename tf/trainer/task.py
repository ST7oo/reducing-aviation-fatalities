import argparse
import tensorflow as tf
import json
import os

import trainer.input as input_module
import trainer.model as model


def _get_session_config_from_env_var():
    """Returns a tf.ConfigProto instance that has appropriate device_filters set."""

    tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))

    if tf_config and 'task' in tf_config and 'type' in tf_config['task'] and 'index' in tf_config['task']:
        # Master should only communicate with itself and ps
        if tf_config['task']['type'] == 'master':
            return tf.ConfigProto(device_filters=['/job:ps', '/job:master'])
        # Worker should only communicate with itself and ps
        elif tf_config['task']['type'] == 'worker':
            return tf.ConfigProto(device_filters=['/job:ps','/job:worker/task:%d' % tf_config['task']['index']])
    return None


def train_and_evaluate(args):
    """Run the training and evaluate using the high level API."""

    def train_input():
        return input_module.input_fn(
            args.train_files,
            num_epochs=args.num_epochs,
            batch_size=args.train_batch_size,
            num_parallel_calls=args.num_parallel_calls,
            prefetch_buffer_size=args.prefetch_buffer_size
        )
    
    def eval_input():
        return input_module.input_fn(
            args.eval_files,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_parallel_calls=args.num_parallel_calls,
            prefetch_buffer_size=args.prefetch_buffer_size
        )

    train_spec = tf.estimator.TrainSpec(train_input, max_steps=args.train_steps)

    exporter = tf.estimator.FinalExporter(args.model_name, input_module.SERVING_FUNCTIONS[args.export_format])
    eval_spec = tf.estimator.EvalSpec(
        eval_input,
        steps=args.eval_steps,
        exporters=[exporter],
        name=args.model_name + '-eval'
    )

    run_config = tf.estimator.RunConfig(session_config=_get_session_config_from_env_var())
    run_config = run_config.replace(model_dir=args.job_dir)
    print('Model dir %s' % run_config.model_dir)

    estimator = model.build_estimator(
        embedding_size=args.embedding_size,
        hidden_units=[max(2, int(args.first_layer_size * args.scale_factor**i)) for i in range(args.num_layers)],
        config=run_config
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()

    PARSER.add_argument(
        '--model-name',
        help='Model name'
    )
    PARSER.add_argument(
        '--train-files',
        help='GCS file or local paths to training data',
        nargs='+'
    )
    PARSER.add_argument(
        '--eval-files',
        help='GCS file or local paths to evaluation data',
        nargs='+'
    )
    PARSER.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        default='./jobs'
    )
    PARSER.add_argument(
        '--train-steps',
        help="""\
            Steps to run the training job for. If --num-epochs is not specified,
            this must be. Otherwise the training job will run indefinitely.""",
        default=100,
        type=int
    )
    PARSER.add_argument(
        '--eval-steps',
        help='Number of steps to run evalution for at each checkpoint',
        default=100,
        type=int
    )
    PARSER.add_argument(
        '--train-batch-size',
        help='Batch size for training steps',
        type=int,
        default=40
    )
    PARSER.add_argument(
        '--eval-batch-size',
        help='Batch size for evaluation steps',
        type=int,
        default=40
    )
    PARSER.add_argument(
        '--num-epochs',
        help="""\
            Maximum number of training data epochs on which to train.
            If both --max-steps and --num-epochs are specified,
            the training job will run for --max-steps or --num-epochs,
            whichever occurs first. If unspecified will run for --max-steps.\
            """,
        type=int
    )
    PARSER.add_argument(
        '--num-parallel-calls',
        help='Number of threads used to read in parallel the training and evaluation',
        default=1,
        type=int
    )
    PARSER.add_argument(
        '--prefetch_buffer_size',
        help='Maximum number of input elements that will be buffered when prefetching',
        type=int
    )
    PARSER.add_argument(
        '--export-format',
        help='The input format of the exported SavedModel binary',
        choices=['JSON', 'CSV', 'EXAMPLE'],
        default='JSON'
    )
    PARSER.add_argument(
        '--embedding-size',
        help='Number of embedding dimensions for categorical columns',
        default=8,
        type=int
    )
    PARSER.add_argument(
        '--first-layer-size',
        help='Number of nodes in the first layer of the DNN',
        default=100,
        type=int
    )
    PARSER.add_argument(
        '--num-layers',
        help='Number of layers in the DNN',
        default=4,
        type=int
    )
    PARSER.add_argument(
        '--scale-factor',
        help='How quickly should the size of the layers in the DNN decay',
        default=0.7,
        type=float
    )
    PARSER.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO'
    )

    ARGUMENTS, _ = PARSER.parse_known_args()

    # Set python level verbosity
    tf.logging.set_verbosity(ARGUMENTS.verbosity)

    # Run the training job
    train_and_evaluate(ARGUMENTS)
