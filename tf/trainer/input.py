import tensorflow as tf
import multiprocessing
from constants import constants
import trainer.featurizer as featurizer


def _decode_csv(line):
    """Takes the string input tensor and returns a dict of rank-2 tensors."""

    row_columns = tf.expand_dims(line, -1)
    columns = tf.decode_csv(row_columns, record_defaults=constants.CSV_COLUMN_DEFAULTS)
    features = dict(zip(constants.CSV_COLUMNS, columns))

    # Remove unused columns
    unused_columns = set(constants.CSV_COLUMNS) - {col.name for col in featurizer.INPUT_COLUMNS} - {constants.LABEL_COLUMN}
    for col in unused_columns:
        features.pop(col)
    
    return features


def input_fn(filenames,
             num_epochs=None,
             batch_size=200,
             shuffle=True,
             skip_header_lines=0,
             num_parallel_calls=None,
             prefetch_buffer_size=1024):

    if num_parallel_calls is None:
        num_parallel_calls = multiprocessing.cpu_count()

    dataset = tf.data.TextLineDataset(filenames)
    dataset = dataset.skip(skip_header_lines)
    dataset = dataset.map(_decode_csv, num_parallel_calls)
    dataset = dataset.prefetch(prefetch_buffer_size)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 10)

    iterator = dataset.repeat(num_epochs).batch(batch_size).make_one_shot_iterator()
    features = iterator.get_next()

    # Build a Hash Table inside the graph
    table = tf.contrib.lookup.index_table_from_tensor(tf.constant(constants.LABELS))
    # Use the hash table to convert string labels to ints and one-hot encode
    label = table.lookup(features.pop(constants.LABEL_COLUMN))

    return features, label


def csv_serving_input_fn():
    """Build the serving inputs."""
    csv_row = tf.placeholder(shape=[None], dtype=tf.string)
    features = _decode_csv(csv_row)
    features.pop(constants.LABEL_COLUMN)
    return tf.estimator.export.ServingInputReceiver(features, {'csv_row': csv_row})


def example_serving_input_fn():
    """Build the serving inputs."""
    example_bytestring = tf.placeholder(
        shape=[None],
        dtype=tf.string,
    )
    features = tf.parse_example(
        example_bytestring,
        tf.feature_column.make_parse_example_spec(featurizer.INPUT_COLUMNS))
    return tf.estimator.export.ServingInputReceiver(
        features, {'example_proto': example_bytestring})


def json_serving_input_fn():
    """Build the serving inputs."""
    inputs = {}
    for feat in featurizer.INPUT_COLUMNS:
        inputs[feat.name] = tf.placeholder(shape=[None], dtype=feat.dtype)

    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


SERVING_FUNCTIONS = {
    'JSON': json_serving_input_fn,
    'EXAMPLE': example_serving_input_fn,
    'CSV': csv_serving_input_fn
}