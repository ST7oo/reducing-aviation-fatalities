import tensorflow as tf

import trainer.featurizer as featurizer


def build_estimator(config, embedding_size=8, hidden_units=[100, 70, 50, 25]):

    (deep_columns, wide_columns) = featurizer.get_deep_and_wide_columns(embedding_size)

    return tf.estimator.DNNLinearCombinedClassifier(
        config=config,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units
    )
