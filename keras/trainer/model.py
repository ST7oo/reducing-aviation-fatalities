import tensorflow as tf


def create_keras_model(input_dim, learning_rate):
    """Creates Keras Model

    Args:
      input_dim: How many features the input has
      learning_rate: Learning rate for training

    Returns:
      The compiled Keras model (still needs to be trained)
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu, kernel_initializer='uniform', input_shape=(input_dim,)))
    model.add(tf.keras.layers.Dense(75, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(50, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(25, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(4, activation=tf.nn.softmax))

    # Custom Optimizer:
    # https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer
    # optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

    # Compile Keras model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def input_fn(features, labels, num_epochs, batch_size):
    """Generates an input function to be used for model training.

    Args:
      features: numpy array of features used for training or inference
      labels: numpy array of labels for each example
      shuffle: boolean for whether to shuffle the data or not (set True for
        training, False for evaluation)
      num_epochs: number of epochs to provide the data for
      batch_size: batch size for training

    Returns:
      A tf.data.Dataset that can provide data to the Keras model for training or
        evaluation
    """
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    print(dataset.output_types)
    print(dataset.output_shapes)

    num_examples = len(features)
    train_size = int(0.8 * num_examples)

    train_dataset = dataset.take(train_size)
    train_dataset = train_dataset.shuffle(buffer_size=train_size)
    train_dataset = train_dataset.repeat(num_epochs)
    train_dataset = train_dataset.batch(batch_size)

    val_dataset = dataset.skip(train_size)
    val_dataset = val_dataset.repeat(num_epochs)
    val_dataset = val_dataset.batch(num_examples - train_size)
    
    return train_dataset, val_dataset
