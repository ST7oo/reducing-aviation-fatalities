import os
import pandas as pd
import numpy as np


LABEL_COLUMN = 'event'


def preprocess(dataframe):
    """Removes unused columns.

    Args:
      dataframe: Pandas dataframe with raw data

    Returns:
      Dataframe with preprocessed data
    """
    dataframe = dataframe.drop(columns=['crew','experiment','seat'])
    dataframe[LABEL_COLUMN] = dataframe[LABEL_COLUMN].astype('category')
    dataframe[[LABEL_COLUMN]] = dataframe[[LABEL_COLUMN]].apply(lambda x: x.cat.codes)

    return dataframe


def standardize(dataframe):
    """Scales numerical columns using their means and standard deviation to get
    z-scores: the mean of each numerical column becomes 0, and the standard
    deviation becomes 1. This can help the model converge during training.

    Args:
      dataframe: Pandas dataframe

    Returns:
      Input dataframe with the numerical columns scaled to z-scores
    """
    dtypes = list(zip(dataframe.dtypes.index, map(str, dataframe.dtypes)))
    # Normalize numeric columns.
    for column, dtype in dtypes:
        if dtype == 'float32':
            dataframe[column] -= dataframe[column].mean()
            dataframe[column] /= dataframe[column].std()
    return dataframe


def load_data(data_dir, training_file):
    training_file_path = os.path.join(data_dir, training_file)

    train_df = pd.read_csv(training_file_path)

    train_df = preprocess(train_df)

    train_x, train_y = train_df, train_df.pop(LABEL_COLUMN)

    train_x = standardize(train_x)

    # Reshape label columns for use with tf.data.Dataset
    train_y = np.asarray(train_y).reshape((-1, 1))

    return train_x, train_y
