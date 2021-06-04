import tensorflow as tf


def parse_tfr_element(element):
    """
    :param element: tfrecord example
    :return: example in the form (tf.float32, tf.int64)
    """

    # use the same structure as the writing
    data = {
      'X': tf.io.FixedLenFeature([], tf.string),
      'y': tf.io.FixedLenFeature([], tf.int64),
    }
    content = tf.io.parse_single_example(element, data)
    X = tf.io.parse_tensor(content['X'], out_type=tf.float64)
    y = content['y']

    return X, y


def read_tfrecords(filenames):
    """Import the tfrecord file into a TFRecordDataset"""
    # create the dataset
    dataset = tf.data.TFRecordDataset(filenames)

    # pass every single feature through our mapping function
    dataset = dataset.map(parse_tfr_element)

    return dataset
