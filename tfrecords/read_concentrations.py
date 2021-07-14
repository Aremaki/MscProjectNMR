import tensorflow as tf


def parse_tfr_element_concentrations(element):
    """
    :param element: tfrecord example
    :return: example in the form (tf.float64, tf.float64)
    """

    # use the same structure as the writing
    data = {
      'X': tf.io.FixedLenFeature([], tf.string),
      'Y': tf.io.FixedLenFeature([], tf.string),
    }
    content = tf.io.parse_single_example(element, data)
    X = tf.io.parse_tensor(content['X'], out_type=tf.float64)
    Y = tf.io.parse_tensor(content['Y'], out_type=tf.float64)

    return X, Y


def parse_tfr_element_concentrations_single(element):
    """
    :param element: tfrecord example
    :return: example in the form (tf.float64, tf.float32)
    """

    # use the same structure as the writing
    data = {
      'X': tf.io.FixedLenFeature([], tf.string),
      'y': tf.io.FixedLenFeature([], tf.float32),
    }
    content = tf.io.parse_single_example(element, data)
    X = tf.io.parse_tensor(content['X'], out_type=tf.float64)
    y = content['y']

    return X, y


def read_tfrecords_concentrations(filenames, batch_size):
    """Import the tfrecord file into a TFRecordDataset"""
    
    # create the dataset
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.experimental.AUTOTUNE)

    # pass every single feature through our mapping function
    dataset = dataset.map(parse_tfr_element_concentrations, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def read_tfrecords_concentrations_single(filenames, batch_size):
    """Import the tfrecord file into a TFRecordDataset"""
    
    # create the dataset
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.experimental.AUTOTUNE)

    # pass every single feature through our mapping function
    dataset = dataset.map(parse_tfr_element_concentrations_single, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
