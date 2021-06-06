import tensorflow as tf
from tfrecords.write import _bytes_feature, serialize_array


def serialize_example_concentrations(X, Y):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {
      'X': _bytes_feature(serialize_array(X)),
      'Y': _bytes_feature(serialize_array(Y)),
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tf_serialize_example_concentrations(X, Y):
    tf_string = tf.py_function(serialize_example_concentrations, (X, Y), tf.string)
    return tf.reshape(tf_string, ()) # The result is a scalar


def write_tfrecords_concentrations(path, dataset=None, size=1000, number=None):
    """
    :param dataset: tf.data object containing the examples (X: array(tf.float32), Y: array(tf.float32))
    :param size: Number of example in each tfrecord file
    :param number: Number of files
    :return: Create the tfrecord files
    """

    serialized_dataset = dataset.map(tf_serialize_example_concentrations)

    if number:
        #The number has priority over the size
        size = len(serialized_dataset) // number
    else:
        number = len(serialized_dataset) // size

    if len(serialized_dataset) % number >= 1:
        number += 1
    
    for i in range(number):
        filename = path + '/data_{}.tfrecord'.format(i)
        data_to_write = serialized_dataset.take(size)
        serialized_dataset = serialized_dataset.skip(size)
        writer = tf.data.experimental.TFRecordWriter(filename)
        writer.write(data_to_write)
