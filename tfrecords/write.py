import tensorflow as tf

# The following three functions can be used to convert a value to a type compatible
# with tf.train.Example.


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_array(array):
    """Returns a serialized string from a tensor"""
    array = tf.io.serialize_tensor(array)
    return array


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(X, y):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {
      'X': _bytes_feature(serialize_array(X)),
      'y': _int64_feature(y),
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tf_serialize_example(X, y):
    tf_string = tf.py_function(serialize_example, (X, y), tf.string)
    return tf.reshape(tf_string, ()) # The result is a scalar


def write_tfrecords(dataset=None, size=1000, number=None):
    """
    :param dataset: tf.data object containing the examples (X: tf.float32, y: tf.int64)
    :param size: Number of example in each tfrecord file
    :param number: Number of files
    :return: Create the tfrecord files
    """

    serialized_dataset = dataset.map(tf_serialize_example)

    if number:
        #The number has priority over the size
        size = len(serialized_dataset) // number
    else:
        number = len(serialized_dataset) // size

    if len(serialized_dataset) % number >= 1:
        number += 1
    
    for i in range(number):
        filename = '../data/tfrecords/data_{}.tfrecord'.format(i)
        data_to_write = serialized_dataset.take(size)
        serialized_dataset = serialized_dataset.skip(size)
        writer = tf.data.experimental.TFRecordWriter(filename)
        writer.write(data_to_write)
