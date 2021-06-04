import tensorflow as tf


def shuffle_and_batch_dataset(dataset, batch_size, shuffle_buffer=None):
    """
    This function is used to shuffle and batch the dataset, using shuffle_buffer
    and batch_size.
    """
    if shuffle_buffer is not None:
        dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size)
    return dataset


def split_dataset(dataset, train_prop=0.8, val_prop=0.2):
    """
    This function takes in the loaded TFRecordDataset, and builds training, validation
    and test TFRecordDataset objects. The test_prop is automatically set up to be equal to
    1 - (train_prop + val_prop).
    """

    dataset_size = sum(1 for _ in dataset)
    train_size = int(train_prop * dataset_size)
    val_size = int(val_prop * dataset_size)
    dataset = dataset.shuffle(dataset_size)
    train_dataset = dataset.take(train_size)
    remaining_dataset = dataset.skip(train_size)
    val_dataset = remaining_dataset.take(val_size)
    test_dataset = remaining_dataset.skip(val_size)
    return train_dataset, val_dataset, test_dataset


def process_dataset(dataset, batch_sizes=None, shuffle_buffers=None, train_prop=0.8, val_prop=0.2):
    """
    :param dataset: TFRecordDataset object
    :param batch_sizes: list of batch_size for train set, validation set and test set
    :param shuffle_buffers: list of shuffle_buffer for train set, validation set and test set
    :param train_prop: the ratio between the full dataset size and the train set size
    :param val_prop: the ratio between the full dataset size and the validation set size
    :return: fully processed train, validation and test TFRecordDataset
    """

    if shuffle_buffers is None:
        shuffle_buffers = [None, None, None]
    if batch_sizes is None:
        batch_sizes = [64, 64, 64]
    if len(shuffle_buffers) != 3 or len(batch_sizes) != 3:
        return "Error: shuffle_buffers and batch_sizes should have a length of 3."

    train_dataset, val_dataset, test_dataset = split_dataset(dataset, train_prop, val_prop)
    train_dataset = shuffle_and_batch_dataset(train_dataset, batch_sizes[0], shuffle_buffers[0])
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = shuffle_and_batch_dataset(val_dataset, batch_sizes[1], shuffle_buffers[1])
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = shuffle_and_batch_dataset(test_dataset, batch_sizes[2], shuffle_buffers[2])
    test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset
