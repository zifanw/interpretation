import tensorflow as tf
import os
import numpy as np
import argparse
import time
import keras.backend as K
import numpy as np
from sklearn.metrics import log_loss
from keras.utils.np_utils import to_categorical
import argparse
from keras.models import load_model
import tensorflow as tf


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    indices = np.unravel_index(indices, ary.shape)
    x, y, z = indices[0], indices[1], indices[2]
    xx = x.reshape((1, x.shape[0]))
    yy = y.reshape((1, y.shape[0]))
    zz = z.reshape((1, z.shape[0]))
    result = [xx, yy, zz]
    result = np.vstack(result)
    return result


def smallest_indices(ary, n):
    """Returns the n smallest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, n)[:n]
    indices = indices[np.argsort(flat[indices])]
    indices = np.unravel_index(indices, ary.shape)
    x, y, z = indices[0], indices[1], indices[2]
    xx = x.reshape((1, x.shape[0]))
    yy = y.reshape((1, y.shape[0]))
    zz = z.reshape((1, z.shape[0]))
    result = [xx, yy, zz]
    result = np.vstack(result)
    return result


def sort_by_influence(args, test_influence):
    '''
    input:

    test_influence can be of two kinds of shapes:
    fc: (Class_number, N, num_of_neurons)
    conv: (Class_number, N, H, W, C)

    class_num is the number of classes.

    return:
    the indices of sorted neurons by influence.

    fc: (Class_number, D, k)
    conv: (Class_number, D, A, k)

    k is the number of highest/lowest neurons the user wants to return and is defined by args.influence_neuron_num
    D = 2. the highest k is stacked with lowest k vertically. index = 0 --> highest. index = 1 --> lowest.
    A = 3. A --> Axis. In convolutional layer, a neuron's location is defined by Height, Width and Channle values.
    '''

    test_influence = np.mean(
        test_influence, axis=0)  # average over the test dataset --> H, W, C
    significant_neuron_ids = None

    if 'fc' in args.layer:
        lowest_significant_neuron_ids = np.argsort(
            test_influence.copy(), axis=0)[:args.influence_neuron_num]
        highest_significant_neuron_ids = np.argsort(
            test_influence.copy(), axis=0)[-args.influence_neuron_num:][::-1]
    else:
        highest_significant_neuron_ids = largest_indices(
            test_influence.copy(), args.influence_neuron_num)  # (3, k)
        lowest_significant_neuron_ids = smallest_indices(
            test_influence.copy(), args.influence_neuron_num)

    significant_neuron_ids = np.vstack([
        highest_significant_neuron_ids[None, :],
        lowest_significant_neuron_ids[None, :]
    ])
    # significant_neuron_ids: fc: (high/low, k) /  conv:(high/low, x/y/z, k)

    return significant_neuron_ids


def sort_by_channel(args, test_influence):

    test_influence = np.max(test_influence, axis=(1, 2))  # N, H, W, C
    test_influence = np.mean(test_influence, axis=0)

    highest_significant_neuron_ids = np.argsort(
        test_influence)[::-1][:args.influence_neuron_num]
    lowest_significant_neuron_ids = np.argsort(
        test_influence)[:args.influence_neuron_num]

    highest_significant_neuron_ids = highest_significant_neuron_ids[None, :]
    lowest_significant_neuron_ids = lowest_significant_neuron_ids[None, :]
    significant_channel_ids = np.vstack(
        [highest_significant_neuron_ids, lowest_significant_neuron_ids])

    return significant_channel_ids


def sort_by_space(args, test_influence):

    test_influence = np.max(test_influence, axis=-1)  # N, H, W, C
    test_influence = np.mean(test_influence, axis=0)

    highest_significant_neuron_ids = np.dstack(
        np.unravel_index(
            np.argsort(test_influence.ravel())[::-1],
            test_influence.shape))[0][:args.influence_neuron_num]
    lowest_significant_neuron_ids = np.dstack(
        np.unravel_index(np.argsort(test_influence.ravel()),
                         test_influence.shape))[0][:args.influence_neuron_num]
    highest_significant_neuron_ids = highest_significant_neuron_ids[None, :]
    lowest_significant_neuron_ids = lowest_significant_neuron_ids[None, :]
    significant_spatial_ids = np.vstack(
        [highest_significant_neuron_ids, lowest_significant_neuron_ids])

    return significant_spatial_ids


def compute_vis(sess,
                model,
                data,
                visualization,
                grad_type=None,
                m=None,
                p=None):
    result = None
    if grad_type is None:
        grad_type = 'Vanila'
    if grad_type == 'Vanila':
        result = sess.run(visualization, feed_dict={model.input: data})
        result = np.asarray(result)
        return result
    elif grad_type == 'Smooth':
        if p is None:
            p = 0.2
        if m is None:
            m = 50
        result = np.zeros_like(data)
        for i, instance in enumerate(data):
            sigma = p * (np.amax(instance) - np.amin(instance))
            for _ in range(m):
                noisy_instance = instance + sigma * \
                    np.random.standard_normal(size=instance.shape)
                value = sess.run(
                    visualization,
                    feed_dict={model.input: noisy_instance[None, ]})
                value = np.asarray(value) / m
                result[i] += value[0][0]
        return result

    return None


def my_generator(x, batch_size, y=None):
    if y is None:
        for i in range(int(x.shape[0] / batch_size) + 1):
            if (i + 1) * batch_size > x.shape[0]:
                yield x[i * batch_size:]
            else:
                yield x[i * batch_size:(i + 1) * batch_size]
    else:
        for i in range(int(x.shape[0] / batch_size) + 1):
            if (i + 1) * batch_size > x.shape[0]:
                yield x[i * batch_size:], y[i * batch_size:]
            else:
                yield x[i * batch_size:(i + 1) *
                        batch_size], y[i * batch_size:(i + 1) * batch_size]
