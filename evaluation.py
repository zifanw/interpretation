import argparse
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
import tensorflow as tfa
import keras
import keras.backend as K
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Softmax
from keras.applications import vgg16 as VGG
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam, SGD
import explainer.QuantityInterests as Q
from explainer.Attribution import KerasAttr
import scipy
from explainer.Influence import KerasInflExp
import explainer.QuantityInterests as Q
from tqdm import trange, tqdm
from scipy.interpolate import BSpline
from sklearn import metrics


def diff_step(N):
    # l1 = [i for i in range(0, 1000, 100)]
    # l2 = [i for i in range(1000, 5000, 500)]
    # l3 = [i for i in range(5000, N, 1000)]
    # return l1 + l2 + l3
    return [i for i in range(0, N, 500)]


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    indices = np.unravel_index(indices, ary.shape)
    x, y = indices[0], indices[1]
    xx = x.reshape((1, x.shape[0]))
    yy = y.reshape((1, y.shape[0]))
    result = [xx, yy]
    result = np.vstack(result)
    return result


def split_attr(attr):

    # attr = np.mean(original_attr, axis=-1)

    pos_attr = np.maximum(attr, np.zeros_like(attr))
    neg_attr = np.minimum(attr, np.zeros_like(attr))

    pos_attr /= (np.sum(pos_attr) + 1e-16)
    neg_attr /= (-np.sum(neg_attr) - 1e-16)
    return pos_attr, neg_attr


def N_Ord(target_input,
          infl_model,
          attr,
          c,
          return_logits=True,
          step=100,
          scoring_layer='dense_2',
          blur=0,
          normalize=True):

    if blur > 0:
        attr = scipy.ndimage.gaussian_filter(attr, sigma=blur)
    pos_sl, _ = split_attr(attr)
    total_pixels = pos_sl.shape[0] * pos_sl.shape[1]
    num_pos = np.sum(pos_sl > 0)

    pixels = diff_step(total_pixels)

    pos_idx = largest_indices(pos_sl, total_pixels)
    removal = []
    share_pixel = []
    for n in pixels:
        if n < num_pos:
            mask = np.ones_like(target_input)
            if n > 0:
                mask[pos_idx.T[:n, 0], pos_idx.T[:n, 1], :] = 0
            img = target_input.copy() * mask
            removal.append(img[None, :])
            if n > 0:
                share_pixel.append(n)
            else:
                share_pixel.append(0)
        else:
            break

    if len(removal) < 2:
        return None, None
    else:
        de_removal = np.vstack(removal)
        de_share_pixel = np.array(share_pixel)
        de_share_pixel = de_share_pixel / de_share_pixel[-1]
        de_pre_softmax = infl_model.get_activation(de_removal,
                                                   scoring_layer)[:, c]
        de_softmax = infl_model.predict(de_removal)[:, c]

        basline = np.zeros((1, 224, 224, 3))
        baseline_output = infl_model.get_activation(basline,
                                                    scoring_layer)[:, c]

        de_pre_softmax = np.maximum(
            de_pre_softmax, baseline_output + np.zeros_like(de_pre_softmax))

        de_pre_softmax -= baseline_output

        if return_logits:
            if normalize:
                de_pre_softmax /= de_pre_softmax[0]

            return de_share_pixel, de_pre_softmax
        else:
            return de_share_pixel, de_softmax


def S_Ord(target_input,
          infl_model,
          attr,
          c,
          return_logits=True,
          step=100,
          scoring_layer='dense_2',
          blur=0,
          normalize=True):

    if blur > 0:
        attr = scipy.ndimage.gaussian_filter(attr, sigma=blur)
    pos_sl, neg_sl = split_attr(attr)
    total_pixels = pos_sl.shape[0] * pos_sl.shape[1]
    num_pos = np.sum(pos_sl > 0)
    pos_idx = largest_indices(pos_sl, total_pixels)
    neg_idx = largest_indices(neg_sl, neg_sl.shape[0] * neg_sl.shape[1])
    removal = []
    share_pixel = []
    pixels = diff_step(total_pixels)
    for n in pixels:
        if n < num_pos:
            mask = np.zeros_like(target_input)
            # mask[neg_sl != 0] = 1
            if n > 0:
                mask[pos_idx.T[:n, 0], pos_idx.T[:n, 1], :] = 1
            img = target_input.copy() * mask
            removal.append(img[None, :])
            if n > 0:
                share_pixel.append(n)
            else:
                share_pixel.append(0)
        else:
            break

    if len(removal) < 2:
        return None, None

    de_removal = np.vstack(removal)
    de_share_pixel = np.array(share_pixel)
    de_share_pixel = de_share_pixel / de_share_pixel[-1]
    de_pre_softmax = infl_model.get_activation(de_removal, scoring_layer)[:, c]
    de_softmax = infl_model.predict(de_removal)[:, c]

    basline = np.zeros((1, 224, 224, 3))
    baseline_output = infl_model.get_activation(basline, scoring_layer)[:, c]

    de_pre_softmax = np.maximum(
        de_pre_softmax, baseline_output + np.zeros_like(de_pre_softmax))

    original_input = target_input[None, :]
    original_output = infl_model.get_activation(original_input,
                                                scoring_layer)[:, c]

    if return_logits:
        if normalize:
            de_pre_softmax /= (original_output + 1e-16)
        return de_share_pixel, de_pre_softmax
    else:
        return de_share_pixel, de_softmax


def TPN(target_input,
        infl_model,
        attr,
        c,
        return_logits=True,
        step=100,
        scoring_layer='dense_2',
        blur=0,
        normalize=True):
    if blur > 0:
        attr = scipy.ndimage.gaussian_filter(attr, sigma=blur)

    pos_sl, _ = split_attr(attr)
    total_pixels = pos_sl.shape[0] * pos_sl.shape[1]
    num_pos = np.sum(pos_sl > 0)
    pos_idx = largest_indices(pos_sl, total_pixels)
    pos_idx = pos_idx.T
    removal = []
    share_attr = []
    pixels = diff_step(total_pixels)
    for n in pixels:
        if n < num_pos:
            mask = np.ones_like(target_input)
            if n > 0:
                mask[pos_idx[:n, 0], pos_idx[:n, 1], :] = 0
            img = target_input.copy() * mask
            removal.append(img[None, :])
            if n > 0:
                share_attr.append(
                    np.sum(pos_sl[pos_idx[:n, 0], pos_idx[:n, 1]]))
            else:
                share_attr.append(0)
        else:
            break

    if len(removal) < 2:
        return None, None, None, None
    de_removal = np.vstack(removal)
    de_share_attr = np.array(share_attr)
    de_pre_softmax = infl_model.get_activation(de_removal, scoring_layer)[:, c]
    de_softmax = infl_model.predict(de_removal)[:, c]

    basline = np.zeros((1, 224, 224, 3))
    baseline_output = infl_model.get_activation(basline, scoring_layer)[:, c]

    de_pre_softmax = np.maximum(
        de_pre_softmax, baseline_output + np.zeros_like(de_pre_softmax))

    removal = []
    share_attr = []
    for n in range(0, total_pixels, step):
        if n < num_pos:
            mask = np.ones_like(target_input)
            if n > 0:
                mask[pos_idx[-n + num_pos:num_pos, 0],
                     pos_idx[-n + num_pos:num_pos, 1], :] = 0
            img = target_input.copy() * mask
            removal.append(img[None, :])
            if n > 0:
                share_attr.append(
                    np.sum(pos_sl[pos_idx[-n + num_pos:num_pos, 0],
                                  pos_idx[-n + num_pos:num_pos, 1]]))
            else:
                share_attr.append(0)
        else:
            break

    in_removal = np.vstack(removal)
    in_share_attr = np.array(share_attr)

    in_pre_softmax = infl_model.get_activation(in_removal, scoring_layer)[:, c]
    in_softmax = infl_model.predict(in_removal)[:, c]
    if return_logits:
        if normalize:
            de_pre_softmax /= de_pre_softmax[0]
            in_pre_softmax /= in_pre_softmax[0]
            de_pre_softmax /= min(
                1, baseline_output /
                (max(de_pre_softmax[-1], baseline_output) + 1e-16))
            in_pre_softmax /= min(
                1, baseline_output /
                (max(in_pre_softmax[-1], baseline_output) + 1e-16))
        return de_share_attr, de_pre_softmax, in_share_attr, in_pre_softmax
    else:
        return de_share_attr, de_softmax, in_share_attr, in_softmax


def TPS(target_input,
        infl_model,
        attr,
        c,
        return_logits=True,
        step=100,
        scoring_layer='dense_2',
        blur=0,
        normalize=True):
    pos_sl, neg_sl = split_attr(attr)
    num_pos = np.sum(pos_sl > 0)
    total_pixels = pos_sl.shape[0] * pos_sl.shape[1]
    pos_idx = largest_indices(pos_sl, total_pixels)
    neg_idx = largest_indices(neg_sl, neg_sl.shape[0] * neg_sl.shape[1])
    removal = []
    share_attr = []
    pos_idx = pos_idx.T
    pixels = diff_step(total_pixels)
    for n in pixels:
        if n < num_pos:
            mask = np.zeros_like(target_input)
            # mask[neg_sl != 0] = 1
            if n > 0:
                mask[pos_idx[:n, 0], pos_idx[:n, 1], :] = 1
            img = target_input.copy() * mask
            removal.append(img[None, :])
            if n > 0:
                share_attr.append(
                    np.sum(pos_sl[pos_idx[:n, 0], pos_idx[:n, 1]]))
            else:
                share_attr.append(0)
        else:
            break

    if len(removal) < 2:
        return None, None, None, None

    de_removal = np.vstack(removal)
    de_share_attr = np.array(share_attr)

    de_pre_softmax = infl_model.get_activation(de_removal, scoring_layer)[:, c]
    de_softmax = infl_model.predict(de_removal)[:, c]

    removal = []
    share_attr = []
    removal = []
    share_attr = []
    for n in range(0, total_pixels, step):
        if n < num_pos:
            mask = np.zeros_like(target_input)
            if n > 0:
                mask[pos_idx[-n + num_pos:num_pos, 0],
                     pos_idx[-n + num_pos:num_pos, 1], :] = 1
            img = target_input.copy() * mask
            removal.append(img[None, :])
            if n > 0:
                share_attr.append(
                    np.sum(pos_sl[pos_idx[-n + num_pos:num_pos, 0],
                                  pos_idx[-n + num_pos:num_pos, 1]]))
            else:
                share_attr.append(0)
        else:
            break

    in_removal = np.vstack(removal)
    in_share_attr = np.array(share_attr)

    in_pre_softmax = infl_model.get_activation(in_removal, scoring_layer)[:, c]
    in_softmax = infl_model.predict(in_removal)[:, c]

    original_input = target_input[None, :]
    original_output = infl_model.get_activation(original_input,
                                                scoring_layer)[:, c]

    if return_logits:
        if normalize:
            de_pre_softmax /= min(original_output, de_pre_softmax[-1])
            in_pre_softmax /= min(original_output, de_pre_softmax[-1])
        return de_share_attr, de_pre_softmax, in_share_attr, in_pre_softmax
    else:
        return in_share_attr, in_softmax, in_share_attr, in_softmax
