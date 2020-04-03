import numpy as np
import cv2
import scipy


def binary_mask(X,
                grads,
                channel_first=False,
                norm=True,
                grad_dir=1,
                threshold=0.2,
                blur=0,
                background=0.1):
    if len(grads.shape) == 4:
        grads = np.mean(grads, axis=-1, keepdims=True)

    if grad_dir == 1:
        grads = np.maximum(grads, np.zeros_like(grads))
    elif grad_dir == -1:
        grads = -np.minimum(grads, np.zeros_like(grads))

    if channel_first:
        X = np.transpose(X, axis=(0, 2, 3, 1))


#     if np.amax(X) <= 1:
#         X *= 255

    result = []
    for score, image in zip(grads, X):
        if norm:
            score = score / (np.max(score) + 1e-9)
        if blur > 0:
            score = scipy.ndimage.filters.gaussian_filter(score, blur)
            score = score / (np.max(score) + 1e-9)

        score[score >= threshold] = 1.0
        score[score <= threshold] = 0

        score[score == 0.0] = background

        binary_map = image * score
        binary_map = 255 * binary_map / (np.max(binary_map) + 1e-9)
        binary_map = np.uint8(binary_map)
        result.append(binary_map[None, :])
    result = np.vstack(result)
    return result


def cv2_heatmap_mask(X,
                     grads,
                     channel_first=False,
                     norm=True,
                     heat_level=0.4,
                     blur=0,
                     threshold=None):

    if channel_first:
        X = np.transpose(X, axis=(0, 2, 3, 1))
        grads = np.transpose(grads, axis=(0, 2, 3, 1))

    if X.shape[-1] == 1:
        X = X[:, :, :, 0]
        X = np.stack((X,) * 3, axis=-1)

    if len(grads.shape) == 4:
        grads = np.mean(grads, axis=-1)

    if np.amax(X) <= 1:
        X *= 255

    result = []
    for score, image in zip(grads, X):
        if norm:
            score = score / (np.max(score,) + 1e-9)

        if blur > 0:
            score = scipy.ndimage.filters.gaussian_filter(score, blur)
            score = score / (np.max(score) + 1e-9)

        if threshold is not None:
            score[score <= threshold] = 0

        heatmap = cv2.applyColorMap(
            np.uint8(255 * (score - score.min()) / np.ptp(score)),
            cv2.COLORMAP_JET)

        heatmap = np.float32(heatmap) * heat_level + np.float32(image)
        heatmap = 255 * heatmap / (np.max(heatmap) + 1e-9)
        heatmap = np.uint8(heatmap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        result.append(heatmap[None, :])
    result = np.vstack(result)
    return result


def point_cloud(grads, threshold=None):
    if len(grads.shape) == 4:
        grads = np.mean(grads, axis=-1)

    if threshold is not None:
        grads[grads < threshold] = 0

    grads /= (np.max(grads, axis=(1, 2), keepdims=True) + 1e-9)
    return grads
