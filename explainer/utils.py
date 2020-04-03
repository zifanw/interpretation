import numpy as np
from scipy import ndimage
from skimage.transform import resize as imresize
from keras.layers import Input
from keras.models import Model
from keras.models import load_model


def shift(X, min=0, max=20, step=1, bg_val=0, dir='right'):

    if len(X.shape) > 3:
        raise ValueError("Can only craete a dataset of shifted \
    images for only one original image")

    W, H, _ = X.shape

    shifted_X = []
    for i in range(min, max, step):
        if dir == 'right':
            shifted = np.pad(X, ((0, 0), (0, i), (0, 0)),
                             'constant',
                             constant_values=bg_val)[:, -H:]

        elif dir == "left":
            shifted = np.pad(X, ((0, 0), (i, 0), (0, 0)),
                             'constant',
                             constant_values=bg_val)[:, :H]

        elif dir == "up":
            shifted = np.pad(X, ((i, 0), (0, 0), (0, 0)),
                             'constant',
                             constant_values=bg_val)[:W]

        elif dir == "down":
            shifted = np.pad(X, ((0, i), (0, 0), (0, 0)),
                             'constant',
                             constant_values=bg_val)[-W:]
        else:
            raise ValueError("Not a supported direction")

        shifted_X.append(shifted[None, :])

    return np.vstack(shifted_X)


def rotate(X, min=0, max=90, step=1, bg_val=0):

    if len(X.shape) > 3:
        raise ValueError("Can only craete a dataset of shifted \
    images for only one original image")

    rotated_X = []
    for i in range(min, max, step):
        rotated = ndimage.rotate(X, i, reshape=False)
        rotated_X.append(rotated[None, :])
    return np.vstack(rotated_X)


def scale(X, start_ratio=0.1, end_ratio=2, num=20, bg_val=0):

    if len(X.shape) > 3:
        raise ValueError("Can only craete a dataset of shifted \
    images for only one original image")

    W, H, _ = X.shape

    scaled_X = []
    for i in np.linspace(start_ratio, end_ratio, num=num):
        new_H, new_W = int(H * i), int(W * i)
        if new_H < H:
            d1, d2 = H - new_H, W - new_W
            up = d1 // 2
            down = d1 - up
            left = d2 // 2
            right = d2 - left
            scaled = np.pad(X, ((up, down), (left, right), (0, 0)),
                            'constant',
                            constant_values=bg_val)[:, :, 0]
            scaled = imresize(scaled, (W, H))[:, :, None]
        elif new_H > H:
            scaled = imresize(X[:, :, 0], (new_H, new_W))[:, :, None]
            d1, d2 = new_H - H, new_W - W
            up = d1 // 2
            down = d1 - up
            left = d2 // 2
            right = d2 - left
            scaled = scaled[left:-right, up:-down]
        else:
            scaled = X.copy()
        scaled_X.append(scaled[None, :])

    return np.vstack(scaled_X)


def split_kerasmodel(net, from_layer, end_layer=-1, load_ori_wts=True):
    # docstring

    #Clone model first
    net.save('model_backup_temp.h5')
    model = load_model('model_backup_temp.h5')

    if isinstance(from_layer, str):
        layer_names = [layer.name for layer in model.layers]
        from_layer = layer_names.index(from_layer)

    if isinstance(end_layer, str):
        layer_names = [layer.name for layer in model.layers]
        end_layer = layer_names.index(end_layer)

    inps = Input(shape=model.get_layer(index=from_layer).output_shape[1:])
    x = model.layers[from_layer + 1](inps)
    for layer in model.layers[from_layer + 2:]:
        x = layer(x)

    submodel = Model(inputs=inps, outputs=x)
    submodel.compile(optimizer='sgd',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

    if load_ori_wts:
        for i, layer in enumerate(model.layers[from_layer + 1:]):
            wts = layer.get_weights()
            submodel.get_layer(index=i + 1).set_weights(wts)

    return submodel
