from vis.input_modifiers import Jitter
import vis.backprop_modifiers

import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from tqdm import tqdm, trange
import scipy
from skimage.transform import resize as imresize
import torch
from keras.models import load_model


class KerasAttr:
    # Test this class

    def __init__(self, model, qoi, verbose=True, presoftmax=None):
        self.model = model
        self.qoi = qoi
        self.verbose = verbose
        self.grad_fn = None
        self.guided = None
        self.presoftmax = presoftmax

    def set_qoi(self, new_qoi):
        self.qoi = new_qoi

    def _saliency_map_fn(self):
        if self.grad_fn is None:
            if self.presoftmax is None:
                output_scalor = self.qoi(self.model.output)
            else:
                output_scalor = self.qoi(
                    self.model.get_layer(self.presoftmax).output)
            grad_list = K.gradients(output_scalor, self.model.inputs)
            self.grad_fn = K.function(self.model.inputs, grad_list)
        return self.grad_fn

    def saliency_map(self, X, batch_size=16, mul_with_input=False):
        num_batch = X.shape[0] // batch_size
        leftover = X.shape[0] % batch_size
        if leftover:
            num_batch += 1

        if self.verbose:
            generator = trange(num_batch)
        else:
            generator = range(num_batch)

        attr_fn = self._saliency_map_fn()
        result = []
        for i in generator:
            if i == num_batch - 1:
                x = X[i * batch_size:]
            else:
                x = X[i * batch_size:(i + 1) * batch_size]
            batch_attr = attr_fn([x])[0]
            if mul_with_input:
                batch_attr *= x
            result.append(batch_attr)
        result = np.vstack(result)
        return result

    def integrated_grad(self,
                        X,
                        batch_size=16,
                        baseline=None,
                        steps=50,
                        mul_with_input=False):
        num_batch = X.shape[0] // batch_size
        leftover = X.shape[0] % batch_size
        if leftover:
            num_batch += 1

        if self.verbose:
            generator = trange(num_batch)
        else:
            generator = range(num_batch)

        attr_fn = self._saliency_map_fn()
        result = []
        for i in generator:
            if i == num_batch - 1:
                x = X[i * batch_size:]
            else:
                x = X[i * batch_size:(i + 1) * batch_size]

            batch_attr = np.zeros_like(x).astype(np.float64)
            for s in range(1, 1 + steps):
                if baseline is None or baseline.shape != x.shape:
                    baseline = np.zeros_like(x)

                x_in = (baseline + (s / steps) * (x - baseline))
                batch_attr += attr_fn([x_in])[0] * (1 / steps)

            if mul_with_input:
                batch_attr *= (x-baseline)

            result.append(batch_attr)
        result = np.vstack(result)
        return result

    def smooth_grad(self,
                    X,
                    batch_size=16,
                    noise_ratio=0.2,
                    steps=50,
                    mul_with_input=False):
        num_batch = X.shape[0] // batch_size
        leftover = X.shape[0] % batch_size
        if leftover:
            num_batch += 1

        if self.verbose:
            generator = trange(num_batch)
        else:
            generator = range(num_batch)

        attr_fn = self._saliency_map_fn()
        result = []
        for i in generator:
            if i == num_batch - 1:
                x = X[i * batch_size:]
            else:
                x = X[i * batch_size:(i + 1) * batch_size]

            batch_attr = np.zeros_like(x).astype(np.float64)
            for _ in range(1, 1 + steps):
                x_in = x + noise_ratio * \
                    np.random.standard_normal(size=x.shape)
                batch_attr += attr_fn([x_in])[0] * (1 / steps)

            if mul_with_input:
                batch_attr *= x

            result.append(batch_attr)
        result = np.vstack(result)
        return result

    def uniform_grad(self,
                     X,
                     batch_size=16,
                     radii=1e-3,
                     steps=50,
                     mul_with_input=False):
        num_batch = X.shape[0] // batch_size
        leftover = X.shape[0] % batch_size
        if leftover:
            num_batch += 1

        if self.verbose:
            generator = trange(num_batch)
        else:
            generator = range(num_batch)

        attr_fn = self._saliency_map_fn()
        result = []
        for i in generator:
            if i == num_batch - 1:
                x = X[i * batch_size:]
            else:
                x = X[i * batch_size:(i + 1) * batch_size]

            batch_attr = np.zeros_like(x).astype(np.float64)
            for _ in range(1, 1 + steps):
                x_in = x + np.random.uniform(0, 1, size=x.shape) * radii
                batch_attr += attr_fn([x_in])[0] * (1 / steps)

            if mul_with_input:
                batch_attr *= x

            result.append(batch_attr)
        result = np.vstack(result)
        return result

    def gradcam(self,
                X,
                layer_name,
                batch_size=16,
                channel_first=False,
                alpha=0.4):
        output_scalor = self.qoi(self.model.output)
        layer_output = self.model.get_layer(layer_name).output
        grad_list = K.gradients(output_scalor, [layer_output])
        layer_infl_fn = K.function(self.model.inputs, grad_list)

        layer_infl = []
        num_batch = X.shape[0] // batch_size
        leftover = X.shape[0] % batch_size
        if leftover:
            num_batch += 1

        if self.verbose:
            generator = trange(num_batch)
        else:
            generator = range(num_batch)

        layer_infl = []
        for i in generator:
            if i == num_batch - 1:
                x = X[i * batch_size:]
            else:
                x = X[i * batch_size:(i + 1) * batch_size]
            batch_attr = layer_infl_fn([x])[0]
            layer_infl.append(batch_attr)
        layer_infl = np.vstack(layer_infl)

        if channel_first:
            _, _, W, H = self.model.input_shape
            layer_infl = np.transpose(layer_infl, (0, 3, 1, 2))
            weight = np.mean(layer_infl, axis=(2, 3))
        else:
            _, W, H, _ = self.model.input_shape
            weight = np.mean(layer_infl, axis=(1, 2))

        weight_tensor = torch.from_numpy(weight).unsqueeze(-1)
        infl_tensor = torch.from_numpy(layer_infl)

        n, w, h, c = infl_tensor.size()
        cam = torch.bmm(infl_tensor.view(n, w * h, c), weight_tensor)
        cam = cam.view(n, h, w)
        cam = cam.numpy()
        cam = np.maximum(np.zeros_like(cam), cam)
        result = []
        for img in cam:
            upsampled = imresize(img, (W, H))
            result.append(upsampled[None, :])
        result = np.vstack(result)
        result /= np.max(result, axis=(1, 2), keepdims=True)
        return result[:, :, :, None]

    def guidedbprop(self, X, batch_size=16, mul_with_input=False):
        if self.guided is None:
            self.model.save("/tmp/model.h5")
            model = load_model("/tmp/model.h5")
            self.guided = GuidedAttr(model, self.qoi, self.verbose)
        return self.guided.saliency_map(X, batch_size=16, mul_with_input=False)

    def deeplift(self):
        raise NotImplementedError


class GuidedAttr(KerasAttr):
    def __init__(self, model, qoi, verbose=True):
        super(GuidedAttr, self).__init__(vis.backprop_modifiers.guided(model),
                                         qoi, verbose)
