import keras.backend as K
import numpy as np
import tensorflow as tf
from tqdm import trange


class KerasAttacker:
    def __init__(self, model, qoi, verbose=True, eps=0.0313726):
        self.model = model
        self.qoi = qoi
        self.verbose = verbose
        self.eps = eps

    def create_saliency_map_tensor(self, normalize=True):
        output_scalor = K.sum(self.qoi(self.model.output))
        first_order_grad = K.gradients(output_scalor, self.model.inputs)[0]
        if normalize:
            len_axises = len(first_order_grad.shape)
            if len_axises == 4:
                reduced_axises = (1, 2, 3)
            elif len_axises == 3:
                reduced_axises = (1, 2)
            first_order_grad = K.abs(first_order_grad) / K.sum(
                K.abs(first_order_grad), axis=reduced_axises, keepdims=True)
        return first_order_grad

    def saliency_map(self, X):
        symbolic_tensor = self.create_saliency_map_tensor()
        generator = K.function(self.model.inputs, [symbolic_tensor])
        return generator([X])[0]

    def _topK_fn(self, first_order_grad, indices):
        flatten_tensor = K.batch_flatten(first_order_grad)
        top_val = tf.gather(flatten_tensor, indices, axis=-1)
        scalor = -K.sum(top_val)
        second_order_grad = K.gradients(scalor, self.model.inputs)
        return K.function(self.model.inputs, second_order_grad)

    def set_reference(self, X, method=0, normalize=True):

        if method == 0:
            It = self.saliency_map(X)

        if normalize:
            len_axises = len(It.shape)
            if len_axises == 4:
                reduced_axises = (1, 2, 3)
            elif len_axises == 3:
                reduced_axises = (1, 2)
            It = abs(It) / (
                np.sum(abs(It), axis=reduced_axises, keepdims=True) + 1e-9)

        self.It = It

    def topK_attack(self, X, lr=1e-3, max_step=300, topK=2000, mode=0):

        if mode == 0:  # Sailency Map
            first_order_grad_tensor = self.create_saliency_map_tensor()
            first_order_grad = self.saliency_map(X)

        first_order_grad = first_order_grad.flatten()
        indices = np.argsort(first_order_grad)[::-1][:topK]
        attacker = self._topK_fn(first_order_grad_tensor, indices)

        Xt = X.copy()
        upper = X + self.eps
        lower = X - self.eps

        result_I = []
        result_X = []
        for _ in trange(max_step):
            perturbation = attacker([Xt])[0]
            Xt += lr * np.sign(perturbation)
            Xt = np.minimum(Xt, upper)
            Xt = np.maximum(Xt, lower)
            pred = self.model.predict(Xt)
            pred = np.argmax(pred, axis=-1)[0]
            if pred == self.qoi._c:
                result_X.append(Xt)
                It = self.saliency_map(Xt)
                result_I.append(It)
        return result_I, result_X
