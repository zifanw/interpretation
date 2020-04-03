import keras.backend as K
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from .utils import split_kerasmodel
from keras.layers import Softmax


class LayerCompress:
    """ Model Compression using the expert units in a given layer """
    def __init__(self, model, layer):
        """__init__ Constructor
        
        Arguments:
            model {Keras.Model} -- Keras Model
            layer {int or str} -- the index of a layer or the name of the layer
        """
        self._model = model
        self.layer2idx = {}
        self.idx2layer = {}
        for idx, l in enumerate(self._model.layers):
            self.idx2layer[idx] = l.name
            self.layer2idx[l.name] = idx
        if isinstance(layer, int):
            self._layer = self.idx2layer[layer]
        else:
            self._layer = layer

        self._split_model()

    def set_model(self, new_model):
        """set_model Set a new model 
        
        Arguments:
            new_model {Keras.Model} -- Keras Model
        """
        self._model = new_model

    def get_model(self):
        """get_model Return the current model"""
        return self._model

    def set_layer(self, new_layer):
        """set_layer Set a new layer to compress

        Arguments:
            new_layer {int or str} -- the index of a layer or the name of the layer
        """
        self._layer = new_layer

    def get_layer(self):
        """get_layer Get the current layer"""
        return self._layer

    def _split_model(self):
        """_split_model Split the model by the given layer. Create three
            sub models: 1) the head model, which propagates from the input to
            the given layer; 2) the tail_logits, which propagates from the 
            given layer to the logits layer. 3) the tail model, which propagates 
            from the given layer to the probits layer. 

        """
        internal_out = self._model.get_layer(self._layer).output
        self.head_model = K.function(self._model.inputs, [internal_out])
        self.sub_model = split_kerasmodel(self._model,
                                          self._layer,
                                          end_layer=-1)
        self.tail_model = K.function(self.sub_model.inputs,
                                     self.sub_model.outputs)
        logits = self.sub_model.get_layer(index=-2).get_output_at(0)
        self.tail_logits = K.function(self.sub_model.inputs, [logits])

    @staticmethod
    def expK(length, K=2):
        """expK Generate a sequence of exponential numbers with base K. 
        
        Arguments:
            length {int} -- The length of all experts
        
        Keyword Arguments:
            K {int} -- base of the exponents (default: {2})
        
        Returns:
            list -- exponents with base of K
        """
        L = []
        for i in range(length):
            if K**i < length:
                L.append(i)
            else:
                break
        L.append(length)
        return L

    @staticmethod
    def uniformK(length, K=10):
        """uniformK Generate a sequence of uniform distribution with stride K
        
        Arguments:
            length {int} -- The length of all experts
        
        Keyword Arguments:
            K {int} -- stride (default: {10})
        
        Returns:
            list -- uniform distributed numbers with stride K
        """
        return np.arange(length + K)[::K]

    @staticmethod
    def fibonacci(length, K=None):
        """fibonacci Generate a fibonacci sequence with a closest end to the length of experts.
        
        Arguments:
            length {int} -- The length of all experts
        
        Keyword Arguments:
            K  -- Ignore (default: {None})
        
        Returns:
            list -- a fibonacci sequence
        """
        alpha = (1.0 + np.sqrt(5.0)) / 2.0
        beta = (1.0 - np.sqrt(5.0)) / 2.0
        N = int(np.log(np.sqrt(5.0) * length) / np.log(alpha))
        L = np.arange(N + 2)
        L = (np.power(alpha, L) - np.power(beta, L)) / np.sqrt(5.0)
        L = L.astype(np.int64)
        return L

    @staticmethod
    def create_mask_fn(seq_shape, axis=None):
        """create_mask_fn Create a function which generates unit masks

        Arguments:
            seq_shape {tuple} -- A tuple of the shape of indices of the experts
        
        Keyword Arguments:
            axis {int, tuple or list} -- The axises on which neurons should be masked. 
                If None, do not mask. (default: {None})
        
        Raises:
            ValueError: The number of axises does not match the dimensions of the experts 
        
        Returns:
            function -- mask_fn: (tuple, np.ndarray, bool) --> np.ndarray
        """
        if axis is None:
            axis = [1]
        elif isinstance(axis, int):
            axis = [axis]
        if len(axis) != seq_shape[1]:
            raise ValueError(
                "Dimensions mismatch between the axis of nuerons to \
                be closed with the provided sequence of unit indices.")

        def mask_fn(array_shape, seq_of_unit=None, turn_off_selected=True):
            """mask_fn Generate a mask to turn off or turn on specific units
            
            Arguments:
                array_shape {tuple} -- the shape of the intermediate layer output
            
            Keyword Arguments:
                seq_of_unit {np.ndarray} -- A sequence of the indices of expert 
                    units (default: {None})
                turn_off_selected {bool} -- If True, turn off the given units. If False, 
                turn on the given units and turn off others (default: {True})
            
            Raises:
                ValueError: The axises can only be a combinations of 1, 2 and 3
            
            Returns:
                np.ndarray -- A mask to turn off or turn on specific units.
            """
            nonlocal axis

            if turn_off_selected:
                mask = np.ones(array_shape)
                if seq_of_unit is not None:
                    for i, a in enumerate(axis):
                        if a == 1:
                            mask[:, seq_of_unit[:, i]] = 0
                        elif a == 2:
                            mask[:, :, seq_of_unit[:, i]] = 0
                        elif a == 3:
                            mask[:, :, :, seq_of_unit[:, i]] = 0
                        else:
                            raise ValueError(
                                "axis can only be 1, 2, and 3, but got %d" % a)
            else:
                mask = np.zeros(array_shape)
                if seq_of_unit is not None:
                    for i, a in enumerate(axis):
                        if a == 1:
                            mask[:, seq_of_unit[:, i]] = 1
                        elif a == 2:
                            mask[:, :, seq_of_unit[:, i]] = 1
                        elif a == 3:
                            mask[:, :, :, seq_of_unit[:, i]] = 1
                        else:
                            raise ValueError(
                                "axis can only be 1, 2, and 3, but got %d" % a)
            return mask

        return mask_fn

    def head_prop(self, X, batch_size=16):
        """head_prop Propagate from the input to the given layer
        
        Arguments:
            X {np.ndarray} -- Input dataset
        
        Keyword Arguments:
            batch_size {int} -- The batch size to run the propagation (default: {16})
        
        Returns:
            np.ndarray -- output of the intermediate layer
        """
        leftover = X.shape[0] % batch_size
        num_of_batch = X.shape[0] // batch_size if not leftover else 1 + (
            X.shape[0] // batch_size)

        layer_ins = []
        for i in range(num_of_batch):
            if i != num_of_batch - 1:
                x = X[i * batch_size:(i + 1) * batch_size]
            else:
                x = X[i * batch_size:]
            if self.layer2idx[self._layer] != 0:
                out = self.head_model([x])[0]
            else:
                out = x
            layer_ins.append(out)
        return layer_ins

    def __call__(self,
                 X,
                 Y,
                 seq_of_unit,
                 batch_size=16,
                 axis=1,
                 sampling='expK',
                 turn_off=True,
                 metric=accuracy_score,
                 return_logits=False,
                 verbose=True,
                 **kwargs):
        """__call__ Run the model compression
        
        Arguments:
            X {np.ndarray} -- Input dataset
            Y {np.ndarray} -- Groundtruth of the input dataset
            seq_of_unit {list or np.ndarray} -- The indices of the expert units
        
        Keyword Arguments:
            batch_size {int} -- Batch size used for the propagation (default: {16})
            axis {int, tuple or list} -- On which axis(es) to run compression. (default: {1})
            sampling {str or fn} -- Sampling function to decide the number of units to 
                compress for each run time. (default: {'expK'})
            turn_off {bool} -- If True, turn off the units identified by seq_of_unit. If False, 
                turn on the units identified by seq_of_unit and turn off others (default: {True})
            metric {fn} -- The metric function to compute the performance of the network (default: {accuracy_score})
            return_logits {bool} -- If True, also return the logits of the model output. (default: {False})
            verbose {bool} -- If True, print the information. (default: {True})
        
        Raises:
            ValueError: The name of sampling function has not been implemented by this class yet.
        
        Returns:
            np.ndarray or tuple -- The performance scores for the model, when its units in that given layer
            are compressed. If return_logits is True, also return the logits. 
        """

        base = kwargs['base'] if 'base' in kwargs else 2

        if isinstance(seq_of_unit, list) or len(seq_of_unit.shape) == 1:
            seq_of_unit = np.asarray(seq_of_unit)[:, np.newaxis]

        layer_outs = self.head_prop(X, batch_size)
        mask_fn = self.create_mask_fn(seq_of_unit.shape, axis=axis)

        if sampling in ['expK', 'uniformK', 'fibonacci']:
            sampled_list = getattr(self, sampling)(len(seq_of_unit), base)
        else:
            if isinstance(sampling, str):
                raise ValueError("Not a supported sampling method")
            else:
                sampled_list = sampling(len(seq_of_unit), base)

        total_logits = []
        scores = []
        for i in tqdm(sampled_list):
            if i != 0:
                unit_idx = seq_of_unit[:i]
            else:
                unit_idx = None

            probits = []
            logits = []

            for internal_in in layer_outs:
                mask = mask_fn(internal_in.shape,
                               unit_idx,
                               turn_off_selected=turn_off)
                internal_in *= mask
                if return_logits:
                    logit = self.tail_logits([internal_in])[0]
                    logits.append(logit)

                p = self.tail_model([internal_in])[0]
                probits.append(p)

            if return_logits:
                logits = np.vstack(logits)
                total_logits.append(logits[np.newaxis, :])

            probits = np.vstack(probits)
            preds = np.argmax(probits, axis=-1)
            s = metric(Y, preds, **kwargs)
            scores.append(s)

        scores = np.asarray(scores)
        if return_logits:
            total_logits = np.vstack(total_logits)
            return scores, total_logits
        return scores
