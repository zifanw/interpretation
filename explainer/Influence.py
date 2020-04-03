import keras
import keras.backend as K
import numpy as np
from .computing_utils import *
from time import time
from tqdm import tqdm, trange
from .QuantityInterests import ClassInterest, ChannelInterest, MultiUnitInterest


def numpy_relu(x):
    """numpy_relu Numpy implementation of relu

    Arguments:
        x {np.ndarray} -- input

    Returns:
        np.ndarray -- output
    """
    return np.maximum(x, np.zeros_like(x))


class KerasInflExp(object):
    """This class implements the Influence Directed Explanations
       Using the Keras backend. This could be running on TF backend
       or Theano backend. I implemeneted the Channel Influence and
       Neuron Influence. Location Influence is still under development.
    """
    def __init__(self,
                 model,
                 channel_first=True,
                 verbose=True,
                 doi_type='internal'):
        """__init__ Constructor

        Arguments:
            model {Keras.Model} -- Keras model object

        Keyword Arguments:
            channel_first {bool} -- If True, the input of the model uses an order
                of channel first. (default: {True})
            verbose {bool} -- If True, print out the infomation.
                (default: {True})

        Raises:
            TypeError: If 'channel_first' is not bool.
        """
        self._model = model
        self._verbose = verbose

        if not isinstance(channel_first, bool):
            raise TypeError(
                "Required type of 'channel_first' is bool, but %s got " %
                str(type(channel_first)))
        self._channel_first = channel_first

        if doi_type in ['internal', 'smooth', 'integrated']:
            self.doi_type = doi_type + '_infl'
        else:
            raise ValueError("Not a supported doi type")

        self.history_grad = {}

    def get_model(self):
        """get_model Return the wrapped model

        Returns:
            Keras.model
        """
        return self._model

    def set_model(self, new_model):
        """set_model Set a new model

        Arguments:
            new_model {Keras.model}
        """
        self._model = new_model

    def load_weights(self, path):
        """load_weights Load the weights from saved keras file

        Arguments:
            path {str} -- Path to the weights
        """
        self._model.load_weights(path)
        if self._verbose:
            print("Model is restored from %s" % path)

    def set_verbose(self, new_verbose):
        """set_verbose Set new verbose """
        self._verbose = new_verbose

    def set_doi_type(self, new_doi_type):
        if new_doi_type in ['internal', 'smooth', 'integrated']:
            self.doi_type = new_doi_type + '_infl'
        else:
            raise ValueError("Not a supported doi type")

    def predict(self, X):
        """predict Predict the labels

        Arguments:
            X {np.ndarray -- Input dataset

        Returns:
            np.ndarray -- Output of the model
        """
        return self._model.predict(X)

    def _infl_function(self, from_layer, wrt_layer, interest_mask=None):
        """_infl_function Internal method to compute the influence. The
        quantities of interests is computed on 'from_layer' and the gradients
        backpropagation stops at 'wrt_layer'.

        Arguments:
            from_layer {str} -- The name of the layer to compute quantities
                of interests
            wrt_layer {str} -- The name of the layer to compute influence

        Keyword Arguments:
            interest_mask {function} -- quantities of interests mask. (default: {None})

        Raises:
            TypeError: If 'from_layer' is not a string variable
            TypeError: If 'wrt_layer' is not a string variable

        Returns:
            K.function: list --> list -- Keras backend function to
                compute the influence
        """

        if interest_mask.qoi_name == 'class' and (
                from_layer, wrt_layer) in self.history_grad:
            return self.history_grad[(from_layer, wrt_layer)]

        if isinstance(from_layer, str):
            if from_layer == 'output':
                from_tensor = self._model.output
            else:
                from_tensor = self._model.get_layer(from_layer).output
        else:
            raise TypeError(
                "from_layer should be a string of a layer in the model")

        if isinstance(wrt_layer, str):
            if wrt_layer == 'input':
                wrt_tensor = self._model.input
            else:
                wrt_tensor = self._model.get_layer(wrt_layer).output
        else:
            raise TypeError(
                "wrt_layer should be a string of a layer in the model or 'input' "
            )

        if interest_mask is None:
            grad = K.gradients(from_tensor.sum(), [wrt_tensor])
        else:
            grad = K.gradients(interest_mask(from_tensor), [wrt_tensor])

        grad_fn = K.function(self._model.inputs, grad)
        self.history_grad[(from_layer, wrt_layer)] = grad_fn
        return grad_fn

    def internal_infl(self,
                      X,
                      from_layer,
                      wrt_layer,
                      interest_mask,
                      batch_size=16,
                      **kwargs):
        """internal_infl Interface function for _infl_function

        Arguments:
            X {np.ndarray or K.Tensor} -- Input dataset
            from_layer {str} -- The name of the layer to compute quantities
                of interests
            wrt_layer {str} -- The name of the layer to compute influence
            interest_mask {function} -- Quantities of interests mask. (default: {None})
            batch_size {int} -- The size of each mini batch (default: {16})

        Returns:
            np.ndarray -- Internal influence
        """

        num_batch = X.shape[0] // batch_size
        leftover = X.shape[0] % batch_size
        if leftover:
            num_batch += 1

        infl_computer = self._infl_function(from_layer, wrt_layer,
                                            interest_mask)
        result = []
        for i in range(num_batch):
            if i == num_batch - 1:
                x = X[i * batch_size:]
            else:
                x = X[i * batch_size:(i + 1) * batch_size]

            infl = infl_computer([x])[0]
            result.append(infl)

        return np.vstack(result)

    def smooth_infl(self,
                    X,
                    from_layer,
                    wrt_layer,
                    interest_mask,
                    batch_size=16,
                    **kwargs):
        """smooth_infl Use SmoothGrad to compute the influence.

        This implementation refers to https://pair-code.github.io/saliency/

        Arguments:
            X {np.ndarray or K.Tensor} -- The input dataset
            from_layer {str} -- The name of the layer to compute QoI
            wrt_layer {str} -- The name of the layer to compute gradients
            interest_mask {function} -- QoI mask

        Keyword Arguments:
            batch_size {int} -- The size of each mini batch (default: {16})
            noise_ratio {float} -- The ratio of the std of noise / range of 
                pixel values (default: {0.2})
            resolution {int} -- The number images with noises to aggregate. 
                (default: {50})

        Returns:
            np.ndarray -- Influence on 'wrt_layer' layer
        """

        noise_ratio = kwargs['noise_ratio'] if 'noise_ratio' in kwargs else 0.2
        resolution = kwargs['resolution'] if 'resolution' in kwargs else 50

        range_of_pixel = np.amax(X) - np.amin(X)
        sigma = noise_ratio * range_of_pixel

        # Old version
        # smoothGradMap = np.zeros_like(X)
        # for _ in range(resolution):
        #     # addint Gaussian Noise
        #     data = X + sigma * np.random.standard_normal(size=X.shape)
        #     saliency_map = self.internal_infl(data, from_layer, wrt_layer,
        #                                       interest_mask)
        #     saliency_map = np.asarray(saliency_map) / resolution
        #     smoothGradMap += saliency_map

        # return smoothGradMap

        # New version, resue the symbolic tensor
        num_batch = X.shape[0] // batch_size
        leftover = X.shape[0] % batch_size
        if leftover:
            num_batch += 1

        infl_computer = self._infl_function(from_layer, wrt_layer,
                                            interest_mask)
        result = []
        for i in range(num_batch):
            if i == num_batch - 1:
                x = X[i * batch_size:]
            else:
                x = X[i * batch_size:(i + 1) * batch_size]

            smoothGradMap = np.zeros_like(x)
            for _ in range(resolution):
                data = x + sigma * np.random.standard_normal(size=x.shape)
                noise_infl = infl_computer([data])[0] / resolution
                smoothGradMap += noise_infl
            result.append(smoothGradMap)

        return np.vstack(result)

    def integrated_infl(self,
                        X,
                        from_layer,
                        wrt_layer,
                        interest_mask,
                        batch_size=16,
                        **kwargs):
        """integrated_infl Use Integrated Gradient to compute the influence.

        Arguments:
            X {np.ndarray or K.Tensor} -- The input dataset
            from_layer {str} -- The name of the layer to compute QoI
            wrt_layer {str} -- The name of the layer to compute gradients
            interest_mask {function} -- QoI mask

        Keyword Arguments:
            batch_size {int} -- The size of each mini batch (default: {16})
            resolution {float} -- The number images with noises to aggregate. 
                (default: {50})
            path {str} -- The path ot integrate.  (default: {'linear'})

        Raises:
            NotImplementedError: Only linear path is implemented

        Returns:
            np.ndarray -- Influence on 'wrt_layer' layer
        """

        baseline = kwargs[
            'baseline'] if 'baseline' in kwargs else np.zeros_like(X)
        resolution = kwargs['resolution'] if 'resolution' in kwargs else 50.0
        path = kwargs['path'] if 'path' in kwargs else 'linear'

        # Old version
        # integrated_map = np.zeros_like(X)
        # for r in range(1, int(resolution + 1)):
        #     if path == 'linear':
        #         data = (r / resolution) * X - baseline
        #     else:
        #         raise NotImplementedError
        #     saliency_map = self.internal_infl(data, from_layer, wrt_layer,
        #                                       interest_mask)
        #     saliency_map = np.asarray(saliency_map) / resolution
        #     integrated_map += saliency_map

        # return integrated_map

        # New version
        num_batch = X.shape[0] // batch_size
        leftover = X.shape[0] % batch_size
        if leftover:
            num_batch += 1

        infl_computer = self._infl_function(from_layer, wrt_layer,
                                            interest_mask)
        result = []
        for i in range(num_batch):
            if i == num_batch - 1:
                x = X[i * batch_size:]
            else:
                x = X[i * batch_size:(i + 1) * batch_size]

            integrated_map = np.zeros_like(x)
            for r in range(1, int(resolution + 1)):
                if path == 'linear':
                    data = (r / resolution) * x - baseline
                else:
                    raise NotImplementedError

                integral_infl = infl_computer([data])[0] / resolution
                integrated_map += integral_infl
            result.append(integrated_map)

        return np.vstack(result)

    def visualization(self,
                      X,
                      wrt_layer,
                      expert_idx,
                      from_layer="output",
                      infl_as_wts=False,
                      batch_size=16,
                      multiply_with_input=False,
                      **kwargs):
        """visualization Interface function for _visualization.

        The visualization of expert unit by computing
            the influence of an intermediate layer w.r.t. the input

        Arguments:
            X {np.ndarray} -- Input dataset
            wrt_layer {str} -- The name of the intermediate layer
            expert_idx {np.ndarray} -- The indices of expert units
            from_layer {str} -- The name of the output layer. (default: {"output})
            batch_size {int} -- The size of each mini batch (default: {16})

        Keyword Arguments:
            infl_as_wts {bool} -- Whether to use the influence as weights
                to combine neurons. (default: {False})

        Raises:
            KeyError: QoI is missing when using influence as wts

        Returns:
            list -- A list of visualization results for all experts
        """
        num_batch = X.shape[0] // batch_size
        leftover = X.shape[0] % batch_size
        if leftover:
            num_batch += 1

        result = []
        for i in range(num_batch):
            if i == num_batch - 1:
                x = X[i * batch_size:]
            else:
                x = X[i * batch_size:(i + 1) * batch_size]

            v = self._visualization(x,
                                    from_layer,
                                    wrt_layer,
                                    expert_idx,
                                    infl_as_wts=infl_as_wts,
                                    batch_size=batch_size,
                                    **kwargs)

            result.append(v)

        result = np.concatenate(result, axis=1)
        return result

    def _visualization(self,
                       X,
                       from_layer,
                       wrt_layer,
                       expert_idx,
                       infl_as_wts=False,
                       batch_size=16,
                       multiply_with_input=False,
                       **kwargs):
        """_visualization The visualization of expert unit by computing
            the influence of an intermediate layer w.r.t. the input

        Arguments:
            X {np.ndarray} -- Input dataset
            from_layer {str} -- The name of the output layer
            wrt_layer {str} -- The name of the intermediate layer
            expert_idx {np.ndarray} -- The indices of expert units
            batch_size {int} -- The size of each mini batch (default: {16})

        Keyword Arguments:
            infl_as_wts {bool} -- Whether to use the influence as weights
                to combine neurons. (default: {False})

        Raises:
            KeyError: QoI is missing when using influence as wts

        Returns:
            np.ndarray -- An array of visualization results for all experts
        """

        ndim = len(self._model.get_layer(wrt_layer).output.shape)
        if infl_as_wts:
            if 'interest_mask' not in kwargs:
                raise KeyError("Key word 'interest_mask' is missing.")
            internal_infl = self.internal_infl(X, from_layer, wrt_layer,
                                               kwargs['interest_mask'])
            if len(internal_infl.shape) == 2:  # FC layer
                internal_infl = numpy_relu(internal_infl[:, expert_idx])
                init_wts = internal_infl / \
                    np.sum(internal_infl, axis=1, keepdims=True)
            else:  # Conv layer
                if self._channel_first:
                    internal_infl = numpy_relu(internal_infl[:, expert_idx])
                else:
                    internal_infl = numpy_relu(
                        internal_infl[:, :, :, expert_idx])
                init_wts = internal_infl / \
                    np.sum(internal_infl, axis=(1, 2), keepdims=True)
        else:
            init_wts = None

        vis = []
        for i in range(len(expert_idx)):
            if ndim == 2:
                vis_mask = ChannelInterest(expert_idx[i], None,
                                           self._channel_first)
            elif self._channel_first:
                if init_wts is not None:
                    vis_mask = ChannelInterest(expert_idx[i], init_wts[:, i],
                                               True)
                else:
                    vis_mask = ChannelInterest(expert_idx[i], None,
                                               self._channel_first)
            else:
                if init_wts is not None:
                    vis_mask = ChannelInterest(expert_idx[i],
                                               init_wts[:, :, :, i], False)
                else:
                    vis_mask = ChannelInterest(expert_idx[i], None,
                                               self._channel_first)

            vis.append(
                getattr(self, self.doi_type)(X,
                                             wrt_layer,
                                             'input',
                                             vis_mask,
                                             batch_size=batch_size)[None, :])
        vis = np.vstack(vis)
        if multiply_with_input:
            vis *= X
        return vis

    def multi_unit_visualizaiton(self,
                                 X,
                                 wrt_layer,
                                 expert_idx,
                                 from_layer="output",
                                 inter_unit_wts=None,
                                 infl_as_wts=False,
                                 batch_size=16,
                                 multiply_with_input=False,
                                 **kwargs):
        """multi_unit_visualizaiton Weighted sum of the visualization 
            of mutilple expert units

        Arguments:
            X {np.ndarray} -- Input dataset
            wrt_layer {str} -- The name of the intermediate layer
            expert_idx {np.ndarray or list} -- The sequence of chose expert nueron. 
                For exmaple, [0, 3] means combine the top 0 and top 3 experts.

        Keyword Arguments:
            from_layer {str} -- Only used if infl_as_wts is True to compute 
                the wts from influence(default: {"output"})
            inter_unit_wts {np.ndarray} -- The weight for each expert. If None, 
                assign the equal importance to each expert (default: {None})
            infl_as_wts {bool} -- If True, use the influence as weights for 
                channel or location experts. (default: {False})
            batch_size {int} -- Batch size for the propagation (default: {16})
            multiply_with_input {bool} -- If True, multiply the input with the 
                gradients (default: {False})

        Raises:
            KeyError: QoI is missing when using influence as wts

        Returns:
            np.ndarray -- An array of visualization results for all instance 
                in the dataset
        """
        if inter_unit_wts is not None:
            if abs(np.sum(inter_unit_wts) - 1.0) > 1e-3:
                inter_unit_wts /= np.sum(inter_unit_wts)

        num_batch = X.shape[0] // batch_size
        leftover = X.shape[0] % batch_size
        if leftover:
            num_batch += 1

        result = []
        if self._verbose:
            generator = trange(num_batch)
        else:
            generator = range(num_batch)

        for i in generator:
            if i == num_batch - 1:
                x = X[i * batch_size:]
            else:
                x = X[i * batch_size:(i + 1) * batch_size]

            intra_unit_wts = None
            if infl_as_wts:
                if 'interest_mask' not in kwargs:
                    raise KeyError("Key word 'interest_mask' is missing.")
                internal_infl = self.internal_infl(x, from_layer, wrt_layer,
                                                   kwargs['interest_mask'])
                if len(internal_infl.shape) == 2:  # FC layer
                    intra_unit_wts = None
                else:  # Conv layer
                    if self._channel_first:
                        intra_unit_wts = numpy_relu(
                            internal_infl[:, expert_idx])
                    else:
                        internal_infl = numpy_relu(
                            internal_infl[:, :, :, expert_idx])
                    intra_unit_wts = internal_infl / \
                        np.sum(internal_infl, axis=(1, 2), keepdims=True)
                    intra_unit_wts = np.transpose(intra_unit_wts, (3, 0, 1, 2))

            multi_unit_qoi = MultiUnitInterest(expert_idx, inter_unit_wts,
                                               intra_unit_wts,
                                               self._channel_first)

            batch_attr = getattr(self,
                                 self.doi_type)(x,
                                                from_layer=wrt_layer,
                                                wrt_layer='input',
                                                interest_mask=multi_unit_qoi,
                                                batch_size=batch_size)
            if multiply_with_input:
                batch_attr *= x
            result.append(batch_attr)

        return np.vstack(result)

    def get_activation(self,
                       X,
                       layer_name,
                       batch_size=16,
                       unit_id=None,
                       axis=1):
        """get_activation Compute the internal activations for a given layer

        Arguments:
            X {np.ndarray} -- Input dataset
            layer_name {str} -- The name of the intermediate layer

        Keyword Arguments:
            batch_size {int} -- Batch size (default: {16})
            unit_id {int or np.ndarray or list} -- Get the activation of a 
            specific unit. If None, return the activation of all units 
            (default: {None})
            axis {int} -- The axis to index the unit (default: {1})

        Raises:
            ValueError: axis can only be 1, 2 or 3

        Returns:
            np.ndarray -- Internal activations
        """
        internal_output = self._model.get_layer(layer_name).output
        activ_fn = K.function(self._model.inputs, [internal_output])

        num_batch = X.shape[0] // batch_size
        leftover = X.shape[0] % batch_size
        if leftover:
            num_batch += 1

        result = []
        for i in range(num_batch):
            if i == num_batch - 1:
                x = X[i * batch_size:]
            else:
                x = X[i * batch_size:(i + 1) * batch_size]

            activation = activ_fn([x])[0]
            result.append(activation)

        activation = np.vstack(result)

        if unit_id is not None:
            if isinstance(axis, int):
                if axis == 1:
                    activation = activation[:, unit_id]
                elif axis == 2:
                    activation = activation[:, :, unit_id]
                elif axis == 3:
                    activation = activation[:, :, :, unit_id]
                else:
                    raise ValueError("Unsupported axis")
            else:
                activation = activation[:, unit_id[0], unit_id[1], unit_id[2]]

        return activation
