import keras.backend as K
import numpy as np


class InterestsMasks(object):
    """Apply an mask to compute the quantity of interets.
       This is an abstact class.

    """
    def __init__(self, qoi_name):
        self.qoi_name = qoi_name

    def __call__(self, y):
        raise NotImplementedError


class ClassInterest(InterestsMasks):
    """Quantity of Interests mask for classification tasks
       The influence will be calculated given a specific class

    """
    def __init__(self, c):
        """__init__ Constructor

        Arguments:
            c {int} -- The class number to compute the influence
        """
        super(ClassInterest, self).__init__("class")
        self._c = c

    def __call__(self, x):
        """__call__

        Arguments:
            x {Keras.tensor} -- The input keras tensor

        Raises:
            IndexError: if the number of dims is not 2.

        Returns:
            Keras.tensor -- A tensor of the class scores
        """
        if K.ndim(x) != 2:
            raise IndexError(
                "The input tensor should have 2 dimensions, but %d got" %
                K.ndim(x))
        return K.sum(x[:, self._c])

    def get_class(self):
        """get_class Return the class

        Returns:
            int -- The class set for qoi
        """
        return self._c


class ChannelInterest(InterestsMasks):
    """ChannelInterest

    Compute the interests of influence on a specific channel with weighted
    sum of each nuerons inside this channel.

    """
    def __init__(self, c_index, init_wts=None, channel_first=True):
        """__init__ Constructor

        Arguments:
            c_index {list or np.ndarray} -- A list of channel indices

        Keyword Arguments:
            init_wts {K.Tensor or np.ndarray} -- A tensor of the initial 
                gradients with the shape shape of the picked channels. If None, no
                initial gradients will be applied. The first dimension should be 
                the number of channels specified by the length of c_index.
                (default: {None})
            channel_first {bool} -- Specify the order of input tensor of the model
                (default: {True})
        """
        super(ChannelInterest, self).__init__("channel")
        self.c_index = c_index
        self.init_wts = init_wts
        self.channel_first = channel_first

    def __call__(self, x):
        """__call__

        Arguments:
            x {K.Tensor} -- The sybolic output of a given layer in the model.

        Raises:
            IndexError: If the dimension of input tensor is not 4 and try to index a 
                conv layer
            IndexError: Try to index a neuron of a conv layer but no enough indices 
                in the c_index. s
            RuntimeError: The dimension of init_wts and input tensor do not match.

        Returns:
            K.Tensor -- a scalor tensor for gradient computation.
        """
        # FC Layer or Channel Influence
        if isinstance(self.c_index, int) or isinstance(self.c_index, np.int64):
            if self.channel_first or K.ndim(x) == 2:
                # Same dims as x because of indexing with list
                x = x[:, self.c_index]

            else:
                if K.ndim(x) != 4:
                    raise IndexError(
                        "The input tensor should have 2 or 4 dimensions, but %d got"
                        % K.ndim(x))
                else:
                    # Same dims as x because of indexing with list
                    x = x[:, :, :, self.c_index]

        # Neuron influence on Conv layer
        elif len(self.c_index) == 3:
            x = x[:, self.c_index[0], self.c_index[1], self.c_index[2]]

        else:
            raise IndexError(
                "The shape of experts should be a 1-D or 2-D array.")

        if self.init_wts is None:
            return K.sum(x)

        elif len(self.init_wts.shape) != K.ndim(x):
            raise RuntimeError(
                "The dims of inti_wts and the input tensor mismatch, got %d and %d"
                % (len(self.init_wts.shape), K.ndim(x)))

        return K.sum(x * self.init_wts)


class MultiUnitInterest(InterestsMasks):
    """MultiUnitInterest 

    Compute the interests of influence on mutiple units for a hidden layer. The final 
    qoi is the weighted sum of all units. 

    """
    def __init__(self,
                 seq_of_index,
                 inter_unit_wts=None,
                 intra_unit_wts=None,
                 channel_first=False):
        """__init__ Constructor

        Arguments:
            seq_of_index {list or np.ndarray} -- A sequence of the indices of units to aggregate

        Keyword Arguments:
            inter_unit_wts {list or np.ndarray} -- The weight assigned to each unit. If None, 
                take the average of all units (default: {None})
            intra_unit_wts {list or np.ndarray} -- The weights assigned to each member of each 
                unit. Only apply for the convolutiona layer. If None, use the average of the 
                whole unit.  (default: {None})
            channel_first {bool} -- If the channel is the first dimension (default: {False})
        """
        super(MultiUnitInterest, self).__init__("multi_unit")
        self.seq_of_index = seq_of_index
        if inter_unit_wts is not None:
            self.inter_unit_wts = inter_unit_wts
        else:
            self.inter_unit_wts = np.array([1 / len(self.seq_of_index)] *
                                           len(self.seq_of_index))
        if intra_unit_wts is not None:
            self.intra_unit_wts = intra_unit_wts
        else:
            self.intra_unit_wts = [None] * len(self.seq_of_index)

        self.channel_first = channel_first

    def __call__(self, y):
        """__call__

        Arguments:
            y {K.tensor} -- The activation of a layer

        Raises:
            IndexError: If the dimension of input tensor is not 4 and try to index a 
                conv layer
            IndexError: Try to index a neuron of a conv layer but no enough indices 
                in the c_index. s
            RuntimeError: The dimension of init_wts and input tensor do not match.

        Returns:
            K.tensor -- weighted sum of all units
        """

        sum_of_unit = K.tf.constant(0, dtype=K.tf.float32)
        for inter_wts, unit, wts in zip(self.inter_unit_wts, self.seq_of_index,
                                        self.intra_unit_wts):

            # FC Layer or Channel Influence
            if isinstance(unit, int) or isinstance(unit, np.int64):
                if self.channel_first or K.ndim(y) == 2:
                    # Same dims as x because of indexing with list
                    x = y[:, unit]

                else:
                    if K.ndim(y) != 4:
                        raise IndexError(
                            "The input tensor should have 2 or 4 dimensions, but %d got"
                            % K.ndim(y))
                    else:
                        # Same dims as x because of indexing with list
                        x = y[:, :, :, unit]

            # Neuron influence on Conv layer
            elif len(unit) == 3:
                x = y[:, unit[0], unit[1], unit[2]]

            else:
                raise IndexError(
                    "The shape of experts should be a 1-D or 2-D array.")

            if wts is None:
                sum_of_unit = sum_of_unit + K.sum(x) * inter_wts

            else:
                if len(wts.shape) != K.ndim(x):
                    raise RuntimeError(
                        "The dims of init_wts and the input tensor mismatch",
                        wts.shape, " and ", x.shape)
                sum_of_unit = sum_of_unit + K.sum(x * wts) * inter_wts

        return sum_of_unit
