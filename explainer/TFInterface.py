# Tensorflow-based Attribution API, including RNN

from typing import Callable
import tensorflow as tf
import numpy as np
import typing.abc as abc

# intended usage:

#   model = ...
#
#   methods = [Saliency(...),
#              IntegratedGradients(...),
#              Whatever(...)]
#
#   visualize = AttributeTimesInput(...)
#
#   for method in methods:
#       result = runner(method(model),
#                       model.input,
#                       instances
#                       )
#       display(visualize(result))
#       )

# spec of quantitiy of interest
# spec of distribution of interest
# spec regarding experts for influence directed

TVar = tf.Tensor
TVal = np.ndarray


def create_runner(batch_size, session):
    def run(vars, feed_dict):
        return session.run(vars, feed_dict=feed_dict)
    return run


class TFRunner(abc):
    def __init__(self, batch_size: int, session: tf.Session):
        self.batch_size = batch_size
        self.session = session

    def __apply__(self, vars, input_tensor, doi):
        # do some batching stuff here
        return self.session.run(vars,
                                feed_dict={input_tensor: doi})


class TDOI(abc):
    # linear interpolation for integrated_grad
    # point for saliency_map
    # others?
    pass


class TQOI(abc):
    # generalize QuantityInterest.py
    pass


class TFModel(abc):
    def __init__(self,
                 input: TVal,
                 output: TVal,
                 ):

        self.input = input
        self.output = output


class ChanneledModel(TFModel):
    def __init__(self,
                 input: TVar,
                 output: TVar,
                 channel_first: bool):
        pass


class RNNModel(TFModel):
    def __init__(self,
                 input: Callable[int, TVar],
                 output: Callable[int, TVar],
                 starting_state: TVar):
        pass


class Attribution(abc):
    def __init__(self):
        pass

    def __apply__(self, model):
        # return the tensor that stores the outcome of the attribution method
        pass

class DOIAttribution(Attribution):
    def __init__(self,
                 doi: TDOI)


def DeepLift(baseline):
    def __apply__(self, model):
        # modify the model here and return the relevant tensor of the modified model


def Saliency(..):
    return

def IntegratedGradients(steps):
    return DOIAttribution(LinearDOI(steps))

class SmoothGradient(Attribution):
    def __init__(self,
                 noise_ratio:float,
                 steps:int):
        pass



class TFAttrib(abc):
    def __init__(self,
                 model: TFModel,
                 session: tf.Session,
                 qoi: TQOI,
                 batch_size: int = 16):
        self.model = model
        self.session = session

    def saliency_map(self, input: TVal) -> TVal:
        pass


class TFVisualize(abc):
    def __init__(self, doi: TDOI): pass
