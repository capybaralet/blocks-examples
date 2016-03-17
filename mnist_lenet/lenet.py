import logging
import numpy
from argparse import ArgumentParser
import os

import numpy
np = numpy

from theano import tensor

from blocks.algorithms import GradientDescent, Scale
from blocks.bricks import (Activation, FeedforwardSequence, Initializable,
                           MLP,
                           Rectifier, Softmax)
from blocks.bricks.conv import (Convolutional, ConvolutionalSequence,
                                Flattener, MaxPooling)
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.stopping import FinishIfNoImprovementAfter
from blocks.extensions.training import TrackTheBest
from blocks.graph import ComputationGraph
from blocks.initialization import Constant, Uniform
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring import aggregation
from toolz.itertoolz import interleave

from blocks.bricks.base import lazy, application

# TODO: what am I actually using from these?????
#from capy.blocks_utils import *
#from capy.theano_utils import *



class LeNet(FeedforwardSequence, Initializable):
    def __init__(self, conv_activations, num_channels, image_shape,
                 filter_sizes, feature_maps, pooling_sizes,
                 top_mlp_activations, top_mlp_dims,
                 conv_step=(1,1), border_mode='valid', **kwargs):
        self.__dict__.update(locals())

        conv_parameters = zip(conv_activations, filter_sizes, feature_maps)

        # Construct convolutional layers with corresponding parameters
        self.layers = list(interleave([ # interleave conv, activation, and pooling layers
            (Convolutional(filter_size=filter_size,
                           num_filters=num_filter,
                           step=self.conv_step,
                           border_mode=self.border_mode,
                           name='conv_{}'.format(i))
             for i, (_, filter_size, num_filter) in enumerate(conv_parameters)),
            # TODO: insert batch-normalization (optional)
            (activation for i, (activation, _, _) in enumerate(conv_parameters)),
            (MaxPooling(size, name='pool_{}'.format(i))
             for i, size in enumerate(pooling_sizes))]))

        self.conv_sequence = ConvolutionalSequence(self.layers, num_channels,
                                                   image_size=image_shape)

        # Construct a top MLP
        self.top_mlp = MLP(top_mlp_activations, top_mlp_dims)

        # We need to flatten the output of the last convolutional layer.
        # This brick accepts a tensor of dimension (batch_size, ...) and
        # returns a matrix (batch_size, features)
        self.flattener = Flattener()
        application_methods = [self.conv_sequence.apply, self.flattener.apply,
                               self.top_mlp.apply]
        super(LeNet, self).__init__(application_methods, **kwargs)

    @property
    def output_dim(self):
        return self.top_mlp_dims[-1]

    @output_dim.setter
    def output_dim(self, value):
        self.top_mlp_dims[-1] = value

    def _push_allocation_config(self):
        self.conv_sequence._push_allocation_config()
        conv_out_dim = self.conv_sequence.get_dim('output')

        self.top_mlp.activations = self.top_mlp_activations
        self.top_mlp.dims = [numpy.prod(conv_out_dim)] + self.top_mlp_dims


# TODO
class ConvBNRelu(Convolutional):
    """
    A version of Convolutions that adds 
    SpatialBatchNormalization and ReLU to the apply method.
    """
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        if self.image_size == (None, None):
            input_shape = None
        else:
            input_shape = (self.batch_size, self.num_channels)
            input_shape += self.image_size

        output = self.conv2d_impl(
            input_, self.W,
            input_shape=input_shape,
            subsample=self.step,
            border_mode=self.border_mode,
            filter_shape=((self.num_filters, self.num_channels) +
                          self.filter_size))
        if self.use_bias:
            if self.tied_biases:
                output += self.b.dimshuffle('x', 0, 'x', 'x')
            else:
                output += self.b.dimshuffle('x', 0, 1, 2)
        return output



"""

# a more generic class for combining ConvolutionalSequences and MLPs
# should take layers that are one of those two classes, and add appropriate reshaping in between
class CNN():
    def __init__(self, layers, num_channels, batch_size=None, image_size=None, 
                             border_mode=None, tied_biases=False, **kwargs): 





"""
