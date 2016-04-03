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

#TODO: BatchNorm, dropout, ImageNet, "view", CNN class

"""
Networks to implement:
MNIST:
    LeNet

CIFAR:
    AlexNet
        numfilters: 32,32,64
        filter shape: 5,5,5
        pad: 2,2,2
        pooling stride: 2,2,2
        OUR MODS:
            pad: 4,4,4
            max pooling
            no LRN
    VGG (BN, dropout)

ImageNet:
    AlexNet (LRN??)
    VGG (BN)


"""

class ConvNet(FeedforwardSequence, Initializable):
    def __init__(self, convolutional_sequence, mlp, **kwargs):
        self.__dict__.update(locals())

        self.flattener = Flattener()
        application_methods = [self.conv_sequence.apply, self.flattener.apply, self.mlp.apply]
        super(ConvNet, self).__init__(application_methods, **kwargs)

    def _push_allocation_config(self):
        self.convolutional_sequence._push_allocation_config()
        conv_out_dim = self.convolutional_sequence.get_dim('output')
        self.mlp.dims = [numpy.prod(conv_out_dim)] + self.mlp.dims

    # necessary??
    @property
    def output_dim(self):
        return self.mlp.dims[-1]

    # necessary??
    @output_dim.setter
    def output_dim(self, value):
        self.mlp.dims[-1] = value


# TODO: how to reshape outputs of MLPs for Convolutional Sequences??
#       I think I can just add another fully connected layer with the right number of units and then reshape the activations
class CNN(FeedforwardSequence, Initializable):
    """
    A stack of alternating Convolutional and feedforward nets.
    Should subsume ConvNet (above)
    """
    def __init__(self, layers, **kwargs):
        self.__dict__.update(locals())
        assert all([type(layer) in [MLP, ConvolutionalSequence] for layer in layers[])
        assert type(layers[-1]) == MLP

        applys = []
        for layer in layers:
            applys.append(layer.apply)
            if type(layer) == ConvolutionalSequence
                applys.append( Flattener().apply )

        super(CNN, self).__init__(applys, **kwargs)

    def _push_allocation_config(self):
        for n,layer in enumerate(layers):
            if type(layer) == ConvolutionalSequence
                layer._push_allocation_config()
            else:
                layer.dims = [numpy.prod(layers[n-1].get_dim('output')] + layer.dims

    # necessary??
    @property
    def output_dim(self):
        return self.layers[-1].dims[-1]

    # necessary??
    @output_dim.setter
    def output_dim(self, value):
        self.layers[-1].dims[-1] = value


###############
lenet_cnn_layers = []
lenet_cnn_layers += [Convolutional(5, 20), Rectifier()]
lenet_cnn_layers += [MaxPooling(2)]
lenet_cnn_layers += [Convolutional(5, 50), Rectifier()]
lenet_cnn_layers += [MaxPooling(2)]
lenet_cnn = ConvolutionalSequence(lenet_cnn_layers, 
                                  num_channels=1,
                                  image_size=(28,28),
                                  border_mode='full')
lenet_mlp = MLP([Rectifier(), Softmax()], [500, 10])
lenet = ConvNet(lenet_cnn, lenet_mlp)

###############
lenet_cnn_layers = []
lenet_cnn_layers += [Convolutional(5, 32), Rectifier()]
lenet_cnn_layers += [MaxPooling(2)]
lenet_cnn_layers += [Convolutional(5, 32), Rectifier()]
lenet_cnn_layers += [MaxPooling(2)]
lenet_cnn_layers += [Convolutional(5, 64), Rectifier()]
lenet_cnn_layers += [MaxPooling(2)]
lenet_cnn = ConvolutionalSequence(lenet_cnn_layers, 
                                  num_channels=3,
                                  image_size=(32,32),
                                  border_mode='full')
lenet_mlp = MLP([Softmax()], [10])
alexnet = ConvNet(lenet_cnn, lenet_mlp)

##########################
vgg_cifar_layers = []
vgg_cifar_layers += [Convolutional(3, 64), Rectifier()]
vgg_cifar_layers += [Convolutional(3, 64), Rectifier()]
vgg_cifar_layers += [MaxPooling(2)]
vgg_cifar_layers += [Convolutional(3, 128), Rectifier()]
vgg_cifar_layers += [Convolutional(3, 128), Rectifier()]
vgg_cifar_layers += [MaxPooling(2)]
vgg_cifar_layers += [Convolutional(3, 256), Rectifier()]
vgg_cifar_layers += [Convolutional(3, 256), Rectifier()]
vgg_cifar_layers += [Convolutional(3, 256), Rectifier()]
vgg_cifar_layers += [MaxPooling(2)]
vgg_cifar_layers += [Convolutional(3, 512), Rectifier()]
vgg_cifar_layers += [Convolutional(3, 512), Rectifier()]
vgg_cifar_layers += [Convolutional(3, 512), Rectifier()]
vgg_cifar_layers += [MaxPooling(2)]
vgg_cifar_layers += [Convolutional(3, 512), Rectifier()]
vgg_cifar_layers += [Convolutional(3, 512), Rectifier()]
vgg_cifar_layers += [Convolutional(3, 512), Rectifier()]
vgg_cifar_layers += [MaxPooling(2)]
# TODO: "view" (position pooling?)
vgg_cifar_cnn = ConvolutionalSequence(vgg_cifar_layers, 
                                  num_channels=3,
                                  image_size=(32,32),
                                  border_mode='full')
vgg_cifar_mlp = MLP([Rectifier(), Softmax()], [512, 10])
vgg_cifar = ConvNet(vgg_cifar_cnn, vgg_cifar_mlp)








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


