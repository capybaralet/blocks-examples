import logging
import numpy
from argparse import ArgumentParser
import os

import numpy
np = numpy

from theano import tensor

from blocks.algorithms import GradientDescent, Scale
from blocks.algorithms import Adam, Momentum
from blocks.bricks import (MLP, Rectifier, Initializable, FeedforwardSequence,
                           Activation, Softmax)
from blocks.bricks.conv import (ConvolutionalSequence,
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


from blocks_utils import get_batch
#from capy.theano_utils import *

from lenet import ConvNet
from get_data_streams import get_data_streams

import inspect

"""
TODO:
    extensions (what do I want to monitor???)
    regularizers (BN, dropout)
"""

# FIXME?: top_mlp is initialized outside ConvNet class

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("parse")
    # TODO: checkme - vs _
    parser.add_argument("--conv_sizes", type=str, default="5_5")
    parser.add_argument("--feature_maps", type=str, default="20_50")
    parser.add_argument("--mlp_hiddens", type=str, default="500")
    parser.add_argument("--pool_sizes", type=str, default="2_2")
    parser.add_argument("--batch_size", type=int, default=128)
    # DK params
    parser.add_argument("--data_set", type=str, default="MNIST")
    parser.add_argument("--init_scale", type=float, default=None)
    parser.add_argument("--learning_rate", type=float, default=.1)
    #parser.add_argument("--momentum", type=float, default=0.)
    momentum = .9
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--percent_noised", type=int, default=0)
    parser.add_argument("--permuted", type=int, default=0) # shuffle input dims
    parser.add_argument("--remove_noised", type=int, default=0)
    args = parser.parse_args()
    args_dict = vars(args)
    locals().update(args_dict)

    save_path = os.path.join(os.environ["SAVE_PATH"], os.path.basename(__file__)[:-3])
    settings_str = '__'.join([arg + "=" + str(args_dict[arg]) for arg in sorted(args_dict.keys())])
    save_path += "/" + settings_str
    print save_path

    # convert argparsed strings to lists of ints
    conv_sizes = [int(item) for item in conv_sizes.split("_")]
    feature_maps = [int(item) for item in feature_maps.split("_")]
    pool_sizes = [int(item) for item in pool_sizes.split("_")]
    if mlp_hiddens == 'None':
        mlp_hiddens = []
    else:
        mlp_hiddens = [int(item) for item in mlp_hiddens.split("_")]

    print mlp_hiddens

    # get data
    input_size, output_size, train_stream, valid_stream, test_stream = get_data_streams(
            data_set, batch_size, percent_noised, remove_noised, permuted)

    print input_size
    if len(input_size) == 3: # CIFAR or other multi-channel input
        num_channels = input_size[2]
        input_size = input_size[:2]
    elif len(input_size) == 2:
        num_channels = 1
    else:
        assert False

    # ------------------------- make CNN ----------------------------- #
    conv_activations = [Rectifier() for _ in feature_maps]
    mlp_activations = [Rectifier() for _ in mlp_hiddens] + [Softmax()]
    if init_scale is not None:
        weights_init = Uniform(width=init_scale)
    else:
        weights_init = Uniform(width=.2)
    convnet = ConvNet(conv_activations, 
                    num_channels=num_channels,
                    image_shape=input_size,
                    filter_sizes=zip(conv_sizes, conv_sizes),
                    feature_maps=feature_maps,
                    pooling_sizes=zip(pool_sizes, pool_sizes),
                    top_mlp_activations=mlp_activations,
                    top_mlp_dims=mlp_hiddens + [output_size],
                    border_mode='full',
                    weights_init=weights_init,
                    biases_init=Constant(0))
    if init_scale is None:
        # We push initialization config so that we can then
        # set different initialization schemes for convolutional layers.
        convnet.push_initialization_config()
        convnet.layers[0].weights_init = Uniform(width=.2)
        convnet.layers[1].weights_init = Uniform(width=.09)
        convnet.top_mlp.linear_transformations[0].weights_init = Uniform(width=.08)
        convnet.top_mlp.linear_transformations[1].weights_init = Uniform(width=.11)
    convnet.initialize()
    # ------------------------- make CNN ----------------------------- #

    logging.info("Input dim: {} {} {}".format(
        *convnet.children[0].get_dim('input_')))
    for i, layer in enumerate(convnet.layers):
        if not Activation in inspect.getmro(type(layer)):
            logging.info("Layer {} ({}) dim: {} {} {}".format(
                i, layer.__class__.__name__, *layer.get_dim('output')))

    x = tensor.tensor4('features')
    x.tag.test_value = get_batch(train_stream)[0]
    y = tensor.lmatrix('targets')
    y.tag.test_value = get_batch(train_stream)[1]

    # Normalize input and apply the convnet
    probs = convnet.apply(x)
    cost = CategoricalCrossEntropy().apply(y.flatten(),
            probs).copy(name='cost')
    error_rate = MisclassificationRate().apply(y.flatten(), probs).copy(
            name='error_rate')

    cg = ComputationGraph([cost, error_rate])

    if optimizer == 'adam':
        algorithm = GradientDescent(step_rule=Adam(learning_rate),
                                    cost=cost,
                                    parameters=cg.parameters)
    elif optimizer == 'momentum':
        algorithm = GradientDescent(step_rule=Momentum(
                                        learning_rate=learning_rate,
                                        momentum=momentum),
                                    cost=cost,
                                    parameters=cg.parameters)
            
    extensions = [Timing(),
                  DataStreamMonitoring(
                      [cost, error_rate],
                      valid_stream,
                      prefix="valid"),
                  TrainingDataMonitoring(
                      [cost, error_rate,
                       aggregation.mean(algorithm.total_gradient_norm)],
                      prefix="train",
                      after_epoch=True),
                  Checkpoint(save_path),
                  ProgressBar(),
                  Printing(),
                  TrackTheBest('valid_error_rate'),
                  FinishIfNoImprovementAfter('valid_error_rate_best_so_far', epochs=10)]

    model = Model(cost)
    main_loop = MainLoop(
        algorithm,
        train_stream,
        model=model,
        extensions=extensions)
    main_loop.run()
