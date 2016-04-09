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
    # model and task
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--net", type=str, default="LeNet")
    # training hyper-params
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--init_scale", type=float, default=.01)
    parser.add_argument("--learning_rate", type=float, default=.01)
    parser.add_argument("--optimizer", type=str, default='momentum')
    # extra weird-ness
    parser.add_argument("--percent_noised", type=int, default=0)
    parser.add_argument("--permuted", type=int, default=0) # shuffle input dims
    parser.add_argument("--remove_noised", type=int, default=0)
    args = parser.parse_args()
    args_dict = vars(args)
    locals().update(args_dict)

    save_path = os.path.join(os.environ["SAVE_PATH"], os.path.basename(__file__)[:-3])
    script_dir = os.path.join(os.environ['SAVE_PATH'], os.path.basename(__file__)[:-3])
    print script_dir
    if not os.path.exists(script_dir): # make sure script_dir exists
        print "making directory:", script_dir
        try:
            os.makedirs(script_dir)
        except:
            pass
    settings_str = '__'.join([arg + "=" + str(args_dict[arg]) for arg in sorted(args_dict.keys())])
    save_path = script_dir + "/" + settings_str
    print save_path

    # convert argparsed strings to lists of ints
    if net == "LeNet":
        conv_sizes = [5,5]
        feature_maps = [20,50]
        pool_sizes = [2,2]
        mlp_hiddens = []
    elif net == "AlexNet":
        conv_sizes = [5,5,5]
        feature_maps = [32, 32, 64]
        pool_sizes = [2,2,2]
        mlp_hiddens = []

    # get data
    input_size, output_size, train_stream, valid_stream, test_stream = get_data_streams(
            dataset, batch_size, percent_noised, remove_noised, permuted)

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
        weights_init = Uniform(width=.01)
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

    # TODO: load weights from file!


    # ------------------------- make CNN ----------------------------- #

    logging.info("Input dim: {} {} {}".format(
        *convnet.children[0].get_dim('input_')))
    for i, layer in enumerate(convnet.layers):
        if not Activation in inspect.getmro(type(layer)):
            logging.info("Layer {} ({}) dim: {} {} {}".format(
                i, layer.__class__.__name__, *layer.get_dim('output')))

    x = tensor.tensor4('features')
    x.tag.test_value = get_batch(train_stream)[0]
    print x.tag.test_value.shape
    y = tensor.lmatrix('targets')
    y.tag.test_value = get_batch(train_stream)[1]
    if 1:
        x.tag.test_value = np.load('/data/lisa/data/mnist/mnist-python/100examples/train100_x.npy').reshape((100,1,28,28))
        y.tag.test_value = np.load('/data/lisa/data/mnist/mnist-python/100examples/train100_y.npy').reshape((100,1))

    # Normalize input and apply the convnet
    probs = convnet.apply(x)
    cost = CategoricalCrossEntropy().apply(y.flatten(),
            probs).copy(name='cost')
    error_rate = MisclassificationRate().apply(y.flatten(), probs).copy(
            name='error_rate')

    cg = ComputationGraph([cost, error_rate])

    outputs = [var for var in cg.variables if var.name is not None and 'output' in var.name]
    predictions = [var for var in cg.variables if var.name == 'argmax'] [0]

    if optimizer == 'adam':
        algorithm = GradientDescent(step_rule=Adam(learning_rate),
                                    cost=cost,
                                    parameters=cg.parameters)
    elif optimizer == 'momentum':
        algorithm = GradientDescent(step_rule=Momentum(
                                        learning_rate=learning_rate,
                                        momentum=.9),
                                    cost=cost,
                                    parameters=cg.parameters)
            
    extensions = [Timing(),
                  DataStreamMonitoring(
                      [cost, error_rate],
                      valid_stream,
                      prefix="valid"),
                  TrainingDataMonitoring(
                      [outputs, 
                       algorithm.gradients,
                       cost, 
                       predictions,
                       error_rate,
                       #aggregation.mean(algorithm.total_gradient_norm),
                       ],
                      prefix="train",
                      #after_batch=True),
                      after_epoch=True),
                  Checkpoint(save_path),
                  #ProgressBar(),
                  Printing(),
                  #Printing(after_batch=True),
                  TrackTheBest('valid_error_rate'),
                  FinishIfNoImprovementAfter('valid_error_rate_best_so_far', epochs=10)]

    model = Model(cost)
    main_loop = MainLoop(
        algorithm,
        train_stream,
        model=model,
        extensions=extensions)
    main_loop.run()



