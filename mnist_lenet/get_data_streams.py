
from collections import OrderedDict

from fuel.datasets import MNIST, CIFAR10, IndexableDataset
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream

import numpy
np = numpy

from pylab import *

"""
TODO:
    whitening, etc.

DOESN'T SEEM TO BE WORKING?
or just need to change params...

"""

# TODO: test me!
def noise_labels(data_set, num_noised, output_size):
    state = data_set.open()
    features, targets = data_set.get_data(
        state=state, request=slice(0, data_set.num_examples))
    data_set.close(state=state)

    # change num_noised examples to have incorrect labels.
    # FIXME!
    targets[:num_noised] += np.random.randint(1, output_size)
    targets[:num_noised] %= output_size
    #import ipdb; ipdb.set_trace()

    preprocessed_data_set = IndexableDataset(
        OrderedDict([('features', features), ('targets', targets)]),
        axis_labels=data_set.axis_labels)

    return preprocessed_data_set


def remove_examples(data_set, num_noised, output_size):
    state = data_set.open()
    features, targets = data_set.get_data(
        state=state, request=slice(0, data_set.num_examples))
    data_set.close(state=state)

    # remove num_noised examples from data_set
    targets = targets[num_noised:]
    features = features[num_noised:]

    preprocessed_data_set = IndexableDataset(
        OrderedDict([('features', features), ('targets', targets)]),
        axis_labels=data_set.axis_labels)

    return preprocessed_data_set

def permute_input_dims(data_set, permutation):
    state = data_set.open()
    features, targets = data_set.get_data(
        state=state, request=slice(0, data_set.num_examples))
    data_set.close(state=state)

    # change num_noised examples to have incorrect labels.
    features_shape = features.shape
    features = features.reshape(len(features), len(permutation))
    features = features[:, permutation]
    features = features.reshape(features_shape)

    preprocessed_data_set = IndexableDataset(
        OrderedDict([('features', features), ('targets', targets)]),
        axis_labels=data_set.axis_labels)

    return preprocessed_data_set

def get_data_streams(data_set, batch_size, percent_noised=0, remove_noised=0, permuted=False, apply_default_transformers=True):

    if data_set == "MNIST":
        input_size = (28, 28)
        output_size = 10
        num_examples = [50000,10000,10000]
        train = MNIST(which_sets=("train",), subset=slice(None, 50000))
        valid = MNIST(which_sets=("train",), subset=slice(50000, None) )
        test = MNIST(("test",))
    elif data_set == "CIFAR10":
        input_size = (32, 32, 3)
        output_size = 10
        num_examples = [40000,10000,10000]
        train = CIFAR10(which_sets=("train",), subset=slice(None, 40000))
        valid = CIFAR10(which_sets=("train",), subset=slice(40000, None) )
        test = CIFAR10(("test",))

    if remove_noised: # does this result in the proper train.num_examples?
        train = remove_examples(train, int(percent_noised * num_examples[0] / 100), output_size)
    elif percent_noised > 0:
        train = noise_labels(train, int(percent_noised * num_examples[0] / 100), output_size)


    if permuted:
        permutation = np.random.permutation(np.prod(input_size))
        train = permute_input_dims(train, permutation)
        valid = permute_input_dims(valid, permutation)
        test = permute_input_dims(test, permutation)

    train_stream = DataStream.default_stream(
        train, iteration_scheme=ShuffledScheme(
            train.num_examples, batch_size))
    valid_stream = DataStream.default_stream(
        valid, iteration_scheme=ShuffledScheme(
            valid.num_examples, batch_size))
    test_stream = DataStream.default_stream(
        test, iteration_scheme=ShuffledScheme(
            test.num_examples, batch_size))

    if apply_default_transformers:
        train.apply_default_transformers(train_stream)
        valid.apply_default_transformers(valid_stream)
        test.apply_default_transformers(test_stream)

    return input_size, output_size, train_stream, valid_stream, test_stream


def test_get_data_streams():
    rval = []
    for data_set in ["MNIST", "CIFAR10"]:
        for batch_size in [1, 10]:
            for percent_noised in [0,99]:
                for remove_noised in [0,1]:
                    for permuted in [0,1]:
                        rval.append(get_data_streams(data_set, batch_size, percent_noised, remove_noised, permuted))
                        print "data_set, batch_size, percent_noised, remove_noised, permuted"
                        print data_set, batch_size, percent_noised, remove_noised, permuted
    #return test2(rval)
    return rval


def test2(rval):
    """ Call on the rval from test_get_data_streams in order to see some of the data from each dataset"""
    exampless = []
    for rr in rval[:8][::2]:
        iter = rr[2].get_epoch_iterator()
        examples = []
        n = 0
        for ex in iter:
            if n < 10:
                examples.append(ex); n += 1
            else:
                continue
        exampless.append(examples)

    for i in range(8)[::2]:
        figure()
        for n in range(9):
            subplot(3,3,n+1)
            imshow(exampless[i][n][0][0,0])
            print exampless[i][n][1]
        print '\n'

    #return rval
    return exampless

if 0:
    rval = test_get_data_streams()
    test2(rval)

