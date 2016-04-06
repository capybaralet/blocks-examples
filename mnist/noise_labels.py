
import h5py
import os
import numpy
np = numpy


# FIXME 
mnist = h5py.File(os.environ['DATAPATH'] + 'mnist/mnist.hdf5', 'r')

def make_dataset(noised_fraction=.5):
    output = h5py.File(os.environ['DATAPATH'] + 'mnist/mnist_' + 'noised_fraction=' + str(noised_fraction) + '.hdf5', 'w')
    features = output.create_dataset('features', (70000, 1, 28, 28), dtype='float32')
    features[...] = np.array(mnist['features'])
    features.dims[0].label = 'batch'
    features.dims[1].label = 'channel'
    features.dims[2].label = 'height'
    features.dims[3].label = 'width'
    # make new targets
    targets = output.create_dataset('targets', (70000, 1), dtype='uint8')
    targets[...] = np.array(replace_labels(mnist['targets'], noised_fraction))
    # save dataset
    output.flush()
    output.close()

def replace_labels(targets, noised_fraction):
    rval = np.array(targets)
    tr_noised = int(60000 * noised_fraction)
    te_noised = int(10000 * noised_fraction)
    rval[:tr_noised] = (rval[:tr_noised] + np.random.randint(1, 10, tr_noised).reshape((-1,1))) % 10
    rval[60000: 60000 + te_noised] = (rval[60000: 60000 + te_noised] + np.random.randint(1, 10, te_noised).reshape((-1,1))) % 10
    return rval


make_dataset()

