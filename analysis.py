import cPickle as pickle
import numpy as np
from pylab import *
import os

def get_all_keys(log):
    keys = []
    for kk in log.keys():
        entry = log[kk]
        for key in entry.keys():
            if not key in keys:
                keys.append(key)
    return keys

def search_log(log, string, txt=False, dtype=None):
    """ Returns a list extracted from the log"""
    if txt:
        return search_log_txt(log, string, dtype)
    rval = []
    for kk in log.keys():
        entry = log[kk]
        try:
            rval.append(entry[string])
        except:
            pass
    return rval

def search_log_txt(log, string, dtype=None):
    """
    return a list extracted from the text log
    assumes dtype is float
    """
    if dtype is None:
        dtype = float
    with open(log, 'r') as file_:
        lines = file_.readlines()
    lines = [line for line in lines if string in line]
    lines = [line.split(":")[1] for line in lines]
    rval = []
    for line in lines:
        try:
            rval.append(dtype(line))
        except:
            pass # TODO: less hacky
    return rval


def lcs_from_log_txt(string='valid_error_rate', filter_in=[''], return_dict=True):
    fns = os.listdir('./')
    fns = [fn for fn in fns if all([ss in fn for ss in filter_in])]
    if return_dict:
        return {fn: search_log_txt(fn, string) for fn in fns}
    else:
        return [search_log_txt(fn, string) for fn in fns]
    

def plot_lcs(lcs):
    """
    takes a dictionary of learning_curves indexed by their filename,
    and plots them in two ways:

    1. in separate subplots with filenames as titles
    2. in the same plot with a legend
    """
    keys = lcs.keys()
    numplots = len(keys)
    numrows = sum([ (numplots - 1) / n**2  > 0 for n in range(1, 11) ]) + 1
    print numplots, numrows
    figure()
    for n in range(numrows**2):
        subplot(numrows, numrows, n+1)
        try:
            plot(lcs[keys[n]])
            title(keys[n])
        except:
            pass
    figure()
    for n in range(numplots):
        plot(lcs[keys[n]], label=keys[n])
    legend()
    

def analyze(paths):
    labels = []
    all_valid_results = []
    for path in paths:
        print path
        try:
            with open(path, 'rb') as f_:
                log = pickle.load(f_)
            print "len(log.keys()) =", len(log.keys())
            loglen = len(log.keys())

            strings = ['valid_per', 'valid_sequence_log_likelihood', 'sequence_log_likelihood']
            results = []
            for string in strings:
                results.append(search_log(log, string))

            keys = get_all_keys(log)
            valid_keys = [k for k in keys if 'valid' in k]

            valid_results = []
            for key in valid_keys:
                res = search_log(log, key)
                valid_results.append(res)
                print key, ":    length =", len(res)

            all_valid_results.append(valid_results)

            figure(51)
            plot(results[0])
            figure(52)
            plot(results[1])
            figure(53)
            plot(results[2][::100])
            labels.append(path)

        except:
            print '\n'
            print "could not load???", path
            #print Exception
            print '\n'

        legend(labels)
    return labels, all_valid_results

