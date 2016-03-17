import cPickle as pickle
import numpy as np
from pylab import *

def get_all_keys(log):
    keys = []
    for kk in log.keys():
        entry = log[kk]
        for key in entry.keys():
            if not key in keys:
                keys.append(key)
    return keys

def search_log(log, str, txt=False, dtype=None):
    """ Returns a list extracted from the log"""
    if txt:
        return search_log_txt(log, str, dtype)
    rval = []
    for kk in log.keys():
        entry = log[kk]
        try:
            rval.append(entry[str])
        except:
            pass
    return rval

def search_log_txt(log, str, dtype=None):
    """
    return a list extracted from the text log
    assumes dtype is float
    """
    if dtype is None:
        dtype = float
    with open(log, 'r') as file_:
        lines = file_.readlines()
    lines = [line for line in lines if str in line]
    lines = [line.split(":")[1] for line in lines]
    rval = []
    for line in lines:
        try:
            rval.append(dtype(line))
        except:
            pass # TODO: less hacky
    return rval

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

            strs = ['valid_per', 'valid_sequence_log_likelihood', 'sequence_log_likelihood']
            results = []
            for str in strs:
                results.append(search_log(log, str))

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

