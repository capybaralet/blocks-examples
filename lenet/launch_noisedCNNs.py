
import numpy
np = numpy
import os

from speechgeneration.wordembeddings.launchers.job_launchers import launch


filename = os.path.basename(__file__)[:-3]
script_path = "./lenet.py"


# TODO
seed = 99

momentum = .99


jobs = []
for data_set in ['MNIST', 'CIFAR10']:
    for percent_noised in range(0,100,10):
        for remove_noised in [0,1]:
            cmd_line_args = []
            # TODO: make this automatic somehow???
            cmd_line_args.append(['data_set', data_set])
            cmd_line_args.append(['percent_noised', percent_noised])
            cmd_line_args.append(['remove_noised', remove_noised])
            jobs.append((script_path, cmd_line_args))

gpus = range(4)
ngpu = gpus[0]
print "njobs =", len(jobs) #30?
print "ngpus =", len(gpus)
print jobs

launch(jobs, gpus, filename, sequential=True)

# log this launch (TODO: make compatible with sequential launch... (or is it already??)
os.system("touch " + os.path.join(os.environ["SAVE_PATH"], "logging", "launched___" + filename))



