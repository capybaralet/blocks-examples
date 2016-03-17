
import numpy
np = numpy
import os

from speechgeneration.wordembeddings.launchers.job_launchers import launch


filename = os.path.basename(__file__)[:-3]
script_path = "./lenet_script.py"


jobs = []
#for seed in range(1,3):
for data_set in ['MNIST', 'CIFAR10']:
    for permuted in [0,1]:
        cmd_line_args = []
        # TODO: make this automatic somehow???
        cmd_line_args.append(['data_set', data_set])
        cmd_line_args.append(['permuted', permuted])
        jobs.append((script_path, cmd_line_args))

gpus = range(8)
ngpu = gpus[0]
print "njobs =", len(jobs) #30?
print "ngpus =", len(gpus)
print jobs

launch(jobs, gpus, filename, sequential=False)#True)

# log this launch (TODO: make compatible with sequential launch... (or is it already??)
os.system("touch " + os.path.join(os.environ["SAVE_PATH"], "logging", "launched___" + filename))



