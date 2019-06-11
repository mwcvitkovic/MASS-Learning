'''
Needs to be run from the project root directory
'''
import copy
import csv
import os
from datetime import datetime

from scripts.utils import evaluate_seeds

checkpoint = '100000'

shared_kwargs = dict(
    seed=141,
    dataset_name='CIFAR10',
    test_experiments_and_kwargs=[
        ('ModelLossAndAccuracy', dict(run_interval=None)),
        ('UncertaintyQuantification', dict(run_interval=None)),
        ('OODDetection', dict(run_interval=None)),
    ],
    normalize_inputs=True,
    batch_size=256,
    device_id='cuda:0'
)

datadir = os.path.join('.', 'runs', 'SmallMLPAccRegUQOOD')

results = []
training_set_sizes = [2500, 10000, 40000]
for training_set_size in training_set_sizes:
    kwargs = copy.deepcopy(shared_kwargs)
    kwargs['model_logdir'] = os.path.join(datadir,
                                          'SoftmaxCE',
                                          'trainsize{}'.format(training_set_size),
                                          'no_reg')
    kwargs['checkpoint'] = checkpoint
    results.append(evaluate_seeds(kwargs))

    kwargs = copy.deepcopy(shared_kwargs)
    kwargs['model_logdir'] = os.path.join(datadir,
                                          'SoftmaxCE',
                                          'trainsize{}'.format(training_set_size),
                                          'wd')
    kwargs['checkpoint'] = checkpoint
    results.append(evaluate_seeds(kwargs))

    kwargs = copy.deepcopy(shared_kwargs)
    kwargs['model_logdir'] = os.path.join(datadir,
                                          'SoftmaxCE',
                                          'trainsize{}'.format(training_set_size),
                                          'dropout')
    kwargs['checkpoint'] = checkpoint
    results.append(evaluate_seeds(kwargs))

for training_set_size in training_set_sizes:
    for beta in [0.1, 0.01, 0.001]:
        kwargs = copy.deepcopy(shared_kwargs)
        kwargs['model_logdir'] = os.path.join(datadir,
                                              'VIB',
                                              'trainsize{}'.format(training_set_size),
                                              'beta{}'.format(str(beta).replace('.', ',')),
                                              'no_dropout')
        kwargs['checkpoint'] = checkpoint
        results.append(evaluate_seeds(kwargs))

        kwargs = copy.deepcopy(shared_kwargs)
        kwargs['model_logdir'] = os.path.join(datadir,
                                              'VIB',
                                              'trainsize{}'.format(training_set_size),
                                              'beta{}'.format(str(beta).replace('.', ',')),
                                              'dropout')
        kwargs['checkpoint'] = checkpoint
        results.append(evaluate_seeds(kwargs))

for training_set_size in training_set_sizes:
    for beta in [0.01, 0.001, 0.0001, 0]:
        kwargs = copy.deepcopy(shared_kwargs)
        kwargs['model_logdir'] = os.path.join(datadir,
                                              'ReducedJacMASSCE',
                                              'trainsize{}'.format(training_set_size),
                                              'beta{}'.format(str(beta).replace('.', ',')),
                                              'no_dropout')
        kwargs['checkpoint'] = checkpoint
        results.append(evaluate_seeds(kwargs))

        kwargs = copy.deepcopy(shared_kwargs)
        kwargs['model_logdir'] = os.path.join(datadir,
                                              'ReducedJacMASSCE',
                                              'trainsize{}'.format(training_set_size),
                                              'beta{}'.format(str(beta).replace('.', ',')),
                                              'dropout')
        kwargs['checkpoint'] = checkpoint
        results.append(evaluate_seeds(kwargs))

results_path = os.path.join('.', 'runs', 'eval_results')
os.makedirs(results_path, exist_ok=True)
results_filename = os.path.join(results_path, '_'.join(['SmallMLPAccRegUQOOD',
                                                        datetime.now().strftime('%b%d_%H-%M-%S'),
                                                        'results.csv'])
                                )
with open(results_filename, 'w') as csvfile:
    fieldnames = list(results[0].keys())
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
        writer.writerow(result)
