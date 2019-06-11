"""
Needs to be run from the project root directory
"""
import copy
import os
from itertools import cycle

from ..utils import run_script_with_kwargs

n_devices = 2
n_seeds = 4

device_cycle = cycle(['cuda:{}'.format(i) for i in range(n_devices)])
shared_kwargs = dict(
    debug_network=False,
    dataset_name='CIFAR10',
    model_kwargs=dict(
        net_name='SmallMLP',
        net_kwargs=dict(
            nonlinearity='elu',
            batch_norm=True,
        ),
    ),
    normalize_inputs=True,
    batch_size=256,
    epochs=None,
    total_batches=60000,
    val_size=0.1,
    optimizer_class_name='Adam',
    optimizer_kwargs=dict(
        lr=0.0005,
    ),
    train_loss_plot_interval=5000,
    train_experiments_and_kwargs=[
        ('ModelLossAndAccuracy', dict(run_interval=5000)),
        ('SaveModelParameters', dict(run_interval=5000)),
        ('UncertaintyQuantification', dict(run_interval=5000)),
        ('OODDetection', dict(run_interval=5000)),
    ],
)

training_set_sizes = [2500, 10000, 40000]
for training_set_size in training_set_sizes:
    for seed in range(n_seeds):
        kwargs = copy.deepcopy(shared_kwargs)
        kwargs['seed'] = seed
        kwargs['model_class_name'] = 'SoftmaxCE'
        kwargs['model_kwargs']['net_kwargs']['out_dim'] = 10
        kwargs['train_size'] = training_set_size

        kwargs['log_dir'] = os.path.join('SmallMLPAccRegUQOOD',
                                         kwargs['model_class_name'],
                                         'trainsize{}'.format(training_set_size),
                                         'no_reg',
                                         'seed{}'.format(seed))
        device_id = device_cycle.__next__()
        kwargs['device_id'] = device_id
        kwargs['optimizer_kwargs']['weight_decay'] = 0.0
        kwargs['model_kwargs']['net_kwargs']['dropout'] = False
        run_script_with_kwargs('./start_training.py', kwargs)

        kwargs['log_dir'] = os.path.join('SmallMLPAccRegUQOOD',
                                         kwargs['model_class_name'],
                                         'trainsize{}'.format(training_set_size),
                                         'wd',
                                         'seed{}'.format(seed))
        device_id = device_cycle.__next__()
        kwargs['device_id'] = device_id
        kwargs['optimizer_kwargs']['weight_decay'] = 0.01
        kwargs['model_kwargs']['net_kwargs']['dropout'] = False
        run_script_with_kwargs('./start_training.py', kwargs)

        kwargs['log_dir'] = os.path.join('SmallMLPAccRegUQOOD',
                                         kwargs['model_class_name'],
                                         'trainsize{}'.format(training_set_size),
                                         'dropout',
                                         'seed{}'.format(seed))
        device_id = device_cycle.__next__()
        kwargs['device_id'] = device_id
        kwargs['optimizer_kwargs']['weight_decay'] = 0.0
        kwargs['model_kwargs']['net_kwargs']['dropout'] = True
        run_script_with_kwargs('./start_training.py', kwargs)

        for beta in [0.1, 0.01, 0.001]:
            kwargs = copy.deepcopy(shared_kwargs)
            kwargs['seed'] = seed
            kwargs['model_class_name'] = 'VIB'
            kwargs['model_kwargs']['net_kwargs']['out_dim'] = 15
            kwargs['train_size'] = training_set_size
            kwargs['model_kwargs']['beta'] = beta
            kwargs['model_kwargs']['n_mixture_components'] = 10
            kwargs['model_kwargs']['covariance_type'] = 'full'
            kwargs['model_kwargs']['train_var_dist_samples'] = 5
            kwargs['model_kwargs']['test_var_dist_samples'] = 10

            kwargs['log_dir'] = os.path.join('SmallMLPAccRegUQOOD',
                                             kwargs['model_class_name'],
                                             'trainsize{}'.format(training_set_size),
                                             'beta{}'.format(str(beta).replace('.', ',')),
                                             'no_dropout',
                                             'seed{}'.format(seed))
            device_id = device_cycle.__next__()
            kwargs['device_id'] = device_id
            kwargs['model_kwargs']['net_kwargs']['dropout'] = False
            run_script_with_kwargs('./start_training.py', kwargs)

            kwargs['log_dir'] = os.path.join('SmallMLPAccRegUQOOD',
                                             kwargs['model_class_name'],
                                             'trainsize{}'.format(training_set_size),
                                             'beta{}'.format(str(beta).replace('.', ',')),
                                             'dropout',
                                             'seed{}'.format(seed))
            device_id = device_cycle.__next__()
            kwargs['device_id'] = device_id
            kwargs['model_kwargs']['net_kwargs']['dropout'] = True
            run_script_with_kwargs('./start_training.py', kwargs)

        for beta in [0.01, 0.001, 0.0001, 0]:
            kwargs = copy.deepcopy(shared_kwargs)
            kwargs['seed'] = seed
            kwargs['model_class_name'] = 'ReducedJacMASSCE'
            kwargs['model_kwargs']['net_kwargs']['out_dim'] = 15
            kwargs['train_size'] = training_set_size
            kwargs['model_kwargs']['var_dist_init_strategy'] = 'random'
            kwargs['model_kwargs']['beta'] = beta
            kwargs['model_kwargs']['n_mixture_components'] = 10
            kwargs['optimizer_kwargs']['var_dist_optimizer_kwargs'] = dict(
                lr=2.5e-5
            )

            kwargs['log_dir'] = os.path.join('SmallMLPAccRegUQOOD',
                                             kwargs['model_class_name'],
                                             'trainsize{}'.format(training_set_size),
                                             'beta{}'.format(str(beta).replace('.', ',')),
                                             'no_dropout',
                                             'seed{}'.format(seed))
            device_id = device_cycle.__next__()
            kwargs['device_id'] = device_id
            kwargs['model_kwargs']['net_kwargs']['dropout'] = False
            run_script_with_kwargs('./start_training.py', kwargs)

            kwargs['log_dir'] = os.path.join('SmallMLPAccRegUQOOD',
                                             kwargs['model_class_name'],
                                             'trainsize{}'.format(training_set_size),
                                             'beta{}'.format(str(beta).replace('.', ',')),
                                             'dropout',
                                             'seed{}'.format(seed))
            device_id = device_cycle.__next__()
            kwargs['device_id'] = device_id
            kwargs['model_kwargs']['net_kwargs']['dropout'] = True
            run_script_with_kwargs('./start_training.py', kwargs)
