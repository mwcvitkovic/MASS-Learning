'''
Needs to be run from the project root directory
'''
import copy
import os
from itertools import cycle

from ..utils import run_script_with_kwargs

n_devices = 2
n_seeds = 5

device_cycle = cycle(['cuda:{}'.format(i) for i in range(n_devices)])
shared_kwargs = dict(
    debug_network=False,
    dataset_name='CIFAR10',
    model_kwargs=dict(
        net_name='SmallMLP',
        net_kwargs=dict(
            out_dim=10,
            nonlinearity='elu',
            batch_norm=False,
            dropout=False
        ),
    ),
    normalize_inputs=True,
    batch_size=256,
    epochs=None,
    total_batches=5e4,
    val_size=0.1,
    train_size='max',
    optimizer_class_name='SGD',
    optimizer_kwargs=dict(
        lr=1e-3,
        momentum=0.9,
    ),
    train_loss_plot_interval=500,
    train_experiments_and_kwargs=[
        ('ModelLossAndAccuracy', dict(run_interval=500)),
        ('LogModelParameters', dict(run_interval=500, log_var_dist=True)),
        ('LogEmbeddings', dict(run_interval=500, stripplot=True)),
        ('MASSLossTerms', dict(run_interval=500,
                               n_mixture_components=10,
                               covariance_type='full',
                               max_batches=15))]
)
for seed in range(5):
    kwargs = copy.deepcopy(shared_kwargs)
    kwargs['seed'] = seed
    kwargs['model_class_name'] = 'SoftmaxCE'

    kwargs['log_dir'] = os.path.join('SmallMLPMASSLossTerms',
                                     kwargs['model_class_name'],
                                     'seed{}'.format(seed))
    kwargs['device_id'] = device_cycle.__next__()
    run_script_with_kwargs('./start_training.py', kwargs)

    kwargs = copy.deepcopy(shared_kwargs)
    kwargs['seed'] = seed
    kwargs['model_class_name'] = 'ReducedJacMASSCE'
    kwargs['model_kwargs']['var_dist_init_strategy'] = 'zeros'
    kwargs['model_kwargs']['beta'] = 0.001
    kwargs['model_kwargs']['n_mixture_components'] = 2
    kwargs['optimizer_kwargs']['var_dist_optimizer_kwargs'] = dict(
        lr=kwargs['optimizer_kwargs']['lr']
    )
    kwargs['log_dir'] = os.path.join('SmallMLPMASSLossTerms',
                                     kwargs['model_class_name'],
                                     'seed{}'.format(seed))
    kwargs['device_id'] = device_cycle.__next__()
    run_script_with_kwargs('./start_training.py', kwargs)
