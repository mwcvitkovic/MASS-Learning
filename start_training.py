import base64
import pickle
import pprint
import sys
from math import ceil

import numpy as np
import torch

import experiments
import models
from models.utils import save_model_kwargs, load_model_from_checkpoint
from utils import get_dataloaders, setup_writer


def train_epoch(writer,
                model,
                train_loader,
                optimizer,
                scheduler,
                epoch,
                total_batches,
                train_experiments):
    for batch_idx, (data, target) in enumerate(train_loader):
        for ex in train_experiments:
            if writer.global_step % ex.run_interval == 0:
                ex.run(batch_idx, epoch)
        model.train()
        data, target = data.to(model.device), target.to(model.device)
        optimizer.zero_grad()
        loss, output = model.net_forward_and_loss(data, target)
        if torch.isnan(loss):
            raise ValueError('Training loss value for {} was NaN'.format(model.__class__.__name__))
        loss.backward()
        optimizer.step()
        if writer.global_step % writer.train_loss_plot_interval == 0:
            writer.add_scalar('Train Loss/Train Loss', loss.item(), writer.global_step)
        writer.global_step += 1
        if total_batches is not None and writer.global_step >= total_batches:
            for ex in train_experiments:
                if writer.global_step % ex.run_interval == 0:
                    ex.run(batch_idx, epoch)
            break
    scheduler.step()


def train_model(
        writer,
        seed,
        dataset_name,
        model_class_name,
        model_kwargs,
        normalize_inputs,
        batch_size,
        train_size,
        val_size,
        epochs,
        total_batches,
        optimizer_class_name,
        optimizer_kwargs,
        lr_scheduler_class_name,
        lr_scheduler_kwargs,
        model_logdir=None,
        checkpoint=None,
        train_experiments_and_kwargs=[],
        device_id='cpu'):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_id if torch.cuda.is_available() else "cpu")

    model_class = models.__dict__[model_class_name]
    train_loader, val_loader, _, in_shape, n_classes = get_dataloaders(dataset_name=dataset_name,
                                                                       batch_size=batch_size,
                                                                       train_size=train_size,
                                                                       val_size=val_size,
                                                                       device_id=device_id,
                                                                       normalize_inputs=normalize_inputs)

    if model_logdir or checkpoint:
        model = load_model_from_checkpoint(writer, model_logdir, checkpoint)
    else:
        model_kwargs['n_classes'] = n_classes
        model_kwargs['net_kwargs']['in_shape'] = in_shape
        model = model_class(writer, **model_kwargs)
    save_model_kwargs(writer, model_class_name, model_kwargs)

    optimizer = model.get_optimizer(optimizer_class_name, optimizer_kwargs)
    scheduler = torch.optim.lr_scheduler.__dict__[lr_scheduler_class_name](optimizer, **lr_scheduler_kwargs)

    train_experiments = []
    for ex in train_experiments_and_kwargs:
        train_experiments.append(experiments.__dict__[ex[0]](writer=writer,
                                                             model=model,
                                                             train_loader=train_loader,
                                                             val_loader=val_loader,
                                                             **ex[1]))
    model.initialize(train_loader)
    model.to(device)
    if epochs is None:
        epochs = ceil(total_batches / len(train_loader))
    for epoch in range(1, epochs + 1):
        train_epoch(writer,
                    model,
                    train_loader,
                    optimizer,
                    scheduler,
                    epoch,
                    total_batches,
                    train_experiments)


def main(kwargs):
    # Workaround for pytorch bug where multiple gpu processes all like to use gpu0
    if 'cuda' in kwargs['device_id'] and torch.cuda.is_available():
        torch.cuda.set_device(int(kwargs['device_id'][-1]))

    assert kwargs['epochs'] is None or kwargs['total_batches'] is None, \
        "Specify either number of epochs to train for, or total batches to train for, not both."

    writer = setup_writer(kwargs.pop('log_dir'),
                          kwargs.pop('debug_network'),
                          kwargs.pop('train_loss_plot_interval'),
                          kwargs.pop('absolute_logdir_path', False))
    writer.add_text('kwargs', pprint.pformat(kwargs).replace('\n', '\t\n'))
    train_model(writer, **kwargs)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        kwargs = dict(
            log_dir='debug',
            debug_network=False,
            seed=2,
            dataset_name='CIFAR10',
            model_class_name='ReducedJacMASSCE',
            model_kwargs=dict(
                net_name='SmallMLP',
                net_kwargs=dict(
                    out_dim=10,
                    nonlinearity='elu',
                    batch_norm=True,
                    dropout=False
                ),
                var_dist_init_strategy='zeros',
                beta=0.001,
                n_mixture_components=2,
            ),
            normalize_inputs=True,
            batch_size=256,
            epochs=None,
            total_batches=1e4,
            val_size=0.1,
            train_size='max',
            optimizer_class_name='Adam',
            optimizer_kwargs=dict(
                lr=3e-4,
                var_dist_optimizer_kwargs=dict(
                    lr=5e-4
                )
            ),
            lr_scheduler_class_name='ExponentialLR',
            lr_scheduler_kwargs=dict(
                 gamma=1.0
            ),
            train_loss_plot_interval=5,
            train_experiments_and_kwargs=[
                ('ModelLossAndAccuracy', dict(run_interval=1000))
            ],
            device_id='cuda:0')
    else:
        kwargs = pickle.loads(base64.b64decode(sys.argv[1]))
    main(kwargs)
