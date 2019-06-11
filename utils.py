import os
import socket
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import SubsetRandomSampler
from torchvision import datasets, transforms


def setup_writer(log_dir, debug_network, train_loss_plot_interval, absolute_path=False):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    if absolute_path:
        log_dir = os.path.join(log_dir, current_time + '_' + socket.gethostname())
    else:
        log_dir = os.path.join('runs', log_dir, current_time + '_' + socket.gethostname())

    writer = SummaryWriter(log_dir=log_dir)
    writer.train_loss_plot_interval = train_loss_plot_interval
    writer.global_step = 0

    if debug_network:
        writer.debug_network = True
        writer.debug_info = writer.add_histogram
    else:
        writer.debug_network = False
        writer.debug_info = lambda *args: None

    return writer


def log_param_values(writer, model):
    for name, param in model.net.named_parameters():
        writer.add_histogram('Parameter Values/{}'.format(name), param, writer.global_step)
        if param.grad is not None:
            writer.add_histogram('Gradients/{}'.format(name), param.grad, writer.global_step)
    if hasattr(model, 'marginal'):
        for name, param in model.marginal.named_parameters():
            writer.add_histogram('Parameter Values/{}'.format(name), param, writer.global_step)
            if param.grad is not None:
                writer.add_histogram('Gradients/{}'.format(name), param.grad, writer.global_step)
    if hasattr(model, 'decoder'):
        for name, param in model.decoder.named_parameters():
            writer.add_histogram('Parameter Values/{}'.format(name), param, writer.global_step)
            if param.grad is not None:
                writer.add_histogram('Gradients/{}'.format(name), param.grad, writer.global_step)


def log_var_dist_parameters(writer, var_dist, comment=''):
    mls = [mog.mixture_logits for mog in var_dist.q]
    mls = torch.stack(mls).squeeze(-2)
    fig = plt.figure(figsize=(8, 8))
    hm = sns.heatmap(mls.detach().cpu(), annot=True, square=True, cbar=False)
    hm.set_ylabel('Output Class')
    hm.set_xlabel('Mixture Component')
    writer.add_figure('Var Dist/mixture_logits {}'.format(comment), fig, writer.global_step)
    plt.close('all')
    for mc in range(var_dist.n_mixture_components):
        locs, scale_trils = [mog.loc[mc] for mog in var_dist.q], [mog.scale_tril[mc] for mog in var_dist.q]
        locs = torch.stack(locs)
        scale_trils = torch.stack(scale_trils)
        fig = plt.figure(figsize=(8, 8))
        hm = sns.heatmap(locs.detach().cpu(), annot=True, square=True, cbar=False, xticklabels=False)
        hm.set_ylabel('Output Class')
        hm.set_xlabel('Representation Dimension')
        writer.add_figure('Var Dist/loc component {} {}'.format(mc, comment), fig, writer.global_step)

        fig, axes = plt.subplots(nrows=scale_trils.shape[0], figsize=(7, 9 * scale_trils.shape[0]))
        for i, st in enumerate(scale_trils):
            axes[i].set_title(str(i))
            sns.heatmap(st.detach().cpu(), annot=True, square=True, cbar=False, yticklabels=False,
                        xticklabels=False, ax=axes[i])
        writer.add_figure('Var Dist/scale tril component {} {}'.format(mc, comment), fig, writer.global_step)
        plt.close('all')


def get_dataloaders(dataset_name, batch_size, train_size, val_size, device_id, normalize_inputs, num_workers=1):
    dataset = datasets.__dict__[dataset_name]

    if dataset_name == 'MNIST':
        mean_std = ((0.1305,), (0.3080,))
        n_classes = 10
    elif dataset_name == 'FashionMNIST':
        mean_std = ((0.2858,), (0.3530,))
        n_classes = 10
    elif dataset_name == 'CIFAR10':
        mean_std = ((0.4912, 0.4818, 0.4460), (0.2470, 0.2434, 0.2614))
        n_classes = 10
    elif dataset_name == 'CIFAR100':
        mean_std = ((0.5073, 0.4868, 0.4411), (0.2673, 0.2563, 0.2762))
        n_classes = 100
    elif dataset_name == 'SVHN':
        mean_std = ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        n_classes = 10
    else:
        raise ValueError('Need to give normalization parameters for this dataset.')
    if not normalize_inputs:
        mean_std = ((0.,), (1.,))

    dataset_kwargs = dict(transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*mean_std)
    ]),
        download=True)
    if dataset_name == 'SVHN':
        dataset_kwargs['split'] = 'train'
    else:
        dataset_kwargs['train'] = True
    trainval_dataset = dataset(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', dataset_name),
                               **dataset_kwargs)
    if dataset_name == 'SVHN':
        dataset_kwargs['split'] = 'test'
    else:
        dataset_kwargs['train'] = False
    test_dataset = dataset(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', dataset_name),
                           **dataset_kwargs)
    in_shape = tuple(trainval_dataset[0][0].shape)

    # Treat floats as percentages
    if isinstance(train_size, float):
        train_size = int(len(trainval_dataset) * train_size)
    if isinstance(val_size, float):
        val_size = int(len(trainval_dataset) * val_size)

    val_size = max(0, val_size)
    if train_size == 'max':
        train_size = len(trainval_dataset) - val_size
    assert train_size + val_size <= len(trainval_dataset), 'Incompatible train_size and val_size specified'
    train_dataset, val_dataset, _ = torch.utils.data.random_split(trainval_dataset, [train_size, val_size, len(
        trainval_dataset) - train_size - val_size])

    dataloader_kwargs = {'num_workers': num_workers, 'pin_memory': True} if (
            'cuda' in device_id and torch.cuda.is_available()) else {}
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               **dataloader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             **dataloader_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              **dataloader_kwargs)
    # # Uncomment to develop when there's no internet
    # train_loader = [(torch.zeros(256, 1, 28, 28).normal_(), torch.tensor([0]))]
    # val_loader = None
    # test_loader = None

    return train_loader, val_loader, test_loader, in_shape, n_classes
