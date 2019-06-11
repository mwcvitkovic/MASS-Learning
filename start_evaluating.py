import base64
import pickle
import sys
from pprint import pprint

import numpy as np
import torch

import experiments
from models.utils import load_model_from_checkpoint
from tests import DummyWriter
from utils import get_dataloaders


def start_evaluating(
        writer,
        seed,
        dataset_name,
        test_experiments_and_kwargs,
        model_logdir,
        checkpoint,
        normalize_inputs,
        batch_size,
        device_id):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, _, test_loader, _, _ = get_dataloaders(dataset_name=dataset_name,
                                                         batch_size=batch_size,
                                                         train_size='max',
                                                         val_size=0,
                                                         device_id=device_id,
                                                         normalize_inputs=normalize_inputs,
                                                         num_workers=0)

    model = load_model_from_checkpoint(writer, model_logdir, checkpoint)
    device = torch.device(device_id if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_experiments = []
    for te, kwargs in test_experiments_and_kwargs:
        test_experiments.append(experiments.__dict__[te](writer=writer,
                                                         model=model,
                                                         train_loader=train_loader,
                                                         val_loader=test_loader,
                                                         **kwargs)
                                )
    results = {}
    for ex in test_experiments:
        results.update(ex.run(0, 0))
    pprint(results)
    return results


def main(kwargs):
    # Workaround for pytorch bug where multiple gpu processes all like to use gpu0
    if 'cuda' in kwargs['device_id'] and torch.cuda.is_available():
        torch.cuda.set_device(int(kwargs['device_id'][-1]))

    writer = DummyWriter()
    return start_evaluating(writer, **kwargs)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        kwargs = dict(
            seed=631,
            dataset_name='CIFAR10',
            test_experiments_and_kwargs=[('OODDetection', dict(run_interval=None))],
            model_logdir='./runs/<directory with data to evaluate>',
            checkpoint='40000',
            normalize_inputs=True,
            batch_size=256,
            device_id='cuda:0')
    else:
        kwargs = pickle.loads(base64.b64decode(sys.argv[1]))
    main(kwargs)
