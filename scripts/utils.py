import base64
import copy
import os
import pickle
import subprocess
import sys

import numpy as np

from start_evaluating import main as evaluate_main


def run_script_with_kwargs(script_name, kwargs):
    session_name = kwargs['log_dir'].replace('/', '_')
    subprocess.run("tmux new-session -d -s {}".format(session_name), shell=True)
    subprocess.run("tmux send-keys -t {} '{} {} {}' C-m ".format(session_name,
                                                                 sys.executable,
                                                                 script_name,
                                                                 base64.b64encode(
                                                                     pickle.dumps(kwargs)).decode()),
                   shell=True,
                   env=os.environ.copy())
    subprocess.run("tmux send-keys -t {} 'exit' C-m ".format(session_name), shell=True)


def evaluate_seeds(orig_kwargs):
    results = []
    for seed in range(4):
        kwargs = copy.deepcopy(orig_kwargs)
        kwargs['model_logdir'] = os.path.join(kwargs['model_logdir'], 'seed{}'.format(seed))
        runs = os.listdir(kwargs['model_logdir'])
        assert len(runs) == 1
        kwargs['model_logdir'] = os.path.join(kwargs['model_logdir'], runs[0])
        results.append(evaluate_main(kwargs))
    agr_results = {'Model': orig_kwargs['model_logdir']}
    for key in results[0].keys():
        vals = [r[key] for r in results]
        if None not in vals:
            agr_results[key + '_values'] = vals
            agr_results[key + '_mean'] = float(np.mean(vals))
            agr_results[key + '_std'] = float(np.std(vals))
        else:
            agr_results[key + '_values'] = None
            agr_results[key + '_mean'] = None
            agr_results[key + '_std'] = None
    return agr_results
