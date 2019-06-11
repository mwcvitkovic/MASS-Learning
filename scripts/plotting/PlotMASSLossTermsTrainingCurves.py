import os

import pandas as pd
import seaborn as sns

sns.set()

sns.set_style("whitegrid")

# Expects data downloaded from tensorboard. You have to do this manually from within tensorboard; the file paths below
#   are example placeholders.
# Your CSVs will have different filenames. You need to set datadir to be wherever you've stored the tensorboard CSVs
# Note that you need to replace the tensorboard seed tag with "seedX" in the filename.

datadir = ''
SoftmaxCE_training_curve_paths_models_terms = [
    (
        'run-SoftmaxCE_seedX_Feb08_19-24-56_ip-0-0-0-0-tag-MASSLossTerms_train__cross_entropy_term.csv',
        'SoftmaxCE',
        '$H(Y | f(X))$ (nats)'
    ),
    (
        'run-SoftmaxCE_seedX_Feb08_19-24-56_ip-0-0-0-0-tag-MASSLossTerms_train__entropy_term.csv',
        'SoftmaxCE',
        '$H(f(X))$ (nats)'
    ),
    (
        'run-SoftmaxCE_seedX_Feb08_19-24-56_ip-0-0-0-0-tag-MASSLossTerms_train__Jacobian_term.csv',
        'SoftmaxCE',
        '$- \mathbb{E}_X[\log \ J_{f}(X)]$'
    ),
    (
        'run-SoftmaxCE_seedX_Feb08_19-24-56_ip-0-0-0-0-tag-ModelLossAndAccuracy_Validation_Accuracy.csv',
        'SoftmaxCE',
        'Validation Accuracy (%)'
    ),
]
ReducedJacMASSCE_training_curve_paths_models_terms = [
    (
        'run-ReducedJacMASSCE_seedX_Feb08_19-24-56_ip-0-0-0-0-tag-MASSLossTerms_train__cross_entropy_term.csv',
        'MASS',
        '$H(Y | f(X))$ (nats)'
    ),
    (
        'run-ReducedJacMASSCE_seedX_Feb08_19-24-56_ip-0-0-0-0-tag-MASSLossTerms_train__entropy_term.csv',
        'MASS',
        '$H(f(X))$ (nats)'
    ),
    (
        'run-ReducedJacMASSCE_seedX_Feb08_19-24-56_ip-0-0-0-0-tag-MASSLossTerms_train__Jacobian_term.csv',
        'MASS',
        '$- \mathbb{E}_X[\log \ J_{f}(X)]$'
    ),
    (
        'run-ReducedJacMASSCE_seedX_Feb08_19-24-56_ip-0-0-0-0-tag-ModelLossAndAccuracy_Validation_Accuracy.csv',
        'MASS',
        'Validation Accuracy (%)'
    ),
]
total_steps = 50000
training_curves = []
for seed in range(5):
    for path, model, term in SoftmaxCE_training_curve_paths_models_terms + ReducedJacMASSCE_training_curve_paths_models_terms:
        path = os.path.join(datadir, path).replace('seedX', 'seed{}'.format(seed))
        df = pd.read_csv(path, usecols=['Step', 'Value'])
        df['Training Method'] = model
        df['Loss Term'] = term
        df['Seed'] = seed
        training_curves.append(df.loc[df['Step'] <= total_steps])

training_curves = pd.concat(training_curves)
training_curves.rename(columns={'Step': 'Training Step'}, inplace=True)

g = sns.lineplot(data=training_curves,
                 x='Training Step',
                 y='Value',
                 hue='Loss Term',
                 style='Training Method',
                 err_style='band',
                 ci='sd')
g.set_xlim(0, 50000)
g.get_legend().set_bbox_to_anchor((1.05, 0.8))
g.get_figure().set_figwidth(10)
g.get_figure().set_figheight(5)
g.get_figure().subplots_adjust(bottom=0.2, right=0.6)
header = g.get_legend().texts[0]
header._fontproperties = header._fontproperties.copy()
g.get_legend().texts[5]._fontproperties = header._fontproperties
header.set_weight('bold')
g.get_figure().savefig('./runs/TrainingCurves.pdf', format='pdf')
