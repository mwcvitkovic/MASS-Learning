# Minimal Achievable Sufficient Statistic Learning

## What is this?

This code reproduces all experiments in the paper [Minimal Achievable Sufficient Statistic Learning](https://arxiv.org/abs/1905.07822).


## License

The code in this repo is licensed under the [MIT license](http://opensource.org/licenses/mit-license.php).


## How do I run your code?
### Installation
Install the [Conda](https://conda.io/docs/index.html) python package manager.  Then follow the instructions [here](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)
using the file `environment.yml` file in this library's root directory to satisfy the python requirements to run this library's code.

In theory our code is operating-system-agnostic, but we ran all our experiments on Ubuntu Linux, so that's where you're most likely to have installation success.

### Tests (optional)
We included as many unit tests as we could.  They're in the `tests` directory, whose directory structure mirrors that of the rest of the library.

You can run them from the library's root directory with `python -m unittest`.

### Running the experiments from the paper
The scripts that run the experiments are in the `scripts` directory.  

Activate your `MASS-Learning` conda environment, adjust the options in the scripts to suit your machine, and run whatever experiments you like from the library's root directory as a module, e.g. `python -m scripts.paper_tables.SmallMLPAccRegUQOOD`.

Experiments log their results in the `runs` directory.  You can activate [tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard) to watch their progress.

Once the experiments are done, the scripts in `scripts/evaluations` or `scripts/plotting` will consume the logs and give you the results from the paper.

## Questions?
Feel free to get in touch with [Milan Cvitkovic](mailto:mwcvitkovic@gmail.com) or any of the other paper authors.  We'd love to hear from you!
