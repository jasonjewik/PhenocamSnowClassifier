<h1 align="center">PhenoCamSnow</h1>

[![Documentation Status](https://readthedocs.org/projects/phenocamsnow/badge/?version=latest)](https://phenocamsnow.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**PhenoCamSnow** is a Python package for quickly building deep learning models to classify [PhenoCam images](https://phenocam.sr.unh.edu/).

## Quickstart

PhenoCamSnow requires Python 3.10+ and can be installed via pip:

```console
pip install phenocam-snow
```

The following code snippets show how to train and evaluate a model on classifying images from the canadaojp site into "snow", "no snow", and "too dark".

```console
python -m phenocam_snow.train \
   canadaojp \
   --new \
   --n_train 120 \
   --n_test 30 \
   --classes snow no_snow too_dark
```
This will print out the file path of the best model, which can be substituted into the next command.

```console
python -m phenocam_snow.predict \
   canadaojp \
   canadaojp_lightning_logs/version_0/checkpoints/epoch=12-step=78.ckpt \
   --categories snow no_snow too_dark \
   --directory 'canadaojp/test'
```

Advanced usage details can be found in the [documentation](http://phenocamsnow.readthedocs.io/).

## Development

PhenoCamSnow uses [Poetry](https://python-poetry.org) for package management. After cloning the repository to your local development environment, install the dependencies with `poetry install`. You can find the location of the virtual environment to set your IDE's Python executable path with `poetry env info`.

Install the pre-commit hooks with `make install-git-hooks`.

## Citation

If you use PhenoCamSnow for your work, please see [`CITATION.cff`](CITATION.cff) or use the citation prompt provided by GitHub in the sidebar.

## Acknowledgements

[Professor Jochen Stutz](https://atmos.ucla.edu/people/faculty/jochen-stutz) and [Zoe Pierrat](https://atmos.ucla.edu/people/graduate-student/zoe-pierrat).
