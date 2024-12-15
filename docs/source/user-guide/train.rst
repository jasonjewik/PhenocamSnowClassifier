Training a Model
================

To train a model to distinguish "snow" from "no snow" from "too dark" for the
"canadaojp" site, use the following command. For a different site, use the
canonical site name as listed at this
`table <https://phenocam.nau.edu/webcam/network/table/>`_. For different
categories, change the class names. Note that the provided names cannot have
spaces: use underscores instead.

.. code-block:: bash
    :linenos:

    python -m phenocam_snow.train \
        canadaojp \
        --new \
        --n_train 120 \
        --n_test 30 \
        --classes snow no_snow too_dark

When you run this command, the program will download 120 random training images
from the canadaojp archive and 30 random testing images. The training images
are the examples the model learns from, and the testing images are for
evaluating the model's performance. After each set of images is downloaded,
the program will prompt you to label the images. To do so, it will create
three sub-directories inside of the current directory, each named after one of
the classes. Move the images into their corresponding directory (e.g., all the
pictures with snow should go into the "snow" directory) to label them. After
the images are labeled, they will be moved out of the sub-directories back into
the original top-level directory.

When the program has finished training the model, it will print out the model's
performance on the testing images and the file path of the saved best model.
This file path will also be written to a file called ``best_model_paths.csv``,
which will look something like this:

.. code-block:: text
    :linenos:

    timestamp,site_name,best_model
    2023-05-24T15:45:59.642605,canadaojp,/home/jason/Development/test/canadaojp_lightning_logs/version_0/checkpoints/epoch=12-step=78.ckpt

To get the model's predictions for a set of images, see the User Guide page
called "Classifying New Images".

Advanced Usage
--------------

Overriding Model and Hyperparameter Defaults
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, PhenoCamSnow only supports training models which use `ResNet <https://arxiv.org/abs/1512.03385/>`_
variants as the feature extraction backbone: ResNet-18, ResNet-34, ResNet-50,
ResNet-101, and ResNet-152. The expected format is "resnet{number}". The larger
the number, the more expressive the model, but the slower it might be to train.
PhenoCamSnow also allows you to specify learning rate, weight decay, and max
epochs hyperparameters. I have picked what I determined to be sensible
defaults, but these all should be tuned for your specific use-case. In short:

* The learning rate affects the speed at which the model learns (too small and the model will learn slowly, too high and the model might overshoot its optimal parameters).

* The weight decay affects the model's generalization abilities (too low and the model might "memorize" the training set and perform poorly on images it has not seen in training).

* The max epochs determines how many training "sessions" the model will run, at most (too few and the model might not have sufficient time to learn well, too many and the model might simply "memorize" the training set). 

Previous example with the ResNet-50 model and some hyperparameter overrides:

.. code-block:: bash
    :linenos:
    
    python -m phenocam_snow.train \
        canadaojp \
        --new \
        --n_train 120 \
        --n_test 30 \
        --classes snow no_snow too_dark
        --model resnet50
        --learning_rate 1e-3
        --weight_decay 1e-3
        --max_epochs 25

Training a Model with Already-Downloaded Images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The "new" flag in the given examples indicates that you need to download the
training and testing images. In this case, we will be getting 120 training
images and 30 testing images.

However, if you already have downloaded and labeled images, then you can train
a model like this (assuming the directories provided for ``train_dir`` and
``test_dir`` actually exist and contain the required images and labels files).

.. code-block:: bash
    :linenos:

    python -m phenocam_snow.train \
        canadaojp \
        --existing
        --train_dir 'canadaojp/train'
        --test_dir 'canadaojp/test'
        --classes snow no_snow too_dark

Getting Help
^^^^^^^^^^^^

You can see a help message displayed by running:

.. code-block:: bash
    :linenos:

    python -m phenocam_snow.train --help

If you have further questions, please raise an
`issue on the GitHub repository <https://github.com/jasonjewik/PhenoCamSnow/issues/new/choose>`_.