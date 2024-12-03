Classifying New Images
======================
Models produced by PhenoCamSnow can be used to generate predictions for all
images in a local directory, or to generate a prediction for images pointed to
by their URLs. 

Local Prediction
----------------

To get predictions for a local directory of canadaojp images, use the
following command.

.. code-block:: bash
    :linenos:

    python -m phenocam_snow.predict \
        canadaojp \
        canadaojp_lightning_logs/version_0/checkpoints/epoch=12-step=78.ckpt \
        --categories snow no_snow too_dark \
        --directory 'canadaojp/test'

The file path provided in the third line is printed by the model training
script. It is also saved in a file called ``best_model_paths.csv``. Ensure that
the categories provided are the same name and in the same order as provided
during training.

The program will then print out its predictions to a CSV file that looks like
this:

.. code-block:: text
    :linenos:

    # Site: canadaojp
    # Categories:
    # 0. snow
    # 1. no_snow
    # 2. too_dark
    filename,label
    canadaojp_2020_04_25_052959.jpg,snow
    canadaojp_2019_11_14_073000.jpg,snow
    canadaojp_2020_01_08_155959.jpg,snow
    canadaojp_2018_10_06_182959.jpg,no_snow
    canadaojp_2019_05_30_040000.jpg,no_snow
    canadaojp_2018_10_23_142959.jpg,no_snow
    canadaojp_2018_04_27_083000.jpg,no_snow
    canadaojp_2020_08_05_100000.jpg,no_snow
    canadaojp_2020_07_13_215959.jpg,no_snow
    canadaojp_2020_04_26_045959.jpg,no_snow
    canadaojp_2020_07_01_205959.jpg,no_snow
    canadaojp_2018_06_01_082959.jpg,no_snow
    canadaojp_2020_05_30_065959.jpg,no_snow
    canadaojp_2020_06_11_033000.jpg,no_snow
    canadaojp_2019_05_10_112959.jpg,no_snow
    canadaojp_2020_04_09_232959.jpg,too_dark
    canadaojp_2018_10_31_182959.jpg,too_dark
    canadaojp_2019_08_31_000000.jpg,too_dark
    canadaojp_2020_02_01_075959.jpg,too_dark
    canadaojp_2018_12_18_185959.jpg,too_dark
    canadaojp_2020_03_05_012959.jpg,too_dark
    canadaojp_2020_04_22_035959.jpg,too_dark
    canadaojp_2020_05_02_032959.jpg,too_dark
    canadaojp_2018_12_29_052959.jpg,too_dark
    canadaojp_2020_04_24_025959.jpg,too_dark
    canadaojp_2020_04_11_232959.jpg,too_dark
    canadaojp_2018_11_20_065959.jpg,too_dark
    canadaojp_2019_02_25_192959.jpg,too_dark
    canadaojp_2020_03_30_232959.jpg,too_dark
    canadaojp_2018_06_15_003000.jpg,too_dark


Online Prediction
-----------------

PhenoCamSnow is also capable of generating a prediction for online images as
as pointed to by their URLs. For example, using canadaojp again:

.. code-block:: bash
    :linenos:

    python -m phenocam_snow.predict \
        canadaojp \
        canadaojp_lightning_logs/version_0/checkpoints/epoch=12-step=78.ckpt \
        resnet18 \
        --categories snow no_snow too_dark
        --urls urls.txt

Here, ``urls.txt`` is a file containing one image URL per line. The program
will write its predictions to a CSV file, formatted similarly to the one shown
for offline prediction.

Getting Help
------------

You can see a help message displayed by running:

.. code-block:: bash
    :linenos:

    python -m phenocam_snow.predict --help


If you have further questions, please raise an
`issue on the GitHub repository <https://github.com/jasonjewik/PhenoCamSnow/issues/new/choose>`_.