"""
A script to train/evaluate/run a Linear SVM that classifies "bad images" (i.e., 
a human cannot reliably identify the image as having snow or not, due to 
lighting conditions) from "good images".
"""

# Standard library imports
from argparse import ArgumentParser
import joblib
import pandas as pd
from pathlib import Path
import sys

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# Local application imports
from sat_meanvar import sat_meanvar
import utils


def main():
    parser = ArgumentParser(
        description='trains/evaluates/runs a saturation classifier')
    parser.add_argument('--train', action='store_true', default=False,
                        help='train a new model: meanvar, labels, and output \
                            must be given')
    parser.add_argument('--eval', action='store', metavar='MODEL',
                        help='evaluates the given model: labels, images must \
                            be given; output is ignored')
    parser.add_argument('--predict', action='store', metavar='MODEL',
                        help='classifies images using the given model: images \
                            and output must be given')
    parser.add_argument('--meanvar', action='store',
                        help='the csv file containing the image means and \
                            variances (see sat_meanvar.py)')
    parser.add_argument('--labels', action='store',
                        help='the csv file containing the image labels (see \
                            label_data.py)')
    parser.add_argument('--images', action='store',
                        help='the path to the image directory')
    parser.add_argument('-o', '--output', action='store',
                        help='see the train/predict options help dialogue')
    args = parser.parse_args()

    run_training = int(args.train)
    run_eval = int(args.eval is not None)
    run_predict = int(args.predict is not None)

    if run_training + run_eval + run_predict > 1:
        utils.eprint('specify just one of train, eval, or predict')

    elif args.train:
        meanvar_csv = utils.validate_file(
            args.meanvar, extension='.csv', panic_on_overwrite=False)
        label_csv = utils.validate_file(
            args.labels, extension='.csv', panic_on_overwrite=False)

        meanvar_df = pd.read_csv(meanvar_csv)
        label_df = pd.read_csv(label_csv)
        joint_df = meanvar_df.merge(label_df)
        X = joint_df[['mean', 'variance']].to_numpy()
        y = joint_df['label'].to_numpy()

        train(X, y, args.output)

    elif args.eval is not None:
        model_path = utils.validate_file(
            args.eval, extension='.joblib', panic_on_overwrite=False)
        label_csv = utils.validate_file(
            args.labels, extension='.csv', panic_on_overwrite=False)
        images_dir = utils.validate_directory(args.images)

        model = joblib.load(model_path)
        label_df = pd.read_csv(label_csv)

        evaluate(model, images_dir, label_df)

    elif args.predict is not None:
        model_path = utils.validate_file(
            args.predict, extension='.joblib', panic_on_overwrite=False)
        images_dir = utils.validate_directory(args.images)
        outpath = utils.validate_file(args.output, extension='.csv')

        model = joblib.load(model_path)
        df = predict(model, images_dir)
        df.to_csv(outpath, index=False)

    else:
        utils.eprint('you need to specify one of train, eval, or predict')


def train(X: np.ndarray, y: np.ndarray, outpath: str, test_size: float = 0.33):
    """ Trains and saves a classification model. """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size)

    clf = make_pipeline(
        StandardScaler(),
        LinearSVC()
    )
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f'Accuracy Score: {acc}')
    f1 = f1_score(y_test, predictions)
    print(f'F1 Score: {f1}')

    outpath = utils.validate_file(outpath, extension='.joblib')
    with open(outpath, 'wb') as f:
        joblib.dump(clf, f)
    print(f'Saved model to {outpath}')


def evaluate(model: LinearSVC, image_dir: Path, labels: pd.DataFrame):
    """ Evaluates the given classification model. """
    meanvars = []
    indices_to_drop = []
    pgbar = utils.ProgressBar(len(labels))

    print('processing images')
    for idx, im_name in enumerate(labels['image']):
        pgbar.display()
        im_path = image_dir.joinpath(im_name)
        if not im_path.exists():
            utils.warn(f'{im_path} does not exist, skipping')
            indices_to_drop.append(idx)
        else:
            im = plt.imread(im_path)
            meanvars.append(sat_meanvar(im))
        pgbar.inc()
        pgbar.display()

    y_true = labels['label'].drop(indices_to_drop)
    y_pred = model.predict(meanvars)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f'Accuracy: {acc}')
    print(f'F1 Score: {f1}')


def predict(model: LinearSVC, image_dir: Path) -> pd.DataFrame:
    """ Output predicted class for each image in the given directory. """
    im_names = []
    meanvars = []
    pgbar = utils.ProgressBar(len(sorted(image_dir.glob('*.jpg'))))

    print('processing images')
    for fp in image_dir.glob('*.jpg'):
        pgbar.display()
        im = plt.imread(fp)
        im_names.append(fp.name)
        meanvars.append(sat_meanvar(im))
        pgbar.inc()
        pgbar.display()

    predictions = model.predict(meanvars)
    df = pd.DataFrame({
        'image': im_names,
        'label': predictions
    })
    return df


if __name__ == '__main__':
    main()