import numpy as np
import matplotlib.pyplot as plt

from scipy import misc
from skimage import io
from os import path, listdir

from lib.ShapeDetector import ShapeDetector
from lib.Dataset import Dataset

SIZE_EXPONENT = 3
SLICE_SIZE = 200


def main():
    # load samples
    samples = load_samples("samples", SIZE_EXPONENT)
    samples_flattened = samples.reshape((len(samples), -1))

    # load datasets, learn, predict
    datasets = load_datasets("data/datasets")

    s = ShapeDetector(datasets)
    p = s.predict(samples_flattened)

    # print results
    for index, prediction in enumerate(p):
        plt.subplot(1, len(p), index + 1)
        plt.imshow(samples[index], cmap=plt.cm.gray_r)
        plt.title(prediction)
    plt.show()


def load_datasets(directory):
    """
    load datasets
    :param directory:
    :return:
    """
    # get dataset folders in base dir
    basedir = [d for d in listdir(directory) if not path.isfile(d)]

    # load files
    sets = []
    for folder in basedir:
        # create path
        dataset_path = path.join(directory, folder)
        d = Dataset(data_path=dataset_path,
                    target=folder,
                    slice_size=SLICE_SIZE,
                    size_exponent=SIZE_EXPONENT)

        d.load_dataset()
        sets.append(d)

    return sets


def load_samples(directory, size_exponent):
    """
    loads samples from directory
    :param directory:
    :return:
    """
    # get files in directory
    files = [f for f in listdir(directory)]

    # print(files)
    samples = []
    for file in files:
        filepath = path.join(directory, file)
        img = io.imread(filepath, as_grey=True)
        img = misc.imresize(img, (2 ** size_exponent, 2 ** size_exponent)) / 16
        samples.append(img)

    return np.asarray(samples)


if __name__ == '__main__':
    main()
