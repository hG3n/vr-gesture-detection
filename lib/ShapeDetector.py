from sklearn import svm
import numpy as np


class ShapeDetector:
    def __init__(self, datasets):
        """
        c'tor
        :param datasets:
        """
        self.datasets = datasets

        images = join_datasets([d.images for d in datasets])
        targets = join_datasets([d.targets for d in datasets])

        # determine number of samples
        n_samples = len(images)
        print("number of samples %i" % n_samples)

        # reshape data
        data = images.reshape((n_samples, -1))

        # create classifier and learn from data
        self.classifier = svm.SVC(gamma=0.001)
        self.classifier.fit(data, targets)

    def predict(self, targets):
        """
        predict input targets using svc's predict function
        :param targets:
        :return:
        """
        return self.classifier.predict(targets)


def join_datasets(datasets):
    """
    join datasets into one
    :param datasets:
    :return:
    """
    first = datasets[0]
    for i in range(1, len(datasets)):
        first += datasets[i]
    return np.asarray(first)
