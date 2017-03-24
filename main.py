from skimage import color, io, transform
from scipy import misc
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics

# globals
IMAGE_PATH = "data/"


def main():
    d = load_dataset('circle.jpeg', 200, 3)
    d2 = load_dataset('triangle.jpeg', 200, 3)

    # for item in d:
    #     plt.imshow(item, cmap=plt.cm.gray_r, interpolation='nearest')
    #     plt.show()

    digits = datasets.load_digits()

    # 8bit subplot
    plt.subplot(1,2,1)
    plt.imshow(d[0], cmap=plt.cm.gray_r)
    plt.title("8bit")

    # 4bit subplot
    plt.subplot(1,2,2)
    plt.imshow(d[0]/16, cmap=plt.cm.gray_r)
    plt.title("4bit")

    # show the plot
    plt.show()

    foo = list(zip(digits.images, digits.target))
    print(foo[0])



def load_dataset(file, slice_size, size_exponent):
    """
    load images from file and return dataset
    :param file:
    :param slice_size:
    :param size_exponent:
    :return:
    """
    # load image atlas as greyscale
    circle_path = IMAGE_PATH + file
    print("loading image: %s" % circle_path)
    atlas = io.imread(circle_path, as_grey=True)

    # check atlas size
    rows = atlas.shape[0]
    cols = atlas.shape[1]
    if (rows % slice_size != 0 or cols % slice_size != 0):
        print("ERROR: wrong image dimensions, should be multiple of 200")
        return

    # segment atlas to single images
    segmented_images = []
    for r in range(0, int(rows / slice_size)):
        for c in range(0, int(cols / slice_size)):
            img = atlas[r * slice_size: r * slice_size + slice_size, c * slice_size: c * slice_size + slice_size]
            segmented_images.append(img)

    # define new size and resize images
    new_size = (2 ** size_exponent, 2**size_exponent)
    for i in range(0, len(segmented_images)):
        # segmented_images[i] = transform.resize(segmented_images[i], new_size)
        segmented_images[i] = misc.imresize(segmented_images[i], new_size)

    ## return segmented image array
    return segmented_images


if __name__ == '__main__':
    main()
