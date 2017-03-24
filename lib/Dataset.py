from scipy import misc
from skimage import io


class Dataset:
    def __init__(self, file, target, slice_size, size_exponent):
        """
        c'tor
        :param file:
        :param target:
        :param slice_size:
        :param size_exponent:
        """
        self.file = file
        self.target = target
        self.slice_size = slice_size
        self.size_exponent = size_exponent
        self.images = None
        self.targets = None

    def load_dataset(self):
        """
        load single dataset and store in obj
        :return:
        """
        # load image atlas as greyscale
        print("loading image: %s" % self.file)
        atlas = io.imread(self.file, as_grey=True)

        # check atlas size
        rows = atlas.shape[0]
        cols = atlas.shape[1]
        if rows % self.slice_size != 0 or cols % self.slice_size != 0:
            print("ERROR: wrong image dimensions, should be multiple of 200")
            return

        # segment atlas to single images
        segmented_images = []
        image_targets = []
        for r in range(0, int(rows / self.slice_size)):
            for c in range(0, int(cols / self.slice_size)):
                r_start = r * self.slice_size
                r_end = r * self.slice_size + self.slice_size
                c_start = c * self.slice_size
                c_end = c * self.slice_size + self.slice_size

                img = atlas[r_start:r_end, c_start:c_end]

                segmented_images.append(img)
                image_targets.append(self.target)

        # define new size and resize images
        new_size = (2 ** self.size_exponent, 2 ** self.size_exponent)
        for i in range(0, len(segmented_images)):
            # segmented_images[i] = transform.resize(segmented_images[i], new_size)
            segmented_images[i] = misc.imresize(segmented_images[i], new_size) / 16

        # return segmented image array
        self.images = segmented_images
        self.targets = image_targets

    def __str__(self):
        """
        overloaded print function
        :return:
        """
        return "<Dataset | file: " + self.file + " | " + self.target + ">"
