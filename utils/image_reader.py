
import os
import random
from collections import namedtuple
import matplotlib.pyplot as plt
from scipy import misc
import click
import numpy as np
from IPython import embed
from keras.preprocessing.image import (
    load_img, img_to_array,
    flip_axis)

# The set of parameters that describes an instance of
# (random) augmentation
TransformParams = namedtuple(
    'TransformParameters',
    ('do_hor_flip', 'do_vert_flip'))

pascal_mean = np.array([102.93, 111.36, 116.52])

label_margin = 186


debug = True








def load_img_array(fname, grayscale=False, target_size=None, dim_ordering='default', data_format=None):
    """Loads and image file and returns an array."""
    img = load_img(fname,
                   grayscale=grayscale,
                   target_size=target_size)
    #x = img_to_array(img, dim_ordering=dim_ordering)
    x = img_to_array(img, data_format=data_format)
    #print("shape of image",x.shape)
    return x


class RandomTransformer:
    """To consistently add data augmentation to image pairs, we split the process in
    two steps. First, we generate a stream of random augmentation parameters, that
    can be zipped together with the images. Second, we do the actual transformation,
    that has no randomness since the parameters are passed in."""

    def __init__(self,
                 horizontal_flip=False,
                 vertical_flip=False):
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

    def random_params_gen(self) -> TransformParams:
        """Returns a generator of random transformation parameters."""
        while True:
            do_hor_flip = self.horizontal_flip and (np.random.random() < 0.5)
            do_vert_flip = self.vertical_flip and (np.random.random() < 0.5)

            yield TransformParams(do_hor_flip=do_hor_flip,
                                  do_vert_flip=do_vert_flip)

    @staticmethod
    def transform(x: np.array, params: TransformParams) -> np.array:
        """Transforms a single image according to the parameters given."""
        if params.do_hor_flip:
            x = flip_axis(x, 1)

        if params.do_vert_flip:
            x = flip_axis(x, 0)

        return x


class SegmentationDataGenerator:
    """A data generator for segmentation tasks, similar to ImageDataGenerator
    in Keras, but with distinct pipelines for images and masks.

    The idea is that this object holds no data, and only knows how to run
    the pipeline to load, augment, and batch samples. The actual data (csv,
    numpy, etc..) must be passed in to the fit/flow functions directly."""

    skipped_count = 0

    def __init__(self,
                 random_transformer: RandomTransformer):
        self.random_transformer = random_transformer

    def get_processed_pairs(self,
                            img_fnames,
                            mask_fnames):
        # Generators for image data
        img_arrs = (load_img_array(f) for f in img_fnames)
        mask_arrs = (load_img_array(f, grayscale=True) for f in mask_fnames)

        # if debug == True:
        #     for count,(image, label) in enumerate(zip(img_arrs, mask_arrs)):
        #         print("count is: ",count)
        #         print("shape of image",image.shape)
        #         print("shape of label",label.shape)
        def add_context_margin(image, margin_size, **pad_kwargs):
            """ Adds a margin-size border around the image, used for
            providing context. """
            return np.pad(image,
                          ((margin_size, margin_size),
                           (margin_size, margin_size),
                           (0, 0)), **pad_kwargs)

        def pad_to_square(image, min_size, **pad_kwargs):
            """ Add padding to make sure that the image is larger than (min_size * min_size).
            This time, the image is aligned to the top left corner. """

            h, w = image.shape[:2]

            if h >= min_size and w >= min_size:
                return image

            top = bottom = left = right = 0

            if h < min_size:
                top = (min_size - h) // 2
                bottom = min_size - h - top
            if w < min_size:
                left = (min_size - w) // 2
                right = min_size - w - left

            return np.pad(image,
                          ((top, bottom),
                           (left, right),
                           (0, 0)), **pad_kwargs)

        def pad_image(image):
            image_pad_kwargs = dict(mode='reflect')
            image = add_context_margin(image, label_margin, **image_pad_kwargs)
            return pad_to_square(image, 256, **image_pad_kwargs)

        def pad_label(image):
            # Same steps as the image, but the borders are constant white
            label_pad_kwargs = dict(mode='constant', constant_values=255)
            image = add_context_margin(image, label_margin, **label_pad_kwargs)
            return pad_to_square(image, 256, **label_pad_kwargs)

        pairs = ((pad_image(image), pad_label(label)) for
                 image, label in zip(img_arrs, mask_arrs))
        #
        # if debug == True:
        #     for count,(image, label) in enumerate(pairs):
        #         print("count is: ",count)
        #         print("shape of image",image.shape)
        #         print("shape of label",label.shape)

        # random/center crop
        def crop_to(image, target_h=256, target_w=256):
            # TODO: random cropping
            h_off = (image.shape[0] - target_h) // 2
            w_off = (image.shape[1] - target_w) // 2
            return image[h_off:h_off + target_h,
                   w_off:w_off + target_w, :]

        pairs = ((crop_to(image), crop_to(label)) for
                 image, label in zip(img_arrs, mask_arrs))

        # if debug == True:
        #     for count,(image, label) in enumerate(pairs):
        #         print("count is: ",count)
        #         print("shape of image",image.shape)
        #         print("shape of label",label.shape)


        # random augmentation
        augmentation_params = self.random_transformer.random_params_gen()
        transf_fn = self.random_transformer.transform
        # pairs = ((transf_fn(image, params), transf_fn(label, params)) for
        #          ((image, label), params) in zip(pairs, augmentation_params))

        # if debug == True:
        #     for count,(image, label) in enumerate(pairs):
        #         print("After augmentation")
        #         print("count is: ",count)
        #         print("shape of image",image.shape)
        #         print("shape of label",label.shape)
        def rgb_to_bgr(image):
            # Swap color channels to use pretrained VGG weights
            return image[:, :, ::-1]

        pairs = ((rgb_to_bgr(image), rgb_to_bgr(label)) for
                 image, label in pairs)

        def remove_mean(image):
            # Note that there's no 0..1 normalization in VGG
            return image - pascal_mean

        pairs = ((remove_mean(image), label) for
                 image, label in pairs)
        # for count,(image, label) in enumerate(pairs):
        #     print("count is: ",count)
        #     print("shape of image",image.shape)
        #     print("shape of label",label.shape)
        def slice_label(image, offset, label_size, stride):
            # Builds label_size * label_size pixels labels, starting from
            # offset from the original image, and stride stride
            # if debug==True:
            #     print("Shape of resized image",image.shape)
            return image[offset:offset + label_size * stride:stride,
                   offset:offset + label_size * stride:stride]

        # pairs = ((image, slice_label(label, label_margin, 256, 1)) for
        #          image, label in pairs)

        # if debug == True:
        #     for count,(image, label) in enumerate(pairs):
        #         print("count is: ",count)
        #         print("shape of image",image.shape)
        #         print("shape of label",label.shape)

        return pairs

    def flow_from_list(self,
                       img_fnames,
                       mask_fnames,
                       batch_size,
                       img_target_size,
                       mask_target_size,
                       shuffle=False):
        assert batch_size > 0

        paired_fnames = list(zip(img_fnames, mask_fnames))

        while True:
            # Starting a new epoch..
            if shuffle:
                random.shuffle(paired_fnames)  # Shuffles in place
            img_fnames, mask_fnames = zip(*paired_fnames)

            pairs = self.get_processed_pairs(img_fnames, mask_fnames)

            i = 0
            img_batch = np.zeros((batch_size, img_target_size[0], img_target_size[1], 3))
            mask_batch = np.zeros((batch_size, mask_target_size[0] * mask_target_size[1], 1))
            #debug information
            # if debug == True:
            #     print("image batch size is: ",img_batch.shape)
            #     print("mask batch size is: ",mask_batch.shape)
                #print("shape of processed pairs",pairs.shape )
            for img, mask in pairs:
                # Fill up the batch one pair at a time
                img_batch[i] = img
                # Pass the label image as 1D array to avoid the problematic Reshape
                # layer after Softmax (see model.py)
                if debug==True:
                    print("Size of mask is: ",mask.shape)
                    #print(np.array(mask))
                mask_batch[i] = np.reshape(mask, (-1, 1))

                # TODO: remove this ugly workaround to skip pairs whose mask
                # has non-labeled pixels.
                if 255. in mask:
                    self.skipped_count += 1
                    continue

                i += 1
                if i == batch_size:
                    i = 0
                    if debug==True:
                        print("Size of batch is: ",mask_batch.shape)
                    yield img_batch, mask_batch


@click.command()
@click.option('--list-fname', type=click.Path(exists=True),
              default='/mnt/pascal_voc/benchmark_RELEASE/dataset/train.txt')
@click.option('--img-root', type=click.Path(exists=True),
              default='/mnt/pascal_voc/benchmark_RELEASE/dataset/img')
@click.option('--mask-root', type=click.Path(exists=True),
              default='/mnt/pascal_voc/benchmark_RELEASE/dataset/pngs')
def test_datagen(list_fname, img_root, mask_root):
    datagen = SegmentationDataGenerator()

    basenames = [l.strip() for l in open(list_fname).readlines()]
    img_fnames = [os.path.join(img_root, f) + '.jpg' for f in basenames]
    mask_fnames = [os.path.join(mask_root, f) + '.png' for f in basenames]

    datagen.flow_from_list(img_fnames, mask_fnames)


if __name__ == '__main__':
    test_datagen()
