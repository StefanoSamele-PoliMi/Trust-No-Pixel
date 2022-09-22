import tensorflow as tf
import eagerpy as ep
import numpy as np
import cv2
from foolbox import TensorFlowModel, utils
from PIL import Image
from tensorflow.keras.preprocessing import image
from os import path
import sys
import config


def main() -> None:
    # Same as preprocessing input of VGG16... If switching model, use the appropriate preprocessing
    """
        See on tf github with is the proprocessing mode for your architecture!!!
        mode: One of "caffe", "tf" or "torch". Defaults to "caffe".
          - caffe: will convert the images from RGB to BGR,
              then will zero-center each color channel with
              respect to the ImageNet dataset,
              without scaling.
          - tf: will scale pixels between -1 and 1,
              sample-wise.
          - torch: will scale pixels between 0 and 1 and then
              will normalize each channel with respect to the
              ImageNet dataset.
    """
    # https://github.com/jonasrauber/foolbox-native-tutorial/blob/master/foolbox-native-tutorial.ipynb
    preprocessing = config.FOOLBOX_PREP
    bounds = config.FOOLBOX_BOUNDS
    model = config.MODEL(weights="imagenet")
    model = TensorFlowModel(model, bounds=bounds, preprocessing=preprocessing)
    model = model.transform_bounds((0, 255))
    print("\nCONFIG --> Model: " + config.MODEL_NAME + ", Attack: " + config.ATTACK_NAME + "\n")
    # Load the images, for the moment we load all in the same array in memory...
    images = []
    progress = 0
    for filename in config.KEPT_IMAGE_NAMES[:config.NUM_IMG_SUBSET]:
        progress += 1
        if progress % int(config.NUM_IMG_SUBSET/10) == 0:
            print("Loading image number " + str(progress))
        if config.MODEL_NAME in ['VGG16', 'ResNet-101', 'ResNet-50']:
            img = cv2.imread(path.join(config.IMG_PATH, filename))
            height, width, _ = img.shape
            new_height = height * 256 // min(img.shape[:2])
            new_width = width * 256 // min(img.shape[:2])
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            height, width, _ = img.shape
            startx = width//2 - (224//2)
            starty = height//2 - (224//2)
            img = img[starty:starty+224,startx:startx+224]
            img = img[..., ::-1]
            images.append(img.astype(np.float32))
        else:
            img = image.load_img(path.join(config.IMG_PATH, filename), target_size=config.NET_SHAPE)
            images.append(image.img_to_array(img))

    #tf_labels = tf.constant(config.KEPT_IMAGE_LABELS[:config.NUM_IMG_SUBSET])
    #tf_images = tf.constant(images)
    #labels_batches = None
    #images_batches = None
    #with tf.device('/cpu:0'):
    #    labels_batches = tf.split(tf_labels, num_or_size_splits=config.NUM_BATCHES)
    #    images_batches = tf.split(tf_images, num_or_size_splits=config.NUM_BATCHES)

    #tf_labels = None
    #tf_images = None
    # Test the accuracy of the model... Should be 1.0
    acc = []
    batch_size = config.NUM_IMG_SUBSET // config.NUM_BATCHES
    for idx in range(0, config.NUM_BATCHES):
        if idx % int(config.NUM_BATCHES/10) == 0:
            print("Testing batch number " + str(idx) + ", partial result: " + str(np.mean(acc)))
        actual_labels = ep.astensor(tf.constant(config.KEPT_IMAGE_LABELS[batch_size*idx:batch_size*idx+batch_size]))
        actual_images = ep.astensor(tf.constant(images[batch_size*idx:batch_size*idx+batch_size]))
        acc.append(utils.accuracy(model, actual_images, actual_labels))
    print("---------------> FINAL Mean accuracy: " + str(np.mean(acc)))

    # Execute the attack
    attack = config.ATTACK()
    defeats = 0
    for idx in range(0, config.NUM_BATCHES):
        if idx % int(config.NUM_BATCHES/10) == 0:
            print("Progress: batch " + str(idx))
        actual_labels = ep.astensor(tf.constant(config.KEPT_IMAGE_LABELS[batch_size*idx:batch_size*idx+batch_size]))
        actual_images = ep.astensor(tf.constant(images[batch_size*idx:batch_size*idx+batch_size]))
        raw_advs, clipped_advs, success = attack(model, actual_images, actual_labels, epsilons=config.ATTACK_EPS)
        for i in range(0, batch_size):
            # Save all the Adversarial Examples
            Image.fromarray(clipped_advs[i].numpy().astype(np.uint8)).save(config.ADV_PATH + path.splitext(
                config.KEPT_IMAGE_NAMES[i + idx*batch_size])[0] + config.FILE_EXT)
        defeats += tf.reduce_sum(tf.cast(success.raw, tf.int32))

    print("Fake accuracy computed only on images not clipped: " + str(1 - defeats/config.NUM_IMG_SUBSET))


if __name__ == "__main__":
    main()
