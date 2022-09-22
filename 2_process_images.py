from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from skimage.feature import peak_local_max
from scipy import ndimage
from os import path
import sys
import config

model = config.MODEL(weights='imagenet')
progress = 0
tot = str(len(config.KEPT_IMAGE_NAMES[:config.NUM_IMG_SUBSET]))

print("\nCONFIG --> Model: " + config.MODEL_NAME + ", Attack: " + config.ATTACK_NAME + "\n")

for filename in config.KEPT_IMAGE_NAMES[:config.NUM_IMG_SUBSET]:
    progress += 1
    if progress % int(int(tot)/10) == 0:
        print("Progress: " + str(progress) + " over " + tot + " images.")
    img = image.load_img(path.join(config.ADV_PATH, path.splitext(filename)[0] + config.FILE_EXT), target_size=config.NET_SHAPE)
    img_array = config.PREPROCESSING(np.expand_dims(image.img_to_array(img), axis=0))
    preds = model.predict(img_array)

    # Generate Class Activation heatMaps for the top classes
    sorted_args = np.argsort(preds, axis=1)
    heatmaps = []
    for i in range(1, config.N+1):
        index = -1 * i
        heatmaps.append(config.CAM_FUNC(
            model=model, img=np.squeeze(img_array), layer_name=config.LAYER_NAME, category_id=sorted_args[0][index]))

    # Load image in its original size, process it and reshape the CAM accordingly and save the image
    img_clean = keras.preprocessing.image.load_img(path.join(config.ADV_PATH, path.splitext(filename)[0] + config.FILE_EXT))
    img_clean = keras.preprocessing.image.img_to_array(img_clean)
    img_clean = np.array(Image.fromarray(img_clean.astype(np.uint8)).resize(config.GAN_SHAPE))
    mask = np.zeros(config.GAN_SHAPE, dtype=bool)
    for hm in heatmaps:
        cam = np.array(Image.fromarray((hm * 255).astype(np.uint8)).resize(config.GAN_SHAPE))
        # coordinates = something like [[5,2], [7,2]]   # a[5][2] ---> the maximum <--- a[tuple(coordinates[0].T)]
        coordinates = peak_local_max(cam, min_distance=2*config.W + 1)
        # process the coordinates array to keep only p holes
        coordinates = coordinates[:config.P]
        out = np.zeros_like(cam, dtype=bool)
        out[tuple(coordinates.T)] = True  # now out is a binary mask that is True only on the peak
        struct = np.ones(config.K).astype(dtype=bool)
        out = ndimage.binary_dilation(out, structure=struct).astype(out.dtype)
        mask = np.logical_or(mask, out)

    # Save the image with holes and the mask, same size as image
    mask3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    emptied_image = img_clean * np.logical_not(mask3d) + mask3d.astype(np.uint8) * 255
    Image.fromarray(mask.astype(np.uint8) * 255).save(config.MASK_PATH + path.splitext(filename)[0] + config.FILE_EXT)
    Image.fromarray(emptied_image.astype(np.uint8)).save(config.HOLED_PATH +
                                                         path.splitext(filename)[0] + config.FILE_EXT)
