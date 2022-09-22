from tensorflow import keras
import numpy as np
from PIL import Image
from os import path
from skimage.restoration import denoise_wavelet
import sys
import config

# Wavelet denoiser, set with parameters of the original paper
def denoiser(img):
    return denoise_wavelet(img / 255.0, sigma=0.01, mode='soft', multichannel=True, convert2ycbcr=True,
                           method='BayesShrink') * 255.0

model = config.MODEL(weights='imagenet')
print("\nCONFIG --> Model: " + config.MODEL_NAME + ", Attack: " + config.ATTACK_NAME + "\n")
correct = [0, 0] if config.TRY_ALSO_FULLINPAINT else [0]
for i in range(len(config.KEPT_IMAGE_NAMES[:config.NUM_IMG_SUBSET])):
    filename = config.KEPT_IMAGE_NAMES[i]

    if not config.TRY_ONLY_FULLINPAINT:
        # Denoise the original input image
        img_clean = keras.preprocessing.image.load_img(path.join(config.ADV_PATH, path.splitext(filename)[0] + config.FILE_EXT))
        img_clean = keras.preprocessing.image.img_to_array(img_clean)
        img_clean_denoised = denoiser(img_clean).astype(np.uint8)
        img_clean_denoised = np.array(Image.fromarray(img_clean_denoised.astype(np.uint8)).resize(config.GAN_SHAPE))
        mask = np.array(Image.open(config.MASK_PATH + path.splitext(filename)[0] + config.FILE_EXT)).astype(bool)
        mask3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    # After having used the GAN to inpaint...
    img_impainted = np.array(Image.open(config.INPAINTED_PATH + path.splitext(filename)[0] + config.FILE_EXT))

    for ctr in range(len(correct)):
        # Fuse the denoised image with the inpainted regions (in this way the inpainted region are not denoised)
        if config.TRY_ONLY_FULLINPAINT:
            finalimage = img_impainted
        elif config.INPAINT_BKGR:
            finalimage = img_impainted * np.logical_not(
                mask3d) + img_clean_denoised * mask3d if ctr == 0 else img_impainted
        else:
            finalimage = img_impainted * mask3d + img_clean_denoised * np.logical_not(
                mask3d) if ctr == 0 else img_impainted

        finalimage = config.PREPROCESSING(np.array(Image.fromarray(finalimage.astype(np.uint8)).resize(config.NET_SHAPE)))

        # Show results for what concerns the top-5 accuracy
        preds = model.predict(np.expand_dims(finalimage, axis=0))
        dec_pred = config.DECODING(preds, top=1)[0]
        if list(config.CLASSES_DICT.keys()).index(dec_pred[0][0]) == config.KEPT_IMAGE_LABELS[i]:
            correct[ctr] += 1
        if i % 1000 == 0:
            print("Processed " + str(i) + " images.")

for ctr in range(len(correct)):
    print("\n\nFrom " + str(len(config.KEPT_IMAGE_NAMES[:config.NUM_IMG_SUBSET])) + " files, " + str(correct[ctr]) + " were correctly classified.\n\n")
    print("Ctr" + str(ctr) + "-accuracy: " + str(correct[ctr] / len(config.KEPT_IMAGE_NAMES[:config.NUM_IMG_SUBSET])))
